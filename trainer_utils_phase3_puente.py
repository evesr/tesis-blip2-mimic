#!/usr/bin/env python3
"""
trainer_utils_phase3_puente.py — Utilidades de entrenamiento para Fase 3 (Puente)
==================================================================================
Re-entrenamiento del COMPONENTE PUENTE de BLIP-2 (Q-Former + language_projection)
sobre el modelo de Fase 2 (dual-LoRA LLM+ViT) ya FUSIONADO permanentemente.

Diferencias clave respecto a `trainer_utils.py` (Fase 2)
--------------------------------------------------------
  • Carga del modelo: NO inyecta LoRA nueva. Carga los adaptadores LoRA de
    Fase 2 y los FUSIONA (merge_and_unload) en los pesos base de ViT y LLM.
  • Componentes entrenables:
        - vision_model            → CONGELADO (LoRA fusionado, requires_grad=False)
        - language_model          → CONGELADO (LoRA fusionado, requires_grad=False)
        - qformer                 → ENTRENABLE (Full FT, sin LoRA)
        - language_projection     → ENTRENABLE (Full FT, sin LoRA)
  • Checkpoints: el `BridgeOnlyTrainer` guarda un .pt con SOLO los parametros
    entrenables (~ 250 MB) en lugar del modelo completo (~ 7 GB).

Responsabilidades
-----------------
1. preparar_modelo_fase_puente(ruta_base, ruta_lora_vit, ruta_lora_llm)
       Función EXPLÍCITA solicitada en el enunciado (ver docstring abajo).

2. build_bridge_optimizer(model)
       AdamW con un solo grupo (qformer + language_projection),
       lr = 1e-4, weight_decay = 0.01, fused.

3. Phase3PuenteAuditCallback
       Guarda el state_dict del puente solo si val_loss mejora.
       Registra history_phase3_{name}.csv y audit_phase3_{name}.txt.

4. BridgeOnlyTrainer (subclase de LengthAwareTrainer)
       Sobreescribe save_model / _load_from_checkpoint / _load_best_model
       para persistir UNICAMENTE qformer + language_projection.

5. train_puente_config()
       Función principal: recibe spec de config, entrena y devuelve dict-resumen.

Autor : Evelyn Silva Rozas
Fecha : Junio 2026
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0. Variables de entorno (ANTES de imports de transformers/tokenizers/torch)
# ─────────────────────────────────────────────────────────────────────────────
import os as _os
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
_os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import gc
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# cv2 DEBE desactivarse ANTES de cualquier import de torch/DataLoader
import cv2
cv2.setNumThreads(0)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch.optim import AdamW
from transformers import (
    Blip2ForConditionalGeneration,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from peft import PeftModel
# (PHASE3 PUENTE) Desactivado: la inferencia de texto en val ya no se ejecuta
# durante el entrenamiento, así que NLTK/Rouge dejaron de ser dependencias
# del callback. Imports preservados como comentario para reactivar fácilmente.
# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
# from rouge_score import rouge_scorer

from config import config
from data_loader import MimicCXRDataset
from model_utils import cargar_procesador

# Reusar utilidades de Fase 2: monkey-patch del ViT (whitelist), TF32,
# precompute_train_lengths y la base LengthAwareTrainer (custom samplers
# + group_by_length sin I/O de imágenes).
from trainer_utils import (
    LengthAwareTrainer,
    precompute_train_lengths,
    # (PHASE3 PUENTE) Desactivado: helpers de generación textual no se usan
    # en el callback de Fase 3 (eval == solo cross-entropy loss).
    # _bleu4,
    # _rougeL,
    # _generate_sample,
)

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES Y RUTAS
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR = config.paths.base_dir

# Datasets del 50 % (mismos splits que Fase 2)
TRAIN_CSV_PUENTE  = BASE_DIR / "train_split_50pct.csv"
VAL_CSV_PUENTE    = BASE_DIR / "val_split_50pct.csv"
TEST_CSV_PUENTE   = BASE_DIR / "test_split_50pct.csv"
IMAGES_DIR_PUENTE = BASE_DIR / "imagenes_50_porciento"

# Pesos LoRA de Fase 2 (modelo ganador Config_B1: LLM r=32, ViT r=8)
PHASE2_BEST_DIR: Path = BASE_DIR / "phase2_results-50pct" / "checkpoints" / "best_model_Config_B1"
PHASE2_LLM_DIR: Path  = PHASE2_BEST_DIR / "llm"
PHASE2_VIT_DIR: Path  = PHASE2_BEST_DIR / "vit"

# Directorio raíz de salida para Fase 3 (Puente)
OUTPUT_DIR: Path      = BASE_DIR / "phase3_puente_results-50pct"
CHECKPOINTS_DIR: Path = OUTPUT_DIR / "checkpoints"
EVAL_OUTPUT_DIR: Path = OUTPUT_DIR / "evaluation"


# ─────────────────────────────────────────────────────────────────────────────
# HIPERPARÁMETROS (Full FT sobre puente liviano: Q-Former 187M + lang_proj 5M)
# ─────────────────────────────────────────────────────────────────────────────

TRAINING_CONFIG: Dict = {
    # Una sola tasa: el puente NO tenía adaptadores; es Full FT desde pesos base.
    # 1e-4 es el mismo LR usado para los LoRA de OPT en Fases 1 y 2 (continuidad).
    "lr_bridge":              1e-4,
    "weight_decay":           0.01,
    # Régimen
    "max_epochs":             20,    # Q-Former + proj convergen más rápido que LoRA dual
    "early_stopping_patience": 8,
    # batch=12, grad_accum=4 → batch EFECTIVO = 48
    # Reducción vs Fase 2 (16×4=64): el Full FT del Q-Former mantiene activaciones
    # del transformer entero del Q-Former (187M params) en backward, así que
    # batch=16 podría OOM con gradient_checkpointing=False.
    "batch_size":             12,
    "gradient_accumulation":  4,
    "eval_steps":             625,
    "warmup_ratio":           0.1,
    # (PHASE3 PUENTE) Desactivado: la inferencia textual en val durante el
    # entrenamiento se quitó. Los CSVs ahora registran SOLO Train_Loss y
    # Val_Loss (la calidad textual se mide en evaluate_phase3_puente.py
    # tras el entrenamiento, igual que en Fase 2).
    # "audit_samples":          10,
    # "max_new_tokens":         400,
}

# Configuración única de Fase 3 (no es un torneo, es un único modelo final)
PUENTE_CONFIG: Dict = {
    "name":           "Phase3_Puente",
    "llm_lora_dir":   PHASE2_LLM_DIR,
    "vit_lora_dir":   PHASE2_VIT_DIR,
    "base_model":     config.model.model_name,
}


# ─────────────────────────────────────────────────────────────────────────────
# 1.  FUNCIÓN PRINCIPAL: preparar_modelo_fase_puente
# ─────────────────────────────────────────────────────────────────────────────

def preparar_modelo_fase_puente(
    ruta_base: str,
    ruta_lora_vit: Union[str, Path],
    ruta_lora_llm: Union[str, Path],
) -> Blip2ForConditionalGeneration:
    """
    Construye el modelo BLIP-2 listo para Fase 3 (Re-entrenamiento del Puente).

    Pasos (estrictamente en este orden):
        1. Carga base : Blip2ForConditionalGeneration desde `ruta_base`
                        en torch.bfloat16, device_map='auto'.
        2. Fusión ViT : Aplica el adaptador LoRA de `ruta_lora_vit` al
                        vision_model y ejecuta merge_and_unload() para
                        fusionar los deltas en los pesos base de forma
                        permanente (devuelve un Blip2VisionModel limpio,
                        sin wrapper PeftModel).
        3. Fusión LLM : Aplica el adaptador LoRA de `ruta_lora_llm` al
                        language_model y ejecuta merge_and_unload(). Tras
                        esto, el LLM es un OPTForCausalLM sin wrapper LoRA.
        4. Congelar extremos : requires_grad = False para todos los
                        parámetros de vision_model y language_model.
        5. Liberar puente : requires_grad = True para qformer y
                        language_projection (Full FT, sin LoRA nueva).
        6. Auditoría : Imprime conteo de parámetros entrenables vs congelados
                        y verifica que la configuración es la esperada.

    Args:
        ruta_base     : Nombre o path del modelo base (ej. "Salesforce/blip2-opt-2.7b").
        ruta_lora_vit : Directorio con el adaptador LoRA del vision_model
                        (debe contener adapter_config.json y adapter_model.*).
        ruta_lora_llm : Directorio con el adaptador LoRA del language_model.

    Returns:
        Blip2ForConditionalGeneration con:
            • ViT y LLM fusionados con sus respectivos LoRA, ambos congelados.
            • Q-Former y language_projection con requires_grad=True.

    Raises:
        FileNotFoundError : Si alguno de los directorios LoRA no existe.
    """
    ruta_lora_vit = Path(ruta_lora_vit)
    ruta_lora_llm = Path(ruta_lora_llm)
    if not ruta_lora_vit.exists():
        raise FileNotFoundError(f"Adaptador LoRA del ViT no encontrado: {ruta_lora_vit}")
    if not ruta_lora_llm.exists():
        raise FileNotFoundError(f"Adaptador LoRA del LLM no encontrado: {ruta_lora_llm}")

    # ── Paso 1: Carga base en bfloat16 ───────────────────────────────────────
    print(f"   [puente] Paso 1 · Cargando modelo base BLIP-2 (bfloat16, device_map='auto')...")
    print(f"   [puente]            base = {ruta_base}")
    base_model = Blip2ForConditionalGeneration.from_pretrained(
        ruta_base,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        # max_memory: limitar el buffer de Accelerate a 14 GB (más que el
        # modelo, ~7.8 GB) deja ~33 GB libres para activaciones del Q-Former
        # full FT + LLM forward, evitando OOM sin gradient_checkpointing.
        max_memory={0: "14GiB"},
    )

    # qformer es un submódulo BASE que nunca se toca → referencia confiable
    # de device/dtype del modelo. Se usa para forzar el dtype/device tras
    # los merge() (los pesos LoRA viven en float32 y podrían upcastear).
    _ref = next(base_model.qformer.parameters())
    target_device, target_dtype = _ref.device, _ref.dtype
    print(f"   [puente]            device={target_device}  dtype={target_dtype}")

    # ── Paso 2: Fusión del ViT ───────────────────────────────────────────────
    print(f"\n   [puente] Paso 2 · Cargando ViT LoRA desde {ruta_lora_vit}")
    vit_peft = PeftModel.from_pretrained(
        base_model.vision_model,
        str(ruta_lora_vit),
        is_trainable=False,                 # solo necesitamos los deltas para fusionar
    )
    print(f"   [puente]          · Ejecutando merge_and_unload() sobre el ViT...")
    base_model.vision_model = vit_peft.merge_and_unload().to(
        device=target_device, dtype=target_dtype
    )
    print(f"   [puente]          ✓ ViT fusionado permanentemente "
          f"(sin wrapper PeftModel) en {target_device} | {target_dtype}")
    del vit_peft

    # ── Paso 3: Fusión del LLM ───────────────────────────────────────────────
    print(f"\n   [puente] Paso 3 · Cargando LLM LoRA desde {ruta_lora_llm}")
    llm_peft = PeftModel.from_pretrained(
        base_model.language_model,
        str(ruta_lora_llm),
        is_trainable=False,
    )
    print(f"   [puente]          · Ejecutando merge_and_unload() sobre el LLM...")
    base_model.language_model = llm_peft.merge_and_unload().to(
        device=target_device, dtype=target_dtype
    )
    print(f"   [puente]          ✓ LLM fusionado permanentemente "
          f"(sin wrapper PeftModel) en {target_device} | {target_dtype}")
    del llm_peft
    gc.collect()
    torch.cuda.empty_cache()

    # ── Paso 4: Congelar extremos (ViT + LLM) ────────────────────────────────
    print(f"\n   [puente] Paso 4 · Congelando vision_model y language_model "
          f"(requires_grad = False)...")
    n_vit_frozen = 0
    for p in base_model.vision_model.parameters():
        p.requires_grad = False
        n_vit_frozen += p.numel()
    n_llm_frozen = 0
    for p in base_model.language_model.parameters():
        p.requires_grad = False
        n_llm_frozen += p.numel()
    print(f"   [puente]          ViT congelado: {n_vit_frozen:>14,} params")
    print(f"   [puente]          LLM congelado: {n_llm_frozen:>14,} params")

    # ── Paso 5: Liberar el puente (Full FT, sin LoRA nueva) ──────────────────
    # En BLIP-2 el "puente" se compone de TRES bloques entrenables:
    #   (a) `query_tokens`        : nn.Parameter raíz de Blip2ForConditionalGeneration
    #                               (las 32 queries aprendibles que entran al Q-Former,
    #                                paper original Li et al. 2023). NO está dentro de
    #                                `qformer.*`; vive en el nivel del modelo principal.
    #   (b) `qformer`             : el Q-Former completo (cross-attn ViT↔text).
    #   (c) `language_projection` : la proyección Q-Former → embedding del LLM.
    # Los tres deben liberarse simultáneamente; congelar query_tokens rompería
    # la fase de aprendizaje del Q-Former (las queries son su entrada principal).
    print(f"\n   [puente] Paso 5 · Liberando puente: query_tokens + qformer + language_projection "
          f"(requires_grad = True)...")
    n_query = 0
    if hasattr(base_model, "query_tokens") and isinstance(base_model.query_tokens, torch.nn.Parameter):
        base_model.query_tokens.requires_grad = True
        n_query = base_model.query_tokens.numel()
    n_qformer = 0
    for p in base_model.qformer.parameters():
        p.requires_grad = True
        n_qformer += p.numel()
    n_proj = 0
    for p in base_model.language_projection.parameters():
        p.requires_grad = True
        n_proj += p.numel()
    print(f"   [puente]          query_tokens        : {n_query:>14,} params (Full FT)")
    print(f"   [puente]          Q-Former            : {n_qformer:>14,} params (Full FT)")
    print(f"   [puente]          language_projection : {n_proj:>14,} params (Full FT)")

    # ── Paso 6: Auditoría final ──────────────────────────────────────────────
    total_params = sum(p.numel() for p in base_model.parameters())
    trainable    = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    frozen       = total_params - trainable
    pct          = 100.0 * trainable / total_params if total_params > 0 else 0.0

    # Verificar que los entrenables son EXACTAMENTE el puente (3 bloques).
    # `query_tokens` es UN parámetro raíz (nombre completo == "query_tokens",
    # sin punto), por eso lo tratamos como prefijo aceptado independiente.
    trainable_names = [n for n, p in base_model.named_parameters() if p.requires_grad]
    expected_prefixes = ("qformer.", "language_projection.", "query_tokens")
    unexpected = [n for n in trainable_names if not n.startswith(expected_prefixes)]
    expected_count = n_query + n_qformer + n_proj

    print("\n" + "═" * 72)
    print("   AUDITORÍA · MODELO FASE 3 (PUENTE)  Full FT de Q-Former + lang_proj")
    print("═" * 72)
    print(f"   Total de parámetros                    : {total_params:>15,}")
    print(f"   Entrenables (query+qformer+proj)       : {trainable:>15,}  ({pct:.3f} %)")
    print(f"   Congelados  (ViT + LLM fusion)         : {frozen:>15,}")
    print("   ─────────────────────────────────────────────────────────────────")
    print(f"   Suma esperada query+qformer+proj       : {expected_count:>15,}   "
          f"{'✓' if expected_count == trainable else '✗ DIFERENCIA'}")
    print(f"   Parámetros entrenables fuera del puente: {len(unexpected):>15,}   "
          f"{'✓' if not unexpected else '✗ INESPERADOS'}")
    if unexpected:
        for n in unexpected[:5]:
            print(f"     · INESPERADO: {n}")
    print("═" * 72 + "\n")

    if unexpected or expected_count != trainable:
        raise RuntimeError(
            "La auditoría de gradientes falló: hay parámetros entrenables fuera "
            "de query_tokens/qformer/language_projection o el conteo no cuadra. "
            "Revisa los pasos 4 y 5 de preparar_modelo_fase_puente()."
        )

    return base_model


# ─────────────────────────────────────────────────────────────────────────────
# 2.  OPTIMIZADOR (un solo grupo: qformer + language_projection)
# ─────────────────────────────────────────────────────────────────────────────

def build_bridge_optimizer(model) -> Tuple[AdamW, None]:
    """
    AdamW para Full FT del puente.

    Un único grupo de parámetros (qformer + language_projection) con
    lr = 1e-4 (mismo orden que LoRA de OPT en Fases 1 y 2 → continuidad).
    El scheduler (cosine + warmup) es delegado a TrainingArguments.
    """
    bridge_params = [p for p in model.parameters() if p.requires_grad]
    if not bridge_params:
        raise RuntimeError("No hay parámetros entrenables. Verifica preparar_modelo_fase_puente().")

    optimizer = AdamW(
        [{"params": bridge_params, "lr": TRAINING_CONFIG["lr_bridge"]}],
        weight_decay=TRAINING_CONFIG["weight_decay"],
        eps=1e-8,
        fused=torch.cuda.is_available(),
    )
    print(f"   [optimizer] Bridge lr={TRAINING_CONFIG['lr_bridge']:.0e}  |  "
          f"weight_decay={TRAINING_CONFIG['weight_decay']}  |  "
          f"Scheduler dinámico delegado a HuggingFace (cosine+warmup)")
    return optimizer, None


# ─────────────────────────────────────────────────────────────────────────────
# 3.  CALLBACK DE AUDITORÍA Y CHECKPOINT (CSV de losses + best-only)
# ─────────────────────────────────────────────────────────────────────────────

class Phase3PuenteAuditCallback(TrainerCallback):
    """
    Callback ejecutado tras cada evaluate() del Trainer.

    Análogo a `Phase2AuditCallback` pero adaptado a Fase 3:
      • Guarda UN ÚNICO state_dict con qformer + language_projection
        (~ 250 MB) en best_model_{name}/puente_state.pt cuando val_loss mejora.
      • Mantiene history_phase3_{name}.csv  (Step | Train_Loss | Val_Loss).
      • RESUME-AWARE: si encuentra historial previo, recupera best_val_loss.

    (PHASE3 PUENTE) — La inferencia de texto durante el entrenamiento se
    desactivó: cada eval ahora solo computa cross-entropy y persiste el CSV
    de pérdidas. Las métricas textuales (BLEU/ROUGE/RadGraph/ClinicalBERT)
    se calculan en `evaluate_phase3_puente.py` tras finalizar el entrenamiento.
    """

    def __init__(
        self,
        val_dataset:           MimicCXRDataset,
        processor,
        config_name:           str,
        output_dir:            Path,
        checkpoints_dir:       Path,
        # (PHASE3 PUENTE) Desactivado: argumentos de generación ya no usados.
        # audit_samples:         int  = 10,
        # max_new_tokens:        int  = 400,
        # skip_text_generation:  bool = True,
    ):
        self.val_dataset          = val_dataset
        self.processor            = processor
        self.config_name          = config_name
        self.output_dir           = output_dir
        self.checkpoints_dir      = checkpoints_dir
        # (PHASE3 PUENTE) Desactivado: state de generación textual.
        # self.audit_samples        = audit_samples
        # self.max_new_tokens       = max_new_tokens
        # self.skip_text_generation = skip_text_generation

        self.history_path   = output_dir / f"history_phase3_{config_name}.csv"
        # (PHASE3 PUENTE) Desactivado: el audit log textual ya no se escribe.
        # self.audit_path     = output_dir / f"audit_phase3_{config_name}.txt"
        self.best_model_dir = checkpoints_dir / f"best_model_{config_name}"

        self.best_val_loss: float       = float("inf")
        self.history:       List[Dict]  = []

        # Resume-aware: recuperar historial previo y mejor val_loss
        if self.history_path.exists():
            try:
                prev = pd.read_csv(self.history_path)
                self.history = prev.to_dict("records")
                valid = prev["Val_Loss"].dropna()
                if len(valid) > 0:
                    self.best_val_loss = float(valid.min())
                print(
                    f"   [RESUME] Historial Fase 3 recuperado: "
                    f"{len(self.history)} evaluaciones, "
                    f"best_val_loss={self.best_val_loss:.4f}"
                )
            except Exception as exc:
                print(f"   [RESUME] No se pudo leer historial previo: {exc}")

        # (PHASE3 PUENTE) Desactivado: cabecera del audit log textual.
        # audit_mode = "a" if self.audit_path.exists() else "w"
        # with open(self.audit_path, audit_mode, encoding="utf-8") as f:
        #     f.write(
        #         f"\n# Audit Log — Fase 3 (Puente) — {config_name}\n"
        #         f"# {'Reanudado' if audit_mode == 'a' else 'Iniciado'}: "
        #         f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        #         f"# Columnas por entrada: Step | BLEU-4 | ROUGE-L | Pred | Real\n\n"
        #     )

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        model = kwargs.get("model", None)
        if model is None:
            return

        step     = state.global_step
        val_loss = (metrics or {}).get("eval_loss", None)
        train_loss = None
        if state.log_history:
            for log in reversed(state.log_history):
                if "loss" in log and "eval_loss" not in log:
                    train_loss = log.get("loss")
                    break

        # (PHASE3 PUENTE) Desactivado: device-pick para generación textual.
        # try:
        #     device = next(p for p in model.parameters() if p.requires_grad).device
        # except StopIteration:
        #     device = next(model.parameters()).device

        model.eval()

        # ─── (PHASE3 PUENTE) BLOQUE DE INFERENCIA DESACTIVADO ───────────────
        # La generación de texto en val durante el entrenamiento se removió:
        # cada eval ahora solo persiste Train_Loss y Val_Loss en el CSV. Las
        # métricas BLEU/ROUGE/RadGraph/ClinicalBERT se calculan después en
        # evaluate_phase3_puente.py sobre el test set, igual que en Fase 2.
        #
        # if self.skip_text_generation:
        #     mean_bleu4  = 0.0
        #     mean_rougeL = 0.0
        #     audit_lines = [f"Step={step} | Text generation SKIPPED (modo fast-train)\n"]
        # else:
        #     n       = min(self.audit_samples, len(self.val_dataset))
        #     indices = np.random.choice(len(self.val_dataset), n, replace=False)
        #     bleu4_scores: List[float] = []
        #     rougeL_scores: List[float] = []
        #     audit_lines:  List[str]   = []
        #     with torch.no_grad():
        #         for idx in indices:
        #             try:
        #                 sample = self.val_dataset[int(idx)]
        #                 gen = _generate_sample(model, sample, self.processor, device,
        #                                        self.max_new_tokens)
        #                 ref = self.processor.batch_decode(
        #                     [sample["labels"]], skip_special_tokens=True
        #                 )[0]
        #                 b4 = _bleu4(ref, gen)
        #                 rl = _rougeL(ref, gen)
        #                 bleu4_scores.append(b4)
        #                 rougeL_scores.append(rl)
        #                 audit_lines.append(
        #                     f"Step={step} | BLEU-4={b4:.4f} | ROUGE-L={rl:.4f}\n"
        #                     f"  Pred : {gen}\n  Real : {ref}\n"
        #                 )
        #             except Exception as exc:
        #                 audit_lines.append(f"Step={step} | Error en muestra {idx}: {exc}\n")
        #     mean_bleu4  = float(np.mean(bleu4_scores))  if bleu4_scores  else 0.0
        #     mean_rougeL = float(np.mean(rougeL_scores)) if rougeL_scores else 0.0
        # ─────────────────────────────────────────────────────────────────────

        # CSV de losses: persistido EN CADA EVALUACIÓN tras cada eval_steps
        # (idéntico al patrón de Phase2AuditCallback en trainer_utils.py).
        self.history.append({
            "Step":       step,
            "Train_Loss": train_loss,
            "Val_Loss":   val_loss,
        })
        pd.DataFrame(self.history).to_csv(self.history_path, index=False)

        train_str = f"{train_loss:.4f}" if train_loss is not None else "N/A"
        val_str   = f"{val_loss:.4f}"   if val_loss   is not None else "N/A"
        print(f"\n   [Step {step}] train_loss={train_str}  val_loss={val_str}"
              f"  → history CSV actualizado en cada eval ({self.history_path.name})")

        # Best-only checkpoint: GUARDA SOLO EL PUENTE
        if val_loss is not None and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_model_dir.mkdir(parents=True, exist_ok=True)

            # state_dict mínimo: solo qformer + language_projection (no LoRA, no LLM/ViT)
            puente_state = {
                n: p.detach().cpu()
                for n, p in model.named_parameters()
                if p.requires_grad
            }
            puente_path = self.best_model_dir / "puente_state.pt"
            torch.save(puente_state, str(puente_path))

            # Metadatos para inferencia (qué pesos cargar y desde qué base)
            meta = {
                "config_name":       self.config_name,
                "step":              step,
                "val_loss":          float(val_loss),
                "param_count":       sum(t.numel() for t in puente_state.values()),
                "trained_modules":   sorted({n.split(".", 1)[0] for n in puente_state.keys()}),
                "base_model":        config.model.model_name,
                "phase2_llm_dir":    str(PHASE2_LLM_DIR),
                "phase2_vit_dir":    str(PHASE2_VIT_DIR),
                "saved_at":          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            (self.best_model_dir / "puente_meta.json").write_text(
                pd.Series(meta).to_json(indent=2), encoding="utf-8"
            )

            print(
                f"   ✓ Checkpoint puente guardado (val_loss={val_loss:.4f}, "
                f"{meta['param_count']:,} params)\n     → {puente_path}"
            )

        # (PHASE3 PUENTE) Desactivado: escritura del audit log textual.
        # with open(self.audit_path, "a", encoding="utf-8") as f:
        #     f.write(f"\n{'─' * 100}\n")
        #     for line in audit_lines:
        #         f.write(line)

        gc.collect()
        torch.cuda.empty_cache()
        model.train()


# ─────────────────────────────────────────────────────────────────────────────
# 4.  TRAINER ESPECIALIZADO — guarda SOLO qformer + language_projection
# ─────────────────────────────────────────────────────────────────────────────

class BridgeOnlyTrainer(LengthAwareTrainer):
    """
    Trainer que sobreescribe save_model / _load_from_checkpoint / _load_best_model
    de transformers/Trainer para persistir UNICAMENTE los parámetros con
    requires_grad=True (qformer + language_projection).

    Motivación
    ----------
    El modelo BLIP-2 tras `preparar_modelo_fase_puente` pesa ~ 7 GB en bfloat16.
    Si se guarda completo cada save_step:
      1. La serialización es CPU-bound y lenta (GPU 0 %).
      2. safetensors falla por pesos atados (lm_head ↔ embed_tokens del OPT).
      3. El disco se llena rápidamente (save_total_limit=2 → 14 GB por config).
    Guardar solo el puente (~ 250 MB) elimina los tres problemas.
    """

    # ── save_model: solo el puente ──────────────────────────────────────────
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        import os
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        puente_state = {
            n: p.detach().cpu()
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }
        torch.save(puente_state, os.path.join(output_dir, "puente_state.pt"))
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    # ── Helper compartido: cargar el puente desde un dir de checkpoint ──────
    def _load_puente_from(self, ckpt_dir: str) -> None:
        import os
        path = os.path.join(ckpt_dir, "puente_state.pt")
        if not os.path.isfile(path):
            return
        # weights_only=True: defensa contra checkpoints maliciosos (PyTorch ≥ 2.4).
        state = torch.load(path, map_location="cpu", weights_only=True)
        # load_state_dict con strict=False: solo cargamos las keys del puente
        # (qformer.* y language_projection.*); el resto del modelo (ViT y LLM
        # fusionados) ya está en su sitio desde preparar_modelo_fase_puente().
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        # `missing` será largo (todos los pesos congelados de ViT+LLM no están
        # en el .pt del puente); eso es correcto y no hay que reportarlo.
        if unexpected:
            print(f"   [resume] WARN: {len(unexpected)} keys inesperadas en puente_state.pt")

    # ── Resume desde checkpoint Trainer (paso intermedio) ───────────────────
    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        import os
        if os.path.isfile(os.path.join(resume_from_checkpoint, "puente_state.pt")):
            self._load_puente_from(resume_from_checkpoint)
        else:
            super()._load_from_checkpoint(resume_from_checkpoint, model)

    # ── Cargar el mejor puente al final (load_best_model_at_end=True) ───────
    def _load_best_model(self):
        import os
        ckpt = self.state.best_model_checkpoint
        if ckpt is None:
            return
        if os.path.isfile(os.path.join(ckpt, "puente_state.pt")):
            self._load_puente_from(ckpt)
        else:
            super()._load_best_model()


# ─────────────────────────────────────────────────────────────────────────────
# 5.  HELPER: cargar datasets de Fase 3 (mismo prompt diseñado que Fase 2)
# ─────────────────────────────────────────────────────────────────────────────

def load_phase3_datasets(processor) -> Tuple[MimicCXRDataset, MimicCXRDataset]:
    """
    Carga train/val del 50 % MIMIC-CXR usando EL PROMPT DISEÑADO
    (config.inference.default_prompt) — IDÉNTICO al de Fases 1 y 2.

    No se usa "self-prompting" ni Chain-of-Thought durante el entrenamiento:
    el Q-Former aprende a alinear el ViT con el LLM bajo el mismo prompt
    que se evaluará después.
    """
    shared_kwargs = dict(
        images_dir          = IMAGES_DIR_PUENTE,
        processor           = processor,
        image_size          = config.data.image_size,
        clahe_clip_limit    = config.data.clahe_clip_limit,
        clahe_tile_grid_size= config.data.clahe_tile_grid_size,
        max_length          = 256,                          # mismo recorte que Fase 2
        padding             = config.data.padding,
        prompt              = config.inference.default_prompt,
    )
    print("   [datasets] Cargando Train (50 %) — CON data augmentation espacial...")
    train_ds = MimicCXRDataset(csv_path=TRAIN_CSV_PUENTE, augment=True, **shared_kwargs)
    print("   [datasets] Cargando Val   (50 %) — SIN augmentation (evaluación limpia)...")
    val_ds   = MimicCXRDataset(csv_path=VAL_CSV_PUENTE, augment=False, **shared_kwargs)
    print(f"   [datasets] Train: {len(train_ds):,}  |  Val: {len(val_ds):,}\n")
    return train_ds, val_ds


# ─────────────────────────────────────────────────────────────────────────────
# 6.  FUNCIÓN PRINCIPAL DE ENTRENAMIENTO
# ─────────────────────────────────────────────────────────────────────────────

def train_puente_config(
    config_dict:    Dict,
    train_dataset:  MimicCXRDataset,
    val_dataset:    MimicCXRDataset,
    processor,
) -> Dict:
    """
    Entrena un único modelo de Fase 3 (puente) de forma aislada.

    Flujo:
      1. preparar_modelo_fase_puente() para construir el modelo Full-FT.
      2. build_bridge_optimizer() (un solo grupo, lr=1e-4).
      3. TrainingArguments (bf16=True, gradient_checkpointing=False,
         group_by_length=True con longitudes pre-computadas).
      4. BridgeOnlyTrainer (guarda solo qformer + language_projection).
      5. EarlyStoppingCallback + Phase3PuenteAuditCallback.
      6. Reanuda automáticamente desde el último checkpoint válido.

    Args:
        config_dict   : claves: name, llm_lora_dir, vit_lora_dir, base_model.
        train_dataset : MimicCXRDataset 50% train (augment=True).
        val_dataset   : MimicCXRDataset 50% val (augment=False).
        processor     : Blip2Processor.

    Returns:
        Dict con métricas de la corrida.
    """
    name = config_dict["name"]
    print(f"\n{'='*80}")
    print(f"  FASE 3 — Re-entrenamiento del Puente (Q-Former + language_projection)")
    print(f"  Config            : {name}")
    print(f"  base_model        : {config_dict['base_model']}")
    print(f"  ViT LoRA (Fase 2) : {config_dict['vit_lora_dir']}")
    print(f"  LLM LoRA (Fase 2) : {config_dict['llm_lora_dir']}")
    print(f"  lr_bridge={TRAINING_CONFIG['lr_bridge']:.0e}  "
          f"batch_ef={TRAINING_CONFIG['batch_size']*TRAINING_CONFIG['gradient_accumulation']}  "
          f"epochs={TRAINING_CONFIG['max_epochs']}")
    print(f"{'='*80}\n")

    gc.collect()
    torch.cuda.empty_cache()

    # ── 1. Modelo (Full FT del puente) ───────────────────────────────────────
    model = preparar_modelo_fase_puente(
        ruta_base     = config_dict["base_model"],
        ruta_lora_vit = config_dict["vit_lora_dir"],
        ruta_lora_llm = config_dict["llm_lora_dir"],
    )

    # ── 2. Optimizador ───────────────────────────────────────────────────────
    optimizer, _ = build_bridge_optimizer(model)

    # ── 3. Directorio de salida ──────────────────────────────────────────────
    config_output_dir = OUTPUT_DIR / name
    config_output_dir.mkdir(parents=True, exist_ok=True)

    # ── 4. TrainingArguments (idénticas a Fase 2 salvo batch_size) ───────────
    training_args = TrainingArguments(
        output_dir=str(config_output_dir),
        num_train_epochs=TRAINING_CONFIG["max_epochs"],
        per_device_train_batch_size=TRAINING_CONFIG["batch_size"],
        per_device_eval_batch_size=24,
        gradient_accumulation_steps=TRAINING_CONFIG["gradient_accumulation"],
        learning_rate=TRAINING_CONFIG["lr_bridge"],
        lr_scheduler_type="cosine",
        warmup_ratio=TRAINING_CONFIG["warmup_ratio"],
        eval_strategy="steps",
        eval_steps=TRAINING_CONFIG["eval_steps"],
        save_strategy="steps",
        save_steps=TRAINING_CONFIG["eval_steps"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=100,
        logging_dir=str(config_output_dir / "logs"),
        fp16=False,
        bf16=True,
        gradient_checkpointing=False,           # tenemos VRAM para activaciones
        report_to="none",
        remove_unused_columns=False,
        label_names=["labels"],
        group_by_length=True,
        dataloader_num_workers=32,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=6,
        dataloader_persistent_workers=False,    # OBLIGATORIO (deadlock al resumir)
        dataloader_drop_last=True,
        bf16_full_eval=True,
        save_total_limit=2,
    )

    # ── 5. Callbacks y collator ──────────────────────────────────────────────
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=processor.tokenizer, padding=True
    )
    audit_cb = Phase3PuenteAuditCallback(
        val_dataset     = val_dataset,
        processor       = processor,
        config_name     = name,
        output_dir      = OUTPUT_DIR,
        checkpoints_dir = CHECKPOINTS_DIR,
        # (PHASE3 PUENTE) Desactivado: parámetros de generación textual.
        # audit_samples         = TRAINING_CONFIG["audit_samples"],
        # max_new_tokens        = TRAINING_CONFIG["max_new_tokens"],
        # skip_text_generation  = True,
    )
    early_stop_cb = EarlyStoppingCallback(
        early_stopping_patience=TRAINING_CONFIG["early_stopping_patience"]
    )

    # ── 6. Pre-computar longitudes (sin I/O de imágenes) ─────────────────────
    train_lengths = precompute_train_lengths(train_dataset)
    eval_lengths  = precompute_train_lengths(val_dataset)

    trainer = BridgeOnlyTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        optimizers=(optimizer, None),
        callbacks=[early_stop_cb, audit_cb],
        train_lengths=train_lengths,
        eval_lengths=eval_lengths,
    )

    # ── 7. Detectar checkpoint previo (rescate) ──────────────────────────────
    all_ckpts = sorted(
        config_output_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1,
    )
    valid_ckpts = [c for c in all_ckpts if (c / "trainer_state.json").exists()]
    incomplete  = [c for c in all_ckpts if c not in valid_ckpts]
    for inc in incomplete:
        print(f"   [RESCATE] ⚠ Checkpoint incompleto descartado: {inc.name}")

    if valid_ckpts:
        last = valid_ckpts[-1]
        print(f"   [RESCATE] {len(valid_ckpts)} checkpoint(s) válido(s). "
              f"Reanudando desde {last.name}...")
        trainer.train(resume_from_checkpoint=str(last))
    else:
        print("   [NUEVO] Sin checkpoints válidos previos. Iniciando desde cero.")
        trainer.train()

    # ── 8. Resumen de la corrida ─────────────────────────────────────────────
    eval_logs  = [l for l in trainer.state.log_history if "eval_loss" in l]
    train_logs = [l for l in trainer.state.log_history
                  if "loss" in l and "eval_loss" not in l]
    best_val_loss = trainer.state.best_metric
    best_step     = None
    if eval_logs and best_val_loss is not None:
        for log in eval_logs:
            if abs(log.get("eval_loss", float("inf")) - best_val_loss) < 1e-6:
                best_step = log.get("step")
                break

    result = {
        "config_name":      name,
        "phase":            "phase3_puente",
        "lr_bridge":        TRAINING_CONFIG["lr_bridge"],
        "total_steps":      trainer.state.global_step,
        "best_val_loss":    best_val_loss,
        "best_step":        best_step,
        "final_train_loss": train_logs[-1]["loss"] if train_logs else None,
        "timestamp":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    print(f"\n   ✓ Completado: steps={result['total_steps']}  "
          f"best_val_loss={result['best_val_loss']}  @  step {result['best_step']}")

    # ── 9. Liberar VRAM ──────────────────────────────────────────────────────
    del model, trainer, audit_cb, early_stop_cb, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 7.  GRÁFICOS DE CURVAS DE PÉRDIDA (igual a Fase 2)
# ─────────────────────────────────────────────────────────────────────────────

def plot_phase3_curves(config_name: str, output_dir: Path, plots_dir: Path) -> None:
    """Genera 1 PNG: Train vs Val Loss para la corrida Fase 3."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / f"history_phase3_{config_name}.csv"
    if not history_path.exists():
        print(f"   [plot] Sin historial: {history_path}")
        return

    df = pd.read_csv(history_path)
    if "Train_Loss" not in df.columns or "Val_Loss" not in df.columns:
        print(f"   [plot] CSV sin columnas Train/Val Loss: {history_path}")
        return
    df = df.dropna(subset=["Train_Loss", "Val_Loss"])
    if df.empty:
        print(f"   [plot] CSV vacío tras dropna: {history_path}")
        return

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(df["Step"], df["Train_Loss"], label="Train Loss",
            color="#2E86AB", linewidth=2, marker="o", markersize=4)
    ax.plot(df["Step"], df["Val_Loss"], label="Val Loss",
            color="#A23B72", linewidth=2, marker="s", markersize=4)
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(f"Train vs Val Loss — {config_name} (Fase 3 · Puente)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11, loc="best")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = plots_dir / f"01_train_val_loss_{config_name}.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   [plot] {out.name} guardado → {out}")
