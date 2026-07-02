#!/usr/bin/env python3
"""
trainer_utils_phase3_text_coadapt.py — Utilidades de entrenamiento para Fase 3 Text Co-adapt
==============================================================================================
Estudio de Ablación — Paso 3: Co-adaptación Textual (Q-Former + LLM LoRA simultáneos).

El ViT se mantiene 100% congelado (ahorro de memoria). Se inyectan adaptadores LoRA
de forma simultánea en el Q-Former (query, value) y en el LLM/OPT (q_proj, v_proj),
invirtiendo la carga cognitiva respecto al Paso 1 (solo Q-Former) y al Paso 2
(co-adaptación visual ViT + Q-Former).

La base de conocimiento proviene íntegramente de Fase 2:
  - LLM  : best_model_Config_B1/llm   → merge_and_unload()
  - ViT  : best_model_Config_B1/vit   → merge_and_unload() → CONGELADO TOTAL

Responsabilidades
-----------------
1. inject_qformer_lora()
       • Carga adaptadores LLM y ViT de Fase 2 y hace merge_and_unload() en ambos.
       • Congela COMPLETAMENTE el modelo fusionado (ViT incluido, sin excepciones).
       • Inyecta LoRA en TODO el modelo base (Q-Former + LLM) usando
         target_modules=["query", "value", "q_proj", "v_proj"].

2. Phase3AuditCallback
       Guarda el modelo PEFT completo (Q-Former + LLM LoRA) en
       checkpoints/best_model_{name}/ SOLO si val_loss mejora.
       Registra history_phase3_text_coadapt_{name}.csv y
       audit_phase3_text_coadapt_{name}.txt.

3. train_single_config_phase3()
       Función principal: recibe spec de config, entrena y devuelve dict-resumen
       con métricas de la corrida.

4. plot_comparative_curves()
       Lee los CSV de historial y genera 3 PNGs comparativos
       (Val Loss / BLEU-4 / ROUGE-L).

Autor : Evelyn Silva Rozas
Fecha : Junio 2026
"""

import gc
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from peft import LoraConfig, PeftModel, get_peft_model
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

from config import config
from data_loader import MimicCXRDataset
from model_utils import cargar_procesador, cargar_modelo_base

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")


# ─────────────────────────────────────────────────────────────────────────────
# 🚨 FIX DEFINITIVO: GLOBAL MONKEY PATCH — Blip2QFormerModel.forward
# ─────────────────────────────────────────────────────────────────────────────
# Problema: PEFT deja pasar kwargs del text decoder (input_ids, labels, etc.)
# hacia el Q-Former, causando TypeError.
# Solución: Whitelist que filtra solo argumentos válidos del Q-Former.

from transformers.models.blip_2.modeling_blip_2 import Blip2QFormerModel

_original_qformer_forward = Blip2QFormerModel.forward

_QFORMER_VALID_KWARGS = frozenset({
    "query_embeds", "attention_mask", "head_mask",
    "encoder_hidden_states", "encoder_attention_mask",
    "past_key_values", "use_cache", "output_attentions",
    "output_hidden_states", "return_dict"
})

def _patched_qformer_forward(self, *args, **kwargs):
    filtered = {k: v for k, v in kwargs.items() if k in _QFORMER_VALID_KWARGS}
    return _original_qformer_forward(self, *args, **filtered)

Blip2QFormerModel.forward = _patched_qformer_forward


# ─────────────────────────────────────────────────────────────────────────────
# PATHS Y CONSTANTES
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR = config.paths.base_dir

# Text Co-adapt trabaja con el 10% del dataset
TRAIN_CSV_10PCT  = BASE_DIR / "train_split_10pct_limpio.csv"
VAL_CSV_10PCT    = BASE_DIR / "val_split_10pct_limpio.csv"
TEST_CSV_10PCT   = BASE_DIR / "test_split_10pct_limpio.csv"
IMAGES_DIR_10PCT = BASE_DIR / "imagenes_10_porciento"

# Pesos duales (LLM + ViT) pre-entrenados de Fase 2
# Solo Base B (best_model_Config_B1) — base de conocimiento para text co-adapt
DUAL_WEIGHTS: Dict[str, Dict[str, Path]] = {
    "B": {
        "llm": BASE_DIR / "phase2_results-10pct" / "checkpoints" / "best_model_Config_B1" / "llm",
        "vit": BASE_DIR / "phase2_results-10pct" / "checkpoints" / "best_model_Config_B1" / "vit",
    },
}

# Directorio raíz de salida para la Fase 3 Text Co-adapt
OUTPUT_DIR      = BASE_DIR / "phase3_results-text-coadapt_lowr"
CHECKPOINTS_DIR = OUTPUT_DIR / "checkpoints"

# ─────────────────────────────────────────────────────────────────────────────
# HIPERPARÁMETROS
# ─────────────────────────────────────────────────────────────────────────────

TRAINING_CONFIG: Dict = {
    # Learning rate con scheduler dinámico cosine
    "lr":                      1e-4,
    # Régimen de entrenamiento
    "max_epochs":              50,
    "early_stopping_patience": 10,      # evaluaciones consecutivas sin mejora
    "batch_size":              4,
    "gradient_accumulation":   16,      # batch efectivo = 64
    "eval_steps":              125,
    "warmup_steps":            500,
    # Generación para auditoría
    "audit_samples":           10,
    "max_new_tokens":          512,
    # LoRA combinado Q-Former + LLM
    "lora_dropout":            0.15,
}

# Módulos objetivo: Q-Former (query, value) + LLM/OPT (q_proj, v_proj)
# Carga cognitiva invertida: tanto el puente de proyección como el decoder
# reciben gradientes simultáneamente.
TARGET_MODULES = ["query", "value", "q_proj", "v_proj"]


# ─────────────────────────────────────────────────────────────────────────────
# MÉTRICAS NLP LIGERAS (solo para el callback de auditoría en tiempo de train)
# ─────────────────────────────────────────────────────────────────────────────

_rouge_inst  = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
_smoothing   = SmoothingFunction()


def _bleu4(reference: str, candidate: str) -> float:
    try:
        return sentence_bleu(
            [reference.lower().split()],
            candidate.lower().split(),
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=_smoothing.method1,
        )
    except Exception:
        return 0.0


def _rougeL(reference: str, candidate: str) -> float:
    try:
        return _rouge_inst.score(reference, candidate)["rougeL"].fmeasure
    except Exception:
        return 0.0


def _generate_sample(model, sample: dict, processor, device: torch.device,
                     max_new_tokens: int) -> str:
    """Genera texto para una sola muestra (greedy, sin grad)."""
    pixel_values = sample["pixel_values"].unsqueeze(0).to(device)
    with torch.no_grad():
        out = model.generate(
            pixel_values=pixel_values,
            max_new_tokens=max_new_tokens,
            num_beams=1,
            use_cache=False,
        )
    return processor.batch_decode(out, skip_special_tokens=True)[0]


# ─────────────────────────────────────────────────────────────────────────────
# 1.  INYECCIÓN LoRA COMBINADO (Q-Former + LLM) — ViT 100% CONGELADO
# ─────────────────────────────────────────────────────────────────────────────

def inject_qformer_lora(
    llm_weights_dir: Path,
    vit_weights_dir: Path,
    qformer_r: int,
    qformer_alpha: int,
) -> "Blip2ForConditionalGeneration":
    """
    Construye el modelo BLIP-2 con adaptadores LoRA simultáneos en Q-Former y LLM.
    El ViT se mantiene 100% congelado (sin co-adaptación visual).

    Estrategia de inyección (Fase Text Co-adapt)
    ---------------------------------------------
    1. Carga modelo BLIP-2 base (bfloat16, sin cuantización).
    2. Carga adaptador LLM LoRA de Fase 2 y hace merge_and_unload()
       → pesos de Fase 2 fusionados en el LLM.
    3. Carga adaptador ViT LoRA de Fase 2 y hace merge_and_unload()
       → pesos de Fase 2 fusionados en el ViT.
    4. Congela TODOS los parámetros del modelo fusionado (requires_grad=False).
       El ViT NO se descongela en ningún momento (ahorro de memoria).
    5. Inyecta LoRA en TODO el modelo base con:
           task_type="CAUSAL_LM"
           target_modules=["query", "value", "q_proj", "v_proj"]
           r=qformer_r, lora_alpha=qformer_alpha
       PEFT detecta automáticamente los módulos compatibles en el Q-Former
       (query, value) y en el LLM/OPT (q_proj, v_proj).

    Args:
        llm_weights_dir : Directorio con adaptador LLM LoRA (Fase 2).
        vit_weights_dir : Directorio con adaptador ViT LoRA (Fase 2).
        qformer_r       : Rango LoRA (128).
        qformer_alpha   : Escalado LoRA (256).

    Returns:
        Modelo BLIP-2 con LoRA combinado Q-Former+LLM listo para entrenamiento.

    Raises:
        FileNotFoundError : Si los directorios de pesos no existen.
    """
    for w_dir, label in [(llm_weights_dir, "LLM"), (vit_weights_dir, "ViT")]:
        if not w_dir.exists():
            raise FileNotFoundError(
                f"Pesos {label} de Fase 2 no encontrados: {w_dir}\n"
                "Asegúrate de que Fase 2 haya concluido exitosamente."
            )

    print(f"   [text_coadapt] Cargando modelo base BLIP-2 (bfloat16, device_map=auto)...")
    base_model = cargar_modelo_base(
        model_name=config.model.model_name,
        use_quantization=False,
        device_map="auto",
    )

    # ── LLM: cargar adaptador de Fase 2 y fusionar ───────────────────────────
    print(f"   [text_coadapt] Cargando LLM LoRA (Fase 2) desde "
          f"{llm_weights_dir.parent.name}/{llm_weights_dir.name}...")
    llm_peft = PeftModel.from_pretrained(
        base_model.language_model,
        str(llm_weights_dir),
        is_trainable=False,
    )
    print(f"   [text_coadapt] Fusionando LLM LoRA (merge_and_unload)...")
    base_model.language_model = llm_peft.merge_and_unload()

    # ── ViT: cargar adaptador de Fase 2 y fusionar ───────────────────────────
    print(f"   [text_coadapt] Cargando ViT LoRA (Fase 2) desde "
          f"{vit_weights_dir.parent.name}/{vit_weights_dir.name}...")
    vit_peft = PeftModel.from_pretrained(
        base_model.vision_model,
        str(vit_weights_dir),
        is_trainable=False,
    )
    print(f"   [text_coadapt] Fusionando ViT LoRA (merge_and_unload)...")
    _ref = next(base_model.qformer.parameters())
    base_model.vision_model = vit_peft.merge_and_unload().to(
        device=_ref.device, dtype=_ref.dtype
    )

    # ── Congelar TODO el modelo fusionado (ViT incluido, sin excepciones) ────
    print(f"   [text_coadapt] Congelando TODOS los parámetros (ViT 100% congelado)...")
    for param in base_model.parameters():
        param.requires_grad = False

    # ── INYECTAR LORA GLOBAL (Q-Former + LLM simultáneamente) ───────────────
    global_lora_cfg = LoraConfig(
        r=qformer_r,
        lora_alpha=qformer_alpha,
        lora_dropout=TRAINING_CONFIG["lora_dropout"],
        bias="none",
        target_modules=TARGET_MODULES,
    )
    print(f"   [text_coadapt] Inyectando LoRA global en: {TARGET_MODULES}...")
    base_model = get_peft_model(base_model, global_lora_cfg)

    # ── Resumen de parámetros ─────────────────────────────────────────────────
    total     = sum(p.numel() for p in base_model.parameters())
    trainable = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    pct       = 100.0 * trainable / total if total > 0 else 0.0

    n_lora_train = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    n_vit_total = sum(p.numel() for p in base_model.vision_model.parameters())

    print(
        f"\n   ┌─── Parámetros del modelo (Fase 3 Text Co-adapt) ──┐\n"
        f"   │  Total        : {total:>15,}                          │\n"
        f"   │  Entrenables  : {trainable:>15,}  ({pct:.4f} %)             │\n"
        f"   │    └─ PEFT Global: {n_lora_train:>15,}                          │\n"
        f"   │  Congelados   : {total - trainable:>15,}                          │\n"
        f"   │    └─ ViT 100% : {n_vit_total:>15,}                          │\n"
        f"   └───────────────────────────────────────────────────────────┘\n"
    )

    return base_model


# ─────────────────────────────────────────────────────────────────────────────
# 2.  CALLBACK DE AUDITORÍA Y CHECKPOINTS
# ─────────────────────────────────────────────────────────────────────────────

class Phase3AuditCallback(TrainerCallback):
    """
    Callback ejecutado después de cada evaluate() del Trainer (Fase Text Co-adapt).

    Responsabilidades
    -----------------
    Checkpoint PEFT completo (Q-Former + LLM LoRA)
        Guarda el modelo PEFT completo en best_model_{name}/
        SOLO si val_loss mejoró respecto al mejor registrado.
        Usando model.save_pretrained() en lugar de model.qformer.save_pretrained()
        para preservar ambos adaptadores (Q-Former y LLM).

    Historial
        Appends a history_phase3_text_coadapt_{config_name}.csv:
            Step | Train_Loss | Val_Loss | BLEU-4 | ROUGE-L

    Audit log
        Appends a audit_phase3_text_coadapt_{config_name}.txt con predicciones
        completas (sin truncar) para las muestras de auditoría.
    """

    def __init__(
        self,
        val_dataset:     MimicCXRDataset,
        processor,
        config_name:     str,
        output_dir:      Path,
        checkpoints_dir: Path,
        audit_samples:   int = 10,
        max_new_tokens:  int = 512,
        skip_text_generation: bool = False,
    ):
        self.val_dataset     = val_dataset
        self.processor       = processor
        self.config_name     = config_name
        self.output_dir      = output_dir
        self.checkpoints_dir = checkpoints_dir
        self.audit_samples   = audit_samples
        self.max_new_tokens  = max_new_tokens
        self.skip_text_generation = skip_text_generation

        self.history_path   = output_dir / f"history_phase3_text_coadapt_{config_name}.csv"
        self.audit_path     = output_dir / f"audit_phase3_text_coadapt_{config_name}.txt"
        self.best_model_dir = checkpoints_dir / f"best_model_{config_name}"

        self.best_val_loss: float      = float("inf")
        self.history:       List[Dict] = []

        # Inicializar audit log limpio para esta config
        self.audit_path.write_text(
            f"# Audit Log — Fase 3 Text Co-adapt (r=128, Q-Former+LLM, ViT congelado, 10%) — {config_name}\n"
            f"# Iniciado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"# Columnas por entrada: Step | BLEU-4 | ROUGE-L | Pred | Real\n\n",
            encoding="utf-8",
        )

    def on_evaluate(
        self,
        args:    TrainingArguments,
        state:   TrainerState,
        control: TrainerControl,
        metrics: Optional[Dict] = None,
        **kwargs,
    ):
        model = kwargs.get("model", None)
        if model is None:
            return

        step     = state.global_step
        val_loss = (metrics or {}).get("eval_loss", None)

        # Extraer train_loss del último log de entrenamiento
        train_loss = None
        if state.log_history:
            for log_entry in reversed(state.log_history):
                if "loss" in log_entry and "eval_loss" not in log_entry:
                    train_loss = log_entry.get("loss")
                    break

        # Dispositivo desde primer param entrenable
        try:
            device = next(p for p in model.parameters() if p.requires_grad).device
        except StopIteration:
            device = next(model.parameters()).device

        model.eval()

        # — Generar predicciones en N muestras aleatorias del val set —
        if self.skip_text_generation:
            mean_bleu4  = 0.0
            mean_rougeL = 0.0
            audit_lines = [f"Step={step} | Text generation SKIPPED (modo fast-train)\n"]
        else:
            n       = min(self.audit_samples, len(self.val_dataset))
            indices = np.random.choice(len(self.val_dataset), n, replace=False)

            bleu4_scores:  List[float] = []
            rougeL_scores: List[float] = []
            audit_lines:   List[str]   = []

            with torch.no_grad():
                for idx in indices:
                    try:
                        sample    = self.val_dataset[int(idx)]
                        generated = _generate_sample(
                            model, sample, self.processor, device, self.max_new_tokens
                        )
                        reference = self.processor.batch_decode(
                            [sample["labels"]], skip_special_tokens=True
                        )[0]
                        b4 = _bleu4(reference, generated)
                        rl = _rougeL(reference, generated)
                        bleu4_scores.append(b4)
                        rougeL_scores.append(rl)
                        audit_lines.append(
                            f"Step={step} | BLEU-4={b4:.4f} | ROUGE-L={rl:.4f}\n"
                            f"  Pred : {generated}\n"
                            f"  Real : {reference}\n"
                        )
                    except Exception as exc:
                        audit_lines.append(f"Step={step} | Error en muestra {idx}: {exc}\n")

            mean_bleu4  = float(np.mean(bleu4_scores))  if bleu4_scores  else 0.0
            mean_rougeL = float(np.mean(rougeL_scores)) if rougeL_scores else 0.0

        # — Persistir historial —
        self.history.append({
            "Step":       step,
            "Train_Loss": train_loss,
            "Val_Loss":   val_loss,
            "BLEU-4":     mean_bleu4,
            "ROUGE-L":    mean_rougeL,
        })
        pd.DataFrame(self.history).to_csv(self.history_path, index=False)

        train_loss_str = f"{train_loss:.4f}" if train_loss is not None else "N/A"
        print(
            f"\n   [TextCoadapt Step {step}] train_loss={train_loss_str}  "
            f"val_loss={val_loss:.4f}  BLEU-4={mean_bleu4:.4f}  ROUGE-L={mean_rougeL:.4f}"
        )

        # — Guardar modelo PEFT completo (Q-Former + LLM LoRA) si val_loss mejoró —
        if val_loss is not None and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_model_dir.mkdir(parents=True, exist_ok=True)

            # Guardado PEFT completo en una sola pasada
            model.save_pretrained(str(self.best_model_dir))

            print(
                f"   ✓ Checkpoint text co-adapt guardado (val_loss={val_loss:.4f})\n"
                f"     Modelo global PEFT → {self.best_model_dir}"
            )

        # — Audit log —
        with open(self.audit_path, "a", encoding="utf-8") as f:
            f.write(f"\n{'─' * 100}\n")
            for line in audit_lines:
                f.write(line)

        # Liberar VRAM y volver a modo train
        gc.collect()
        torch.cuda.empty_cache()
        model.train()


# ─────────────────────────────────────────────────────────────────────────────
# 3.  FUNCIÓN PRINCIPAL DE ENTRENAMIENTO
# ─────────────────────────────────────────────────────────────────────────────

def train_single_config_phase3(
    config_dict:    Dict,
    config_num:     int,
    total_configs:  int,
    train_dataset:  MimicCXRDataset,
    val_dataset:    MimicCXRDataset,
    processor,
) -> Dict:
    """
    Entrena una configuración de Fase Text Co-adapt de forma completamente aislada.

    Flujo
    -----
    1. Llama a inject_qformer_lora() para construir el modelo con LoRA combinado
       (Q-Former + LLM) y ViT 100% congelado.
    2. Configura optimizador AdamW sobre TODOS los params entrenables (Q-Former + LLM).
    3. Configura TrainingArguments (bf16=True, gradient_checkpointing=True, cosine).
    4. Lanza Trainer con Phase3AuditCallback + EarlyStoppingCallback.
    5. Detecta checkpoints previos para reanudación automática (rescate).
    6. Al terminar, destruye modelo, trainer y optimizador, y libera VRAM.

    Args:
        config_dict   : Spec de la config con claves:
                        name, qformer_r, qformer_alpha, base_weights_key ("A" | "B").
        config_num    : Índice 1-based para logging.
        total_configs : Total de configs en el torneo.
        train_dataset : Dataset de entrenamiento (10% MIMIC-CXR).
        val_dataset   : Dataset de validación   (10% MIMIC-CXR).
        processor     : Blip2Processor compartido.

    Returns:
        Dict con:
            config_name, qformer_r, qformer_alpha, lr,
            total_steps, best_val_loss, best_step,
            final_train_loss, timestamp
    """
    name            = config_dict["name"]
    qformer_r       = config_dict["qformer_r"]
    qformer_alpha   = config_dict["qformer_alpha"]
    base_key        = config_dict["base_weights_key"]
    llm_weights_dir = DUAL_WEIGHTS[base_key]["llm"]
    vit_weights_dir = DUAL_WEIGHTS[base_key]["vit"]

    print(f"\n{'='*80}")
    print(f"  FASE 3 TEXT CO-ADAPT (10%) — Config {config_num}/{total_configs}: {name}")
    print(f"  Base: {base_key} (LLM+ViT fusionados de Fase 2)")
    print(f"  LoRA combinado: r={qformer_r}, alpha={qformer_alpha}")
    print(f"  Targets: {TARGET_MODULES}")
    print(f"  ViT: 100% congelado (sin co-adaptación visual)")
    print(f"  lr={TRAINING_CONFIG['lr']:.0e}  scheduler=cosine  warmup_steps={TRAINING_CONFIG['warmup_steps']}")
    print(f"  batch_ef={TRAINING_CONFIG['batch_size']*TRAINING_CONFIG['gradient_accumulation']}")
    print(f"{'='*80}\n")

    gc.collect()
    torch.cuda.empty_cache()

    # ── 1. Construir modelo con LoRA combinado Q-Former + LLM ────────────────
    model = inject_qformer_lora(
        llm_weights_dir=llm_weights_dir,
        vit_weights_dir=vit_weights_dir,
        qformer_r=qformer_r,
        qformer_alpha=qformer_alpha,
    )

    # ── 2. Optimizador (Q-Former LoRA + LLM LoRA) ────────────────────────────
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No hay parámetros entrenables. Verifica inject_qformer_lora().")

    optimizer = AdamW(
        trainable_params,
        lr=TRAINING_CONFIG["lr"],
        weight_decay=0.05,
        eps=1e-8,
    )
    print(f"   [optimizer] lr={TRAINING_CONFIG['lr']:.0e}  scheduler=cosine  "
          f"warmup_steps={TRAINING_CONFIG['warmup_steps']}")

    # ── 3. Directorio de salida por config ────────────────────────────────────
    config_output_dir = OUTPUT_DIR / name
    config_output_dir.mkdir(parents=True, exist_ok=True)

    # ── 4. TrainingArguments ──────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=str(config_output_dir),
        num_train_epochs=TRAINING_CONFIG["max_epochs"],
        per_device_train_batch_size=TRAINING_CONFIG["batch_size"],
        per_device_eval_batch_size=TRAINING_CONFIG["batch_size"],
        gradient_accumulation_steps=TRAINING_CONFIG["gradient_accumulation"],
        learning_rate=TRAINING_CONFIG["lr"],
        # Scheduler dinámico cosine con warmup
        lr_scheduler_type="cosine",
        warmup_steps=TRAINING_CONFIG["warmup_steps"],
        eval_strategy="steps",
        eval_steps=TRAINING_CONFIG["eval_steps"],
        save_strategy="steps",
        save_steps=TRAINING_CONFIG["eval_steps"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=TRAINING_CONFIG["eval_steps"],  # Sincronizado con eval_steps
        logging_dir=str(config_output_dir / "logs"),
        # Precisión y memoria
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
        # Otros
        report_to="none",
        remove_unused_columns=False,
        label_names=["labels"],
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        save_total_limit=2,
    )

    # ── 5. Callbacks y collator ───────────────────────────────────────────────
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=processor.tokenizer, padding=True
    )

    audit_cb = Phase3AuditCallback(
        val_dataset=val_dataset,
        processor=processor,
        config_name=name,
        output_dir=OUTPUT_DIR,
        checkpoints_dir=CHECKPOINTS_DIR,
        audit_samples=TRAINING_CONFIG["audit_samples"],
        max_new_tokens=TRAINING_CONFIG["max_new_tokens"],
        skip_text_generation=True,  # Fast-train mode
    )
    early_stop_cb = EarlyStoppingCallback(
        early_stopping_patience=TRAINING_CONFIG["early_stopping_patience"]
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        optimizers=(optimizer, None),
        callbacks=[early_stop_cb, audit_cb],
    )

    # ── 6. Detectar checkpoint previo para rescate ────────────────────────────
    existing_ckpts = sorted(config_output_dir.glob("checkpoint-*"))
    if existing_ckpts:
        last = existing_ckpts[-1]
        print(f"   [RESCATE] {len(existing_ckpts)} checkpoint(s) detectados. "
              f"Reanudando desde {last.name}...")
        trainer.train(resume_from_checkpoint=True)
    else:
        print("   [NUEVO] Sin checkpoints previos. Iniciando text co-adapt desde cero.")
        trainer.train()

    # ── 7. Extraer métricas de la corrida ─────────────────────────────────────
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
        "qformer_r":        qformer_r,
        "qformer_alpha":    qformer_alpha,
        "base":             base_key,
        "lr":               TRAINING_CONFIG["lr"],
        "total_steps":      trainer.state.global_step,
        "best_val_loss":    best_val_loss,
        "best_step":        best_step,
        "final_train_loss": train_logs[-1]["loss"] if train_logs else None,
        "timestamp":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    print(
        f"\n   ✓ Text co-adapt completado: steps={result['total_steps']}  "
        f"best_val_loss={result['best_val_loss']}  @  step {result['best_step']}"
    )

    # ── 8. Liberar VRAM antes de la siguiente config ──────────────────────────
    del model, trainer, audit_cb, early_stop_cb, optimizer
    gc.collect()
    torch.cuda.empty_cache()

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 4.  GRÁFICOS COMPARATIVOS
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparative_curves(
    config_names: List[str],
    output_dir:   Path,
    plots_dir:    Path,
) -> None:
    """
    Genera 3 PNGs comparativos (Val Loss / BLEU-4 / ROUGE-L) para todas
    las configs del experimento de text co-adaptación.

    Lee los archivos history_phase3_text_coadapt_{name}.csv escritos por
    Phase3AuditCallback.

    Args:
        config_names : Nombres de las configuraciones a graficar.
        output_dir   : Directorio donde residen los CSV de historial.
        plots_dir    : Directorio de destino para los PNG.
    """
    plots_dir.mkdir(parents=True, exist_ok=True)
    palette = ["steelblue", "coral"]

    histories: Dict[str, pd.DataFrame] = {}
    for name in config_names:
        p = output_dir / f"history_phase3_text_coadapt_{name}.csv"
        if p.exists():
            histories[name] = pd.read_csv(p)
        else:
            print(f"   [plot] Sin historial para {name}: {p}")

    if not histories:
        print("   [plot] Sin historiales. Saltando gráficos.")
        return

    specs = [
        ("Val_Loss", "Val Loss",  "01_val_loss_phase3_text_coadapt.png"),
        ("BLEU-4",   "BLEU-4",   "02_bleu4_phase3_text_coadapt.png"),
        ("ROUGE-L",  "ROUGE-L",  "03_rougeL_phase3_text_coadapt.png"),
    ]

    for col, ylabel, filename in specs:
        fig, ax = plt.subplots(figsize=(13, 6))
        for i, (name, df) in enumerate(histories.items()):
            if col not in df.columns:
                continue
            ax.plot(
                df["Step"], df[col],
                label=name,
                color=palette[i % len(palette)],
                linewidth=2, marker="o", markersize=4,
            )
        ax.set_xlabel("Step", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(
            f"{ylabel} vs Steps — Fase 3 Text Co-adapt: Q-Former+LLM r=128 (ViT congelado, 10%)",
            fontsize=13, fontweight="bold",
        )
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        out_path = plots_dir / filename
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"   [plot] {filename} guardado → {out_path}")

    # Gráficos individuales: Train Loss vs Val Loss para cada config
    for i, (name, df) in enumerate(histories.items()):
        if "Train_Loss" not in df.columns or "Val_Loss" not in df.columns:
            continue

        df_filtered = df.dropna(subset=["Train_Loss", "Val_Loss"])
        if df_filtered.empty:
            continue

        fig, ax = plt.subplots(figsize=(13, 6))
        ax.plot(
            df_filtered["Step"], df_filtered["Train_Loss"],
            label="Train Loss",
            color="#2E86AB",
            linewidth=2, marker="o", markersize=4,
        )
        ax.plot(
            df_filtered["Step"], df_filtered["Val_Loss"],
            label="Val Loss",
            color="#A23B72",
            linewidth=2, marker="s", markersize=4,
        )
        ax.set_xlabel("Step", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title(
            f"Train Loss vs Val Loss — {name} (Fase 3 Text Co-adapt: Q-Former+LLM r=128)",
            fontsize=13, fontweight="bold",
        )
        ax.legend(fontsize=11, loc="best")
        ax.grid(alpha=0.3)
        plt.tight_layout()

        filename_individual = f"04_train_val_loss_text_coadapt_{name}.png"
        out_path_individual = plots_dir / filename_individual
        plt.savefig(out_path_individual, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"   [plot] {filename_individual} guardado → {out_path_individual}")

    print(f"\n   Todos los gráficos en: {plots_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  HELPER DE CARGA DE DATASETS
# ─────────────────────────────────────────────────────────────────────────────

def load_phase3_datasets(processor) -> Tuple[MimicCXRDataset, MimicCXRDataset]:
    """
    Carga los datasets de train y val del 10% de MIMIC-CXR (Fase Text Co-adapt).

    Args:
        processor : Blip2Processor ya instanciado.

    Returns:
        (train_dataset, val_dataset)
    """
    shared_kwargs = dict(
        images_dir           = IMAGES_DIR_10PCT,
        processor            = processor,
        image_size           = config.data.image_size,
        clahe_clip_limit     = config.data.clahe_clip_limit,
        clahe_tile_grid_size = config.data.clahe_tile_grid_size,
        max_length           = 512,
        padding              = config.data.padding,
        prompt               = config.inference.default_prompt,
    )

    print("   [datasets] Cargando Train (10%) — Text Co-adapt...")
    train_ds = MimicCXRDataset(csv_path=TRAIN_CSV_10PCT, **shared_kwargs)

    print("   [datasets] Cargando Val   (10%) — Text Co-adapt...")
    val_ds   = MimicCXRDataset(csv_path=VAL_CSV_10PCT,   **shared_kwargs)

    print(f"   [datasets] Train: {len(train_ds):,}  |  Val: {len(val_ds):,}\n")
    return train_ds, val_ds
