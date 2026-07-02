#!/usr/bin/env python3
"""
trainer_utils.py — Utilidades de entrenamiento para Fase 2.2
=============================================================
Escalamiento de finalistas: Config_A2 y Config_B1 al 10% de MIMIC-CXR.

Responsabilidades
-----------------
1. inject_dual_lora()
       • LLM (OPT)   : carga pesos pre-entrenados de Fase 1 como entrenables.
       • ViT          : inyecta adaptador LoRA NUEVO (alpha = r → scaling = 1.0).
       • Q-Former     : completamente congelado (requires_grad = False).

2. build_differential_optimizer()
       • ViT LoRA  params → lr = 5e-5  (evita olvido catastrófico del ViT)
       • LLM LoRA  params → lr = 1e-4  (continuidad con Fase 1)
       Devuelve (AdamW, LinearWarmup+Decay) listo para Trainer(optimizers=...).

3. Phase2AuditCallback
       Guarda AMBOS adaptadores (llm/ y vit/) solo si val_loss mejora.
       Registra history_phase2_{name}.csv y audit_phase2_{name}.txt.
       OPTIMIZACIÓN: skip_text_generation=True desactiva inferencia de texto
       durante evaluación (ahorra ~70% del tiempo en val).

4. train_single_config_phase2()
       Función principal: recibe una spec de config, entrena y devuelve
       un dict-resumen con métricas de la corrida.

5. plot_comparative_curves()
       Lee los CSV de historial y genera 3 PNGs comparativos.

Autor : Evelyn Silva Rozas
Fecha : Mayo 2026 (actualizado para Fase 2.2)
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
from tqdm.auto import tqdm as _tqdm
from transformers import (
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    get_linear_schedule_with_warmup,
)
from transformers.trainer_pt_utils import LengthGroupedSampler
from peft import LoraConfig, PeftModel, get_peft_model
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

from config import config
from data_loader import MimicCXRDataset
from model_utils import cargar_procesador, cargar_modelo_base

# ─────────────────────────────────────────────────────────────────────────────
# 🚨 FIX DEFINITIVO: GLOBAL MONKEY PATCH — Blip2VisionModel.forward
# ─────────────────────────────────────────────────────────────────────────────
# Raíz del problema
# -----------------
# Cuando PEFT envuelve el ViT con get_peft_model(), Accelerate/Trainer
# puede intentar pasar kwargs textuales (input_ids, attention_mask,
# inputs_embeds, labels, …) al forward() del ViT durante el paso de
# forward completo del modelo BLIP-2.  Blip2VisionModel.forward NO acepta
# ninguno de esos argumentos → TypeError.
#
# Solución: WHITELIST estricto en lugar de blacklist.
# En vez de listar qué kwargs eliminar (lista que puede crecer), declaramos
# EXACTAMENTE cuáles parámetros acepta Blip2VisionModel.forward (verificado
# contra la firma oficial de la versión instalada) y descartamos todo lo demás.
#
# Firma oficial (transformers ≥ 4.36):
#   forward(self,
#           pixel_values            : Optional[FloatTensor] = None,
#           output_attentions       : Optional[bool]        = None,
#           output_hidden_states    : Optional[bool]        = None,
#           return_dict             : Optional[bool]        = None,
#           interpolate_pos_encoding: bool                  = False)
# ─────────────────────────────────────────────────────────────────────────────
from transformers.models.blip_2.modeling_blip_2 import Blip2VisionModel

_original_vit_forward = Blip2VisionModel.forward

# Conjunto fijo de kwargs que Blip2VisionModel.forward acepta realmente.
# Si en el futuro se añade un parámetro a la firma bastará con añadirlo aquí.
_VIT_VALID_KWARGS = frozenset({
    "pixel_values",
    "output_attentions",
    "output_hidden_states",
    "return_dict",
    "interpolate_pos_encoding",
})


def _patched_vit_forward(self, *args, **kwargs):
    """
    Wrapper que filtra por WHITELIST antes de llamar al forward real del ViT.
    Cualquier kwarg no reconocido (input_ids, attention_mask, inputs_embeds,
    labels, decoder_input_ids, decoder_attention_mask, etc.) se descarta
    silenciosamente, evitando TypeError independientemente de qué intente
    inyectar Accelerate o PEFT.
    """
    filtered = {k: v for k, v in kwargs.items() if k in _VIT_VALID_KWARGS}
    return _original_vit_forward(self, *args, **filtered)


Blip2VisionModel.forward = _patched_vit_forward
# ─────────────────────────────────────────────────────────────────────────────

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

# ─────────────────────────────────────────────────────────────────────────────
# ⚡ OPTIMIZACIONES DE VELOCIDAD GLOBALES — L40S (SM 8.9, Ada Lovelace)
# ─────────────────────────────────────────────────────────────────────────────
# TOKENIZERS_PARALLELISM=false: los DataLoader workers hacen fork() DESPUÉS
# de que HuggingFace tokenizers ya inicializó su pool de threads.  Sin este
# flag, cada worker lanza un warning y deshabilita el pool en caliente,
# añadiendo overhead de inicialización por worker por epoch.
import os as _os
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# expandable_segments: el allocator de CUDA reserva bloques expandibles en
# lugar de bloques fijos.  Elimina OOM por fragmentación cuando hay memoria
# reservada pero no contigua (el mensaje "660 MiB reserved but unallocated").
_os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# ─────────────────────────────────────────────────────────────────────────────
# TF32: los Tensor Cores Ada realizan multiplicaciones de matrices en TF32
# (10-bit mantissa) con el mismo rango float32.  Impacto: ~2× en matmul
# sin cambio visible en loss (los pesos viven en bfloat16 de todos modos).
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32      = True
torch.set_float32_matmul_precision("high")   # PyTorch 2.0+
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: pre-computar longitudes desde el CSV (sin cargar imágenes)
# ─────────────────────────────────────────────────────────────────────────────

def precompute_train_lengths(dataset: "MimicCXRDataset") -> List[int]:
    """
    Calcula las longitudes de cada muestra a partir del texto en el CSV,
    sin cargar ninguna imagen.  Se usa para group_by_length=True con una
    barra de progreso visible en lugar del pase silencioso y lento del
    Trainer (que carga imagen por imagen).

    La longitud se aproxima como el número de palabras del reporte, que es
    un proxy suficiente para agrupar secuencias similares.
    """
    print(f"   [group_by_length] Pre-computando longitudes de {len(dataset):,} muestras...")
    col = "report_text" if "report_text" in dataset.data.columns else dataset.data.columns[-1]
    lengths = [
        len(str(txt).split())
        for txt in _tqdm(
            dataset.data[col],
            desc="   Longitudes",
            unit="muestras",
            ncols=90,
        )
    ]
    print(f"   [group_by_length] Listo. Longitud media: {int(np.mean(lengths))} palabras.")
    return lengths


class LengthAwareTrainer(Trainer):
    """
    Trainer que recibe longitudes pre-computadas y las pasa directamente
    a LengthGroupedSampler, evitando que HuggingFace itere el dataset
    completo (con carga de imágenes) para calcularlas en caliente.

    ADEMÁS sobreescribe el guardado/carga de checkpoints para persistir
    ÚNICAMENTE los dos adapters LoRA (LLM + ViT, ~14M params, ~560 MB) en
    lugar del modelo BLIP-2 completo (3.7B params, 7.1 GB).

    Motivación
    ----------
    El modelo NO es un PeftModel en su raíz (los adapters están en submódulos
    language_model y vision_model), por lo que el Trainer estándar serializa
    los 3.7B params completos cada save_step.  Eso causaba:
      1. Serialización CPU-bound lentísima (GPU 0%, 1 core al 100%).
      2. CUELGUE en safetensors por los pesos ATADOS de OPT
         (lm_head.weight ↔ embed_tokens.weight no se pueden serializar juntos).
      3. Checkpoints de 7.2 GB → presión de disco + corrupción si el proceso
         muere a mitad de escritura.
    Guardar solo los adapters elimina los tres problemas: es instantáneo,
    sin pesos atados, y los .safetensors pesan ~560 MB.
    """

    def __init__(self, *args, train_lengths: Optional[List[int]] = None,
                 eval_lengths: Optional[List[int]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._train_lengths = train_lengths
        self._eval_lengths = eval_lengths

    def _get_train_sampler(self):
        if (
            self.args.group_by_length
            and self._train_lengths is not None
            and self.train_dataset is not None
        ):
            return LengthGroupedSampler(
                batch_size=self.args.per_device_train_batch_size
                           * self.args.gradient_accumulation_steps,
                dataset=self.train_dataset,
                lengths=self._train_lengths,
            )
        return super()._get_train_sampler()

    def _get_eval_sampler(self, eval_dataset):
        """
        ⚠️ FIX DEADLOCK DE EVAL (cuelgue determinista en cada evaluación):
        Con group_by_length=True, el _get_eval_sampler ESTÁNDAR de transformers
        crea un LengthGroupedSampler con lengths=None (porque nuestro dataset NO
        es un datasets.Dataset).  Eso obliga al sampler a RECORRER las ~39K
        muestras de validación UNA A UNA cargando+tokenizando cada imagen en el
        HILO PRINCIPAL → spin determinista al 100 % de un core, GPU 0 %, sin
        avance (mismo bug que ya resolvimos en train con precompute_train_lengths).

        Solución: si tenemos longitudes pre-computadas desde el CSV (sin I/O de
        imágenes), las inyectamos directamente; si no, caemos a SequentialSampler
        (eval no necesita agrupar por longitud para ser correcto).
        """
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        if (
            self.args.group_by_length
            and self._eval_lengths is not None
            and eval_dataset is not None
            and len(self._eval_lengths) == len(eval_dataset)
        ):
            return LengthGroupedSampler(
                batch_size=self.args.per_device_eval_batch_size,
                dataset=eval_dataset,
                lengths=self._eval_lengths,
            )
        from torch.utils.data import SequentialSampler
        return SequentialSampler(eval_dataset)

    # ── Guardado de checkpoint: SOLO adapters LoRA ───────────────────────────
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """Guarda únicamente los dos adapters LoRA, no el modelo completo."""
        import os
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # PeftModel.save_pretrained escribe solo los deltas LoRA (instantáneo).
        self.model.language_model.save_pretrained(os.path.join(output_dir, "llm_adapter"))
        self.model.vision_model.save_pretrained(os.path.join(output_dir, "vit_adapter"))

        # training_args.bin: lo espera el Trainer para algunas verificaciones.
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    # ── Carga de adapters (helper compartido por resume y best-model) ────────
    def _load_adapters_from(self, ckpt_dir: str) -> None:
        import os
        from safetensors.torch import load_file
        from peft import set_peft_model_state_dict

        for subdir, peft_module in [
            ("llm_adapter", self.model.language_model),
            ("vit_adapter", self.model.vision_model),
        ]:
            adapter_dir = os.path.join(ckpt_dir, subdir)
            if not os.path.isdir(adapter_dir):
                continue
            # peft guarda como .safetensors (default) o .bin (versiones viejas).
            st_path  = os.path.join(adapter_dir, "adapter_model.safetensors")
            bin_path = os.path.join(adapter_dir, "adapter_model.bin")
            if os.path.exists(st_path):
                state_dict = load_file(st_path)
            elif os.path.exists(bin_path):
                state_dict = torch.load(bin_path, map_location="cpu")
            else:
                continue
            set_peft_model_state_dict(peft_module, state_dict)

    # ── Resume: cargar adapters en vez del modelo completo ───────────────────
    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        import os
        # Formato NUEVO (adapters) si existe el subdir; si no, checkpoint
        # ANTIGUO con modelo completo → delegamos al cargador estándar para
        # no perder el progreso de corridas previas (compatibilidad).
        if os.path.isdir(os.path.join(resume_from_checkpoint, "llm_adapter")):
            self._load_adapters_from(resume_from_checkpoint)
        else:
            super()._load_from_checkpoint(resume_from_checkpoint, model)

    # ── load_best_model_at_end: cargar los adapters del mejor checkpoint ─────
    def _load_best_model(self):
        import os
        ckpt = self.state.best_model_checkpoint
        if ckpt is None:
            return
        if os.path.isdir(os.path.join(ckpt, "llm_adapter")):
            self._load_adapters_from(ckpt)
        else:
            super()._load_best_model()


# ─────────────────────────────────────────────────────────────────────────────
# PATHS Y CONSTANTES
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR = config.paths.base_dir

# Fase 2.2 trabaja con el 10 % del dataset (escalamiento de finalistas)
TRAIN_CSV_10PCT  = BASE_DIR / "train_split_50pct.csv"
VAL_CSV_10PCT    = BASE_DIR / "val_split_50pct.csv"
TEST_CSV_10PCT   = BASE_DIR / "test_split_50pct.csv"
IMAGES_DIR_10PCT = BASE_DIR / "imagenes_50_porciento"

# Pesos LLM pre-entrenados de Fase 1 (Gran Final)
LLM_WEIGHTS: Dict[str, Path] = {
    "r16": BASE_DIR / "final_results" / "checkpoints" / "best_model_Config_A",
    "r32": BASE_DIR / "final_results" / "checkpoints" / "best_model_Config_B",
}

# Directorio raíz de salida para toda la Fase 2.1
OUTPUT_DIR      = BASE_DIR / "phase2_results-50pct"
CHECKPOINTS_DIR = OUTPUT_DIR / "checkpoints"

# ─────────────────────────────────────────────────────────────────────────────
# HIPERPARÁMETROS
# ─────────────────────────────────────────────────────────────────────────────

TRAINING_CONFIG: Dict = {
    # Learning rates diferenciales
    "lr_llm":                  1e-4,   # anclado al valor de Fase 1
    "lr_vit":                  5e-5,   # conservador para domain adaptation visual
    # Régimen de entrenamiento
    # max_epochs=30: el horizonte del cosine scheduler.  50 epochs sobre 133K
    # imágenes (= ver cada radiografía 50 veces) es excesivo para LoRA fine-tuning
    # desde pesos Fase 1.  Con early_stopping_patience=10 (6250 steps sin mejora)
    # el modelo converge mucho antes; reducir a 30 NO cambia la dinámica per-step
    # (mismo LR, batch efectivo, ranks) y mejora el decaimiento del cosine.
    # Worst-case: 68h (50 ep) → 41h (30 ep), ahorro ~27h.
    "max_epochs":             30,
    "early_stopping_patience": 10,      # evaluaciones consecutivas sin mejora
    # batch=16, grad_accum=4 → batch EFECTIVO = 64.
    # Se probó batch=32/accum=2 (mismo efectivo) pero hace OOM: con
    # gradient_checkpointing=False (apagado a propósito para -34 % tiempo) el
    # forward del OPT-2.7B retiene TODAS las activaciones para el backward;
    # batch=32 × 288 tokens llega a 44.1/44.4 GB → desborda.  El cuello de
    # botella real NO era el batch sino el I/O de imágenes (rglob), ya resuelto
    # con el índice O(1) + 32 workers, que saturan la GPU ya con batch=16.
    "batch_size":              16,
    "gradient_accumulation":   4,       # batch efectivo = 64 (16×4)
    "eval_steps":              625,
    "warmup_steps":            1500,
    # Generación para auditoría
    "audit_samples":           10,
    "max_new_tokens":          512,
    # LoRA
    "lora_dropout":            0.05,
}

# Módulos objetivo por sub-red
# OPT (LLM) usa proyecciones q_proj / v_proj en su mecanismo de atención
LLM_TARGET_MODULES = ["q_proj", "v_proj"]
# ViT BLIP-2 (EVA-CLIP) concatena Query, Key y Value en una sola capa llamada 'qkv'
VIT_TARGET_MODULES = ["qkv"]


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
    """Genera texto para una sola muestra (greedy, sin grad, compatible con
    gradient_checkpointing gracias a use_cache=False)."""
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
# 1.  INYECCIÓN DUAL LoRA
# ─────────────────────────────────────────────────────────────────────────────

def inject_dual_lora(
    llm_weights_dir: Path,
    vit_r: int,
    llm_r: int,
) -> "Blip2ForConditionalGeneration":
    """
    Construye el modelo BLIP-2 con dos adaptadores LoRA independientes.

    Estrategia de inyección
    -----------------------
    LLM (OPT)
        PeftModel.from_pretrained(..., is_trainable=True) sobre
        base_model.language_model, exactamente como en evaluate_final.py pero
        habilitando el gradiente para continuar el ajuste.

    ViT (Blip2VisionModel)
        get_peft_model() con un LoraConfig nuevo:
            task_type    = "FEATURE_EXTRACTION"
            target       = ["query", "value"]
            lora_alpha   = vit_r   →  scaling = alpha/r = 1.0
        Usar alpha = r es la práctica estándar para domain adaptation visual:
        evita amplificar perturbaciones en los pesos pre-entrenados del ViT.

    Q-Former
        Todos los parámetros se congelan explícitamente con requires_grad=False.

    Args:
        llm_weights_dir : Directorio con los adaptadores LoRA del LLM (Fase 1).
        vit_r           : Rango LoRA para el ViT (8 o 16).
        llm_r           : Rango LoRA del LLM (solo para logging).

    Returns:
        Modelo BLIP-2 listo para entrenamiento (bfloat16, device_map=auto).

    Raises:
        FileNotFoundError : Si llm_weights_dir no existe.
    """
    if not llm_weights_dir.exists():
        raise FileNotFoundError(
            f"Pesos LLM de Fase 1 no encontrados: {llm_weights_dir}\n"
            "Asegúrate de que train_semifinal.py haya concluido exitosamente."
        )

    print(f"   [dual_lora] Cargando modelo base BLIP-2 (bfloat16, device_map=auto)...")
    base_model = cargar_modelo_base(
        model_name=config.model.model_name,
        use_quantization=False,
        device_map="auto",
        # max_memory: Accelerate por defecto reserva el 90 % de VRAM (42.9 GB)
        # como buffer para el modelo, aunque este solo pesa ~7.8 GB.
        # Eso deja <5 GB para activaciones → fuerza gradient_checkpointing.
        # Limitando el buffer a 14 GB (más que suficiente para el modelo)
        # se liberan ~33 GB para activaciones → GC puede apagarse → -34 % tiempo.
        max_memory={0: "14GiB"},
    )

    # ── LLM LoRA: pesos de Fase 1, reactivados para entrenamiento ────────────
    print(f"   [dual_lora] Cargando LLM LoRA (r={llm_r}) desde {llm_weights_dir.name}...")
    base_model.language_model = PeftModel.from_pretrained(
        base_model.language_model,
        str(llm_weights_dir),
        is_trainable=True,      # <- CRÍTICO: habilita requires_grad en los adapters
    )
    n_llm = sum(p.numel() for p in base_model.language_model.parameters()
                if p.requires_grad)
    print(f"   [dual_lora] LLM entrenables : {n_llm:,} params")

    # ── ViT LoRA: adaptador nuevo (alpha = r → scaling = 1.0) ────────────────
    vit_lora_cfg = LoraConfig(
        r=vit_r,
        lora_alpha=vit_r,           # alpha == r  →  scaling factor = 1.0
        lora_dropout=TRAINING_CONFIG["lora_dropout"],
        bias="none",
        task_type="FEATURE_EXTRACTION",
        target_modules=VIT_TARGET_MODULES,
    )
    print(f"   [dual_lora] Inyectando ViT LoRA (r={vit_r}, alpha={vit_r}, "
          f"targets={VIT_TARGET_MODULES})...")
    base_model.vision_model = get_peft_model(base_model.vision_model, vit_lora_cfg)
    n_vit = sum(p.numel() for p in base_model.vision_model.parameters()
                if p.requires_grad)
    print(f"   [dual_lora] ViT entrenables : {n_vit:,} params")

    # ── Q-Former: congelar COMPLETAMENTE ─────────────────────────────────────
    n_qformer = 0
    for param in base_model.qformer.parameters():
        param.requires_grad = False
        n_qformer += param.numel()
    print(f"   [dual_lora] Q-Former congelado: {n_qformer:,} params")

    # ── Resumen de parámetros ─────────────────────────────────────────────────
    total     = sum(p.numel() for p in base_model.parameters())
    trainable = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    pct       = 100.0 * trainable / total if total > 0 else 0.0
    print(
        f"\n   ┌─── Parámetros del modelo ──────────────────────────┐\n"
        f"   │  Total        : {total:>15,}                      │\n"
        f"   │  Entrenables  : {trainable:>15,}  ({pct:.3f} %)          │\n"
        f"   │  Congelados   : {total - trainable:>15,}                      │\n"
        f"   └────────────────────────────────────────────────────┘\n"
    )

    return base_model


# ─────────────────────────────────────────────────────────────────────────────
# 2.  OPTIMIZADOR DIFERENCIAL
# ─────────────────────────────────────────────────────────────────────────────

def build_differential_optimizer(
    model,
) -> Tuple[AdamW, object]:
    """
    AdamW con dos grupos de parámetros y tasas de aprendizaje asimétricas.

    Grupos
    ------
    Grupo 0 (ViT LoRA)  lr = 5e-5
        Conservador: domain adaptation visual suave sin sobreescribir features
        pre-entrenados del ViT en ImageNet/LAION.

    Grupo 1 (LLM LoRA)  lr = 1e-4
        Igual que Fase 1 para mantener continuidad en la dirección de descenso.

    Nota: El scheduler se retorna como None para delegar su construcción 
    automática (Cosine + Warmup) directamente a los TrainingArguments del Trainer.
    """
    vit_params = [p for p in model.vision_model.parameters()  if p.requires_grad]
    llm_params = [p for p in model.language_model.parameters() if p.requires_grad]

    if not vit_params:
        raise RuntimeError("No hay parámetros entrenables en vision_model. "
                           "Verifica que inject_dual_lora() haya funcionado.")
    if not llm_params:
        raise RuntimeError("No hay parámetros entrenables en language_model. "
                           "Verifica que PeftModel.from_pretrained se llamó con "
                           "is_trainable=True.")

    optimizer = AdamW(
        [
            {"params": vit_params, "lr": TRAINING_CONFIG["lr_vit"]},
            {"params": llm_params, "lr": TRAINING_CONFIG["lr_llm"]},
        ],
        weight_decay=0.01,
        eps=1e-8,
        # fused=True: kernel único de CUDA para el paso del optimizador.
        # Reduce lecturas/escrituras de memoria de O(2N) a O(1) por parámetro.
        # Requiere PyTorch >= 2.0 y CUDA.  ~10-15% speedup en el optimizer step.
        fused=torch.cuda.is_available(),
    )

    print(
        f"   [optimizer] ViT lr={TRAINING_CONFIG['lr_vit']:.0e}  |  "
        f"LLM lr={TRAINING_CONFIG['lr_llm']:.0e}  |  "
        f"Scheduler dinámico delegado a Hugging Face"
    )
    
    return optimizer, None


# ─────────────────────────────────────────────────────────────────────────────
# 3.  CALLBACK DE AUDITORÍA Y CHECKPOINTS
# ─────────────────────────────────────────────────────────────────────────────

class Phase2AuditCallback(TrainerCallback):
    """
    Callback ejecutado después de cada evaluate() del Trainer.

    Responsabilidades
    -----------------
    Checkpoint dual
        Guarda language_model LoRA en  best_model_{name}/llm/
        Guarda vision_model   LoRA en  best_model_{name}/vit/
        SOLO si val_loss mejoró respecto al mejor registrado.

    Historial
        Appends a history_phase2_{config_name}.csv:
            Step | Val_Loss | BLEU-4 | ROUGE-L

    Audit log
        Appends a audit_phase2_{config_name}.txt con predicciones completas
        (sin truncar) para las muestras de auditoría.
    """

    def __init__(
        self,
        val_dataset:     MimicCXRDataset,
        processor,
        config_name:     str,
        output_dir:      Path,
        checkpoints_dir: Path,
        audit_samples:   int = 10,
        max_new_tokens:  int = 400,
        skip_text_generation: bool = False,  # ← FASE 2.2: desactiva inferencia en eval
    ):
        self.val_dataset     = val_dataset
        self.processor       = processor
        self.config_name     = config_name
        self.output_dir      = output_dir
        self.checkpoints_dir = checkpoints_dir
        self.audit_samples   = audit_samples
        self.max_new_tokens  = max_new_tokens
        self.skip_text_generation = skip_text_generation

        self.history_path   = output_dir / f"history_phase2_{config_name}.csv"
        self.audit_path     = output_dir / f"audit_phase2_{config_name}.txt"
        self.best_model_dir = checkpoints_dir / f"best_model_{config_name}"

        self.best_val_loss:      float      = float("inf")
        self.history:            List[Dict] = []

        # ── RESUME-AWARE: si ya existe historial de una corrida previa
        # (el contenedor mató el proceso y reanudamos desde checkpoint),
        # recuperamos la curva y el best_val_loss para NO sobrescribirlos.
        if self.history_path.exists():
            try:
                prev = pd.read_csv(self.history_path)
                self.history = prev.to_dict("records")
                valid_losses = prev["Val_Loss"].dropna()
                if len(valid_losses) > 0:
                    self.best_val_loss = float(valid_losses.min())
                print(
                    f"   [RESUME] Historial previo recuperado: "
                    f"{len(self.history)} evaluaciones, "
                    f"best_val_loss={self.best_val_loss:.4f}"
                )
            except Exception as exc:
                print(f"   [RESUME] No se pudo leer historial previo: {exc}")

        # Audit log: append si reanudamos, crear si es nuevo
        audit_mode = "a" if self.audit_path.exists() else "w"
        with open(self.audit_path, audit_mode, encoding="utf-8") as f:
            f.write(
                f"\n# Audit Log — Fase 2.1 — {config_name}\n"
                f"# {'Reanudado' if audit_mode == 'a' else 'Iniciado'}: "
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"# Columnas por entrada: Step | BLEU-4 | ROUGE-L | Pred | Real\n\n"
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

        # Dispositivo desde primer param entrenable (evita error si no hay ninguno)
        try:
            device = next(p for p in model.parameters() if p.requires_grad).device
        except StopIteration:
            device = next(model.parameters()).device

        model.eval()

        # — Generar predicciones en N muestras aleatorias del val set —
        # FASE 2.2 OPTIMIZACIÓN: skip_text_generation=True → solo track val_loss
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

        # Formatear train_loss condicionalmente
        train_loss_str = f"{train_loss:.4f}" if train_loss is not None else "N/A"
        print(
            f"\n   [Step {step}] train_loss={train_loss_str}  "
            f"val_loss={val_loss:.4f}  BLEU-4={mean_bleu4:.4f}  ROUGE-L={mean_rougeL:.4f}"
        )

        # — Guardar AMBOS adaptadores si val_loss mejoró —
        if val_loss is not None and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_model_dir.mkdir(parents=True, exist_ok=True)

            # LLM LoRA → best_model_{name}/llm/
            llm_save = self.best_model_dir / "llm"
            model.language_model.save_pretrained(str(llm_save))

            # ViT LoRA → best_model_{name}/vit/
            vit_save = self.best_model_dir / "vit"
            model.vision_model.save_pretrained(str(vit_save))

            print(
                f"   ✓ Checkpoint guardado (val_loss={val_loss:.4f})\n"
                f"     LLM → {llm_save}\n"
                f"     ViT → {vit_save}"
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
# 4.  FUNCIÓN PRINCIPAL DE ENTRENAMIENTO
# ─────────────────────────────────────────────────────────────────────────────

def train_single_config_phase2(
    config_dict:    Dict,
    config_num:     int,
    total_configs:  int,
    train_dataset:  MimicCXRDataset,
    val_dataset:    MimicCXRDataset,
    processor,
) -> Dict:
    """
    Entrena una configuración de Fase 2.1 de forma completamente aislada.

    Flujo
    -----
    1. Llama a inject_dual_lora() para construir el modelo con dual-LoRA.
    2. Calcula num_training_steps y llama a build_differential_optimizer().
    3. Configura TrainingArguments (bf16=True, gradient_checkpointing=True).
    4. Lanza Trainer con Phase2AuditCallback + EarlyStoppingCallback.
    5. Detecta checkpoints previos para reanudación automática (rescate).
    6. Al terminar, destruye modelo, trainer y optimizador, y libera VRAM.

    Args:
        config_dict   : Spec de la config con claves:
                        name, llm_r, vit_r, llm_weights_key ("r16" | "r32").
        config_num    : Índice 1-based para logging.
        total_configs : Total de configs en el torneo.
        train_dataset : Dataset de entrenamiento (1 % MIMIC-CXR).
        val_dataset   : Dataset de validación   (1 % MIMIC-CXR).
        processor     : Blip2Processor compartido.

    Returns:
        Dict con:
            config_name, llm_r, vit_r, lr_llm, lr_vit,
            total_steps, best_val_loss, best_step,
            final_train_loss, timestamp
    """
    name            = config_dict["name"]
    llm_r           = config_dict["llm_r"]
    vit_r           = config_dict["vit_r"]
    llm_weights_dir = LLM_WEIGHTS[config_dict["llm_weights_key"]]

    print(f"\n{'='*80}")
    print(f"  FASE 2.1 — Config {config_num}/{total_configs}: {name}")
    print(f"  LLM LoRA r={llm_r} (pesos Fase 1)  |  ViT LoRA r={vit_r} (nuevo)")
    print(f"  lr_LLM={TRAINING_CONFIG['lr_llm']:.0e}  "
          f"lr_ViT={TRAINING_CONFIG['lr_vit']:.0e}  "
          f"batch_ef={TRAINING_CONFIG['batch_size']*TRAINING_CONFIG['gradient_accumulation']}")
    print(f"{'='*80}\n")

    gc.collect()
    torch.cuda.empty_cache()

    # ── 1. Construir modelo con dual-LoRA ─────────────────────────────────────
    model = inject_dual_lora(
        llm_weights_dir=llm_weights_dir,
        vit_r=vit_r,
        llm_r=llm_r,
    )

    # ── 2. Optimizador diferencial ────────────────────────────────────────────
    steps_per_epoch    = max(1, len(train_dataset) // TRAINING_CONFIG["batch_size"])
    # num_training_steps es el número total de llamadas a optimizer.step()
    # (ya descontada la acumulación de gradiente)
    num_training_steps = (
        steps_per_epoch // TRAINING_CONFIG["gradient_accumulation"]
    ) * TRAINING_CONFIG["max_epochs"]

    optimizer, scheduler = build_differential_optimizer(model)

    # ── 3. Directorio de salida por config ────────────────────────────────────
    config_output_dir = OUTPUT_DIR / name
    config_output_dir.mkdir(parents=True, exist_ok=True)

    # ── 4. TrainingArguments ──────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=str(config_output_dir),
        num_train_epochs=TRAINING_CONFIG["max_epochs"],
        per_device_train_batch_size=TRAINING_CONFIG["batch_size"],
        per_device_eval_batch_size=32,      # eval sin gradientes (forward-only) → 2× el batch de train, seguro
        gradient_accumulation_steps=TRAINING_CONFIG["gradient_accumulation"],
        # learning_rate es requerido por TrainingArguments; el valor real
        # viene del optimizador diferencial que pasamos en `optimizers=`.
        learning_rate=TRAINING_CONFIG["lr_llm"],
        # Learning rate scheduler con decaimiento coseno y warmup
        lr_scheduler_type="cosine",         # decaimiento suave para convergencia estable
        warmup_ratio=0.1,                   # 10% de pasos iniciales para warmup gradual
        eval_strategy="steps",
        eval_steps=TRAINING_CONFIG["eval_steps"],
        save_strategy="steps",
        save_steps=TRAINING_CONFIG["eval_steps"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=100,
        logging_dir=str(config_output_dir / "logs"),
        # Precisión y memoria
        fp16=False,
        bf16=True,                          # nativo en L40S (Ada Lovelace)
        # gradient_checkpointing=False: con max_memory={0:"14GiB"} Accelerate
        # deja ~33 GB libres para activaciones. No necesitamos recomputar
        # el forward pass → ahorramos ~34 % del tiempo de cómputo.
        gradient_checkpointing=False,
        # torch.compile acelera ~10-15 % adicional en PyTorch 2.4, pero
        # puede causar retracing con formas dinámicas y el monkey-patch en
        # Blip2VisionModel.  Activar solo si el entrenamiento es estable.
        # torch_compile=True,
        # Otros
        report_to="none",
        remove_unused_columns=False,
        label_names=["labels"],
        # group_by_length=True: LengthAwareTrainer inyecta las longitudes
        # pre-computadas desde el CSV (sin cargar imágenes), por lo que
        # el muestreo agrupa secuencias similares SIN la espera de 4-6 min.
        group_by_length=True,
        # DataLoader: 32 workers (la máquina tiene 128 cores).  CLAHE + Data
        # Augmentation espacial son CPU-bound; con solo 12 workers el GPU se
        # quedaba hambriento esperando datos.  Con el índice O(1) de imágenes
        # (sin rglob) + 32 workers, la preparación deja de ser el cuello de
        # botella.  prefetch_factor=6: 32×6 = 192 muestras en RAM ahead del GPU.
        #
        # ⚠️ persistent_workers=False (OBLIGATORIO):
        # Con persistent_workers=True el entrenamiento se cuelga de forma
        # determinista en el PRIMER eval tras un *resume* (deadlock: el hilo
        # principal gira al 100 % de un core en userspace mientras los workers
        # quedan ociosos en do_poll esperando una petición que nunca llega).
        # Causa: al reanudar, los workers persistentes del train-loader nacen en
        # estado "skip_first_batches" y su handshake colisiona con la creación
        # del eval-loader (también persistente) bajo accelerate. Sin persistencia
        # los workers se crean/destruyen limpiamente por cada iteración; el costo
        # extra de fork (~5-10 s por epoch) es despreciable frente a epochs de horas.
        dataloader_num_workers=32,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=6,
        dataloader_persistent_workers=False,
        # drop_last: descarta el último batch parcial de cada época (<16 de 133K
        # muestras, distintas cada época por el shuffle → impacto despreciable).
        # Evita un batch de forma irregular que infrautiliza los Tensor Cores.
        dataloader_drop_last=True,
        # bf16_full_eval: evalúa en bf16 (igual que el training) → eval ~2× más
        # rápido y consistente con la precisión de entrenamiento.
        bf16_full_eval=True,
        save_total_limit=2,
    )

    # ── 5. Callbacks y collator ───────────────────────────────────────────────
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=processor.tokenizer, padding=True
    )

    audit_cb = Phase2AuditCallback(
        val_dataset=val_dataset,
        processor=processor,
        config_name=name,
        output_dir=OUTPUT_DIR,
        checkpoints_dir=CHECKPOINTS_DIR,
        audit_samples=TRAINING_CONFIG["audit_samples"],
        max_new_tokens=TRAINING_CONFIG["max_new_tokens"],
        skip_text_generation=True,  # ← FASE 2.2: desactiva inferencia para acelerar
    )
    early_stop_cb = EarlyStoppingCallback(
        early_stopping_patience=TRAINING_CONFIG["early_stopping_patience"]
    )

    # Pre-computar longitudes desde el CSV (sin I/O de imágenes, con tqdm)
    # para que group_by_length funcione sin el pase silencioso del Trainer.
    # CRÍTICO: hay que pre-computar TAMBIÉN las de validación, porque el
    # _get_eval_sampler estándar de transformers, con group_by_length=True,
    # recorrería las ~39K imágenes de val una a una en el hilo principal
    # (lengths=None) → spin/deadlock en CADA evaluación.
    train_lengths = precompute_train_lengths(train_dataset)
    eval_lengths  = precompute_train_lengths(val_dataset)

    trainer = LengthAwareTrainer(
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

    # ── 6. Detectar checkpoint previo para rescate ────────────────────────────
    # Orden NUMÉRICO (no lexicográfico): checkpoint-9999 < checkpoint-10000.
    # CRÍTICO: solo se consideran checkpoints VÁLIDOS (con trainer_state.json).
    # Si el contenedor mató el proceso mientras escribía un checkpoint, ese
    # directorio queda incompleto (sin trainer_state.json) y reanudar desde él
    # crashearía.  Se descartan y se usa el último checkpoint íntegro.
    all_ckpts = sorted(
        config_output_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1,
    )
    valid_ckpts = [c for c in all_ckpts if (c / "trainer_state.json").exists()]
    incomplete  = [c for c in all_ckpts if c not in valid_ckpts]

    for inc in incomplete:
        print(f"   [RESCATE] ⚠ Checkpoint incompleto descartado: {inc.name} "
              f"(sin trainer_state.json, proceso muerto durante el guardado)")

    if valid_ckpts:
        last = valid_ckpts[-1]
        print(f"   [RESCATE] {len(valid_ckpts)} checkpoint(s) válido(s). "
              f"Reanudando desde {last.name}...")
        trainer.train(resume_from_checkpoint=str(last))
    else:
        print("   [NUEVO] Sin checkpoints válidos previos. Iniciando desde cero.")
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
        "llm_r":            llm_r,
        "vit_r":            vit_r,
        "lr_llm":           TRAINING_CONFIG["lr_llm"],
        "lr_vit":           TRAINING_CONFIG["lr_vit"],
        "total_steps":      trainer.state.global_step,
        "best_val_loss":    best_val_loss,
        "best_step":        best_step,
        "final_train_loss": train_logs[-1]["loss"] if train_logs else None,
        "timestamp":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    print(
        f"\n   ✓ Completado: steps={result['total_steps']}  "
        f"best_val_loss={result['best_val_loss']}  @  step {result['best_step']}"
    )

    # ── 8. Liberar VRAM antes de la siguiente config ──────────────────────────
    del model, trainer, audit_cb, early_stop_cb, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 5.  GRÁFICOS COMPARATIVOS
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparative_curves(
    config_names: List[str],
    output_dir:   Path,
    plots_dir:    Path,
) -> None:
    """
    Genera 3 PNGs comparativos (Val Loss / BLEU-4 / ROUGE-L) para todas
    las configs del torneo.

    Lee los archivos history_phase2_{name}.csv escritos por Phase2AuditCallback.

    Args:
        config_names : Nombres de las configuraciones a graficar.
        output_dir   : Directorio donde residen los CSV de historial.
        plots_dir    : Directorio de destino para los PNG.
    """
    plots_dir.mkdir(parents=True, exist_ok=True)
    palette = ["steelblue", "coral", "seagreen", "darkorchid"]

    histories: Dict[str, pd.DataFrame] = {}
    for name in config_names:
        p = output_dir / f"history_phase2_{name}.csv"
        if p.exists():
            histories[name] = pd.read_csv(p)
        else:
            print(f"   [plot] Sin historial para {name}: {p}")

    if not histories:
        print("   [plot] Sin historiales. Saltando gráficos.")
        return

    # Gráficos comparativos entre configs (una métrica, todas las configs)
    specs = [
        ("Val_Loss", "Val Loss",  "01_val_loss_phase2.png"),
        ("BLEU-4",   "BLEU-4",   "02_bleu4_phase2.png"),
        ("ROUGE-L",  "ROUGE-L",  "03_rougeL_phase2.png"),
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
            f"{ylabel} vs Steps — Fase 2.1 Mini-Torneo (ViT+LLM LoRA)",
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
        
        # Filtrar valores nulos
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
            f"Train Loss vs Val Loss — {name} (Fase 2.2)",
            fontsize=13, fontweight="bold",
        )
        ax.legend(fontsize=11, loc="best")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        
        filename_individual = f"04_train_val_loss_{name}.png"
        out_path_individual = plots_dir / filename_individual
        plt.savefig(out_path_individual, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"   [plot] {filename_individual} guardado → {out_path_individual}")

    print(f"\n   Todos los gráficos en: {plots_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  HELPER DE CARGA DE DATASETS (compartido con run_tournament.py)
# ─────────────────────────────────────────────────────────────────────────────

def load_phase2_datasets(processor) -> Tuple[MimicCXRDataset, MimicCXRDataset]:
    """
    Carga los datasets de train y val del 10 % de MIMIC-CXR (Fase 2.2).

    Los CSVs y el directorio de imágenes son los del 10 %:
        train_split_10pct_limpio.csv   →  TRAIN_CSV_10PCT
        val_split_10pct_limpio.csv     →  VAL_CSV_10PCT
        imagenes_10_porciento/         →  IMAGES_DIR_10PCT

    Args:
        processor : Blip2Processor ya instanciado.

    Returns:
        (train_dataset, val_dataset)
    """
    shared_kwargs = dict(
        images_dir          = IMAGES_DIR_10PCT,
        processor           = processor,
        image_size          = config.data.image_size,
        clahe_clip_limit    = config.data.clahe_clip_limit,
        clahe_tile_grid_size= config.data.clahe_tile_grid_size,
        # max_length=256: el p99 real del dataset es 201 palabras (~241 tokens
        # BPE).  Reducir 512→256 da 4× menos cómputo de atención (O(n²)) y
        # libera ~3 GB de VRAM por forward pass, permitiendo batch más grande.
        max_length          = 256,
        padding             = config.data.padding,
        prompt              = config.inference.default_prompt,
    )

    print("   [datasets] Cargando Train (50 %) — CON data augmentation espacial...")
    train_ds = MimicCXRDataset(csv_path=TRAIN_CSV_10PCT, augment=True, **shared_kwargs)

    print("   [datasets] Cargando Val   (50 %) — SIN augmentation (evaluación limpia)...")
    val_ds   = MimicCXRDataset(csv_path=VAL_CSV_10PCT, augment=False, **shared_kwargs)

    print(f"   [datasets] Train: {len(train_ds):,}  |  Val: {len(val_ds):,}\n")
    return train_ds, val_ds
