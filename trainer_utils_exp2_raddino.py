#!/usr/bin/env python3
"""
trainer_utils_exp2_raddino.py — Experimento 2 (Fase 4): RadDINO + Puente Full-FT
================================================================================
Arquitectura multimodal personalizada basada en BLIP-2 que SUSTITUYE el
codificador visual genérico (EVA-CLIP) por un ViT médico preentrenado
(RadDINO, microsoft/rad-dino) y re-entrena ÚNICAMENTE el puente (Q-Former +
language_projection) por Full Fine-Tuning, con LoRA exclusivamente en RadDINO
y el LLM completamente congelado.

────────────────────────────────────────────────────────────────────────────
MAPA DE GRADIENTES (resumen ejecutivo)
────────────────────────────────────────────────────────────────────────────
  Componente                       | requires_grad | Técnica
  ---------------------------------+---------------+----------------------------
  vision_model.backbone (RadDINO)  | parcial       | LoRA (query, value)
  vision_model.projection (768→1408)| TRUE (100%)  | Full FT (capa NUEVA, ver ⚠)
  language_model (OPT-2.7b)        | FALSE (0%)    | LoRA Fase 2 fusionado + frozen
  qformer (Q-Former)               | TRUE (100%)   | Full FT, SIN LoRA
  language_projection              | TRUE (100%)   | Full FT, SIN LoRA
  query_tokens (32×768)            | TRUE (100%)   | Full FT (entrada del Q-Former)

⚠ PROYECTOR VISUAL 768→1408 — POR QUÉ EXISTE (decisión de arquitectura)
────────────────────────────────────────────────────────────────────────────
El Q-Former de BLIP-2 tiene su cross-attention dimensionada para recibir
features visuales de `encoder_hidden_size = 1408` (la salida del EVA-CLIP
original). RadDINO (DINOv2 ViT-B/14) produce embeddings de 768 dimensiones.

Para cumplir el requisito CRÍTICO de "conservar los pesos preentrenados del
Q-Former (NO inicializar con pesos aleatorios)", NO podemos reinicializar las
capas key/value del cross-attention del Q-Former a 768. La alternativa correcta
y mínimamente invasiva es interponer una proyección lineal entrenable
(768 → 1408) entre RadDINO y el Q-Former. Así:
    • RadDINO  → solo LoRA entrenable (backbone congelado).
    • proyector→ capa nueva entrenable (necesaria, ~1.08M params).
    • Q-Former → TODOS sus pesos preentrenados intactos y entrenables (Full FT).
Esta capa es el único añadido respecto al enunciado; se documenta y audita por
separado para total transparencia.

NOTA SOBRE EL TAMAÑO DE IMAGEN / NORMALIZACIÓN
────────────────────────────────────────────────────────────────────────────
RadDINO fue entrenado a 518×518 con su propia normalización. El pipeline del
proyecto entrega 224×224 (Blip2Processor + CLAHE). DINOv2 admite tamaños
arbitrarios vía `interpolate_pos_encoding=True` (forzado en el wrapper). La
LoRA del backbone + el proyector aprenden a alinear la distribución; aun así,
para máxima fidelidad se RECOMIENDA evaluar usar la normalización de RadDINO
en el data_loader en una iteración futura (fuera del alcance de esta función).

PROMPT
────────────────────────────────────────────────────────────────────────────
Se usa EXCLUSIVAMENTE el prompt diseñado del proyecto:
    DEFAULT_PROMPT = config.inference.default_prompt
No se utiliza self.prompt ni ningún otro prompt alternativo.

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
import inspect
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# cv2 DEBE desactivarse ANTES de cualquier import de torch/DataLoader
import cv2
cv2.setNumThreads(0)

import torch
import torch.nn as nn
from torch.optim import AdamW

from transformers import (
    AutoModel,
    Blip2ForConditionalGeneration,
)
from transformers.modeling_outputs import BaseModelOutputWithPooling
from peft import LoraConfig, PeftModel, get_peft_model

from config import config

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# 1. CONSTANTES, RUTAS E HIPERPARÁMETROS
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR: Path = config.paths.base_dir

# ── Modelo base BLIP-2 y codificador visual médico ──
BASE_MODEL_NAME: str = config.model.model_name          # "Salesforce/blip2-opt-2.7b"
RADDINO_MODEL_ID: str = "microsoft/rad-dino"            # ViT médico COMPLETO (DINOv2 ViT-B/14)

# ── Pesos LoRA de Fase 2 (modelo ganador Config_B1: LLM r=32, ViT r=8) ──
# Solo se reutiliza el adaptador del LLM (se fusiona y se congela). El ViT de
# Fase 2 NO se usa: el codificador visual se reemplaza por RadDINO.
PHASE2_BEST_DIR: Path = BASE_DIR / "phase2_results-50pct" / "checkpoints" / "best_model_Config_B1"
PHASE2_LLM_DIR:  Path = PHASE2_BEST_DIR / "llm"

# ── PROMPT DISEÑADO (único permitido) ──
DEFAULT_PROMPT: str = config.inference.default_prompt

# ── Datasets del 50% (mismos splits que Fase 2/3) ──
TRAIN_CSV:  Path = BASE_DIR / "train_split_50pct.csv"
VAL_CSV:    Path = BASE_DIR / "val_split_50pct.csv"
TEST_CSV:   Path = BASE_DIR / "test_split_50pct.csv"
IMAGES_DIR: Path = BASE_DIR / "imagenes_50_porciento"

# ── Salidas ──
OUTPUT_DIR:      Path = BASE_DIR / "exp2_raddino_results-50pct"
CHECKPOINTS_DIR: Path = OUTPUT_DIR / "checkpoints"

# ── LoRA para RadDINO (DINOv2: capas q/k/v separadas → ["query","value"]) ──
# Consistente con la convención del proyecto (Q-Former Fase 3 usó ["query","value"]).
# r=8 / alpha=16 (scaling=2.0): RadDINO ya es un encoder médico fuerte; basta una
# adaptación de bajo rango para alinearlo con el puente que se entrena de cero.
RADDINO_TARGET_MODULES: List[str] = ["query", "value"]
RADDINO_LORA_R:        int   = 8
RADDINO_LORA_ALPHA:    int   = 16
RADDINO_LORA_DROPOUT:  float = 0.05

# ── Hiperparámetros de entrenamiento (heredados de Fase 2/3 + ajuste de ViT) ──
# El usuario autorizó variar el LR de la LoRA del ViT. Se mantiene CONSERVADOR
# (5e-5, igual que el grupo ViT de la Fase 2) porque RadDINO ya está
# preentrenado en dominio médico: se prioriza preservar sus features y dejar
# que el PUENTE (lr=1e-4) haga el aprendizaje pesado de alineación.
TRAINING_CONFIG: Dict = {
    "lr_vit_lora":            5e-5,   # LoRA de RadDINO (conservador, dominio ya médico)
    "lr_bridge":              1e-4,   # Q-Former + language_projection + query_tokens + proyector
    "weight_decay":           0.01,
    "max_epochs":             30,
    "early_stopping_patience": 10,
    "batch_size":             12,     # Full FT del Q-Former → activaciones grandes
    "gradient_accumulation":  4,      # batch efectivo = 48
    "eval_steps":             625,
    "warmup_ratio":           0.1,
}

# Dimensión que el cross-attention del Q-Former espera (encoder_hidden_size).
QFORMER_ENCODER_HIDDEN_SIZE: int = 1408


# ─────────────────────────────────────────────────────────────────────────────
# 2. WRAPPER VISUAL: RadDINO + proyector 768 → 1408 (interfaz Blip2VisionModel)
# ─────────────────────────────────────────────────────────────────────────────

class RadDinoVisionTower(nn.Module):
    """
    Envuelve el backbone RadDINO (con LoRA) y añade un proyector lineal
    entrenable (hidden_raddino → encoder_hidden_size del Q-Former), exponiendo
    la MISMA interfaz que `Blip2VisionModel` para que el resto del modelo
    BLIP-2 funcione sin cambios.

    Interfaz replicada:
        forward(pixel_values, output_attentions, output_hidden_states,
                return_dict, interpolate_pos_encoding) -> BaseModelOutputWithPooling
        donde `last_hidden_state` ya está proyectado a `out_dim` (=1408).

    Detalles clave:
        • `interpolate_pos_encoding` se FUERZA a True internamente (las
          imágenes del pipeline son 224×224 y RadDINO es nativo 518×518).
        • Se ignoran kwargs textuales (input_ids, attention_mask, …) que
          Accelerate/PEFT pudieran inyectar (whitelist implícita por firma).
        • Expone `.config`, `.dtype`, `.main_input_name` y delega
          `gradient_checkpointing_enable` al backbone si existe.
    """

    main_input_name = "pixel_values"

    def __init__(
        self,
        backbone: nn.Module,
        in_dim: int,
        out_dim: int = QFORMER_ENCODER_HIDDEN_SIZE,
        force_interpolate: bool = True,
    ):
        super().__init__()
        self.backbone = backbone                       # RadDINO envuelto en PeftModel (LoRA)
        self.projection = nn.Linear(in_dim, out_dim)   # 768 → 1408 (capa NUEVA entrenable)
        self._force_interpolate = force_interpolate
        # Exponer la config del backbone para cualquier acceso a vision_model.config
        self.config = getattr(backbone, "config", None)

        # ── Detectar qué kwargs acepta el forward REAL del backbone ──────────
        # En transformers 4.46.3, Dinov2Model.forward NO acepta
        # `interpolate_pos_encoding` (la interpolación la hace SIEMPRE
        # Dinov2Embeddings según el H/W real → 224×224 funciona automáticamente).
        # Otras versiones sí lo aceptan. Se inspecciona una sola vez para pasar
        # solo los kwargs soportados y ser robusto entre versiones.
        try:
            _base_fwd = self.backbone.get_base_model().forward
            _params = set(inspect.signature(_base_fwd).parameters.keys())
        except (ValueError, TypeError, AttributeError):
            _params = {"pixel_values", "output_hidden_states", "return_dict"}
        self._accepts_ipe = "interpolate_pos_encoding" in _params
        self._accepts_ohs = "output_hidden_states" in _params

    @property
    def dtype(self) -> torch.dtype:
        return self.projection.weight.dtype

    def gradient_checkpointing_enable(self, **kwargs):
        """Delegación segura: activa GC en el backbone RadDINO si lo soporta."""
        fn = getattr(self.backbone, "gradient_checkpointing_enable", None)
        if callable(fn):
            try:
                fn(**kwargs)
            except TypeError:
                fn()

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        **kwargs,   # absorbe y descarta cualquier kwarg textual no visual
    ) -> BaseModelOutputWithPooling:
        # Construir kwargs solo con los que el backbone realmente acepta.
        call_kwargs = {"pixel_values": pixel_values, "return_dict": True}
        if self._accepts_ohs:
            call_kwargs["output_hidden_states"] = output_hidden_states
        if self._accepts_ipe:
            call_kwargs["interpolate_pos_encoding"] = (
                True if self._force_interpolate else bool(interpolate_pos_encoding)
            )

        # RadDINO (Dinov2Model) → last_hidden_state (B, N, in_dim=768).
        # Si el backbone no expone el flag, Dinov2Embeddings interpola SIEMPRE
        # las posiciones según el tamaño real de entrada (224×224 OK).
        vision_outputs = self.backbone(**call_kwargs)
        last_hidden = vision_outputs.last_hidden_state          # (B, N, 768)

        # Proyección al espacio que espera el cross-attention del Q-Former
        projected = self.projection(last_hidden)                # (B, N, 1408)

        return BaseModelOutputWithPooling(
            last_hidden_state=projected,
            pooler_output=None,
            hidden_states=getattr(vision_outputs, "hidden_states", None),
            attentions=None,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3. HELPER DE CONTEO DE PARÁMETROS (para la auditoría)
# ─────────────────────────────────────────────────────────────────────────────

def _contar(modulo: nn.Module) -> Tuple[int, int]:
    """Devuelve (entrenables, total) de un submódulo."""
    total = sum(p.numel() for p in modulo.parameters())
    train = sum(p.numel() for p in modulo.parameters() if p.requires_grad)
    return train, total


# ─────────────────────────────────────────────────────────────────────────────
# 4. FUNCIÓN PRINCIPAL: build_experiment_2_model
# ─────────────────────────────────────────────────────────────────────────────

def build_experiment_2_model(
    base_model_name: str = BASE_MODEL_NAME,
    raddino_model_id: str = RADDINO_MODEL_ID,
    ruta_lora_llm: Union[str, Path] = PHASE2_LLM_DIR,
    raddino_lora_r: int = RADDINO_LORA_R,
    raddino_lora_alpha: int = RADDINO_LORA_ALPHA,
    raddino_lora_dropout: float = RADDINO_LORA_DROPOUT,
    raddino_target_modules: Optional[List[str]] = None,
    train_query_tokens: bool = True,
    max_memory_gib: int = 16,
) -> Blip2ForConditionalGeneration:
    """
    Construye la arquitectura del Experimento 2 (RadDINO + puente Full-FT).

    Pasos (estrictamente en este orden):
        1. Carga base : Blip2ForConditionalGeneration en bfloat16 (device_map='auto').
        2. LLM        : carga LoRA de Fase 2 → merge_and_unload() → CONGELA todo
                        el language_model (requires_grad = False).
        3. ViT (RadDINO):
              a) carga RadDINO COMPLETO (AutoModel, bfloat16).
              b) aplica LoRA SOLO a RadDINO (target = query, value).
              c) lo envuelve en RadDinoVisionTower (añade proyector 768→1408)
                 y reemplaza base_model.vision_model.
        4. Puente     : qformer + language_projection conservan los pesos
                        preentrenados de BLIP-2 y se liberan (requires_grad=True,
                        Full FT, SIN LoRA). query_tokens también (configurable).
        5. Auditoría  : imprime parámetros entrenables de ViT, LLM y Q-Former
                        (+ proyector y query_tokens) y verifica las invariantes.

    Args:
        base_model_name      : ID del modelo base BLIP-2.
        raddino_model_id     : ID de RadDINO en Hugging Face (modelo COMPLETO).
        ruta_lora_llm        : Directorio del adaptador LoRA del LLM (Fase 2).
        raddino_lora_r       : Rango LoRA para RadDINO.
        raddino_lora_alpha   : Alpha LoRA para RadDINO.
        raddino_lora_dropout : Dropout LoRA para RadDINO.
        raddino_target_modules: Módulos objetivo de LoRA en RadDINO.
        train_query_tokens   : Si liberar query_tokens (entrada del Q-Former).
                               Por defecto True (precedente Fase 3 PUENTE).
        max_memory_gib       : Tope de buffer de Accelerate al cargar la base.

    Returns:
        Blip2ForConditionalGeneration listo para entrenar el Experimento 2.

    Raises:
        FileNotFoundError : si no existe el adaptador LoRA del LLM.
        RuntimeError      : si la auditoría de gradientes no cuadra.
    """
    raddino_target_modules = raddino_target_modules or RADDINO_TARGET_MODULES
    ruta_lora_llm = Path(ruta_lora_llm)
    if not ruta_lora_llm.exists():
        raise FileNotFoundError(
            f"Adaptador LoRA del LLM (Fase 2) no encontrado: {ruta_lora_llm}\n"
            "Verifica phase2_results-50pct/checkpoints/best_model_Config_B1/llm/."
        )

    print("\n" + "═" * 78)
    print("  BUILD · EXPERIMENTO 2 — RadDINO + Puente Full-FT (BLIP-2 personalizado)")
    print("═" * 78)
    print(f"  base_model   : {base_model_name}")
    print(f"  RadDINO      : {raddino_model_id}")
    print(f"  LLM LoRA F2  : {ruta_lora_llm}")
    print("═" * 78)

    # ── Paso 1: Carga base BLIP-2 en bfloat16 ────────────────────────────────
    print("\n  [1/5] Cargando modelo base BLIP-2 (bfloat16, device_map='auto')...")
    model = Blip2ForConditionalGeneration.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory={0: f"{max_memory_gib}GiB"},
    )
    # qformer es un submódulo base que nunca cambia de device → referencia fiable
    _ref = next(model.qformer.parameters())
    target_device, target_dtype = _ref.device, _ref.dtype
    print(f"        device={target_device}  dtype={target_dtype}")

    # ── Paso 2: LLM — cargar LoRA Fase 2, fusionar y CONGELAR ─────────────────
    print(f"\n  [2/5] LLM · cargando LoRA Fase 2 → merge_and_unload() → congelar...")
    llm_peft = PeftModel.from_pretrained(
        model.language_model, str(ruta_lora_llm), is_trainable=False
    )
    model.language_model = llm_peft.merge_and_unload().to(
        device=target_device, dtype=target_dtype
    )
    del llm_peft
    gc.collect()
    torch.cuda.empty_cache()
    n_llm_frozen = 0
    for p in model.language_model.parameters():
        p.requires_grad = False
        n_llm_frozen += p.numel()
    print(f"        ✓ LLM fusionado y CONGELADO: {n_llm_frozen:,} params (requires_grad=False)")

    # ── Paso 3: ViT — reemplazar EVA-CLIP por RadDINO + LoRA + proyector ──────
    print(f"\n  [3/5] ViT · cargando RadDINO COMPLETO ({raddino_model_id})...")
    raddino = AutoModel.from_pretrained(
        raddino_model_id,
        torch_dtype=target_dtype,
    )
    raddino_hidden = int(getattr(raddino.config, "hidden_size", 768))
    print(f"        RadDINO hidden_size = {raddino_hidden}  "
          f"(Q-Former espera {QFORMER_ENCODER_HIDDEN_SIZE} → se interpone proyector)")

    # LoRA EXCLUSIVAMENTE sobre RadDINO (backbone congelado, solo LoRA entrenable).
    # task_type=None (NO "FEATURE_EXTRACTION"): el wrapper PeftModelForFeatureExtraction
    # está pensado para encoders de TEXTO e inyecta `input_ids/attention_mask/inputs_embeds`
    # al reenviar, lo que rompe Dinov2Model.forward (TypeError: unexpected 'input_ids').
    # Con task_type=None, PeftModel.forward hace passthrough puro de kwargs visuales y la
    # LoRA sigue activa (las capas query/value ya son lora.Linear). El congelamiento del
    # backbone y el marcado de la LoRA como entrenable lo realiza get_peft_model igual.
    raddino_lora_cfg = LoraConfig(
        r=raddino_lora_r,
        lora_alpha=raddino_lora_alpha,
        lora_dropout=raddino_lora_dropout,
        bias="none",
        task_type=None,
        target_modules=raddino_target_modules,
    )
    print(f"        Inyectando LoRA en RadDINO (r={raddino_lora_r}, "
          f"alpha={raddino_lora_alpha}, targets={raddino_target_modules})...")
    raddino_peft = get_peft_model(raddino, raddino_lora_cfg)

    # Envolver en la torre visual (añade el proyector 768→1408) y reemplazar
    vision_tower = RadDinoVisionTower(
        backbone=raddino_peft,
        in_dim=raddino_hidden,
        out_dim=QFORMER_ENCODER_HIDDEN_SIZE,
        force_interpolate=True,
    ).to(device=target_device, dtype=target_dtype)
    model.vision_model = vision_tower
    # Mantener coherente la config del modelo (informativo; el forward usa la
    # salida real del proyector, no este campo).
    try:
        model.config.vision_config.hidden_size = QFORMER_ENCODER_HIDDEN_SIZE
    except Exception:
        pass

    n_vit_train, n_vit_total = _contar(model.vision_model)
    n_proj_train, n_proj_total = _contar(model.vision_model.projection)
    print(f"        ✓ RadDINO + proyector instalados. "
          f"ViT entrenable={n_vit_train:,} / total={n_vit_total:,}")

    # ── Paso 4: Puente — liberar qformer + language_projection (+ query_tokens) ─
    print(f"\n  [4/5] Puente · liberando qformer + language_projection "
          f"(Full FT, SIN LoRA)...")
    n_qformer = 0
    for p in model.qformer.parameters():
        p.requires_grad = True
        n_qformer += p.numel()
    n_proj_lang = 0
    for p in model.language_projection.parameters():
        p.requires_grad = True
        n_proj_lang += p.numel()

    n_query = 0
    if hasattr(model, "query_tokens") and isinstance(model.query_tokens, nn.Parameter):
        model.query_tokens.requires_grad = bool(train_query_tokens)
        if train_query_tokens:
            n_query = model.query_tokens.numel()
    print(f"        Q-Former            : {n_qformer:,} params (Full FT)")
    print(f"        language_projection : {n_proj_lang:,} params (Full FT)")
    print(f"        query_tokens        : {n_query:,} params "
          f"({'Full FT' if train_query_tokens else 'CONGELADO'})")
    print(f"        proyector visual    : {n_proj_train:,} params (Full FT, NUEVO)")

    # ── Paso 5: AUDITORÍA DE PARÁMETROS ──────────────────────────────────────
    _auditar_experiment_2(model)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 5. AUDITORÍA DE GRADIENTES (las 3 métricas requeridas + verificación)
# ─────────────────────────────────────────────────────────────────────────────

def _auditar_experiment_2(model: Blip2ForConditionalGeneration) -> None:
    """
    Imprime las tres métricas solicitadas y verifica las invariantes:
        a) ViT entrenable  → SOLO LoRA (+ proyector, contabilizado aparte).
        b) LLM entrenable  → 0.
        c) Q-Former entren.→ 100% del submódulo.
    """
    # Backbone RadDINO (sin contar el proyector) → solo LoRA debe ser entrenable
    backbone = model.vision_model.backbone
    n_bb_train, n_bb_total = _contar(backbone)
    n_proj_v_train, n_proj_v_total = _contar(model.vision_model.projection)

    n_llm_train, n_llm_total = _contar(model.language_model)
    n_qf_train,  n_qf_total  = _contar(model.qformer)
    n_lp_train,  n_lp_total  = _contar(model.language_projection)

    n_query = (model.query_tokens.numel()
               if (hasattr(model, "query_tokens")
                   and model.query_tokens.requires_grad) else 0)

    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = 100.0 * train / total if total else 0.0

    print("\n" + "═" * 78)
    print("  AUDITORÍA DE PARÁMETROS · EXPERIMENTO 2 (RadDINO + Puente Full-FT)")
    print("═" * 78)
    print("  (a) ViT — RadDINO backbone (esperado: solo LoRA)")
    print(f"        entrenables : {n_bb_train:>14,}  / total {n_bb_total:,}  "
          f"({100.0*n_bb_train/max(n_bb_total,1):.3f}%)")
    print(f"        + proyector visual 768→1408 (NUEVO, Full FT): "
          f"{n_proj_v_train:>12,} / {n_proj_v_total:,}")
    print("  (b) LLM — OPT-2.7b (esperado: 0)")
    print(f"        entrenables : {n_llm_train:>14,}  / total {n_llm_total:,}")
    print("  (c) Q-Former (esperado: 100% del submódulo)")
    print(f"        entrenables : {n_qf_train:>14,}  / total {n_qf_total:,}  "
          f"({100.0*n_qf_train/max(n_qf_total,1):.3f}%)")
    print("  (+) language_projection (Full FT) : "
          f"{n_lp_train:>12,} / {n_lp_total:,}")
    print("  (+) query_tokens (Full FT)        : "
          f"{n_query:>12,}")
    print("  ──────────────────────────────────────────────────────────────────")
    print(f"  TOTAL modelo        : {total:>15,}")
    print(f"  TOTAL entrenable    : {train:>15,}  ({pct:.3f}%)")
    print(f"  TOTAL congelado     : {total-train:>15,}")
    print("═" * 78)

    # ── Verificación de invariantes ──────────────────────────────────────────
    errores = []
    # (b) LLM debe estar 100% congelado
    if n_llm_train != 0:
        errores.append(f"LLM tiene {n_llm_train:,} params entrenables (esperado 0).")
    # (c) Q-Former debe estar 100% entrenable
    if n_qf_train != n_qf_total:
        errores.append(f"Q-Former entrenable {n_qf_train:,} ≠ total {n_qf_total:,} (esperado 100%).")
    # language_projection 100% entrenable
    if n_lp_train != n_lp_total:
        errores.append(f"language_projection entrenable {n_lp_train:,} ≠ total {n_lp_total:,}.")
    # (a) RadDINO backbone: solo una fracción (LoRA) entrenable, nunca el 100%
    if n_bb_train == 0:
        errores.append("RadDINO no tiene LoRA entrenable (esperado solo-LoRA > 0).")
    if n_bb_train >= n_bb_total:
        errores.append(f"RadDINO entrenable {n_bb_train:,} == total: ¿backbone no congelado?")

    if errores:
        print("  ✗ AUDITORÍA FALLIDA:")
        for e in errores:
            print(f"      · {e}")
        raise RuntimeError(
            "La auditoría de gradientes del Experimento 2 no cumple las "
            "invariantes (ViT=solo LoRA, LLM=0, Q-Former=100%). Revisa "
            "build_experiment_2_model()."
        )
    print("  ✓ Invariantes OK: ViT=solo LoRA · LLM=0 · Q-Former=100% · proj/query Full-FT")
    print("═" * 78 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# 6. OPTIMIZADOR DIFERENCIAL (LoRA ViT vs. Puente Full-FT)
# ─────────────────────────────────────────────────────────────────────────────

def build_experiment_2_optimizer(model: Blip2ForConditionalGeneration) -> Tuple[AdamW, None]:
    """
    AdamW con dos grupos de parámetros y tasas asimétricas:

        Grupo 0 (LoRA de RadDINO)        lr = TRAINING_CONFIG["lr_vit_lora"] (5e-5)
            Conservador: RadDINO ya está preentrenado en dominio médico; una
            adaptación suave evita degradar sus features.

        Grupo 1 (Puente Full-FT)         lr = TRAINING_CONFIG["lr_bridge"] (1e-4)
            proyector visual + query_tokens + qformer + language_projection:
            el bloque que debe APRENDER a conectar RadDINO con OPT (congelado).

    El scheduler (cosine + warmup) se delega a TrainingArguments.
    """
    # Grupo 0: SOLO los parámetros LoRA del backbone RadDINO
    vit_lora_params = [
        p for p in model.vision_model.backbone.parameters() if p.requires_grad
    ]
    # Grupo 1: puente = proyector + query_tokens + qformer + language_projection
    bridge_params: List[nn.Parameter] = []
    bridge_params += [p for p in model.vision_model.projection.parameters() if p.requires_grad]
    if hasattr(model, "query_tokens") and model.query_tokens.requires_grad:
        bridge_params.append(model.query_tokens)
    bridge_params += [p for p in model.qformer.parameters() if p.requires_grad]
    bridge_params += [p for p in model.language_projection.parameters() if p.requires_grad]

    if not vit_lora_params:
        raise RuntimeError("No hay parámetros LoRA entrenables en RadDINO.")
    if not bridge_params:
        raise RuntimeError("No hay parámetros entrenables en el puente.")

    optimizer = AdamW(
        [
            {"params": vit_lora_params, "lr": TRAINING_CONFIG["lr_vit_lora"]},
            {"params": bridge_params,   "lr": TRAINING_CONFIG["lr_bridge"]},
        ],
        weight_decay=TRAINING_CONFIG["weight_decay"],
        eps=1e-8,
        fused=torch.cuda.is_available(),
    )
    n_vit  = sum(p.numel() for p in vit_lora_params)
    n_brid = sum(p.numel() for p in bridge_params)
    print(f"   [optimizer] Grupo 0 · RadDINO LoRA : lr={TRAINING_CONFIG['lr_vit_lora']:.0e}  "
          f"({n_vit:,} params)")
    print(f"   [optimizer] Grupo 1 · Puente FullFT: lr={TRAINING_CONFIG['lr_bridge']:.0e}  "
          f"({n_brid:,} params)")
    print(f"   [optimizer] weight_decay={TRAINING_CONFIG['weight_decay']}  "
          f"| scheduler (cosine+warmup) delegado a HuggingFace")
    return optimizer, None


# ─────────────────────────────────────────────────────────────────────────────
# 7. CONSTRUCCIÓN DE DATASETS (usa EXCLUSIVAMENTE el prompt diseñado)
# ─────────────────────────────────────────────────────────────────────────────

def build_experiment_2_datasets(processor):
    """
    Construye los datasets de train/val del 50% inyectando ÚNICAMENTE el
    prompt diseñado (DEFAULT_PROMPT = config.inference.default_prompt).
    NO se usa self.prompt ni ningún otro prompt.
    """
    from data_loader import MimicCXRDataset

    print(f"   [datasets] prompt diseñado = {DEFAULT_PROMPT!r}")
    train_ds = MimicCXRDataset(
        csv_path             = TRAIN_CSV,
        images_dir           = IMAGES_DIR,
        processor            = processor,
        image_size           = config.data.image_size,
        clahe_clip_limit     = config.data.clahe_clip_limit,
        clahe_tile_grid_size = config.data.clahe_tile_grid_size,
        max_length           = config.data.max_length,
        padding              = config.data.padding,
        prompt               = DEFAULT_PROMPT,   # ← prompt diseñado (único permitido)
        augment              = True,
    )
    val_ds = MimicCXRDataset(
        csv_path             = VAL_CSV,
        images_dir           = IMAGES_DIR,
        processor            = processor,
        image_size           = config.data.image_size,
        clahe_clip_limit     = config.data.clahe_clip_limit,
        clahe_tile_grid_size = config.data.clahe_tile_grid_size,
        max_length           = config.data.max_length,
        padding              = config.data.padding,
        prompt               = DEFAULT_PROMPT,   # ← prompt diseñado (único permitido)
        augment              = False,
    )
    return train_ds, val_ds


# ─────────────────────────────────────────────────────────────────────────────
# 8. SMOKE TEST (construcción + auditoría sin entrenar)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Smoke test de build_experiment_2_model (construye + audita)."
    )
    parser.add_argument("--no-query-tokens", action="store_true",
                        help="Congelar query_tokens (solo qformer + language_projection).")
    parser.add_argument("--forward", action="store_true",
                        help="Probar un forward dummy con pixel_values aleatorios.")
    args = parser.parse_args()

    print("Construyendo modelo del Experimento 2 (esto descarga RadDINO si falta)...")
    model = build_experiment_2_model(train_query_tokens=not args.no_query_tokens)

    # Optimizador (verifica que los grupos de parámetros son coherentes)
    build_experiment_2_optimizer(model)

    if args.forward:
        print("\n  [smoke] Probando forward visual dummy (1×3×224×224)...")
        dev = next(model.qformer.parameters()).device
        dummy = torch.randn(1, 3, 224, 224, dtype=model.vision_model.dtype, device=dev)
        with torch.no_grad():
            vout = model.vision_model(pixel_values=dummy)
        print(f"  [smoke] last_hidden_state proyectado: {tuple(vout.last_hidden_state.shape)} "
              f"(esperado dim final = {QFORMER_ENCODER_HIDDEN_SIZE})")
        assert vout.last_hidden_state.shape[-1] == QFORMER_ENCODER_HIDDEN_SIZE

    print("✓ Smoke test completado.")
