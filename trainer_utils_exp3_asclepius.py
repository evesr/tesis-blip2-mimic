#!/usr/bin/env python3
"""
trainer_utils_exp3_asclepius.py — Experimento 3 (Fase 5): ViT(Fase 3) + Asclepius-R + Puente Full-FT
=====================================================================================================
Arquitectura multimodal basada en BLIP-2 que:

  • Conserva el CODIFICADOR VISUAL afinado en la Fase 3 (EVA-CLIP de BLIP-2 con
    la LoRA del ViT de la Fase 2 ya FUSIONADA) y lo CONGELA al 100 %.
  • SUSTITUYE el decodificador genérico (OPT-2.7b) por un LLM clínico
    especializado, **Asclepius-R** (LLaMA pre-entrenado y afinado sobre notas
    REALES de MIMIC-III), y lo CONGELA al 100 % (SIN LoRA).
  • Re-entrena ÚNICAMENTE el PUENTE (Q-Former + language_projection + query_tokens)
    por Full Fine-Tuning, conservando sus pesos preentrenados de BLIP-2 (sin
    inicialización aleatoria y SIN LoRA).

────────────────────────────────────────────────────────────────────────────
MAPA DE GRADIENTES (resumen ejecutivo)
────────────────────────────────────────────────────────────────────────────
  Componente                         | requires_grad | Técnica
  -----------------------------------+---------------+--------------------------
  vision_model (EVA-CLIP + LoRA F3)  | FALSE (0%)    | LoRA Fase 2/3 fusionada + frozen
  language_model (Asclepius-R/LLaMA) | FALSE (0%)    | Reemplazo total, frozen, SIN LoRA
  qformer (Q-Former)                 | TRUE (100%)   | Full FT, SIN LoRA (pesos BLIP-2)
  language_projection[0] (768→2560)  | TRUE (100%)   | Full FT (pesos BLIP-2 preservados)
  language_projection[1] (2560→H_llm)| TRUE (100%)   | Full FT (adaptador NUEVO, ver ⚠)
  query_tokens (32×768)              | TRUE (100%)   | Full FT (entrada del Q-Former)

⚠ ADAPTADOR DE PROYECCIÓN 2560 → H_llm — POR QUÉ EXISTE (decisión de arquitectura)
────────────────────────────────────────────────────────────────────────────
La `language_projection` original de BLIP-2 es `Linear(768 → 2560)`: mapea la
salida del Q-Former al espacio de embeddings de OPT-2.7b (hidden_size = 2560).
Asclepius-R está basado en LLaMA, cuyo hidden_size es 4096 (7B) o 5120 (13B).

El requisito CRÍTICO es CONSERVAR los pesos preentrenados de `language_projection`
(NO reinicializar). No podemos por tanto cambiar su dimensión de salida a 4096.
La solución correcta y mínimamente invasiva (idéntica en espíritu al proyector
visual 768→1408 del Experimento 2) es interponer un adaptador lineal entrenable
`Linear(2560 → H_llm)` DESPUÉS de la proyección preentrenada:

    language_projection = nn.Sequential(
        Linear(768 → 2560)   # ← PESOS PREENTRENADOS DE BLIP-2 (intactos, Full FT)
        Linear(2560 → H_llm) # ← CAPA NUEVA entrenable (única añadida; se audita aparte)
    )

Así, `model.language_projection(query_output)` produce embeddings de dimensión
H_llm y el `forward()`/`generate()` de stock de BLIP-2 funcionan SIN cambios
(ambos solo invocan `self.language_projection(query_output)`). Los pesos
preentrenados quedan en el elemento [0]; el adaptador [1] es el único añadido
respecto al enunciado, documentado y auditado por separado para total
transparencia. Ambos sub-módulos son 100 % entrenables (Full FT).

CONSISTENCIA CON EL forward()/generate() DE BLIP-2 (transformers 4.46.3)
────────────────────────────────────────────────────────────────────────────
  • La pérdida usa `self.config.text_config.vocab_size`  → se ACTUALIZA
    `model.config.text_config = asclepius.config` (vocab 32000 de LLaMA).
  • `generate()` antepone `self.config.text_config.bos_token_id`  → queda el de
    LLaMA (=1) tras la actualización de config.
  • `image_token_index` se FUERZA a None en build_experiment_3_model. ⚠ En
    transformers ≥4.46 el config base de blip2-opt-2.7b trae
    `image_token_index = 50265` (≠ None), lo que activaría la ruta
    `masked_scatter` (exige placeholders en input_ids). Al forzar None se usa la
    ruta de CONCATENACIÓN [visual ; texto], única compatible con nuestro
    data_loader (input_ids de solo texto) y con el cálculo de la pérdida.
  • El TOKENIZER debe ser el de Asclepius/LLaMA (use_fast=False). Se provee
    `preparar_procesador_exp3()` para intercambiar `processor.tokenizer`.

PROMPT
────────────────────────────────────────────────────────────────────────────
Se usa EXCLUSIVAMENTE el prompt diseñado del proyecto:
    DEFAULT_PROMPT = config.inference.default_prompt
No se utiliza self.prompt ni ningún prompt dinámico/nativo de Asclepius.

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
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# cv2 DEBE desactivarse ANTES de cualquier import de torch/DataLoader
import cv2
cv2.setNumThreads(0)

import torch
import torch.nn as nn
from torch.optim import AdamW

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Blip2ForConditionalGeneration,
)
from peft import PeftModel

from config import config

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# 1. CONSTANTES, RUTAS E HIPERPARÁMETROS
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR: Path = config.paths.base_dir

# ── Modelo base BLIP-2 ──
BASE_MODEL_NAME: str = config.model.model_name          # "Salesforce/blip2-opt-2.7b"

# ── ViT de la Fase 3 = EVA-CLIP de BLIP-2 + LoRA del ViT de Fase 2 (Config_B1, r=8) ──
# En la Fase 3 (Puente) este ViT se fusionó (merge_and_unload) y se CONGELÓ. El
# Experimento 3 reutiliza EXACTAMENTE ese codificador visual afinado.
PHASE2_BEST_DIR: Path = BASE_DIR / "phase2_results-50pct" / "checkpoints" / "best_model_Config_B1"
PHASE2_VIT_DIR:  Path = PHASE2_BEST_DIR / "vit"        # adaptador LoRA del ViT (Fase 2/3)

# ── Asclepius-R (LLaMA clínico entrenado con notas REALES de MIMIC-III) ──
# ⚠ MODELO DE ACCESO CREDENCIADO (PhysioNet). NO se distribuye con el repo.
#   El usuario debe descargarlo (ver ACCION OBLIGATORIA en run_experiment_3.py)
#   y colocarlo en este directorio (formato HuggingFace: config.json + *.bin).
#   Variante por defecto: 7B (hidden=4096). Para 13B, cambiar a "Asclepius-R-13B".
ASCLEPIUS_VARIANT: str = _os.environ.get("ASCLEPIUS_VARIANT", "Asclepius-R-7B")
ASCLEPIUS_R_DIR:   Path = Path(
    _os.environ.get("ASCLEPIUS_R_DIR", str(BASE_DIR / "models" / ASCLEPIUS_VARIANT))
)

# ── PROMPT DISEÑADO (único permitido) ──
DEFAULT_PROMPT: str = config.inference.default_prompt

# ── Datasets del 50% (mismos splits que Fase 2/3/Exp2) ──
TRAIN_CSV:  Path = BASE_DIR / "train_split_50pct.csv"
VAL_CSV:    Path = BASE_DIR / "val_split_50pct.csv"
TEST_CSV:   Path = BASE_DIR / "test_split_50pct.csv"
IMAGES_DIR: Path = BASE_DIR / "imagenes_50_porciento"

# ── Salidas ──
OUTPUT_DIR:      Path = BASE_DIR / "exp3_asclepius_results-50pct"
CHECKPOINTS_DIR: Path = OUTPUT_DIR / "checkpoints"

# ── Hiperparámetros de entrenamiento ──
# El Experimento 3 entrena SOLO el puente (igual que la Fase 3 PUENTE: ViT y LLM
# congelados, Full FT del Q-Former). Se REUTILIZAN los hiperparámetros de esa
# fase para continuidad y comparabilidad directa. El batch por dispositivo se deja
# configurable (default conservador) porque el backward debe propagarse a través
# del LLM congelado (LLaMA-7B) para alcanzar el puente → activaciones grandes.
TRAINING_CONFIG: Dict = {
    "lr_bridge":               1e-4,   # Q-Former + language_projection(+adaptador) + query_tokens
    "weight_decay":            0.01,
    "max_epochs":              30,
    "early_stopping_patience": 10,
    "batch_size":              6,      # micro-batch: cabe SIN GC con LLaMA-7B (pico ~32GB + conv fp32 de accelerate)
    "gradient_accumulation":   8,      # batch EFECTIVO = 48 (idntico a Exp2 RadDino y Fase 3 PUENTE)
    "eval_steps":              625,
    "warmup_ratio":            0.1,
    # GC desactivado: hay VRAM de sobra (pico 37.9/46GB) → evitar el recómputo del
    # forward del LLaMA-7B acelera ~26%. El gradiente al puente es idéntico con GC
    # ON/OFF (verificado con sonda: el backward atraviesa el LLM congelado igual).
    "gradient_checkpointing":  False,
}


# ─────────────────────────────────────────────────────────────────────────────
# 2. HELPER DE CONTEO DE PARÁMETROS (para la auditoría)
# ─────────────────────────────────────────────────────────────────────────────

def _contar(modulo: nn.Module) -> Tuple[int, int]:
    """Devuelve (entrenables, total) de un submódulo."""
    total = sum(p.numel() for p in modulo.parameters())
    train = sum(p.numel() for p in modulo.parameters() if p.requires_grad)
    return train, total


# ─────────────────────────────────────────────────────────────────────────────
# 3. FUNCIÓN PRINCIPAL: build_experiment_3_model
# ─────────────────────────────────────────────────────────────────────────────

def build_experiment_3_model(
    base_model_name: str = BASE_MODEL_NAME,
    asclepius_dir: Union[str, Path] = ASCLEPIUS_R_DIR,
    ruta_lora_vit: Union[str, Path] = PHASE2_VIT_DIR,
    train_query_tokens: bool = True,
    max_memory_gib: int = 38,
    gradient_checkpointing: bool = True,
) -> Blip2ForConditionalGeneration:
    """
    Construye la arquitectura del Experimento 3 (ViT Fase 3 + Asclepius-R + Puente Full-FT).

    Pasos (estrictamente en este orden):
        1. Carga base : Blip2ForConditionalGeneration en bfloat16 (device_map='auto').
        2. ViT (Fase 3): aplica la LoRA del ViT (Fase 2/3) → merge_and_unload() →
                        CONGELA el vision_model completo (requires_grad=False).
        3. LLM (Asclepius-R):
              a) carga Asclepius-R COMPLETO (AutoModelForCausalLM, bfloat16) desde
                 `asclepius_dir` (NO sucedáneos, NO cuantización).
              b) sustituye base_model.language_model por Asclepius-R.
              c) CONGELA el language_model completo (requires_grad=False). SIN LoRA.
              d) actualiza model.config.text_config = asclepius.config (vocab/bos/eos)
                 y reemplaza language_projection por Sequential(LP_preentrenada,
                 adaptador 2560→H_llm) para casar la dimensión del LLM.
        4. Puente     : qformer + language_projection (ambos sub-módulos) conservan
                        los pesos preentrenados de BLIP-2 y se LIBERAN
                        (requires_grad=True, Full FT, SIN LoRA). query_tokens también.
        5. Auditoría  : imprime parámetros entrenables de ViT (=0), LLM (=0) y
                        Q-Former (=100%) y verifica las invariantes.

    Returns:
        Blip2ForConditionalGeneration listo para entrenar el Experimento 3.

    Raises:
        FileNotFoundError : si falta el adaptador LoRA del ViT o el modelo Asclepius-R.
        RuntimeError      : si la auditoría de gradientes no cuadra.
    """
    asclepius_dir = Path(asclepius_dir)
    ruta_lora_vit = Path(ruta_lora_vit)

    # ── Verificación de pre-requisitos (mensajes claros si faltan) ───────────
    if not (ruta_lora_vit / "adapter_config.json").exists():
        raise FileNotFoundError(
            f"Adaptador LoRA del ViT (Fase 2/3) no encontrado: {ruta_lora_vit}\n"
            "Verifica phase2_results-50pct/checkpoints/best_model_Config_B1/vit/."
        )
    if not _asclepius_disponible(asclepius_dir):
        raise FileNotFoundError(
            "═" * 78 + "\n"
            "  ACCION OBLIGATORIA — Falta el modelo Asclepius-R (acceso credenciado)\n"
            + "═" * 78 + "\n"
            f"  No se encontró un modelo HuggingFace válido en:\n    {asclepius_dir}\n\n"
            "  Asclepius-R es de ACCESO CREDENCIADO en PhysioNet (no se puede\n"
            "  redistribuir). Descárgalo con tus credenciales y colócalo ahí.\n"
            "  Pasos detallados en run_experiment_3.py (sección ACCION OBLIGATORIA)\n"
            "  o ejecuta:  python run_experiment_3.py --print-accion-obligatoria\n"
            + "═" * 78
        )

    print("\n" + "═" * 78)
    print("  BUILD · EXPERIMENTO 3 — ViT(Fase 3) + Asclepius-R + Puente Full-FT")
    print("═" * 78)
    print(f"  base_model   : {base_model_name}")
    print(f"  ViT LoRA F3  : {ruta_lora_vit}")
    print(f"  Asclepius-R  : {asclepius_dir}")
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

    # ── FIX CRÍTICO (transformers ≥4.46): forzar la RUTA DE CONCATENACIÓN ─────
    # En 4.46.3 el config de `Salesforce/blip2-opt-2.7b` trae
    # `image_token_index = 50265` (≠ None). Con ese valor, forward()/generate()
    # toman la ruta `masked_scatter`, que ESPERA tokens placeholder dentro de
    # `input_ids` (uno por query token) para inyectar ahí los embeds del puente.
    # Nuestro data_loader produce input_ids de SOLO TEXTO (vocab LLaMA, sin
    # placeholders) → el scatter inyecta en 0 posiciones → los 32 embeds del
    # puente se descartan → gradiente del puente = 0 (NO aprende nada).
    # Forzamos None para usar la ruta `torch.cat([visual ; texto])` (609 tokens),
    # que es la que asumen el data_loader, el enmascarado de labels y el cálculo
    # de la pérdida `logits[:, -labels.size(1):]`. Verificado con sonda de
    # gradientes: con None el |grad| del puente pasa de 0 a valores reales.
    model.config.image_token_index = None
    print("        ✓ image_token_index → None (ruta CONCAT, no masked_scatter)")

    # ── Paso 2: ViT — fusionar LoRA Fase 2/3 y CONGELAR (100%) ───────────────
    print(f"\n  [2/5] ViT · aplicando LoRA Fase 3 → merge_and_unload() → congelar...")
    vit_peft = PeftModel.from_pretrained(
        model.vision_model, str(ruta_lora_vit), is_trainable=False
    )
    model.vision_model = vit_peft.merge_and_unload().to(
        device=target_device, dtype=target_dtype
    )
    del vit_peft
    gc.collect()
    torch.cuda.empty_cache()
    n_vit_frozen = 0
    for p in model.vision_model.parameters():
        p.requires_grad = False
        n_vit_frozen += p.numel()
    print(f"        ✓ ViT (EVA-CLIP + LoRA F3) fusionado y CONGELADO: "
          f"{n_vit_frozen:,} params (requires_grad=False)")

    # ── Paso 3: LLM — reemplazar OPT por Asclepius-R + adaptador de proyección ─
    print(f"\n  [3/5] LLM · cargando Asclepius-R COMPLETO ({asclepius_dir.name})...")
    # Dimensión de salida de la LP preentrenada (= hidden_size de OPT-2.7b = 2560)
    opt_hidden = int(model.language_projection.out_features)

    # attn_implementation="sdpa": usa el kernel FlashAttention nativo de PyTorch
    # (torch.nn.functional.scaled_dot_product_attention). Acelera el forward/
    # backward del LLaMA-7B (el cuello de botella) y reduce la VRAM de atención
    # (no materializa la matriz seq×seq). Si el entorno no lo soporta, se cae a
    # "eager" sin abortar. LlamaModel._supports_sdpa = True.
    try:
        asclepius = AutoModelForCausalLM.from_pretrained(
            str(asclepius_dir),
            torch_dtype=target_dtype,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )
        print("        ✓ atención = SDPA (FlashAttention nativo de PyTorch)")
    except Exception as exc:
        print(f"        ⚠ SDPA no disponible ({exc}); usando atención por defecto.")
        asclepius = AutoModelForCausalLM.from_pretrained(
            str(asclepius_dir),
            torch_dtype=target_dtype,
            low_cpu_mem_usage=True,
        )
    llama_hidden = int(asclepius.config.hidden_size)
    print(f"        Asclepius-R hidden_size = {llama_hidden}  "
          f"(LP preentrenada produce {opt_hidden} → se interpone adaptador)")

    # Sustituir el decodificador y CONGELARLO al 100% (SIN LoRA)
    del model.language_model
    gc.collect()
    torch.cuda.empty_cache()
    model.language_model = asclepius.to(device=target_device, dtype=target_dtype)
    n_llm_frozen = 0
    for p in model.language_model.parameters():
        p.requires_grad = False
        n_llm_frozen += p.numel()
    print(f"        ✓ Asclepius-R instalado y CONGELADO: {n_llm_frozen:,} params "
          f"(requires_grad=False, SIN LoRA)")

    # Actualizar la config de texto del wrapper BLIP-2 (vocab_size para la pérdida,
    # bos_token_id para generate). CRÍTICO para que forward()/generate() de stock
    # operen con el espacio de tokens de LLaMA y no con el de OPT.
    model.config.text_config = asclepius.config
    model.config.use_decoder_only_language_model = True
    # Asegurar pad_token_id en el LLM (LLaMA no trae pad por defecto) → evita
    # warnings y comportamiento indefinido en generate().
    if getattr(asclepius.config, "pad_token_id", None) is None:
        asclepius.config.pad_token_id = asclepius.config.eos_token_id
    try:
        if asclepius.generation_config is not None and \
           asclepius.generation_config.pad_token_id is None:
            asclepius.generation_config.pad_token_id = asclepius.config.eos_token_id
    except Exception:
        pass

    # Reemplazar language_projection por Sequential(LP_preentrenada, adaptador NUEVO).
    # ⚠ La LP preentrenada (768→2560) se PRESERVA intacta (elemento [0]); el
    #   adaptador (2560→H_llm) es la única capa nueva (se inicializa con la init
    #   por defecto de nn.Linear y se entrena por Full FT). Ver docstring del módulo.
    print(f"\n        Reconstruyendo language_projection: "
          f"Linear({model.language_projection.in_features}→{opt_hidden}) [preentrenada] "
          f"+ Linear({opt_hidden}→{llama_hidden}) [adaptador NUEVO]...")
    lp_preentrenada = model.language_projection            # Linear(768→2560), pesos BLIP-2
    adaptador = nn.Linear(opt_hidden, llama_hidden)        # NUEVO, entrenable
    model.language_projection = nn.Sequential(lp_preentrenada, adaptador).to(
        device=target_device, dtype=target_dtype
    )

    if gradient_checkpointing:
        # GC en el LLM congelado: el backward DEBE atravesarlo para llegar al
        # puente; recomputar activaciones ahorra mucha VRAM con LLaMA-7B/13B.
        try:
            model.language_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            model.language_model.config.use_cache = False
            print("        ✓ gradient_checkpointing habilitado en el LLM (use_reentrant=False)")
        except Exception as exc:
            print(f"        ⚠ No se pudo habilitar gradient_checkpointing en el LLM: {exc}")

    # ── Paso 4: Puente — liberar qformer + language_projection (+ query_tokens) ─
    print(f"\n  [4/5] Puente · liberando qformer + language_projection "
          f"(Full FT, SIN LoRA)...")
    n_qformer = 0
    for p in model.qformer.parameters():
        p.requires_grad = True
        n_qformer += p.numel()
    n_lp = 0
    for p in model.language_projection.parameters():
        p.requires_grad = True
        n_lp += p.numel()
    n_query = 0
    if hasattr(model, "query_tokens") and isinstance(model.query_tokens, nn.Parameter):
        model.query_tokens.requires_grad = bool(train_query_tokens)
        if train_query_tokens:
            n_query = model.query_tokens.numel()
    n_adapt = sum(p.numel() for p in adaptador.parameters())
    print(f"        Q-Former                  : {n_qformer:,} params (Full FT)")
    print(f"        language_projection (tot) : {n_lp:,} params (Full FT)")
    print(f"          · LP preentrenada 768→{opt_hidden} (pesos BLIP-2 preservados)")
    print(f"          · adaptador {opt_hidden}→{llama_hidden} (NUEVO): {n_adapt:,} params")
    print(f"        query_tokens              : {n_query:,} params "
          f"({'Full FT' if train_query_tokens else 'CONGELADO'})")

    # ── Paso 5: AUDITORÍA DE PARÁMETROS ──────────────────────────────────────
    _auditar_experiment_3(model, opt_hidden, llama_hidden)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 4. AUDITORÍA DE GRADIENTES (las 3 métricas requeridas + verificación)
# ─────────────────────────────────────────────────────────────────────────────

def _auditar_experiment_3(
    model: Blip2ForConditionalGeneration,
    opt_hidden: int,
    llama_hidden: int,
) -> None:
    """
    Imprime las tres métricas solicitadas y verifica las invariantes:
        a) ViT entrenable       → 0   (100% congelado).
        b) LLM entrenable       → 0   (100% congelado, sin LoRA).
        c) Q-Former entrenable  → 100% del submódulo.
    (+) language_projection (LP preentrenada + adaptador) y query_tokens → Full FT.
    """
    n_vit_train,  n_vit_total  = _contar(model.vision_model)
    n_llm_train,  n_llm_total  = _contar(model.language_model)
    n_qf_train,   n_qf_total   = _contar(model.qformer)
    n_lp_train,   n_lp_total   = _contar(model.language_projection)

    n_query = (model.query_tokens.numel()
               if (hasattr(model, "query_tokens")
                   and model.query_tokens.requires_grad) else 0)

    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = 100.0 * train / total if total else 0.0

    print("\n" + "═" * 78)
    print("  AUDITORÍA DE PARÁMETROS · EXPERIMENTO 3 (ViT F3 + Asclepius-R + Puente Full-FT)")
    print("═" * 78)
    print("  (a) ViT — EVA-CLIP + LoRA Fase 3 fusionada (esperado: 0 entrenables)")
    print(f"        entrenables : {n_vit_train:>15,}  / total {n_vit_total:,}")
    print("  (b) LLM — Asclepius-R / LLaMA (esperado: 0 entrenables, SIN LoRA)")
    print(f"        entrenables : {n_llm_train:>15,}  / total {n_llm_total:,}")
    print("  (c) Q-Former (esperado: 100% del submódulo)")
    print(f"        entrenables : {n_qf_train:>15,}  / total {n_qf_total:,}  "
          f"({100.0*n_qf_train/max(n_qf_total,1):.3f}%)")
    print("  (+) language_projection (Full FT) : "
          f"{n_lp_train:>13,} / {n_lp_total:,}")
    print(f"        · LP preentrenada 768→{opt_hidden} (pesos BLIP-2) + "
          f"adaptador {opt_hidden}→{llama_hidden} (NUEVO)")
    print(f"  (+) query_tokens (Full FT)        : {n_query:>13,}")
    print("  ──────────────────────────────────────────────────────────────────")
    print(f"  TOTAL modelo        : {total:>15,}")
    print(f"  TOTAL entrenable    : {train:>15,}  ({pct:.3f}%)")
    print(f"  TOTAL congelado     : {total-train:>15,}")
    print("═" * 78)

    # ── Verificación de invariantes ──────────────────────────────────────────
    errores = []
    if n_vit_train != 0:
        errores.append(f"ViT tiene {n_vit_train:,} params entrenables (esperado 0).")
    if n_llm_train != 0:
        errores.append(f"LLM tiene {n_llm_train:,} params entrenables (esperado 0).")
    if n_qf_train != n_qf_total:
        errores.append(f"Q-Former entrenable {n_qf_train:,} ≠ total {n_qf_total:,} (esperado 100%).")
    if n_lp_train != n_lp_total:
        errores.append(f"language_projection entrenable {n_lp_train:,} ≠ total {n_lp_total:,}.")

    if errores:
        print("  ✗ AUDITORÍA FALLIDA:")
        for e in errores:
            print(f"      · {e}")
        raise RuntimeError(
            "La auditoría de gradientes del Experimento 3 no cumple las "
            "invariantes (ViT=0, LLM=0, Q-Former=100%). Revisa build_experiment_3_model()."
        )
    print("  ✓ Invariantes OK: ViT=0 · LLM=0 · Q-Former=100% · language_projection/query Full-FT")
    print("═" * 78 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# 5. OPTIMIZADOR (Puente Full-FT: un solo grupo, igual que Fase 3 PUENTE)
# ─────────────────────────────────────────────────────────────────────────────

def build_experiment_3_optimizer(model: Blip2ForConditionalGeneration) -> Tuple[AdamW, None]:
    """
    AdamW con UN ÚNICO grupo de parámetros (el puente, Full FT):
        qformer + language_projection(LP preentrenada + adaptador) + query_tokens.
    lr = TRAINING_CONFIG["lr_bridge"] (1e-4, idéntico a Fase 3 PUENTE).
    El scheduler (cosine + warmup) se delega a TrainingArguments.
    """
    bridge_params: List[nn.Parameter] = []
    bridge_params += [p for p in model.qformer.parameters() if p.requires_grad]
    bridge_params += [p for p in model.language_projection.parameters() if p.requires_grad]
    if hasattr(model, "query_tokens") and model.query_tokens.requires_grad:
        bridge_params.append(model.query_tokens)

    if not bridge_params:
        raise RuntimeError("No hay parámetros entrenables en el puente (Experimento 3).")

    optimizer = AdamW(
        bridge_params,
        lr=TRAINING_CONFIG["lr_bridge"],
        weight_decay=TRAINING_CONFIG["weight_decay"],
        eps=1e-8,
        fused=torch.cuda.is_available(),
    )
    n_bridge = sum(p.numel() for p in bridge_params)
    print(f"   [optimizer] Puente Full-FT : lr={TRAINING_CONFIG['lr_bridge']:.0e}  "
          f"({n_bridge:,} params)")
    print(f"   [optimizer] weight_decay={TRAINING_CONFIG['weight_decay']}  "
          f"| scheduler (cosine+warmup) delegado a HuggingFace")
    return optimizer, None


# ─────────────────────────────────────────────────────────────────────────────
# 6. TOKENIZER / PROCESADOR (intercambia el tokenizer de OPT por el de Asclepius)
# ─────────────────────────────────────────────────────────────────────────────

def cargar_tokenizer_asclepius(asclepius_dir: Union[str, Path] = ASCLEPIUS_R_DIR):
    """
    Carga el tokenizer de Asclepius-R (LLaMA, use_fast=False) y garantiza pad_token.
    """
    tok = AutoTokenizer.from_pretrained(str(asclepius_dir), use_fast=False)
    if tok.pad_token is None:
        # LLaMA no trae pad por defecto → reutilizar eos (estándar; los labels de
        # padding se enmascaran con -100 por el DataCollatorForSeq2Seq).
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"   # teacher forcing (entrenamiento)
    return tok


def preparar_procesador_exp3(processor, asclepius_dir: Union[str, Path] = ASCLEPIUS_R_DIR):
    """
    Intercambia `processor.tokenizer` por el tokenizer de Asclepius/LLaMA, dejando
    intacto el `image_processor` de BLIP-2. El data_loader del proyecto tokeniza
    vía `processor.tokenizer` / `processor(...)`, de modo que este intercambio basta
    para que labels e inferencia operen en el espacio de tokens de LLaMA.
    """
    processor.tokenizer = cargar_tokenizer_asclepius(asclepius_dir)
    return processor


# ─────────────────────────────────────────────────────────────────────────────
# 7. CONSTRUCCIÓN DE DATASETS (usa EXCLUSIVAMENTE el prompt diseñado)
# ─────────────────────────────────────────────────────────────────────────────

def build_experiment_3_datasets(processor):
    """
    Construye los datasets de train/val del 50% inyectando ÚNICAMENTE el prompt
    diseñado (DEFAULT_PROMPT = config.inference.default_prompt). NO se usa
    self.prompt ni ningún otro prompt.

    IMPORTANTE: `processor` debe traer YA el tokenizer de Asclepius (llamar antes
    a preparar_procesador_exp3), para que los labels se tokenicen en vocab LLaMA.
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
        # padding="longest": cada muestra se tokeniza SIN padding (longest de 1
        # secuencia = su largo real). El DataCollatorForSeq2Seq(padding=True) +
        # group_by_length agrupan y rellenan por-batch al mximo del batch, no al
        # 577 global. El 66% de tokens eran padding (real p50=194 vs 609 fijo) →
        # ~1.8x menos cmputo en el LLaMA-7B. Matemticamente IDNTICO: la loss
        # solo cuenta tokens reales (padding → -100 en labels, attention_mask=0).
        padding              = "longest",
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
        padding              = "longest",   # dinmico por-batch (ver nota en train_ds)
        prompt               = DEFAULT_PROMPT,   # ← prompt diseñado (único permitido)
        augment              = False,
    )
    return train_ds, val_ds


# ─────────────────────────────────────────────────────────────────────────────
# 8. UTILIDAD: ¿está disponible un modelo HuggingFace válido en el directorio?
# ─────────────────────────────────────────────────────────────────────────────

def _asclepius_disponible(asclepius_dir: Union[str, Path]) -> bool:
    """True si `asclepius_dir` contiene un config.json + pesos (*.bin/*.safetensors)."""
    d = Path(asclepius_dir)
    if not d.is_dir():
        return False
    if not (d / "config.json").exists():
        return False
    pesos = list(d.glob("*.bin")) + list(d.glob("*.safetensors")) + \
            list(d.glob("*.bin.index.json")) + list(d.glob("*.safetensors.index.json"))
    return len(pesos) > 0


# ─────────────────────────────────────────────────────────────────────────────
# 9. SMOKE TEST (construcción + auditoría sin entrenar)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Smoke test de build_experiment_3_model (construye + audita)."
    )
    parser.add_argument("--no-query-tokens", action="store_true",
                        help="Congelar query_tokens (solo qformer + language_projection).")
    parser.add_argument("--forward", action="store_true",
                        help="Probar un forward dummy (pixel_values + input_ids aleatorios).")
    args = parser.parse_args()

    print("Construyendo modelo del Experimento 3 (requiere Asclepius-R descargado)...")
    model = build_experiment_3_model(train_query_tokens=not args.no_query_tokens)
    build_experiment_3_optimizer(model)

    if args.forward:
        print("\n  [smoke] Probando forward dummy (1×3×224×224 + prompt)...")
        dev = next(model.qformer.parameters()).device
        tok = cargar_tokenizer_asclepius()
        enc = tok(DEFAULT_PROMPT, return_tensors="pt").to(dev)
        dummy = torch.randn(1, 3, 224, 224, dtype=model.dtype, device=dev)
        labels = enc["input_ids"].clone()
        with torch.no_grad():
            out = model(pixel_values=dummy, input_ids=enc["input_ids"],
                        attention_mask=enc["attention_mask"], labels=labels)
        print(f"  [smoke] loss={float(out.loss):.4f}  logits={tuple(out.logits.shape)}")

    print("✓ Smoke test completado.")
