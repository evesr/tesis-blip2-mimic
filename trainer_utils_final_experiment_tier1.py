#!/usr/bin/env python3
"""
trainer_utils_final_experiment_tier1.py — EXPERIMENTO FINAL + TIER 1 (pérdida clínica auxiliar)
===============================================================================================
RÉPLICA de trainer_utils_final_experiment.py (el ORIGINAL NO se modifica) que
añade la estrategia TIER 1 contra el "language prior dominance": una pérdida
clínica auxiliar CheXpert de 14 clases combinada con la pérdida de lenguaje:

        L_total = L_LM  +  λ · L_clin      (λ = TRAINING_CONFIG["lambda_clin"])

  • L_clin = BCE multietiqueta entre una cabeza lineal (sobre el pooling de los
    32 query tokens del Q-Former) y el GT oficial CheXpert unido por study_id.
  • Fuerza al puente (Q-Former + language_projection) a PRESERVAR la señal
    diagnóstica que los linear-probes demostraron presente en las features,
    en vez de gastarse solo en imitar la prosa del corpus.

CAMBIOS RESPECTO AL ORIGINAL (mínimos y documentados)
-----------------------------------------------------
  1. OUTPUT_DIR → final_experiment_tier1_results-50pct (NO pisa la corrida original).
  2. TRAINING_CONFIG: se AÑADEN 2 claves nuevas (lambda_clin, chexpert_uncertain).
     NINGÚN hiperparámetro existente (lr, batch, accum, epochs, warmup, early
     stopping, eval_steps) se modifica → comparabilidad directa con el final.
  3. Cabeza clínica auxiliar Linear(768→14) — única capa nueva (~11K params).
  4. Dataset aumentado con etiquetas CheXpert (lookup O(1); overhead ≈ 0).
El compute_loss multitarea y el hook del Q-Former viven en run_final_experiment_tier1.py.

Arquitectura multimodal basada en BLIP-2 que combina lo mejor de los dos
experimentos previos y CORRIGE el learning rate para evitar la explosión de
gradientes:

  • CODIFICADOR VISUAL = RadDINO (microsoft/rad-dino, DINOv2 ViT-B/14) con la
    LoRA YA AJUSTADA en el Experimento 2 (Config ganadora) — se carga, se FUSIONA
    (merge_and_unload) y se CONGELA al 100 % (incluida la proyección 768→1408).
  • DECODIFICADOR = Asclepius-R (LLaMA-7B clínico afinado sobre notas REALES de
    MIMIC-III), CONGELADO al 100 % (SIN LoRA, sin cuantización, modelo COMPLETO).
  • PUENTE = Q-Former + language_projection (+ adaptador 2560→H_llm) + query_tokens
    por Full Fine-Tuning, conservando los pesos preentrenados de BLIP-2 (NO
    inicialización aleatoria, SIN LoRA).

────────────────────────────────────────────────────────────────────────────
MAPA DE GRADIENTES (resumen ejecutivo)
────────────────────────────────────────────────────────────────────────────
  Componente                              | requires_grad | Técnica
  ----------------------------------------+---------------+-----------------------
  vision_model.backbone (RadDINO+LoRA E2) | FALSE (0%)    | LoRA Exp2 fusionada + frozen
  vision_model.projection (768→1408)      | FALSE (0%)    | Pesos Exp2 + frozen
  language_model (Asclepius-R / LLaMA-7B) | FALSE (0%)    | Reemplazo total, frozen, SIN LoRA
  qformer (Q-Former)                      | TRUE (100%)   | Full FT, SIN LoRA (pesos BLIP-2)
  language_projection[0] (768→2560)       | TRUE (100%)   | Full FT (pesos BLIP-2 preservados)
  language_projection[1] (2560→H_llm)     | TRUE (100%)   | Full FT (adaptador NUEVO, ver ⚠)
  query_tokens (32×768)                   | TRUE (100%)   | Full FT (entrada del Q-Former)

────────────────────────────────────────────────────────────────────────────
⚠ CORRECCIÓN CRÍTICA DEL LEARNING RATE (motivo de este experimento)
────────────────────────────────────────────────────────────────────────────
El Q-Former entra en Full Fine-Tuning (~188 M params). Con lr = 1e-4 (usado en
las fases previas) la pérdida de validación REBOTA al salir del warmup: el paso
de optimización es demasiado agresivo para un módulo tan grande entrenado de
forma densa y DESTRUYE el espacio latente que conecta la visión con el LLM
congelado. Se fija el learning_rate máximo de forma ESTRICTA en 1e-5
(TRAINING_CONFIG["lr_bridge"]), con un techo admisible de 2e-5, para garantizar
una convergencia suave y monótona.

────────────────────────────────────────────────────────────────────────────
⚠ ADAPTADOR DE PROYECCIÓN 2560 → H_llm (idéntico al Experimento 3)
────────────────────────────────────────────────────────────────────────────
La `language_projection` de BLIP-2 es `Linear(768 → 2560)` (espacio de OPT-2.7b).
Asclepius-R (LLaMA-7B) tiene hidden_size = 4096. El requisito de CONSERVAR los
pesos preentrenados de `language_projection` impide cambiar su dimensión de
salida. La solución mínimamente invasiva (igual que en el Experimento 3) es
interponer un adaptador lineal entrenable DESPUÉS de la proyección preentrenada:

    language_projection = nn.Sequential(
        Linear(768 → 2560)   # ← PESOS PREENTRENADOS DE BLIP-2 (intactos, Full FT)
        Linear(2560 → H_llm) # ← CAPA NUEVA entrenable (única añadida; se audita aparte)
    )

Así `model.language_projection(query_output)` produce embeddings de dimensión
H_llm y el forward()/generate() de stock de BLIP-2 funcionan SIN cambios.

────────────────────────────────────────────────────────────────────────────
CONSISTENCIA CON forward()/generate() DE BLIP-2 (transformers 4.46.3)
────────────────────────────────────────────────────────────────────────────
  • `model.config.text_config = asclepius.config`  → vocab 32000 / bos=1 de LLaMA.
  • `image_token_index` se FUERZA a None → ruta de CONCATENACIÓN [visual ; texto]
    (la única compatible con nuestro data_loader de solo-texto y con la pérdida).
    En 4.46 el config base trae image_token_index=50265, que activaría la ruta
    masked_scatter y descartaría los 32 embeds del puente (gradiente = 0).
  • El TOKENIZER debe ser el de Asclepius/LLaMA (use_fast=False) → se provee
    `preparar_procesador_exp3()` (reutilizado del Experimento 3).

PROMPT
────────────────────────────────────────────────────────────────────────────
Se usa EXCLUSIVAMENTE el prompt diseñado del proyecto:
    DEFAULT_PROMPT = config.inference.default_prompt
inyectado de forma EXPLÍCITA en build_final_experiment_datasets (NO self.prompt
ni ningún prompt dinámico).

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
    AutoModel,
    AutoModelForCausalLM,
    Blip2ForConditionalGeneration,
)
from peft import LoraConfig, PeftModel, get_peft_model

from config import config

# ── TIER 1: dependencias para la pérdida clínica auxiliar ────────────────────
import numpy as np
import pandas as pd
from data_loader import MimicCXRDataset

# ── Reutilización del bloque visual RadDINO del Experimento 2 (SIN cambios) ──
#   RadDinoVisionTower expone la MISMA interfaz que Blip2VisionModel (forward que
#   devuelve last_hidden_state ya proyectado a 1408). Los hiperparámetros de la
#   LoRA se importan para reconstruir EXACTAMENTE la estructura con la que se
#   entrenó en el Experimento 2 (necesario para cargar sus pesos por nombre).
from trainer_utils_exp2_raddino import (
    RadDinoVisionTower,
    RADDINO_MODEL_ID,
    RADDINO_TARGET_MODULES,
    RADDINO_LORA_R,
    RADDINO_LORA_ALPHA,
    RADDINO_LORA_DROPOUT,
    QFORMER_ENCODER_HIDDEN_SIZE,
)

# ── Reutilización de helpers del Experimento 3 (Asclepius-R, tokenizer, rutas) ──
from trainer_utils_exp3_asclepius import (
    BASE_MODEL_NAME,
    ASCLEPIUS_VARIANT,
    ASCLEPIUS_R_DIR,
    DEFAULT_PROMPT,
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
    IMAGES_DIR,
    cargar_tokenizer_asclepius,
    preparar_procesador_exp3,
    _asclepius_disponible,
)

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# 1. CONSTANTES, RUTAS E HIPERPARÁMETROS
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR: Path = config.paths.base_dir

# ── Pesos del Experimento 2 (RadDINO): de aquí se extraen SOLO los del bloque
#    visual (vision_model.backbone.* LoRA + vision_model.projection.*). El resto
#    de exp2_state.pt (qformer/language_projection/query_tokens entrenados para
#    OPT) NO se usa: el puente se re-entrena desde los pesos BASE de BLIP-2.
EXP2_BEST_DIR:  Path = BASE_DIR / "exp2_raddino_results-50pct" / "checkpoints" / "best_model_Exp2_RadDINO"
EXP2_STATE_PT:  Path = EXP2_BEST_DIR / "exp2_state.pt"

# ── PROMPT DISEÑADO (único permitido) ──
# (DEFAULT_PROMPT se importa de trainer_utils_exp3_asclepius = config.inference.default_prompt)

# ── Datasets del 50% (mismos splits que Fase 2/3/Exp2/Exp3) ──
# (TRAIN_CSV / VAL_CSV / TEST_CSV / IMAGES_DIR se importan de trainer_utils_exp3_asclepius)

# ── Salidas (TIER 1: carpeta SEPARADA para no pisar la corrida final original) ──
OUTPUT_DIR:      Path = BASE_DIR / "final_experiment_tier1_results-50pct"
CHECKPOINTS_DIR: Path = OUTPUT_DIR / "checkpoints"

# ── Hiperparámetros de entrenamiento ──
# Se REUTILIZAN los hiperparámetros de control de las fases pasadas (batch,
# eval_steps, warmup, accum, epochs, early stopping) para comparabilidad directa.
# El micro-batch = 6 con accum = 8 → batch EFECTIVO = 48 (idéntico a Exp2/Exp3 y
# Fase 3 PUENTE). Se toma el batch de Exp3 porque el cuello de botella de memoria
# es Asclepius-R (LLaMA-7B), no RadDINO (ViT-B, mucho más liviano que EVA-CLIP).
#
# ⚠ ÚNICO cambio respecto a las fases previas: lr_bridge = 1e-5 (NO 1e-4) para
#   evitar la explosión de gradientes / rebote de la val_loss del Q-Former Full-FT.
TRAINING_CONFIG: Dict = {
    "lr_bridge":               1e-5,   # ⚠ CRÍTICO: 1e-4 destruía el espacio latente al salir del warmup
    "weight_decay":            0.01,
    "max_epochs":              30,
    "early_stopping_patience": 10,
    "batch_size":              6,      # micro-batch (Asclepius-R domina la VRAM)
    "gradient_accumulation":   8,      # batch EFECTIVO = 48 (idéntico a Exp2/Exp3)
    "eval_steps":              625,
    "warmup_ratio":            0.1,
    # GC desactivado: RadDINO (ViT-B) es liviano y hay VRAM de sobra → evitar el
    # recómputo del forward de LLaMA-7B acelera. El gradiente al puente es idéntico
    # con GC ON/OFF (el backward atraviesa el LLM congelado igual).
    "gradient_checkpointing":  False,
    # ─────────────────────────────────────────────────────────────────────────
    # TIER 1 — claves NUEVAS (NO modifican ningún hiperparámetro existente)
    # ─────────────────────────────────────────────────────────────────────────
    # lambda_clin: peso λ de la pérdida clínica auxiliar en L = L_LM + λ·L_clin.
    #   0.5 = punto medio conservador: da señal clínica fuerte sin ahogar el
    #   objetivo lingüístico. Es un hiperparámetro AÑADIDO, no una modificación de
    #   lr/batch/epochs/warmup/early-stopping (que quedan intactos).
    "lambda_clin":             0.5,
    # chexpert_uncertain: política para la etiqueta -1 (incierto) del GT MIMIC-CXR.
    #   "positive" (−1→1) replica EXACTAMENTE la convención canónica del proyecto
    #   (_chexpert_comparison_metrics / eval_raddino_probe) → la señal auxiliar es
    #   consistente con la métrica de evaluación. Con esta política la máscara es
    #   todo-unos para los estudios con GT.
    "chexpert_uncertain":      "positive",
}


# ─────────────────────────────────────────────────────────────────────────────
# 1-bis. TIER 1 — ETIQUETAS CheXpert PARA LA PÉRDIDA CLÍNICA AUXILIAR
# ─────────────────────────────────────────────────────────────────────────────
# Se une el GT oficial CheXpert (mimic-cxr-2.0.0-chexpert.csv) a cada muestra por
# study_id y se precomputa, UNA sola vez en __init__, el vector objetivo de 14
# clases + su máscara. En __getitem__ el coste es un lookup O(1) en un dict → el
# overhead por paso de entrenamiento es DESPRECIABLE (ninguna I/O extra).
CHEXPERT_CSV: Path = BASE_DIR / "mimic-cxr-2.0.0-chexpert.csv"

# Orden EXACTO de las 14 columnas del CSV oficial (define el orden de los logits
# de la cabeza auxiliar). Consistente con CHEXBERT_LABELS del proyecto.
CHEXPERT_LABELS_14: List[str] = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", "Lung Opacity",
    "No Finding", "Pleural Effusion", "Pleural Other", "Pneumonia",
    "Pneumothorax", "Support Devices",
]
NUM_CHEXPERT: int = len(CHEXPERT_LABELS_14)


def _norm_sid(s) -> str:
    """Normaliza study_id a str (quita 's' inicial y decimales .0). Idéntico a la
    convención de eval_raddino_probe / Fase 3 / Exp2."""
    if isinstance(s, float) and s.is_integer():
        s = int(s)
    s = str(s).strip()
    return s[1:] if s.startswith("s") else s


def construir_label_map(
    df_split: pd.DataFrame,
    df_chex: pd.DataFrame,
    uncertain: str = "positive",
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Devuelve {study_id_norm: (labels[14] float32, mask[14] float32)} replicando la
    codificación canónica del proyecto:
        1.0 → positivo · 0.0 → negativo · -1.0 → incierto · NaN → no mencionado(0)
    Política de incertidumbre (uncertain):
        'positive' (def.): -1 → 1   (máscara todo-unos; = _chexpert_comparison_metrics)
        'negative'       : -1 → 0
        'ignore'         : -1 → mask=0 (excluido de la pérdida)
    Estudios SIN GT tras el merge → (zeros, zeros): máscara 0 (no aportan señal).
    """
    cols = [c for c in CHEXPERT_LABELS_14 if c in df_chex.columns]
    chex = df_chex.copy()
    chex["_sid"] = chex["study_id"].map(_norm_sid)
    chex = chex.drop_duplicates("_sid").set_index("_sid")

    label_map: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    zeros = torch.zeros(NUM_CHEXPERT, dtype=torch.float32)
    for sid in df_split["study_id"].map(_norm_sid).unique():
        if sid in chex.index:
            raw = chex.loc[sid, cols].to_numpy(dtype=np.float64)
            lab = np.zeros(NUM_CHEXPERT, dtype=np.float32)
            msk = np.ones(NUM_CHEXPERT, dtype=np.float32)
            nan_mask = np.isnan(raw)
            lab[raw == 1.0] = 1.0
            unc = (raw == -1.0)
            if uncertain == "positive":
                lab[unc] = 1.0
            elif uncertain == "negative":
                lab[unc] = 0.0
            elif uncertain == "ignore":
                msk[unc] = 0.0
            else:
                raise ValueError(f"chexpert_uncertain inválido: {uncertain}")
            lab[nan_mask] = 0.0
            label_map[sid] = (torch.from_numpy(lab), torch.from_numpy(msk))
        else:
            label_map[sid] = (zeros.clone(), zeros.clone())
    return label_map


class CheXpertAugmentedDataset(MimicCXRDataset):
    """
    MimicCXRDataset + etiquetas CheXpert de 14 clases para la pérdida auxiliar
    (Tier 1). Idéntico al dataset original salvo que __getitem__ añade dos claves:
        chexpert_labels : FloatTensor(14)  objetivo binario multietiqueta
        chexpert_mask   : FloatTensor(14)  1 = cuenta en la BCE · 0 = ignorar

    El mapa study_id→(labels,mask) se precomputa una sola vez en __init__ (join
    pandas), por lo que __getitem__ solo hace un lookup O(1): overhead ≈ 0.
    """

    def __init__(self, *args, chexpert_csv: Union[str, Path] = CHEXPERT_CSV,
                 uncertain: str = "positive", **kwargs):
        super().__init__(*args, **kwargs)
        df_chex = pd.read_csv(chexpert_csv)
        self._label_map = construir_label_map(self.data, df_chex, uncertain)
        self._zeros = torch.zeros(NUM_CHEXPERT, dtype=torch.float32)
        cubiertos = sum(
            1 for sid in self.data["study_id"].map(_norm_sid)
            if float(self._label_map.get(sid, (None, self._zeros))[1].sum()) > 0
        )
        print(f"   [tier1/dataset] {self.csv_path.name}: GT CheXpert para "
              f"{cubiertos:,}/{len(self.data):,} muestras "
              f"(política incierto='{uncertain}')")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        enc = super().__getitem__(idx)
        sid = _norm_sid(self.data.iloc[idx]["study_id"])
        lab, msk = self._label_map.get(sid, (self._zeros, self._zeros))
        enc["chexpert_labels"] = lab
        enc["chexpert_mask"] = msk
        return enc


# ─────────────────────────────────────────────────────────────────────────────
# 2. HELPER DE CONTEO DE PARÁMETROS (para la auditoría)
# ─────────────────────────────────────────────────────────────────────────────

def _contar(modulo: nn.Module) -> Tuple[int, int]:
    """Devuelve (entrenables, total) de un submódulo."""
    total = sum(p.numel() for p in modulo.parameters())
    train = sum(p.numel() for p in modulo.parameters() if p.requires_grad)
    return train, total


# ─────────────────────────────────────────────────────────────────────────────
# 3. BLOQUE VISUAL: RadDINO + LoRA(Exp2) fusionada + proyección(Exp2) — CONGELADO
# ─────────────────────────────────────────────────────────────────────────────

def _construir_vision_tower_raddino_frozen(
    model: Blip2ForConditionalGeneration,
    exp2_state_pt: Path,
    target_device: torch.device,
    target_dtype: torch.dtype,
) -> int:
    """
    Reemplaza `model.vision_model` por el bloque visual RadDINO del Experimento 2,
    con sus pesos LoRA YA AJUSTADOS, y lo CONGELA al 100 %.

    Secuencia (en este orden):
        1) Carga RadDINO COMPLETO (AutoModel, bfloat16).
        2) Inyecta la MISMA LoRA del Experimento 2 (r=8, alpha=16, targets
           query/value) con get_peft_model → reproduce la estructura de claves
           con la que se guardó exp2_state.pt.
        3) Envuelve en RadDinoVisionTower (añade la proyección 768→1408) y lo
           asigna a model.vision_model.
        4) Carga del exp2_state.pt ÚNICAMENTE las claves `vision_model.*`
           (LoRA del backbone + proyección) con strict=False.
        5) merge_and_unload() del backbone → funde la LoRA entrenada en RadDINO
           (encoder visual liviano y rápido, sin overhead de adaptadores).
        6) requires_grad = False para TODO el bloque visual (Protección total).

    Returns:
        Nº de parámetros del bloque visual (todos congelados).
    """
    exp2_state_pt = Path(exp2_state_pt)
    if not exp2_state_pt.exists():
        raise FileNotFoundError(
            f"No se encontró exp2_state.pt (pesos RadDINO del Experimento 2):\n"
            f"    {exp2_state_pt}\n"
            "Verifica exp2_raddino_results-50pct/checkpoints/best_model_Exp2_RadDINO/."
        )

    # ── 1. RadDINO COMPLETO ──────────────────────────────────────────────────
    print(f"        Cargando RadDINO COMPLETO ({RADDINO_MODEL_ID})...")
    raddino = AutoModel.from_pretrained(RADDINO_MODEL_ID, torch_dtype=target_dtype)
    raddino_hidden = int(getattr(raddino.config, "hidden_size", 768))

    # ── 2. Misma LoRA del Experimento 2 (estructura de claves idéntica) ──────
    raddino_lora_cfg = LoraConfig(
        r=RADDINO_LORA_R,
        lora_alpha=RADDINO_LORA_ALPHA,
        lora_dropout=RADDINO_LORA_DROPOUT,
        bias="none",
        task_type=None,
        target_modules=list(RADDINO_TARGET_MODULES),
    )
    raddino_peft = get_peft_model(raddino, raddino_lora_cfg)

    # ── 3. Torre visual (proyección 768→1408) + asignación ───────────────────
    vision_tower = RadDinoVisionTower(
        backbone=raddino_peft,
        in_dim=raddino_hidden,
        out_dim=QFORMER_ENCODER_HIDDEN_SIZE,
        force_interpolate=True,
    ).to(device=target_device, dtype=target_dtype)
    model.vision_model = vision_tower
    try:
        model.config.vision_config.hidden_size = QFORMER_ENCODER_HIDDEN_SIZE
    except Exception:
        pass

    # ── 4. Cargar SOLO los pesos visuales entrenados del Experimento 2 ───────
    print(f"        Cargando pesos RadDINO (LoRA + proyección) de {exp2_state_pt.name}...")
    exp2_sd = torch.load(str(exp2_state_pt), map_location="cpu", weights_only=True)
    vision_sd = {
        k: v.to(device=target_device, dtype=target_dtype)
        for k, v in exp2_sd.items()
        if k.startswith("vision_model.")
    }
    missing, unexpected = model.load_state_dict(vision_sd, strict=False)
    # `missing` será LARGO (todo lo no-visual) → es CORRECTO. `unexpected` debe
    # ser vacío: cada clave vision_model.* debe casar con la torre recién creada.
    vis_unexpected = [k for k in unexpected if k.startswith("vision_model.")]
    if vis_unexpected:
        raise RuntimeError(
            "Claves visuales INESPERADAS al cargar exp2_state.pt "
            f"(¿estructura LoRA distinta?): {vis_unexpected[:4]} ..."
        )
    n_proj = sum(v.numel() for k, v in vision_sd.items() if "projection" in k)
    n_lora = sum(v.numel() for k, v in vision_sd.items() if "backbone" in k)
    print(f"        ✓ Pesos visuales del Exp2 cargados: "
          f"LoRA={n_lora:,} params · proyección={n_proj:,} params "
          f"({len(vision_sd)} tensores)")
    del exp2_sd, vision_sd

    # ── 5. Fusionar la LoRA entrenada en RadDINO (merge_and_unload) ──────────
    print(f"        Fusionando LoRA del Exp2 en RadDINO (merge_and_unload)...")
    model.vision_model.backbone = model.vision_model.backbone.merge_and_unload().to(
        device=target_device, dtype=target_dtype
    )
    gc.collect()
    torch.cuda.empty_cache()

    # ── 6. CONGELAR el 100 % del bloque visual (backbone fusionado + proyección)
    n_vit_frozen = 0
    for p in model.vision_model.parameters():
        p.requires_grad = False
        n_vit_frozen += p.numel()
    print(f"        ✓ Bloque visual RadDINO (backbone+proyección) CONGELADO: "
          f"{n_vit_frozen:,} params (requires_grad=False)")
    return n_vit_frozen


# ─────────────────────────────────────────────────────────────────────────────
# 4. FUNCIÓN PRINCIPAL: build_final_experiment_model
# ─────────────────────────────────────────────────────────────────────────────

def build_final_experiment_model(
    base_model_name: str = BASE_MODEL_NAME,
    asclepius_dir: Union[str, Path] = ASCLEPIUS_R_DIR,
    exp2_state_pt: Union[str, Path] = EXP2_STATE_PT,
    train_query_tokens: bool = True,
    max_memory_gib: int = 38,
    gradient_checkpointing: bool = True,
) -> Blip2ForConditionalGeneration:
    """
    Construye la arquitectura del EXPERIMENTO FINAL
    (RadDINO[LoRA Exp2, frozen] + Asclepius-R[frozen] + Puente Full-FT).

    Pasos (estrictamente en este orden):
        1. Carga base : Blip2ForConditionalGeneration en bfloat16 (device_map='auto').
        2. ViT (RadDINO): carga RadDINO COMPLETO + LoRA AJUSTADA del Exp2 →
                          merge_and_unload() → CONGELA el 100 % (backbone+proyección).
        3. LLM (Asclepius-R):
              a) carga Asclepius-R COMPLETO (AutoModelForCausalLM, bfloat16). NO
                 sucedáneos, NO cuantización, NO versiones livianas.
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
        Blip2ForConditionalGeneration listo para entrenar el Experimento FINAL.

    Raises:
        FileNotFoundError : si falta exp2_state.pt (RadDINO) o el modelo Asclepius-R.
        RuntimeError      : si la auditoría de gradientes no cuadra.
    """
    asclepius_dir = Path(asclepius_dir)
    exp2_state_pt = Path(exp2_state_pt)

    # ── Verificación de pre-requisitos (mensajes claros si faltan) ───────────
    if not exp2_state_pt.exists():
        raise FileNotFoundError(
            f"Pesos RadDINO del Experimento 2 no encontrados: {exp2_state_pt}\n"
            "Verifica exp2_raddino_results-50pct/checkpoints/best_model_Exp2_RadDINO/exp2_state.pt."
        )
    if not _asclepius_disponible(asclepius_dir):
        raise FileNotFoundError(
            "═" * 78 + "\n"
            "  ACCION OBLIGATORIA — Falta el modelo Asclepius-R (acceso credenciado)\n"
            + "═" * 78 + "\n"
            f"  No se encontró un modelo HuggingFace válido en:\n    {asclepius_dir}\n\n"
            "  Asclepius-R es de ACCESO CREDENCIADO en PhysioNet (no se puede\n"
            "  redistribuir). Descárgalo con tus credenciales y colócalo ahí.\n"
            "  Pasos detallados en run_final_experiment.py "
            "(--print-accion-obligatoria).\n"
            + "═" * 78
        )

    print("\n" + "═" * 78)
    print("  BUILD · EXPERIMENTO FINAL — RadDINO + Asclepius-R + Puente Full-FT")
    print("═" * 78)
    print(f"  base_model    : {base_model_name}")
    print(f"  RadDINO       : {RADDINO_MODEL_ID}  (+ LoRA Exp2, frozen)")
    print(f"  RadDINO weights: {exp2_state_pt}")
    print(f"  Asclepius-R   : {asclepius_dir}")
    print(f"  lr_bridge     : {TRAINING_CONFIG['lr_bridge']:.0e}  (CORREGIDO ⚠)")
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
    # En 4.46.3 el config de blip2-opt-2.7b trae image_token_index=50265 (≠ None),
    # que activaría la ruta masked_scatter (espera placeholders en input_ids).
    # Nuestro data_loader produce input_ids de SOLO TEXTO → forzamos None para usar
    # la ruta torch.cat([visual ; texto]), única compatible con la pérdida y el
    # data_loader. Sin esto, los 32 embeds del puente se descartan → gradiente = 0.
    model.config.image_token_index = None
    print("        ✓ image_token_index → None (ruta CONCAT, no masked_scatter)")

    # ── Paso 2: ViT — RadDINO + LoRA(Exp2) fusionada + CONGELADO (100%) ───────
    print(f"\n  [2/5] ViT · RadDINO + LoRA Exp2 (ajustada) → merge_and_unload() → congelar...")
    _construir_vision_tower_raddino_frozen(
        model=model,
        exp2_state_pt=exp2_state_pt,
        target_device=target_device,
        target_dtype=target_dtype,
    )

    # ── Paso 3: LLM — reemplazar OPT por Asclepius-R + adaptador de proyección ─
    print(f"\n  [3/5] LLM · cargando Asclepius-R COMPLETO ({asclepius_dir.name})...")
    # Dimensión de salida de la LP preentrenada (= hidden_size de OPT-2.7b = 2560)
    opt_hidden = int(model.language_projection.out_features)

    # attn_implementation="sdpa": kernel FlashAttention nativo de PyTorch.
    # Acelera el forward/backward de LLaMA-7B (cuello de botella) y reduce la VRAM
    # de atención. Si el entorno no lo soporta, se cae a "eager" sin abortar.
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
    # bos_token_id para generate). CRÍTICO para operar en el espacio de LLaMA.
    model.config.text_config = asclepius.config
    model.config.use_decoder_only_language_model = True
    # Asegurar pad_token_id en el LLM (LLaMA no trae pad por defecto).
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
    #   adaptador (2560→H_llm) es la única capa nueva (init por defecto de nn.Linear,
    #   entrenada por Full FT). Ver docstring del módulo.
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
        # puente; recomputar activaciones ahorra VRAM con LLaMA-7B.
        try:
            model.language_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            model.language_model.config.use_cache = False
            print("        ✓ gradient_checkpointing habilitado en el LLM (use_reentrant=False)")
        except Exception as exc:
            print(f"        ⚠ No se pudo habilitar gradient_checkpointing en el LLM: {exc}")

    # ── TIER 1 · torch.compile del LLM — DESACTIVADO por defecto (evidencia empírica) ─
    # Se MIDIÓ en esta arquitectura: torch.compile(LLM) da -10% (2.67 vs 2.42 s/it)
    # porque el padding "longest" produce secuencias de longitud VARIABLE por batch
    # → recompilaciones/guardas dinámicas cuyo overhead supera la ganancia. Queda el
    # gancho disponible SOLO si se fuerza TIER1_COMPILE_LLM=1 (no recomendado aquí).
    if _os.environ.get("TIER1_COMPILE_LLM", "0") == "1":
        try:
            model.language_model = torch.compile(
                model.language_model, dynamic=True, mode="default"
            )
            print("        ✓ [TIER 1 opt] torch.compile(LLM, dynamic=True) ACTIVADO "
                  "(⚠ medido -10% aquí; solo por petición explícita)")
        except Exception as exc:
            print(f"        ⚠ [TIER 1 opt] torch.compile no disponible ({exc}); "
                  "se continúa en modo eager.")

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

    # ── Paso 4-bis (TIER 1): cabeza clínica auxiliar CheXpert ─────────────────
    # Linear(hidden_qformer → 14) sobre el pooling de los 32 query tokens del
    # Q-Former (la MISMA representación puente que consume el LLM). Es la única
    # capa nueva que introduce Tier 1: ~11K params → coste de cómputo ≈ 0.
    qf_hidden = int(getattr(model.qformer.config, "hidden_size",
                            model.query_tokens.shape[-1]))
    aux_head = nn.Linear(qf_hidden, NUM_CHEXPERT).to(
        device=target_device, dtype=target_dtype
    )
    for p in aux_head.parameters():
        p.requires_grad = True
    model.chexpert_aux_head = aux_head
    print(f"\n  [4-bis · TIER 1] Cabeza clínica auxiliar: "
          f"Linear({qf_hidden}→{NUM_CHEXPERT})  "
          f"({sum(p.numel() for p in aux_head.parameters()):,} params, entrenable)")

    # ── Paso 5: AUDITORÍA DE PARÁMETROS ──────────────────────────────────────
    _auditar_final_experiment(model, opt_hidden, llama_hidden)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 5. AUDITORÍA DE GRADIENTES (las 3 métricas requeridas + verificación)
# ─────────────────────────────────────────────────────────────────────────────

def _auditar_final_experiment(
    model: Blip2ForConditionalGeneration,
    opt_hidden: int,
    llama_hidden: int,
) -> None:
    """
    Imprime OBLIGATORIAMENTE las tres métricas solicitadas y verifica invariantes:
        a) Total entrenables en el ViT  → 0   (100% congelado, incluida la LoRA).
        b) Total entrenables en el LLM  → 0   (100% congelado, sin LoRA).
        c) Total entrenables en Q-Former→ 100% del submódulo.
    (+) language_projection (LP preentrenada + adaptador) y query_tokens → Full FT.
    """
    n_vit_train,  n_vit_total  = _contar(model.vision_model)
    n_llm_train,  n_llm_total  = _contar(model.language_model)
    n_qf_train,   n_qf_total   = _contar(model.qformer)
    n_lp_train,   n_lp_total   = _contar(model.language_projection)

    # Desglose informativo del bloque visual (backbone RadDINO vs proyección)
    backbone = getattr(model.vision_model, "backbone", None)
    proj_v   = getattr(model.vision_model, "projection", None)
    n_bb_train, n_bb_total   = _contar(backbone) if backbone is not None else (0, 0)
    n_pv_train, n_pv_total   = _contar(proj_v)   if proj_v   is not None else (0, 0)

    n_query = (model.query_tokens.numel()
               if (hasattr(model, "query_tokens")
                   and model.query_tokens.requires_grad) else 0)

    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = 100.0 * train / total if total else 0.0

    print("\n" + "═" * 78)
    print("  AUDITORÍA DE PARÁMETROS · EXPERIMENTO FINAL (RadDINO + Asclepius-R + Puente Full-FT)")
    print("═" * 78)
    print("  (a) ViT — RadDINO + LoRA Exp2 fusionada + proyección (esperado: 0 entrenables)")
    print(f"        entrenables : {n_vit_train:>15,}  / total {n_vit_total:,}")
    print(f"          · backbone RadDINO   : {n_bb_train:>12,} / {n_bb_total:,}")
    print(f"          · proyección 768→1408: {n_pv_train:>12,} / {n_pv_total:,}")
    print("  (b) LLM — Asclepius-R / LLaMA-7B (esperado: 0 entrenables, SIN LoRA)")
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
            "La auditoría de gradientes del Experimento FINAL no cumple las "
            "invariantes (ViT=0, LLM=0, Q-Former=100%). Revisa "
            "build_final_experiment_model()."
        )
    print("  ✓ Invariantes OK: ViT=0 · LLM=0 · Q-Former=100% · language_projection/query Full-FT")
    print("═" * 78 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# 6. OPTIMIZADOR (Puente Full-FT: un solo grupo, lr = 1e-5 CORREGIDO)
# ─────────────────────────────────────────────────────────────────────────────

def build_final_experiment_optimizer(model: Blip2ForConditionalGeneration) -> Tuple[AdamW, None]:
    """
    AdamW con UN ÚNICO grupo de parámetros (el puente, Full FT):
        qformer + language_projection(LP preentrenada + adaptador) + query_tokens.
    lr = TRAINING_CONFIG["lr_bridge"] = 1e-5 (CORREGIDO para evitar la explosión
    de gradientes del Q-Former Full-FT). El scheduler (cosine + warmup) se delega
    a TrainingArguments.
    """
    bridge_params: List[nn.Parameter] = []
    bridge_params += [p for p in model.qformer.parameters() if p.requires_grad]
    bridge_params += [p for p in model.language_projection.parameters() if p.requires_grad]
    # TIER 1: la cabeza clínica auxiliar entrena en el MISMO grupo/lr que el puente
    # (no se añade ningún learning-rate nuevo; hereda lr_bridge = 1e-5).
    if hasattr(model, "chexpert_aux_head"):
        bridge_params += [p for p in model.chexpert_aux_head.parameters() if p.requires_grad]
    if hasattr(model, "query_tokens") and model.query_tokens.requires_grad:
        bridge_params.append(model.query_tokens)

    if not bridge_params:
        raise RuntimeError("No hay parámetros entrenables en el puente (Experimento FINAL).")

    optimizer = AdamW(
        bridge_params,
        lr=TRAINING_CONFIG["lr_bridge"],
        weight_decay=TRAINING_CONFIG["weight_decay"],
        eps=1e-8,
        fused=torch.cuda.is_available(),
    )
    n_bridge = sum(p.numel() for p in bridge_params)
    print(f"   [optimizer] Puente Full-FT : lr={TRAINING_CONFIG['lr_bridge']:.0e}  "
          f"({n_bridge:,} params)  ⚠ lr CORREGIDO (1e-5)")
    print(f"   [optimizer] weight_decay={TRAINING_CONFIG['weight_decay']}  "
          f"| scheduler (cosine+warmup) delegado a HuggingFace")
    return optimizer, None


# ─────────────────────────────────────────────────────────────────────────────
# 7. CONSTRUCCIÓN DE DATASETS (usa EXPLÍCITAMENTE el prompt diseñado)
# ─────────────────────────────────────────────────────────────────────────────

def build_final_experiment_datasets(processor):
    """
    Construye los datasets de train/val del 50% inyectando de forma EXPLÍCITA el
    prompt diseñado (DEFAULT_PROMPT = config.inference.default_prompt). NO se usa
    self.prompt ni ningún prompt dinámico/nativo.

    IMPORTANTE: `processor` debe traer YA el tokenizer de Asclepius (llamar antes
    a preparar_procesador_exp3), para que los labels se tokenicen en vocab LLaMA.
    """
    # TIER 1: CheXpertAugmentedDataset = MimicCXRDataset + etiquetas CheXpert(14).
    uncertain = TRAINING_CONFIG["chexpert_uncertain"]
    print(f"   [datasets] prompt diseñado (EXPLÍCITO) = {DEFAULT_PROMPT!r}")
    print(f"   [datasets] TIER 1 · etiquetas CheXpert (incierto='{uncertain}') desde {CHEXPERT_CSV.name}")
    train_ds = CheXpertAugmentedDataset(
        csv_path             = TRAIN_CSV,
        images_dir           = IMAGES_DIR,
        processor            = processor,
        image_size           = config.data.image_size,
        clahe_clip_limit     = config.data.clahe_clip_limit,
        clahe_tile_grid_size = config.data.clahe_tile_grid_size,
        max_length           = config.data.max_length,
        # padding="longest": tokeniza sin padding al 577 global; el
        # DataCollatorForSeq2Seq(padding=True) + group_by_length rellenan por-batch
        # al máximo real del batch (~1.8x menos cómputo en LLaMA-7B). La loss solo
        # cuenta tokens reales (padding → -100), por lo que es matemáticamente igual.
        padding              = "longest",
        prompt               = DEFAULT_PROMPT,   # ← prompt diseñado EXPLÍCITO (único permitido)
        augment              = True,
        chexpert_csv         = CHEXPERT_CSV,
        uncertain            = uncertain,
    )
    val_ds = CheXpertAugmentedDataset(
        csv_path             = VAL_CSV,
        images_dir           = IMAGES_DIR,
        processor            = processor,
        image_size           = config.data.image_size,
        clahe_clip_limit     = config.data.clahe_clip_limit,
        clahe_tile_grid_size = config.data.clahe_tile_grid_size,
        max_length           = config.data.max_length,
        padding              = "longest",        # dinámico por-batch (ver nota en train_ds)
        prompt               = DEFAULT_PROMPT,    # ← prompt diseñado EXPLÍCITO (único permitido)
        augment              = False,
        chexpert_csv         = CHEXPERT_CSV,
        uncertain            = uncertain,
    )
    return train_ds, val_ds


# ─────────────────────────────────────────────────────────────────────────────
# 8. SMOKE TEST (construcción + auditoría sin entrenar)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Smoke test de build_final_experiment_model (construye + audita)."
    )
    parser.add_argument("--no-query-tokens", action="store_true",
                        help="Congelar query_tokens (solo qformer + language_projection).")
    parser.add_argument("--forward", action="store_true",
                        help="Probar un forward dummy (pixel_values + input_ids aleatorios).")
    args = parser.parse_args()

    print("Construyendo modelo del Experimento FINAL "
          "(requiere RadDINO + exp2_state.pt + Asclepius-R)...")
    model = build_final_experiment_model(train_query_tokens=not args.no_query_tokens)
    build_final_experiment_optimizer(model)

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
