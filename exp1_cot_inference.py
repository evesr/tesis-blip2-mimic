#!/usr/bin/env python3
"""
exp1_cot_inference.py — Experimento 1: Inferencia Chain-of-Thought (CoT) + CheXpert
====================================================================================
Experimento SIN entrenamiento sobre el modelo de Fase 2 (50% MIMIC-CXR).

Objetivo
--------
Forzar al modelo a realizar una CLASIFICACIÓN explícita (Task 1) ANTES de generar
el reporte (Task 2), mediante un prompt estructurado de Chain-of-Thought. La
hipótesis es que "pensar en voz alta" qué patologías están presentes mejora la
fidelidad clínica de las etiquetas CheXpert extraídas del texto generado.

Modelo (FASE 2 · dual-LoRA)
---------------------------
  • LLM  → PeftModel (adaptador LoRA de Fase 2; se MANTIENE el wrapper).
  • ViT  → merge_and_unload() (fusión OBLIGATORIA para generate()).
  • Q-Former / language_projection → pesos BASE (Fase 2 NO los entrena).

Prompt CoT exacto (inyectado por cada muestra del Test Set)
----------------------------------------------------------
    "[Context: {vista_mapeada} view] Task 1: Identify the presence or absence of
     Cardiomegaly, Edema, Consolidation, Atelectasis, Pleural Effusion,
     Pneumothorax, and Support Devices. Task 2: {default_prompt} "

donde {default_prompt} = config.inference.default_prompt
      {vista_mapeada}  = Frontal | Lateral | Unknown  (MimicCXRDataset._mapear_vista)

Generación
----------
  • num_beams = 3        (solicitado por el usuario; más rápido que beam=5)
  • max_new_tokens = 512 (espacio para cumplir AMBAS tareas en una sola pasada)

Salidas (en phase2_results-50pct/exp1_cot/)
-------------------------------------------
  1. exp1_raw_inference_results.csv
        dicom_id, subject_id, study_id, reference_report, generated_report
  2. exp1_chexpert_per_report.csv      (granular fila por fila — misma lógica Fase 3)
        + pred_chexpert_* (CheXbert) + GT crudo + chexpert_precision/recall/f1 + matches
  3. exp1_chexpert_per_pathology.csv   (granular por patología — NaN-aware estricto)
        categoria, TP, FP, FN, TN, precision, recall, f1, ...
  4. exp1_chexpert_general.csv         (resumen global micro + macro)

TODO el cómputo pesado (inferencia BLIP-2 + CheXbert) corre en GPU.

Uso
---
    python exp1_cot_inference.py                       # Test Set completo (19,137)
    python exp1_cot_inference.py --num-samples 200     # smoke test rápido
    python exp1_cot_inference.py --skip-inference      # reutiliza el raw CSV y solo evalúa
    python exp1_cot_inference.py --batch-size 4

Autor : Evelyn Silva Rozas
Fecha : Junio 2026
"""

# ─────────────────────────────────────────────────────────────────────────────
# Variables de entorno (ANTES de cualquier import de transformers/tokenizers)
# ─────────────────────────────────────────────────────────────────────────────
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import gc
import logging
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# cv2 debe desactivar hilos ANTES de DataLoader/torch para evitar contención CPU
import cv2
cv2.setNumThreads(0)

# ── Módulos del proyecto ──────────────────────────────────────────────────────
from config import config
from model_utils import cargar_procesador
from data_loader import MimicCXRDataset
from preprocessing import estandarizar_imagen
from chexbert_official import CheXbertLabeler, CHEXBERT_LABELS

# Paths y pesos duales de Fase 3 (50%) — reutilizados sin modificar el trainer
# ── Rutas y pesos de FASE 2 (50%) ──────────────────────────────────────
# CAMBIO Fase 3 → Fase 2: este experimento usa el modelo de FASE 2 (dual-LoRA
# LLM+ViT). Por eso ya NO se importa de trainer_utils_phase3_50pct; las rutas se
# definen directamente aquí apuntando a phase2_results-50pct.
#
# ANTES (Fase 3):
#   from trainer_utils_phase3_50pct import (
#       OUTPUT_DIR, CHECKPOINTS_DIR, TEST_CSV_10PCT, IMAGES_DIR_10PCT, DUAL_WEIGHTS,
#   )
BASE_DIR         = config.paths.base_dir
TEST_CSV_10PCT   = BASE_DIR / "test_split_50pct.csv"      # test set 50% (nombre histórico)
IMAGES_DIR_10PCT = BASE_DIR / "imagenes_50_porciento"     # imágenes 50% (nombre histórico)
PHASE2_BEST_DIR  = BASE_DIR / "phase2_results-50pct" / "checkpoints" / "best_model_Config_B1"
LLM_DIR          = PHASE2_BEST_DIR / "llm"               # adaptador LLM LoRA (Fase 2)
VIT_DIR          = PHASE2_BEST_DIR / "vit"               # adaptador ViT LoRA (Fase 2)
OUTPUT_DIR       = BASE_DIR / "phase2_results-50pct"      # raíz de salidas del experimento

# Lógica de métricas CheXpert por patología (NaN-aware) + re-etiquetado CheXbert.
# Se reutiliza la MISMA lógica usada en la limpieza/evaluación de Fase 3.
from limpiar_tail_babbling import (
    evaluar_metricas_estrictas,
    reetiquetar_con_chexbert,
    _canon,
)

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  CONSTANTES DEL EXPERIMENTO
# ─────────────────────────────────────────────────────────────────────────────

# Prompt por defecto del proyecto (Task 2 del CoT)
DEFAULT_PROMPT: str = config.inference.default_prompt

# Las 7 condiciones que el modelo debe clasificar primero (Task 1)
SEVEN_CONDITIONS: str = (
    "Cardiomegaly, Edema, Consolidation, Atelectasis, "
    "Pleural Effusion, Pneumothorax, and Support Devices"
)


def construir_prompt_cot(vista: str) -> str:
    """
    Construye el prompt Chain-of-Thought EXACTO solicitado para el experimento.

        "[Context: {vista} view] Task 1: Identify the presence or absence of
         Cardiomegaly, Edema, Consolidation, Atelectasis, Pleural Effusion,
         Pneumothorax, and Support Devices. Task 2: {default_prompt} "

    (Se respeta el espacio final intencional tras Task 2.)
    """
    return (
        f"[Context: {vista} view] "
        f"Task 1: Identify the presence or absence of {SEVEN_CONDITIONS}. "
        f"Task 2: {DEFAULT_PROMPT} "
    )


# Parámetros de generación del experimento (Fase 2).
#   • num_beams=3        → solicitado por el usuario (más rápido que beam=5)
#   • max_new_tokens=512 → espacio para que el modelo cumpla AMBAS tareas CoT
EXP1_GEN_PARAMS: Dict = dict(
    num_beams            = 3,                            # ← beam=3 (requisito del usuario)
    do_sample            = False,
    repetition_penalty   = 1.1,
    no_repeat_ngram_size = 3,
    max_new_tokens       = 512,                          # ← REQUISITO del experimento
    use_cache            = True,
)

# Directorio de salida del experimento
EXP1_DIR: Path = OUTPUT_DIR / "exp1_cot"
RAW_CSV: Path = EXP1_DIR / "exp1_raw_inference_results.csv"
PER_REPORT_CSV: Path = EXP1_DIR / "exp1_chexpert_per_report.csv"
PER_PATHOLOGY_CSV: Path = EXP1_DIR / "exp1_chexpert_per_pathology.csv"
GENERAL_CSV: Path = EXP1_DIR / "exp1_chexpert_general.csv"

# Configuración del modelo de Fase 2 al 50% (única corrida: dual-LoRA LLM+ViT)
CONFIG_NAME: str = "Config_B1"


# ─────────────────────────────────────────────────────────────────────────────
# 2.  CARGA DEL MODELO DE FASE 2 (dual-LoRA: LLM PeftModel + ViT fusionado)
# ─────────────────────────────────────────────────────────────────────────────
# CAMBIO Fase 3 → Fase 2: ANTES esta función (cargar_modelo_phase3) fusionaba 4
# componentes (LLM+ViT merge, Q-Former LoRA merge, language_projection.pt). El
# modelo de FASE 2 solo tiene adaptadores duales LLM+ViT; el Q-Former y la
# language_projection conservan sus pesos BASE. La carga replica EXACTAMENTE
# inferencia_profunda_fase2.cargar_modelo_dual_lora.

def cargar_modelo_phase2(model_dir: Path):
    """
    Reconstruye el modelo BLIP-2 de FASE 2 (dual-LoRA: LLM + ViT).

    Réplica EXACTA de inferencia_profunda_fase2.cargar_modelo_dual_lora:
        1. Base BLIP-2 (bfloat16, device_map=auto)
        2. LLM LoRA (Fase 2) → PeftModel (se MANTIENE el wrapper; NO se fusiona,
           porque generate() invoca el LLM por kwargs y el wrapper es compatible).
        3. ViT LoRA (Fase 2) → merge_and_unload() (fusión OBLIGATORIA: generate()
           llama al ViT con pixel_values POSICIONAL, incompatible con PeftModel).
        4. Q-Former            → SIN tocar (pesos base; Fase 2 no lo entrena).
        5. language_projection → SIN tocar (pesos base; Fase 2 no la entrena).

    Returns:
        (model, device)
    """
    from transformers import Blip2ForConditionalGeneration
    from peft import PeftModel

    llm_dir = Path(model_dir) / "llm"
    vit_dir = Path(model_dir) / "vit"
    for sub, lbl in [(llm_dir, "LLM"), (vit_dir, "ViT")]:
        if not sub.exists():
            raise FileNotFoundError(
                f"Adaptador {lbl} de Fase 2 no encontrado: {sub}\n"
                "Verifica phase2_results-50pct/checkpoints/best_model_Config_B1/."
            )

    logger.info(f"\n{'='*65}")
    logger.info(f"  CARGA MODELO FASE 2 (50%): {CONFIG_NAME}  ·  dual-LoRA")
    logger.info(f"  LLM ← {llm_dir}   (PeftModel, sin fusionar)")
    logger.info(f"  ViT ← {vit_dir}   (merge_and_unload)")
    logger.info(f"  Q-Former / language_projection: pesos BASE (Fase 2 no los entrena)")
    logger.info(f"{'='*65}")

    logger.info("  Cargando modelo base BLIP-2 (bfloat16, device_map=auto)...")
    base_model = Blip2ForConditionalGeneration.from_pretrained(
        config.model.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    logger.info("  Inyectando LLM LoRA (PeftModel, is_trainable=False)...")
    base_model.language_model = PeftModel.from_pretrained(
        base_model.language_model, str(llm_dir), is_trainable=False
    )

    logger.info("  Inyectando ViT LoRA y fusionando (merge_and_unload)...")
    vit_peft = PeftModel.from_pretrained(
        base_model.vision_model, str(vit_dir), is_trainable=False
    )
    # qformer nunca se toca → refleja el device/dtype reales del modelo base
    _ref = next(base_model.qformer.parameters())
    base_model.vision_model = vit_peft.merge_and_unload().to(
        device=_ref.device, dtype=_ref.dtype
    )

    base_model.eval()
    logger.info(f"  ✓ Modelo dual-LoRA listo en {_ref.device} | dtype={_ref.dtype}\n")
    return base_model, _ref.device


# ─────────────────────────────────────────────────────────────────────────────
# 3.  FASE DE INFERENCIA CoT (GPU, batched)
# ─────────────────────────────────────────────────────────────────────────────

def fase_inferencia_cot(
    model,
    processor,
    device,
    num_samples: Optional[int] = None,
    batch_size: int = 16,
    resume: bool = False,
) -> Path:
    """
    Genera reportes crudos para el Test Set del 50% usando el prompt CoT y los
    exporta a exp1_raw_inference_results.csv (escritura incremental).

    Si resume=True y el CSV ya contiene filas, NO se reescribe el header: se
    cargan los dicom_id ya generados y se SALTAN en el bucle, reanudando la
    inferencia exactamente donde quedó (p. ej. tras una pausa del pod).
    """
    logger.info("  Cargando Test Set (50%)...")
    test_dataset = MimicCXRDataset(
        csv_path            = TEST_CSV_10PCT,
        images_dir          = IMAGES_DIR_10PCT,
        processor           = processor,
        image_size          = config.data.image_size,
        clahe_clip_limit    = config.data.clahe_clip_limit,
        clahe_tile_grid_size= config.data.clahe_tile_grid_size,
        max_length          = config.data.max_length,
        padding             = config.data.padding,
        prompt              = config.inference.default_prompt,  # no se usa para gen (prompt manual)
        augment             = False,                            # inferencia determinista
    )

    n_total = len(test_dataset)
    n_eval  = n_total if num_samples is None else min(num_samples, n_total)
    logger.info(f"  Test Set: {n_total:,} muestras | A evaluar: {n_eval:,}")
    logger.info(f"  Prompt CoT ejemplo: {construir_prompt_cot('Frontal')!r}")
    logger.info(f"  GEN_PARAMS: {EXP1_GEN_PARAMS}")

    EXP1_DIR.mkdir(parents=True, exist_ok=True)

    COLS = ["dicom_id", "subject_id", "study_id", "reference_report", "generated_report"]

    # ── Reanudación (resume) ─────────────────────────────────────────────────
    # Si --resume está activo y el CSV ya tiene filas, NO se reescribe el header:
    # se cargan los dicom_id ya guardados para saltarlos en el bucle. El criterio
    # se basa en dicom_id REALES (no en el índice), por lo que es robusto incluso
    # si en la corrida previa se saltaron muestras por errores de imagen.
    procesados: set = set()
    if resume and RAW_CSV.exists():
        try:
            _prev = pd.read_csv(RAW_CSV, usecols=["dicom_id"])
            procesados = set(_prev["dicom_id"].astype(str).tolist())
        except Exception as exc:
            logger.warning(f"  [resume] no se pudo leer {RAW_CSV.name}: {exc}; se empieza de cero.")
            procesados = set()

    if procesados:
        logger.info(f"  [RESUME] {len(procesados):,} reportes ya guardados → se reanuda en modo append "
                    f"(faltan ~{max(0, n_eval - len(procesados)):,}).")
    else:
        # Comportamiento original: crear CSV nuevo con solo el header.
        pd.DataFrame(columns=COLS).to_csv(RAW_CSV, index=False)

    batch_images:  List = []
    batch_prompts: List[str] = []
    batch_meta:    List[Dict] = []
    errores       = 0
    n_guardadas   = 0

    t0 = time.time()
    for idx in tqdm(range(n_eval), desc="  Inferencia CoT", unit="img"):
        row      = test_dataset.data.iloc[idx]
        dicom_id = str(row["dicom_id"])

        # [resume] saltar muestras ya generadas en una corrida previa
        if procesados and dicom_id in procesados:
            continue

        study_id = row.get("study_id", "")
        subj_id  = row.get("subject_id", "")
        ref_text = str(row["report_text"]).strip()

        # Vista mapeada (idéntico criterio que data_loader / run_phase3)
        view_raw = row.get("ViewPosition", None)
        if view_raw is not None and not (
            isinstance(view_raw, float) and math.isnan(view_raw)
        ):
            vista = test_dataset._mapear_vista(str(view_raw))
        else:
            vista = "Unknown"

        # ── PROMPT CoT (la ÚNICA diferencia clave vs inferencia estándar) ────
        prompt = construir_prompt_cot(vista)

        # Imagen
        imagen_path = test_dataset._buscar_imagen(dicom_id)
        if imagen_path is None:
            errores += 1
            continue
        imagen_pil = estandarizar_imagen(
            imagen_path,
            target_size          = config.data.image_size,
            clahe_clip_limit     = config.data.clahe_clip_limit,
            clahe_tile_grid_size = config.data.clahe_tile_grid_size,
        )
        if imagen_pil is None:
            errores += 1
            continue

        batch_images.append(imagen_pil)
        batch_prompts.append(prompt)
        batch_meta.append({
            "dicom_id":   dicom_id,
            "subject_id": subj_id,
            "study_id":   study_id,
            "ref_text":   ref_text,
        })

        # Vaciar lote cuando esté lleno o sea la última iteración
        if len(batch_images) == batch_size or idx == n_eval - 1:
            if not batch_images:
                continue
            try:
                inputs = processor(
                    images=batch_images,
                    text=batch_prompts,
                    return_tensors="pt",
                    padding=True,
                ).to(device)

                with torch.no_grad():
                    generated_ids = model.generate(**inputs, **EXP1_GEN_PARAMS)

                decoded = processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )

                rows = [
                    {
                        "dicom_id":         m["dicom_id"],
                        "subject_id":       m["subject_id"],
                        "study_id":         m["study_id"],
                        "reference_report": m["ref_text"],
                        "generated_report": rep.strip(),
                    }
                    for m, rep in zip(batch_meta, decoded)
                ]
                pd.DataFrame(rows, columns=COLS).to_csv(
                    RAW_CSV, mode="a", header=False, index=False
                )
                n_guardadas += len(rows)

            except Exception as exc:
                logger.exception(f"  Error en lote idx={idx}: {exc}")
                errores += len(batch_images)

            batch_images.clear()
            batch_prompts.clear()
            batch_meta.clear()

    dt = time.time() - t0
    logger.info(
        f"  Inferencia CoT OK: {n_guardadas:,} reportes | {errores} errores | {dt/60:.1f} min"
    )
    logger.info(f"  → {RAW_CSV}")

    del test_dataset
    gc.collect()
    torch.cuda.empty_cache()
    return RAW_CSV


# ─────────────────────────────────────────────────────────────────────────────
# 4.  EVALUACIÓN CheXpert (misma lógica de Fase 3) — GT oficial vs CheXbert
# ─────────────────────────────────────────────────────────────────────────────

def _merge_gt_oficial(df: pd.DataFrame, chexpert_csv: Path) -> pd.DataFrame:
    """
    Une el Ground Truth oficial (mimic-cxr-2.0.0-chexpert.csv) por study_id.

    Réplica de la lógica de merge de evaluate_final.fase2_evaluacion:
    normaliza study_id (quita prefijo 's') y trae las columnas crudas de las
    14 etiquetas CheXpert (1.0/0.0/-1.0/NaN, sin imputar).
    """
    def _norm_sid(s):
        s = str(s).strip()
        return s.lstrip("s") if s.startswith("s") else s

    df_chex = pd.read_csv(chexpert_csv, compression="infer")
    df = df.copy()
    df["_sid"]      = df["study_id"].astype(str).apply(_norm_sid)
    df_chex["_sid"] = df_chex["study_id"].astype(str).apply(_norm_sid)

    cols_gt = [lbl for lbl in CHEXBERT_LABELS if lbl in df_chex.columns]
    merged = df.merge(
        df_chex[["_sid"] + cols_gt].drop_duplicates(subset=["_sid"]),
        on="_sid", how="left",
    )
    merged.drop(columns=["_sid"], inplace=True)

    n_match = merged[cols_gt[0]].notna().sum() if cols_gt else 0
    logger.info(f"  GT CheXpert tras merge: {n_match:,}/{len(merged):,} filas con etiquetas")
    return merged


def _chexpert_comparison_per_report(
    gt_row: pd.Series,
    pred_dict: Dict[str, float],
) -> Dict[str, float]:
    """
    Compara GT (CSV oficial, nombres crudos) vs Pred (CheXbert, prefijo
    pred_chexpert_) clase por clase → micro Precision/Recall/F1 por reporte.

    Réplica EXACTA de evaluate_final._chexpert_comparison_metrics:
        1.0  → positivo  | 0.0 → negativo | -1.0 → positivo (incierto=mencionado)
        NaN  → se imputa a 0 (no mencionado = negativo) para el match binario.
    """
    tp = fp = fn = 0
    metrics: Dict[str, float] = {}

    for lbl in CHEXBERT_LABELS:
        col_gt   = lbl
        col_pred = f"pred_chexpert_{_canon(lbl)}"

        gt_val   = gt_row.get(col_gt,   float("nan"))
        pred_val = pred_dict.get(col_pred, float("nan"))

        def _to_bin(v):
            if isinstance(v, float) and math.isnan(v):
                return None
            return 1 if float(v) != 0.0 else 0   # 1.0 y -1.0 → positivo

        gt_bin   = _to_bin(gt_val)
        pred_bin = _to_bin(pred_val)

        if gt_bin is None and pred_bin is None:
            metrics[f"chexpert_match_{_canon(lbl)}"] = float("nan")
            continue
        if gt_bin   is None: gt_bin   = 0
        if pred_bin is None: pred_bin = 0

        metrics[f"chexpert_match_{_canon(lbl)}"] = 1.0 if gt_bin == pred_bin else 0.0

        if   gt_bin == 1 and pred_bin == 1: tp += 1
        elif gt_bin == 0 and pred_bin == 1: fp += 1
        elif gt_bin == 1 and pred_bin == 0: fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall    = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    if not (math.isnan(precision) or math.isnan(recall)) and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = float("nan")

    metrics["chexpert_precision"] = precision
    metrics["chexpert_recall"]    = recall
    metrics["chexpert_f1"]        = f1
    return metrics


def evaluar_chexpert(
    raw_csv: Path,
    chexbert: CheXbertLabeler,
    chexpert_csv: Optional[Path],
    chexbert_batch: int = 16,
) -> None:
    """
    Pipeline CheXpert completo sobre el CSV crudo del experimento:

        1. CheXbert (GPU) sobre generated_report → columnas pred_chexpert_*
        2. Merge GT oficial (study_id) → columnas crudas de las 14 etiquetas
        3. Granular POR REPORTE   → exp1_chexpert_per_report.csv
        4. Granular POR PATOLOGÍA → exp1_chexpert_per_pathology.csv  (NaN-aware)
        5. Resumen GENERAL        → exp1_chexpert_general.csv        (micro+macro)
    """
    logger.info(f"\n{'='*65}")
    logger.info(f"  EVALUACIÓN CheXpert — {CONFIG_NAME} (CoT)")
    logger.info(f"{'='*65}")

    df = pd.read_csv(raw_csv)
    logger.info(f"  Reportes a etiquetar: {len(df):,}")

    # ── 1. CheXbert sobre generated_report (GPU, batched) ────────────────────
    df["generated_report"] = df["generated_report"].fillna("").astype(str)
    df = reetiquetar_con_chexbert(
        df,
        col_texto="generated_report",
        prefijo_pred="pred_chexpert_",
        labeler=chexbert,
        batch_size=chexbert_batch,
        verbose=True,
    )

    # ── 2. Merge GT oficial ──────────────────────────────────────────────────
    if chexpert_csv is None or not Path(chexpert_csv).exists():
        logger.warning("  CSV CheXpert oficial no disponible — se omite comparación GT.")
        df.to_csv(PER_REPORT_CSV, index=False)
        return
    df = _merge_gt_oficial(df, Path(chexpert_csv))

    # ── 3. Granular POR REPORTE (micro P/R/F1 + match por clase) ─────────────
    pred_cols = [f"pred_chexpert_{_canon(l)}" for l in CHEXBERT_LABELS]
    rows_out: List[Dict] = []
    for _, row in tqdm(df.iterrows(), total=len(df),
                       desc="  CheXpert por reporte", unit="rep"):
        pred_dict = {c: row.get(c, float("nan")) for c in pred_cols}
        out: Dict = {
            "dicom_id":   row.get("dicom_id", ""),
            "subject_id": row.get("subject_id", ""),
            "study_id":   row.get("study_id", ""),
        }
        out.update(pred_dict)
        out.update(_chexpert_comparison_per_report(row, pred_dict))
        rows_out.append(out)

    df_per_report = pd.DataFrame(rows_out)
    df_per_report.to_csv(PER_REPORT_CSV, index=False)
    logger.info(f"  ✓ Granular por reporte → {PER_REPORT_CSV}  ({len(df_per_report):,} filas)")

    # ── 4. Granular POR PATOLOGÍA (NaN-aware estricto, MISMA lógica Fase 3) ──
    df_pat = evaluar_metricas_estrictas(
        df,
        prefijo_pred="pred_chexpert_",
        uncertain_as_positive=True,
        labels=CHEXBERT_LABELS,
        verbose=True,
        titulo="EXP1 CoT — CheXpert por patología (GT oficial vs CheXbert)",
    )
    df_pat.to_csv(PER_PATHOLOGY_CSV, index=False)
    logger.info(f"  ✓ Granular por patología → {PER_PATHOLOGY_CSV}  ({len(df_pat)} categorías)")

    # ── 5. Resumen GENERAL (micro pooled + macro nanmean) ────────────────────
    tp_t = float(np.nansum(df_pat["TP"]))
    fp_t = float(np.nansum(df_pat["FP"]))
    fn_t = float(np.nansum(df_pat["FN"]))
    tn_t = float(np.nansum(df_pat["TN"]))

    micro_p = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else float("nan")
    micro_r = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else float("nan")
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)
                if not (np.isnan(micro_p) or np.isnan(micro_r)) and (micro_p + micro_r) > 0
                else float("nan"))

    with np.errstate(invalid="ignore"):
        macro_p  = float(np.nanmean(df_pat["precision"]))
        macro_r  = float(np.nanmean(df_pat["recall"]))
        macro_f1 = float(np.nanmean(df_pat["f1"]))

    accuracy = (tp_t + tn_t) / (tp_t + fp_t + fn_t + tn_t) if (tp_t + fp_t + fn_t + tn_t) > 0 else float("nan")

    df_general = pd.DataFrame([
        {"scope": "micro (pooled 14 clases)", "precision": micro_p, "recall": micro_r, "f1": micro_f1,
         "TP": tp_t, "FP": fp_t, "FN": fn_t, "TN": tn_t, "accuracy_binaria": accuracy},
        {"scope": "macro (nanmean 14 clases)", "precision": macro_p, "recall": macro_r, "f1": macro_f1,
         "TP": np.nan, "FP": np.nan, "FN": np.nan, "TN": np.nan, "accuracy_binaria": np.nan},
    ])
    df_general.to_csv(GENERAL_CSV, index=False)
    logger.info(f"  ✓ Resumen general → {GENERAL_CSV}")
    logger.info(
        f"\n  RESUMEN CheXpert (CoT):"
        f"\n    micro  P={micro_p:.4f}  R={micro_r:.4f}  F1={micro_f1:.4f}"
        f"\n    macro  P={macro_p:.4f}  R={macro_r:.4f}  F1={macro_f1:.4f}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5.  CLI / MAIN
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Experimento 1: Inferencia Chain-of-Thought + evaluación CheXpert (GPU).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--num-samples", type=int, default=None,
                   help="Limitar a N muestras del Test Set (None = todas, 19,137).")
    p.add_argument("--batch-size", type=int, default=16,
                   help="Tamaño de lote de inferencia (beam=3, max_new=512 → 16 óptimo: "
                        "~2.70 s/img vs 3.29 con batch=4; la GPU L40S de 45GB lo soporta).")
    p.add_argument("--chexbert-batch", type=int, default=16,
                   help="Tamaño de lote de CheXbert.")
    p.add_argument("--model-dir", type=str,
                   default=str(PHASE2_BEST_DIR),
                   help="Directorio del checkpoint de Fase 2 (subdirs llm/ + vit/).")
    p.add_argument("--chexpert-csv", type=str,
                   default=str(config.paths.base_dir / "mimic-cxr-2.0.0-chexpert.csv"),
                   help="CSV oficial de etiquetas CheXpert (.csv o .csv.gz).")
    p.add_argument("--skip-inference", action="store_true",
                   help="Reutilizar exp1_raw_inference_results.csv y solo (re)evaluar CheXpert.")
    p.add_argument("--resume", action="store_true",
                   help="Reanudar la inferencia: conserva el CSV existente y solo genera los "
                        "reportes faltantes (útil tras una pausa/reinicio del pod).")
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    print("\n" + "=" * 80)
    print("  EXPERIMENTO 1 — INFERENCIA CHAIN-OF-THOUGHT (CoT)  ·  Fase 2 (50%)")
    print("  Task 1: clasificación de 7 patologías  →  Task 2: generación del reporte")
    print("=" * 80)
    print(f"  Modelo      : {CONFIG_NAME} (Fase 2 dual-LoRA: LLM PeftModel + ViT merge)")
    print(f"  max_new_tokens: {EXP1_GEN_PARAMS['max_new_tokens']}  | num_beams: {EXP1_GEN_PARAMS['num_beams']}")
    print(f"  Salidas     : {EXP1_DIR}")
    print(f"  GPU         : {'disponible ✓' if torch.cuda.is_available() else 'NO disponible ✗'}")
    print("=" * 80 + "\n")

    if not torch.cuda.is_available():
        logger.warning("CUDA no disponible: la inferencia será MUY lenta en CPU.")

    chexpert_csv = Path(args.chexpert_csv) if args.chexpert_csv else None
    # Auto-fallback a la variante .gz si la ruta exacta no existe
    if chexpert_csv is not None and not chexpert_csv.exists():
        gz_alt = Path(str(chexpert_csv) + ".gz")
        if gz_alt.exists():
            chexpert_csv = gz_alt

    # ── Procesador BLIP-2 ─────────────────────────────────────────────────────
    logger.info("Cargando Blip2Processor...")
    processor = cargar_procesador(model_name=config.model.model_name)

    # ── FASE INFERENCIA ───────────────────────────────────────────────────────
    if args.skip_inference and RAW_CSV.exists():
        logger.info(f"--skip-inference: reutilizando {RAW_CSV}")
    else:
        model, device = cargar_modelo_phase2(Path(args.model_dir))
        fase_inferencia_cot(
            model, processor, device,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            resume=args.resume,
        )
        # Liberar el modelo BLIP-2 antes de cargar CheXbert
        del model
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("  VRAM del modelo BLIP-2 liberada.\n")

    # ── EVALUACIÓN CheXpert (GPU) ─────────────────────────────────────────────
    logger.info("Cargando CheXbert oficial (Stanford AIMI)...")
    chexbert = CheXbertLabeler()   # autodetecta CUDA
    evaluar_chexpert(
        RAW_CSV, chexbert, chexpert_csv,
        chexbert_batch=args.chexbert_batch,
    )

    print("\n" + "=" * 80)
    print("  EXPERIMENTO 1 — INFERENCIA + CheXpert COMPLETADO")
    print("=" * 80)
    print(f"  Raw            : {RAW_CSV}")
    print(f"  Por reporte    : {PER_REPORT_CSV}")
    print(f"  Por patología  : {PER_PATHOLOGY_CSV}")
    print(f"  General        : {GENERAL_CSV}")
    print("\n  Siguiente paso : python llm_formatter_pipeline.py")
    print("=" * 80 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
