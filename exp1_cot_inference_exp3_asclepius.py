#!/usr/bin/env python3
"""
exp1_cot_inference_exp3_asclepius.py — Experimento 1 (CoT) sobre EXPERIMENTO 3
================================================================================

Misma idea que exp1_cot_inference_exp2_raddino.py, pero ejecutado sobre el
checkpoint del EXPERIMENTO 3 (ViT Fase 3 + Asclepius-R + Puente Full-FT). El
modelo se reconstruye con build_experiment_3_model() + exp3_state.pt
(strict=False); el resto del pipeline (prompt CoT, GEN_PARAMS y las 5 métricas)
es IDÉNTICO al de Fase 3 PUENTE / Experimento 2 → resultados DIRECTAMENTE
comparables.

  ► CRÍTICO: el LLM es Asclepius-R (LLaMA). El `processor.tokenizer` se
    INTERCAMBIA por el de Asclepius (preparar_procesador_exp3) ANTES de generar.

Pipeline
--------
FASE 1 — Inferencia con prompt Chain-of-Thought (PROMPT NUEVO)
    • Modelo Experimento 3 (cargar_modelo_exp3_asclepius)
    • Prompt CoT idéntico a exp1_cot_inference.construir_prompt_cot:
          "[Context: {vista} view] Task 1: Identify the presence or absence
           of Cardiomegaly, Edema, Consolidation, Atelectasis, Pleural
           Effusion, Pneumothorax, and Support Devices. Task 2: {prompt} "
    • Generación con num_beams=3, max_new_tokens=512 (EXP1_GEN_PARAMS)
    • Test Set 50% (mismas imágenes/CSV que evaluate_exp3_asclepius)
    • Reanudable; Output: exp3_asclepius_results-50pct/exp1_cot/
                          exp1_raw_inference_exp3_asclepius.csv

FASE 2 — Evaluación granular fila por fila (las 5 métricas requeridas)
    • CheXpert GT / Pred (CheXbert), BLEU-4, ROUGE-L, F1-RadGraph,
      Clinical BERTScore.
    • Outputs:
        - resultados_granulares_Exp3_Asclepius_CoT.csv
        - metricas_globales_Exp3_Asclepius_CoT.csv

Uso
---
    python exp1_cot_inference_exp3_asclepius.py
    python exp1_cot_inference_exp3_asclepius.py --num-samples 200    # smoke
    python exp1_cot_inference_exp3_asclepius.py --skip-inference     # reusa raw
    python exp1_cot_inference_exp3_asclepius.py --resume             # reanuda
    python exp1_cot_inference_exp3_asclepius.py --skip-evaluation    # solo raw (→ Llama 8b)

NOTA: NO usa entornos virtuales — corre sobre el Python del sistema.

Autor : Evelyn Silva Rozas
Fecha : Junio 2026
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0. Patch tokenizer ANTES de importar radgraph (heredado del proyecto)
# ─────────────────────────────────────────────────────────────────────────────
from transformers import PreTrainedTokenizerBase

if not hasattr(PreTrainedTokenizerBase, "encode_plus"):
    def _ep(self, text, text_pair=None, add_special_tokens=True,
            padding=False, truncation=False, max_length=None,
            stride=0, return_tensors=None, **kwargs):
        return self.__call__(text, text_pair=text_pair,
                             add_special_tokens=add_special_tokens,
                             padding=padding, truncation=truncation,
                             max_length=max_length, stride=stride,
                             return_tensors=return_tensors, **kwargs)
    PreTrainedTokenizerBase.encode_plus = _ep

if not hasattr(PreTrainedTokenizerBase, "build_inputs_with_special_tokens"):
    def _bi(self, t0, t1=None):
        if t1 is None:
            return [self.cls_token_id] + t0 + [self.sep_token_id]
        return [self.cls_token_id] + t0 + [self.sep_token_id] + t1 + [self.sep_token_id]
    PreTrainedTokenizerBase.build_inputs_with_special_tokens = _bi


# ─────────────────────────────────────────────────────────────────────────────
# 1. Imports
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

import pandas as pd
import torch
from tqdm import tqdm

# cv2 sin hilos para no competir con DataLoader/torch
import cv2
cv2.setNumThreads(0)

import nltk
from radgraph import F1RadGraph

from config import config
from model_utils import cargar_procesador
from data_loader import MimicCXRDataset
from preprocessing import estandarizar_imagen
from chexbert_official import CheXbertLabeler, CHEXBERT_LABELS

# ── Reutilización de helpers ya validados ─────────────────────────────────────
# (carga del modelo Experimento 3 + métricas idénticas a la evaluación con
#  prompt diseñado). evaluate_exp3_asclepius re-exporta los helpers de métrica de
#  evaluate_phase3_puente, por lo que la lógica es EXACTAMENTE la misma.
from evaluate_exp3_asclepius import (
    cargar_modelo_exp3_asclepius,
    _bleu4,
    _rouge_l,
    _radgraph_f1,
    _chexpert_pred_cols,
    _chexpert_comparison_metrics,
    _truncar_para_bert,
    TEST_CSV_50PCT,
    IMAGES_DIR_50PCT,
    EXP3_STATE,
    _CHEX_COLS,
    _PRED_PREFIX,
    SEP,
)

# ── Prompt Chain-of-Thought y gen-params del experimento (compartidos) ───────
from exp1_cot_inference import (
    construir_prompt_cot,
    EXP1_GEN_PARAMS,
    SEVEN_CONDITIONS,
)

# ── Helpers de Clinical BERTScore (los mismos que evaluate_exp3_asclepius) ──
from reevaluar_sucio_vs_limpio import (
    bertscore_clinico_por_fila,
    cargar_bertscorer,
    CLINICAL_BERT_MODEL,
    CLINICAL_BERT_LAYERS,
)

# ── Rutas + tokenizer del Experimento 3 ───────────────────────────────────────
from trainer_utils_exp3_asclepius import (
    OUTPUT_DIR,
    ASCLEPIUS_R_DIR,
    ASCLEPIUS_VARIANT,
    preparar_procesador_exp3,
    _asclepius_disponible,
)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Constantes propias del experimento (paths exclusivos del CoT Exp3)
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR: Path = config.paths.base_dir

EXP1_DIR: Path = OUTPUT_DIR / "exp1_cot"

# IMPORTANTE: el nombre EXACTO de RAW_CSV debe coincidir con el path declarado
# en vllm_formatter.TASKS["nuevo_exp3asclepius"].in_csv. Si lo cambias, actualiza
# también vllm_formatter.py.
RAW_CSV:      Path = EXP1_DIR / "exp1_raw_inference_exp3_asclepius.csv"
GRANULAR_CSV: Path = EXP1_DIR / "resultados_granulares_Exp3_Asclepius_CoT.csv"
GLOBAL_CSV:   Path = EXP1_DIR / "metricas_globales_Exp3_Asclepius_CoT.csv"

CONFIG_NAME: str = "Exp3_Asclepius_CoT"

nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)


# ─────────────────────────────────────────────────────────────────────────────
# 4. FASE 1 — Inferencia CoT (GPU, batched, con resume)
# ─────────────────────────────────────────────────────────────────────────────

def fase_inferencia_cot_exp3_asclepius(
    model,
    processor,
    device,
    num_samples: Optional[int] = None,
    batch_size: int = 16,
    resume: bool = False,
) -> Path:
    """
    Genera reportes con el prompt CoT y los exporta a RAW_CSV (escritura
    incremental). Si resume=True y el CSV ya tiene filas, salta los dicom_id
    ya generados. Réplica de exp1_cot_inference_exp2_raddino con paths del Exp3.
    """
    logger.info("  Cargando Test Set (50%)...")
    test_dataset = MimicCXRDataset(
        csv_path             = TEST_CSV_50PCT,
        images_dir           = IMAGES_DIR_50PCT,
        processor            = processor,
        image_size           = config.data.image_size,
        clahe_clip_limit     = config.data.clahe_clip_limit,
        clahe_tile_grid_size = config.data.clahe_tile_grid_size,
        max_length           = config.data.max_length,
        padding              = config.data.padding,
        prompt               = config.inference.default_prompt,  # solo placeholder
        augment              = False,                            # determinista
    )

    n_total = len(test_dataset)
    n_eval  = n_total if num_samples is None else min(num_samples, n_total)
    logger.info(f"  Test Set: {n_total:,} muestras | A evaluar: {n_eval:,}")
    logger.info(f"  Prompt CoT ejemplo: {construir_prompt_cot('Frontal')!r}")
    logger.info(f"  GEN_PARAMS: {EXP1_GEN_PARAMS}")

    EXP1_DIR.mkdir(parents=True, exist_ok=True)

    COLS = ["dicom_id", "subject_id", "study_id",
            "reference_report", "generated_report"]

    # ── Reanudación ──────────────────────────────────────────────────────────
    procesados: set = set()
    if resume and RAW_CSV.exists():
        try:
            _prev = pd.read_csv(RAW_CSV, usecols=["dicom_id"])
            procesados = set(_prev["dicom_id"].astype(str).tolist())
        except Exception as exc:
            logger.warning(f"  [resume] no se pudo leer {RAW_CSV.name}: {exc}; "
                           "se empieza de cero.")
            procesados = set()

    if procesados:
        logger.info(f"  [RESUME] {len(procesados):,} reportes ya guardados → "
                    f"se reanuda en modo append "
                    f"(faltan ~{max(0, n_eval - len(procesados)):,}).")
    else:
        pd.DataFrame(columns=COLS).to_csv(RAW_CSV, index=False)

    batch_images:  List = []
    batch_prompts: List[str] = []
    batch_meta:    List[Dict] = []
    errores       = 0
    n_guardadas   = 0

    t0 = time.time()
    for idx in tqdm(range(n_eval), desc="  Inferencia CoT (Exp3 Asclepius)", unit="img"):
        row      = test_dataset.data.iloc[idx]
        dicom_id = str(row["dicom_id"])

        if procesados and dicom_id in procesados:
            continue

        study_id = row.get("study_id", "")
        subj_id  = row.get("subject_id", "")
        ref_text = str(row["report_text"]).strip()

        # Vista mapeada (mismo criterio que exp1_cot_inference)
        view_raw = row.get("ViewPosition", None)
        if view_raw is not None and not (
            isinstance(view_raw, float) and math.isnan(view_raw)
        ):
            vista = test_dataset._mapear_vista(str(view_raw))
        else:
            vista = "Unknown"

        prompt = construir_prompt_cot(vista)

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

                with torch.inference_mode():
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
        f"  Inferencia CoT OK: {n_guardadas:,} reportes | {errores} errores "
        f"| {dt/60:.1f} min"
    )
    logger.info(f"  → {RAW_CSV}")

    del test_dataset
    gc.collect()
    torch.cuda.empty_cache()
    return RAW_CSV


# ─────────────────────────────────────────────────────────────────────────────
# 5. FASE 2 — Evaluación granular completa (5 métricas)
# ─────────────────────────────────────────────────────────────────────────────

def _merge_gt_chexpert(df: pd.DataFrame, chexpert_csv: Path) -> pd.DataFrame:
    """Une el GT oficial (mimic-cxr-2.0.0-chexpert.csv) por study_id (sin 's')."""
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
    logger.info(f"  GT CheXpert tras merge: {n_match:,}/{len(merged):,} "
                "filas con etiquetas")
    return merged


def fase2_evaluacion_cot(
    raw_csv:        Path,
    chexpert_csv:   Optional[Path],
    chexbert_batch: int = 16,
    bert_batch:     int = 64,
    no_bertscore:   bool = False,
) -> pd.DataFrame:
    """
    Evaluación fila por fila SOBRE el RAW_CSV de inferencia CoT (idéntica a
    exp1_cot_inference_exp2_raddino.fase2_evaluacion_cot):
        1. CheXbert sobre generated_report → pred_chexpert_*
        2. Merge GT oficial (study_id)
        3. BLEU-4 / ROUGE-L / F1-RadGraph fila por fila
        4. CheXpert P/R/F1 + match por enfermedad
        5. Clinical BERTScore (Bio_ClinicalBERT) batched al final
    """
    logger.info(f"\n{SEP}")
    logger.info(f"  EVALUACIÓN GRANULAR — {CONFIG_NAME}")
    logger.info(f"{SEP}")

    df = pd.read_csv(raw_csv)
    df["generated_report"]  = df["generated_report"].fillna("").astype(str)
    df["reference_report"]  = df["reference_report"].fillna("").astype(str)
    logger.info(f"  Reportes a evaluar: {len(df):,}")

    # ── Merge GT oficial CheXpert ────────────────────────────────────────────
    has_chexpert_gt = False
    if chexpert_csv is not None and Path(chexpert_csv).exists():
        df = _merge_gt_chexpert(df, Path(chexpert_csv))
        has_chexpert_gt = any(c in df.columns for c in _CHEX_COLS)
    else:
        logger.warning("  CSV CheXpert oficial no disponible — solo se calculará "
                       "CheXbert sobre Pred (sin comparación GT vs Pred).")

    # ── Cargar modelos pesados (CheXbert + RadGraph) ─────────────────────────
    logger.info("  Cargando CheXbert oficial (Stanford AIMI)...")
    chexbert = CheXbertLabeler()  # autodetecta CUDA
    logger.info("  Cargando F1RadGraph (radgraph-xl, reward_level='all')...")
    f1rg = F1RadGraph(reward_level="all", model_type="radgraph-xl")

    # ── Bucle granular fila por fila ─────────────────────────────────────────
    rows_out: List[Dict] = []
    for idx, row in tqdm(df.iterrows(), total=len(df),
                         desc="  Métricas granulares", unit="rep"):
        ref = row["reference_report"]
        hyp = row["generated_report"]

        out: Dict = {
            "dicom_id":         row.get("dicom_id", ""),
            "subject_id":       row.get("subject_id", ""),
            "study_id":         row.get("study_id", ""),
            "reference_report": ref,
            "generated_report": hyp,
        }

        if has_chexpert_gt:
            for lbl in _CHEX_COLS:
                if lbl in df.columns:
                    out[lbl] = row.get(lbl, float("nan"))

        out["bleu_4"]  = _bleu4(ref, hyp) if (ref and hyp) else 0.0
        out["rouge_l"] = _rouge_l(ref, hyp) if (ref and hyp) else 0.0
        out["radgraph_f1"] = (
            _radgraph_f1(f1rg, ref, hyp) if (ref and hyp) else float("nan")
        )

        try:
            pred_chex = _chexpert_pred_cols(chexbert, hyp)
        except Exception as exc:
            logger.warning(f"  [row {idx}] CheXbert pred: {exc}")
            pred_chex = {
                f"{_PRED_PREFIX}{lbl.lower().replace(' ', '_')}": float("nan")
                for lbl in _CHEX_COLS
            }
        out.update(pred_chex)

        if has_chexpert_gt:
            try:
                cmp = _chexpert_comparison_metrics(row, pred_chex)
            except Exception as exc:
                logger.warning(f"  [row {idx}] CheXpert comparison: {exc}")
                cmp = {}
            out.update(cmp)

        rows_out.append(out)

    df_out = pd.DataFrame(rows_out)

    # ── Liberar CheXbert + RadGraph antes de cargar BERTScore ────────────────
    del chexbert, f1rg
    gc.collect()
    torch.cuda.empty_cache()

    # ── Clinical BERTScore (Bio_ClinicalBERT, batched al final) ──────────────
    if not no_bertscore:
        logger.info(f"\n  Cargando Bio_ClinicalBERT...")
        bert_scorer = cargar_bertscorer(
            CLINICAL_BERT_MODEL, CLINICAL_BERT_LAYERS, device=None
        )
        bert_tokenizer = getattr(bert_scorer, "_tokenizer", None)

        logger.info(f"  Calculando Clinical BERTScore sobre {len(df_out):,} reportes...")
        refs_full = df_out["reference_report"].astype(str).fillna(".").tolist()
        hyps_full = df_out["generated_report"].astype(str).fillna(".").tolist()
        if bert_tokenizer is not None:
            refs_t = _truncar_para_bert(refs_full, bert_tokenizer)
            hyps_t = _truncar_para_bert(hyps_full, bert_tokenizer)
        else:
            refs_t, hyps_t = refs_full, hyps_full
        bs = bertscore_clinico_por_fila(
            bert_scorer, refs_t, hyps_t,
            batch_size=bert_batch, etiqueta=f"[{CONFIG_NAME}]",
        )
        df_out["bertscore_p"]  = bs["p"]
        df_out["bertscore_r"]  = bs["r"]
        df_out["bertscore_f1"] = bs["f1"]
        logger.info(
            f"  ClinBERT → P={bs['p'].mean():.4f}  "
            f"R={bs['r'].mean():.4f}  F1={bs['f1'].mean():.4f}"
        )
        del bert_scorer
        gc.collect()
        torch.cuda.empty_cache()

    df_out.to_csv(GRANULAR_CSV, index=False)
    logger.info(f"\n  ✓ Granular: {GRANULAR_CSV}  "
                f"({len(df_out):,} filas, {len(df_out.columns)} cols)")
    return df_out


# ─────────────────────────────────────────────────────────────────────────────
# 6. Resumen global (medias agregadas)
# ─────────────────────────────────────────────────────────────────────────────

def calcular_resumen_global(df: pd.DataFrame) -> pd.DataFrame:
    """Promedios globales de TODAS las columnas numéricas del granular."""
    skip_cols = {"dicom_id", "subject_id", "study_id",
                 "reference_report", "generated_report"}
    numeric_cols = [
        c for c in df.columns
        if c not in skip_cols and pd.api.types.is_numeric_dtype(df[c])
    ]
    row = {"config": CONFIG_NAME, "n_samples": len(df)}
    for col in numeric_cols:
        row[col] = df[col].mean(skipna=True)
    df_summary = pd.DataFrame([row])
    df_summary.to_csv(GLOBAL_CSV, index=False)
    logger.info(f"  ✓ Resumen global: {GLOBAL_CSV}")
    return df_summary


# ─────────────────────────────────────────────────────────────────────────────
# 7. CLI / main
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=("Experimento 1 (CoT) sobre Experimento 3 (Asclepius-R): "
                     "inferencia con Chain-of-Thought + evaluación granular "
                     "(BLEU/ROUGE/RadGraph/CheXpert/Clinical BERTScore)."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--num-samples", type=int, default=None,
                   help="Limitar a N muestras del Test Set (None = todas).")
    p.add_argument("--batch-size", type=int, default=16,
                   help="Tamaño de lote de inferencia (beam=3, max_new=512).")
    p.add_argument("--chexbert-batch", type=int, default=16,
                   help="Tamaño de lote de CheXbert.")
    p.add_argument("--bert-batch", type=int, default=64,
                   help="Tamaño de lote de Bio_ClinicalBERT (BERTScore).")
    p.add_argument("--exp3-state", type=str, default=str(EXP3_STATE),
                   help="exp3_state.pt entrenado en el Experimento 3 "
                        "(solo params entrenables; se carga con strict=False).")
    p.add_argument("--chexpert-csv", type=str,
                   default=str(BASE_DIR / "mimic-cxr-2.0.0-chexpert.csv"),
                   help="CSV oficial de etiquetas CheXpert (.csv o .csv.gz).")
    p.add_argument("--skip-inference", action="store_true",
                   help="Reutilizar exp1_raw_inference_exp3_asclepius.csv y "
                        "solo (re)evaluar.")
    p.add_argument("--resume", action="store_true",
                   help="Reanudar la inferencia: conserva el CSV existente y "
                        "solo genera los reportes faltantes.")
    p.add_argument("--no-bertscore", action="store_true",
                   help="Saltar Clinical BERTScore (smoke tests rápidos).")
    p.add_argument("--skip-evaluation", action="store_true",
                   help="Solo realizar la inferencia (no evaluar). Útil cuando "
                        "se va a aplicar primero el formateo Llama 8b.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    print("\n" + "=" * 80)
    print("  EXPERIMENTO 1 — CHAIN-OF-THOUGHT  ·  EXPERIMENTO 3 (ViT F3 + Asclepius-R)")
    print("  Task 1: clasificación de 7 patologías  →  Task 2: generar reporte")
    print("=" * 80)
    print(f"  CONFIG_NAME    : {CONFIG_NAME}")
    print(f"  num_beams      : {EXP1_GEN_PARAMS['num_beams']}  "
          f"max_new_tokens : {EXP1_GEN_PARAMS['max_new_tokens']}")
    print(f"  Asclepius-R    : {ASCLEPIUS_R_DIR}  (variante {ASCLEPIUS_VARIANT})")
    print(f"  exp3_state.pt  : {args.exp3_state}")
    print(f"  RAW_CSV        : {RAW_CSV}")
    print(f"  GRANULAR_CSV   : {GRANULAR_CSV}")
    print(f"  GLOBAL_CSV     : {GLOBAL_CSV}")
    print(f"  GPU            : {'disponible ✓' if torch.cuda.is_available() else 'NO ✗'}")
    print("=" * 80 + "\n")

    if not torch.cuda.is_available():
        logger.warning("CUDA no disponible: la inferencia será MUY lenta en CPU.")

    chexpert_csv = Path(args.chexpert_csv) if args.chexpert_csv else None
    if chexpert_csv is not None and not chexpert_csv.exists():
        gz_alt = Path(str(chexpert_csv) + ".gz")
        if gz_alt.exists():
            chexpert_csv = gz_alt

    # ── Procesador BLIP-2 (tokenizer se intercambia a Asclepius antes de generar)
    logger.info("Cargando Blip2Processor...")
    processor = cargar_procesador(model_name=config.model.model_name)

    # ── FASE 1: Inferencia ────────────────────────────────────────────────────
    if args.skip_inference and RAW_CSV.exists():
        logger.info(f"--skip-inference: reutilizando {RAW_CSV}")
    else:
        if not _asclepius_disponible(ASCLEPIUS_R_DIR):
            logger.error(
                "Asclepius-R no disponible. Ejecuta "
                "`python run_experiment_3.py --print-accion-obligatoria` "
                "para las instrucciones de descarga (acceso credenciado)."
            )
            return 2
        # ⚠ Intercambiar tokenizer OPT → Asclepius/LLaMA ANTES de generar.
        processor = preparar_procesador_exp3(processor, ASCLEPIUS_R_DIR)

        model, device = cargar_modelo_exp3_asclepius(
            ruta_exp3_state = Path(args.exp3_state),
        )
        fase_inferencia_cot_exp3_asclepius(
            model, processor, device,
            num_samples = args.num_samples,
            batch_size  = args.batch_size,
            resume      = args.resume,
        )
        del model
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("  VRAM del modelo Experimento 3 liberada.\n")

    # ── FASE 2: Evaluación (puede saltarse si se va a formatear primero) ─────
    if args.skip_evaluation:
        logger.info("--skip-evaluation: NO se realizará la evaluación granular.")
        logger.info(f"  RAW_CSV listo: {RAW_CSV}")
        logger.info("  Próximo paso sugerido: "
                    "bash run_vllm_formatter_tmux.sh --tasks=nuevo_exp3asclepius")
        return 0

    df_granular = fase2_evaluacion_cot(
        RAW_CSV, chexpert_csv,
        chexbert_batch = args.chexbert_batch,
        bert_batch     = args.bert_batch,
        no_bertscore   = args.no_bertscore,
    )
    calcular_resumen_global(df_granular)

    print("\n" + "=" * 80)
    print("  EXP1 (CoT)  ·  EXPERIMENTO 3 (Asclepius-R)  ·  COMPLETADO")
    print("=" * 80)
    print(f"  Raw          : {RAW_CSV}")
    print(f"  Granular     : {GRANULAR_CSV}")
    print(f"  Global       : {GLOBAL_CSV}")
    print("\n  Siguiente   : bash run_vllm_formatter_tmux.sh --tasks=nuevo_exp3asclepius")
    print("=" * 80 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
