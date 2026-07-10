#!/usr/bin/env python3
"""
evaluate_final.py — Pipeline de Producción: Inferencia + Evaluación Granular.

FASE 1 — Inferencia Visual y Generación
    • Carga Config_A (best_model_Config_A) en bfloat16 via model_utils.py
    • Itera el Test Set con MimicCXRDataset (data_loader.py)
    • Genera reportes: num_beams=5, do_sample=False, rep_penalty=1.0,
      max_new_tokens=400, use_cache=True
    • Exporta predicciones_Config_A.csv  (dicom_id, study_id,
      reference_report, generated_report)
    • gc.collect() + torch.cuda.empty_cache() entre configs
    • Repite para Config_B

FASE 2 — Evaluación Granular Fila por Fila (Gold Standard)
    • CheXpert Ground Truth: carga mimic-cxr-2.0.0-chexpert.csv y hace
      merge con las predicciones por study_id  (NO se labela el reference)
    • CheXpert Predicted: CheXbert oficial Stanford AIMI sobre generated_report
    • BLEU-4 y ROUGE-L calculados sobre (reference_report, generated_report)
    • F1-RadGraph calculado sobre (reference_report, generated_report)
    • Guarda resultados_granulares_Config_X.csv con columnas métricas añadidas
    • Calcula promedios globales → metricas_globales_comparacion.csv

Uso:
    python evaluate_final.py [--output-dir DIR] [--num-samples N]
                             [--chexpert-csv PATH] [--skip-inference]
                             [--configs Config_A Config_B]
"""

# ─────────────
# 0. Patch de compatibilidad (antes de cualquier import de radgraph)
# ─────────────
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

# ─────────────
# 1. Imports
# ───────────────────
import argparse
import gc
import logging
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer as rouge_scorer_module
from radgraph import F1RadGraph

# ───────
# 2. Módulos del proyecto (REGLA CERO: arquitectura modular)
# ───────────────
from config import config
from model_utils import cargar_modelo_entrenado, cargar_procesador
from data_loader import MimicCXRDataset
from preprocessing import estandarizar_imagen

# CheXbert oficial Stanford AIMI
from chexbert_official import CheXbertLabeler, CHEXBERT_LABELS

# ───────
# 3. Logging
# ─────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─────────────────
# 4. Constantes
# ────────────────────
BASE_DIR = config.paths.base_dir

CONFIGS = [
    {
        "name":      "Config_A",
        "model_dir": BASE_DIR / "final_results" / "checkpoints" / "best_model_Config_A",
    },
    {
        "name":      "Config_B",
        "model_dir": BASE_DIR / "final_results" / "checkpoints" / "best_model_Config_B",
    },
]

# Parámetros de generación ESTRICTOS Y BLINDADOS
GEN_PARAMS = dict(
    num_beams            = 5,
    do_sample            = False,
    repetition_penalty   = 1.1,
    no_repeat_ngram_size = 3,
    max_new_tokens       = 400,
    use_cache            = True,
)

# Columnas CheXpert en el CSV oficial de MIMIC-CXR
_CHEX_COLS = CHEXBERT_LABELS  # mismo orden que el CSV oficial

# Prefijos para columnas en el CSV de salida
_GT_PREFIX  = "gt_chexpert_"    # ground truth del CSV oficial
_PRED_PREFIX = "pred_chexpert_" # predicción de CheXbert sobre generated_report

nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)

SEP = "-" * 65


# ──────────
# 5. Helpers de métricas (BLEU-4, ROUGE-L, RadGraph)
# ──────────────────────────

def _bleu4(reference: str, hypothesis: str) -> float:
    ref_tok  = nltk.word_tokenize(reference.lower())
    hyp_tok  = nltk.word_tokenize(hypothesis.lower())
    return sentence_bleu(
        [ref_tok], hyp_tok,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=SmoothingFunction().method1,
    )


_rouge_scorer = rouge_scorer_module.RougeScorer(["rougeL"], use_stemmer=True)


def _rouge_l(reference: str, hypothesis: str) -> float:
    return _rouge_scorer.score(reference, hypothesis)["rougeL"].fmeasure


def _radgraph_f1(f1rg: F1RadGraph, reference: str, hypothesis: str) -> float:
    try:
        mean_reward, _, _, _ = f1rg(refs=[reference], hyps=[hypothesis])
        if isinstance(mean_reward, (tuple, list)):
            return float(mean_reward[2])   # RG_ERbar
        return float(mean_reward)
    except Exception as exc:
        logger.warning(f"      RadGraph error: {exc}")
        return float("nan")


# ─────────────────────────────────────────────────
# 6. Helper CheXpert — comparación GT (CSV oficial) vs Pred (CheXbert)
# ───────────────────────────────────────────────

def _chexpert_pred_cols(labeler: CheXbertLabeler, report: str) -> Dict[str, float]:
    """Ejecuta CheXbert sobre el reporte generado y devuelve dict con prefijo pred_."""
    raw = labeler.label(report)
    return {f"{_PRED_PREFIX}{lbl.lower().replace(' ', '_')}": v
            for lbl, v in raw.items()}


def _chexpert_comparison_metrics(
    gt_row: pd.Series,
    pred_dict: Dict[str, float],
) -> Dict[str, float]:
    """
    Compara el vector GT (del CSV oficial) con el vector Predicted (CheXbert)
    clase por clase y calcula micro Precision, Recall y F1.

    Tratamiento de valores:
        1.0  → positivo
        0.0  → negativo
       -1.0  → se trata como positivo (incierto = mencionado)
        NaN  → se excluye del cálculo (no mencionado en GT o Pred)

    Returns:
        Diccionario con claves:
            chexpert_precision, chexpert_recall, chexpert_f1
            + per-class agreement (chexpert_match_{label})
    """
    tp = fp = fn = 0
    metrics: Dict[str, float] = {}

    for lbl in _CHEX_COLS:
        col_gt   = lbl                                             # columna del CSV oficial
        col_pred = f"{_PRED_PREFIX}{lbl.lower().replace(' ', '_')}"

        # Obtener valores
        gt_val   = gt_row.get(col_gt,   float("nan"))
        pred_val = pred_dict.get(col_pred, float("nan"))

        # Convertir a binario (mencionado = positivo/incierto; NaN = skip)
        def _to_bin(v):
            if isinstance(v, float) and math.isnan(v):
                return None
            return 1 if float(v) != 0.0 else 0

        gt_bin   = _to_bin(gt_val)
        pred_bin = _to_bin(pred_val)

        # Si ambos son NaN → skip
        if gt_bin is None and pred_bin is None:
            metrics[f"chexpert_match_{lbl.lower().replace(' ', '_')}"] = float("nan")
            continue

        # Imputar NaN como 0 (no mencionado = negativo)
        if gt_bin   is None: gt_bin   = 0
        if pred_bin is None: pred_bin = 0

        match = 1.0 if gt_bin == pred_bin else 0.0
        metrics[f"chexpert_match_{lbl.lower().replace(' ', '_')}"] = match

        if gt_bin == 1 and pred_bin == 1:
            tp += 1
        elif gt_bin == 0 and pred_bin == 1:
            fp += 1
        elif gt_bin == 1 and pred_bin == 0:
            fn += 1

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


# ───────────────
# 7. FASE 1 — Inferencia Visual y Generación
# ───────────────────────────────────────

def fase1_inferencia(
    config_dict: Dict,
    output_dir: Path,
    num_samples: Optional[int] = None,
    wrap_full_model: bool = False,
) -> Path:
    """
    Genera reportes para el Test Set completo y los exporta a predicciones_X.csv.

    Columnas del CSV de salida:
        dicom_id, study_id, reference_report, generated_report
    """
    cfg_name  = config_dict["name"]
    model_dir = Path(config_dict["model_dir"])

    logger.info(f"\n{'='*65}")
    logger.info(f"  FASE 1 — Inferencia: {cfg_name}")
    logger.info(f"  Checkpoint : {model_dir}")
    logger.info(f"{'='*65}")

    if not model_dir.exists():
        raise FileNotFoundError(
            f"Modelo no encontrado: {model_dir}\n"
            f"Asegurate de haber completado el entrenamiento Gran Final."
        )

    # Cargar procesador y modelo
    logger.info("  Cargando procesador...")
    processor = cargar_procesador()

    # // MODIFICADO PARA INCRUSTAR LORA CORRECTAMENTE
    logger.info("  Cargando modelo base FRESCO en bfloat16...")
    from transformers import Blip2ForConditionalGeneration
    from peft import PeftModel

    base_model = Blip2ForConditionalGeneration.from_pretrained(
        config.model.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # Auto-detectar el nivel de wrapping según las claves del adapter:
    #   • claves con '.language_model.'  -> LoRA entrenada sobre el MODELO COMPLETO (Blip2)
    #   • en caso contrario              -> LoRA sobre el language_model (submodulo OPT)
    def _needs_full_wrap(mdir: Path) -> bool:
        st = mdir / "adapter_model.safetensors"
        if not st.exists():
            return False
        try:
            from safetensors import safe_open
            with safe_open(str(st), framework="pt") as f:
                return any(".language_model." in k for k in f.keys())
        except Exception:
            return False

    if wrap_full_model or _needs_full_wrap(model_dir):
        logger.info("  Incrustando LoRA sobre el MODELO COMPLETO (Blip2) y fusionando (merge_and_unload)...")
        peft_model = PeftModel.from_pretrained(base_model, str(model_dir))
        model = peft_model.merge_and_unload()
    else:
        logger.info("  Incrustando adaptadores LoRA ESTRICTAMENTE en el language_model...")
        # LA LLAVE MÁGICA: Conectar LoRA exactamente en la misma sub-red donde se entrenó
        base_model.language_model = PeftModel.from_pretrained(
            base_model.language_model,
            str(model_dir)
        )
        model = base_model
    # // FIN MODIFICACIÓN
    model.eval()
    device = next(model.parameters()).device
    logger.info(f"  Modelo en: {device} | dtype: {next(model.parameters()).dtype}")

    # Construir Test Dataset
    logger.info("  Cargando Test Set...")
    test_dataset = MimicCXRDataset(
        csv_path            = config.paths.test_csv,
        images_dir          = config.paths.images_dir,
        processor           = processor,
        image_size          = config.data.image_size,
        clahe_clip_limit    = config.data.clahe_clip_limit,
        clahe_tile_grid_size= config.data.clahe_tile_grid_size,
        max_length          = config.data.max_length,
        padding             = config.data.padding,
    )

    n_total = len(test_dataset)
    n_eval  = n_total if num_samples is None else min(num_samples, n_total)
    logger.info(f"  Muestras Test Set: {n_total} | A evaluar: {n_eval}")

    COLS = ["dicom_id", "study_id", "reference_report", "generated_report"]
    errores = 0
    n_guardadas = 0

    preds_path = output_dir / f"predicciones_{cfg_name}.csv"
    # Crear CSV con encabezados (sobreescribe cualquier archivo previo incompleto)
    pd.DataFrame(columns=COLS).to_csv(preds_path, index=False)
    logger.info(f"  CSV incremental iniciado: {preds_path}")

    # // MODIFICADO PARA BATCHING — acumuladores de lote
    batch_size     = 4
    batch_images   = []
    batch_prompts  = []
    batch_metadata = []   # lista de dicts con dicom_id, study_id, ref_text
    # // FIN DECLARACIÓN ACUMULADORES

    for idx in tqdm(range(n_eval), desc=f"  Inferencia {cfg_name}", unit="img"):
        row      = test_dataset.data.iloc[idx]
        dicom_id = str(row["dicom_id"])
        study_id = str(row.get("study_id", ""))
        ref_text = str(row["report_text"]).strip()

        # Prompt con vista real
        view_raw = row.get("ViewPosition", None)
        if view_raw is not None and not (isinstance(view_raw, float) and math.isnan(view_raw)):
            vista = test_dataset._mapear_vista(str(view_raw))
        else:
            vista = "Unknown"
        prompt = f"[Context: {vista} view] {config.inference.default_prompt}"

        # Buscar y preprocesar imagen
        imagen_path = test_dataset._buscar_imagen(dicom_id)
        if imagen_path is None:
            logger.warning(f"  Imagen no encontrada: {dicom_id}")
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

        # // MODIFICADO PARA BATCHING — acumular en lugar de inferir de a 1
        batch_images.append(imagen_pil)
        batch_prompts.append(prompt)
        batch_metadata.append({
            "dicom_id": dicom_id,
            "study_id": study_id,
            "ref_text": ref_text,
        })

        # Vaciar el lote cuando está lleno O es la última iteración válida
        if len(batch_images) == batch_size or idx == n_eval - 1:
            try:
                inputs = processor(
                    images   = batch_images,
                    text     = batch_prompts,
                    return_tensors = "pt",
                    padding  = True,
                ).to(device)

                with torch.no_grad():
                    generated_ids = model.generate(**inputs, **GEN_PARAMS)

                decoded_reports = processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )

                # Iterar sobre resultados y metadatos juntos
                batch_rows = []
                for meta, generated_report in zip(batch_metadata, decoded_reports):
                    batch_rows.append({
                        "dicom_id":         meta["dicom_id"],
                        "study_id":         meta["study_id"],
                        "reference_report": meta["ref_text"],
                        "generated_report": generated_report.strip(),
                    })
                    n_guardadas += 1

                pd.DataFrame(batch_rows, columns=COLS).to_csv(
                    preds_path, mode="a", header=False, index=False
                )

            except Exception as exc:
                logger.error(f"  Error inferencia lote (idx={idx}): {exc}")
                errores += len(batch_images)

            # Limpiar acumuladores para el siguiente lote
            batch_images.clear()
            batch_prompts.clear()
            batch_metadata.clear()
        # // FIN MODIFICACIÓN BATCHING

    logger.info(f"  Inferencia OK: {n_guardadas} reportes guardados | {errores} errores")
    logger.info(f"  Predicciones: {preds_path}")

    # Liberar VRAM
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("  VRAM liberada")

    return preds_path


# ────────
# 8. FASE 2 — Evaluación Granular Fila por Fila (Gold Standard)
# ─────────────

def fase2_evaluacion(
    preds_path:      Path,
    cfg_name:        str,
    output_dir:      Path,
    f1rg:            F1RadGraph,
    chexbert:        CheXbertLabeler,
    chexpert_csv:    Optional[Path],
) -> pd.DataFrame:
    """
    Evaluación granular fila por fila sobre el CSV de predicciones.

    Estrategia CheXpert:
        Ground Truth → mimic-cxr-2.0.0-chexpert.csv  (merge por study_id)
        Predicted    → CheXbert oficial sobre generated_report

    Args:
        preds_path   : CSV generado por fase1_inferencia().
        cfg_name     : Nombre de la config (ej. 'Config_A').
        output_dir   : Directorio de salida.
        f1rg         : Instancia de F1RadGraph ya cargada.
        chexbert     : Instancia de CheXbertLabeler ya cargada.
        chexpert_csv : Ruta al CSV oficial mimic-cxr-2.0.0-chexpert.csv.
                       Si es None, se omite la evaluación CheXpert.
    """
    logger.info(f"\n{'='*65}")
    logger.info(f"  FASE 2 — Evaluacion Granular: {cfg_name}")
    logger.info(f"{'='*65}")

    df = pd.read_csv(preds_path)
    logger.info(f"  Filas: {len(df)}")

    # ── Cargar y hacer merge con CSV oficial de CheXpert etiquetas ─
    # pd.read_csv soporta .gz nativamente; buscamos también la variante .csv.gz
    # si el usuario pasó la ruta sin extensión comprimida.
    has_chexpert_gt = False
    if chexpert_csv is not None:
        chexpert_csv = Path(chexpert_csv)
        # Auto-fallback: si no existe la ruta exacta, probar la variante .gz
        if not chexpert_csv.exists():
            gz_alt = Path(str(chexpert_csv) + ".gz") if not str(chexpert_csv).endswith(".gz") else Path(str(chexpert_csv)[:-3])
            if gz_alt.exists():
                logger.info(f"  Usando variante: {gz_alt}")
                chexpert_csv = gz_alt
    if chexpert_csv is not None and Path(chexpert_csv).exists():
        logger.info(f"  Cargando CSV oficial CheXpert: {chexpert_csv}")
        df_chex = pd.read_csv(chexpert_csv, compression="infer")

        # Normalizar study_id (puede ser 's50691028' o 50691028)
        def _norm_sid(s):
            s = str(s).strip()
            return s.lstrip("s") if s.startswith("s") else s

        df["_sid"]       = df["study_id"].astype(str).apply(_norm_sid)
        df_chex["_sid"]  = df_chex["study_id"].astype(str).apply(_norm_sid)

        # Renombrar columnas CheXpert en df_chex para evitar colisiones
        rename_map = {lbl: lbl for lbl in _CHEX_COLS if lbl in df_chex.columns}
        df = df.merge(
            df_chex[["_sid"] + list(rename_map.keys())],
            on="_sid", how="left"
        )
        df.drop(columns=["_sid"], inplace=True)

        has_chexpert_gt = True
        n_matched = df[_CHEX_COLS[0]].notna().sum() if _CHEX_COLS[0] in df.columns else 0
        logger.info(f"  Filas con GT CheXpert tras merge: {n_matched}/{len(df)}")
    else:
        if chexpert_csv is None:
            logger.warning("  --chexpert-csv no proporcionado: se omite GT CheXpert.")
        else:
            logger.warning(f"  CSV no encontrado: {chexpert_csv} — se omite GT CheXpert.")

    # ── Bucle de evaluación fila por fila ─────────────────────────────────────
    rows_out = []

    for idx, row in tqdm(df.iterrows(), total=len(df),
                         desc=f"  Metricas {cfg_name}", unit="row"):
        ref  = str(row["reference_report"]).strip() or "no findings"
        hyp  = str(row["generated_report"]).strip()  or "no findings"

        out: Dict = {
            "dicom_id":         row.get("dicom_id", ""),
            "study_id":         row.get("study_id", ""),
            "reference_report": ref,
            "generated_report": hyp,
        }

        # BLEU-4
        try:
            out["bleu_4"] = _bleu4(ref, hyp)
        except Exception as exc:
            logger.warning(f"  [row {idx}] BLEU-4: {exc}")
            out["bleu_4"] = float("nan")

        # ROUGE-L
        try:
            out["rouge_l"] = _rouge_l(ref, hyp)
        except Exception as exc:
            logger.warning(f"  [row {idx}] ROUGE-L: {exc}")
            out["rouge_l"] = float("nan")

        # F1-RadGraph
        out["radgraph_f1"] = _radgraph_f1(f1rg, ref, hyp)

        # CheXpert — Predicted Vector (CheXbert sobre generated_report)
        try:
            pred_chex = _chexpert_pred_cols(chexbert, hyp)
        except Exception as exc:
            logger.warning(f"  [row {idx}] CheXbert pred: {exc}")
            pred_chex = {
                f"{_PRED_PREFIX}{lbl.lower().replace(' ', '_')}": float("nan")
                for lbl in _CHEX_COLS
            }
        out.update(pred_chex)

        # CheXpert — Comparación GT (CSV oficial) vs Pred
        if has_chexpert_gt:
            try:
                cmp = _chexpert_comparison_metrics(row, pred_chex)
            except Exception as exc:
                logger.warning(f"  [row {idx}] CheXpert comparison: {exc}")
                cmp = {}
            out.update(cmp)

        rows_out.append(out)

    df_out = pd.DataFrame(rows_out)

    # Guardar CSV granular
    granular_path = output_dir / f"resultados_granulares_{cfg_name}.csv"
    df_out.to_csv(granular_path, index=False)
    logger.info(f"  CSV granular: {granular_path}  ({len(df_out.columns)} cols)")

    return df_out


# ──────
# 9. Resumen global
# ─────

def calcular_resumen_global(
    dfs:        Dict[str, pd.DataFrame],
    output_dir: Path,
) -> pd.DataFrame:
    """Calcula promedios globales y guarda metricas_globales_comparacion.csv."""

    # Columnas a promediar: todas las numéricas menos dicom_id/study_id/textos
    skip_cols = {"dicom_id", "study_id", "reference_report", "generated_report"}

    rows = []
    for cfg_name, df in dfs.items():
        numeric_cols = [c for c in df.columns
                        if c not in skip_cols and pd.api.types.is_numeric_dtype(df[c])]
        row = {"config": cfg_name, "n_samples": len(df)}
        for col in numeric_cols:
            row[col] = df[col].mean(skipna=True)
        rows.append(row)

    df_summary = pd.DataFrame(rows)
    summary_path = output_dir / "metricas_globales_comparacion.csv"
    df_summary.to_csv(summary_path, index=False)
    logger.info(f"  Resumen global: {summary_path}")
    return df_summary


# ───────────────────────────────────────
# 10. CLI
# ────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline final de evaluacion BLIP-2 (Config_A y Config_B).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=str(BASE_DIR / "evaluation_results"),
        help="Directorio donde guardar todos los CSV de salida.",
    )
    parser.add_argument(
        "--chexpert-csv", type=str,
        default=str(BASE_DIR / "mimic-cxr-2.0.0-chexpert.csv.gz"),
        help="Ruta al CSV oficial de etiquetas CheXpert de MIMIC-CXR (.csv o .csv.gz).",
    )
    parser.add_argument(
        "--num-samples", type=int, default=None,
        help="Limitar evaluacion a N muestras del Test Set.",
    )
    parser.add_argument(
        "--skip-inference", action="store_true",
        help="Saltar Fase 1 y reutilizar predicciones_X.csv existentes.",
    )
    parser.add_argument(
        "--configs", nargs="+",
        choices=["Config_A", "Config_B"], default=["Config_A", "Config_B"],
        help="Configs a evaluar.",
    )
    return parser.parse_args()


# ────────────────────────────────────────────
# 11. Main
# ──

def main() -> int:
    args = _parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Directorio de salida: {output_dir}")

    chexpert_csv_path = Path(args.chexpert_csv) if args.chexpert_csv else None

    configs_a_evaluar = [c for c in CONFIGS if c["name"] in args.configs]

    # Pre-cargar modelos de evaluacion UNA sola vez (costoso en RAM)
    logger.info("\nCargando F1RadGraph (reward_level='all', model_type='radgraph-xl')...")
    f1rg = F1RadGraph(reward_level="all", model_type="radgraph-xl")
    logger.info("  F1RadGraph listo")

    logger.info("Cargando CheXbert oficial (Stanford AIMI — chexbert.pth)...")
    chexbert = CheXbertLabeler()   # autodetecta GPU/CPU
    logger.info("  CheXbert listo\n")

    granular_dfs: Dict[str, pd.DataFrame] = {}

    for cfg_dict in configs_a_evaluar:
        cfg_name   = cfg_dict["name"]
        preds_path = output_dir / f"predicciones_{cfg_name}.csv"

        # FASE 1
        if args.skip_inference and preds_path.exists():
            logger.info(f"[Fase 1] --skip-inference: usando {preds_path}")
        else:
            preds_path = fase1_inferencia(
                config_dict = cfg_dict,
                output_dir  = output_dir,
                num_samples = args.num_samples,
            )

        # FASE 2
        df_gran = fase2_evaluacion(
            preds_path   = preds_path,
            cfg_name     = cfg_name,
            output_dir   = output_dir,
            f1rg         = f1rg,
            chexbert     = chexbert,
            chexpert_csv = chexpert_csv_path,
        )
        granular_dfs[cfg_name] = df_gran

    # Resumen global
    if granular_dfs:
        logger.info(f"\n{'='*65}")
        logger.info("  Calculando resumen global de metricas...")
        df_summary = calcular_resumen_global(granular_dfs, output_dir)

        # Mostrar columnas clave
        show_cols = ["config", "n_samples", "bleu_4", "rouge_l",
                     "radgraph_f1", "chexpert_precision", "chexpert_recall", "chexpert_f1"]
        show_cols = [c for c in show_cols if c in df_summary.columns]
        print(f"\n{'='*65}")
        print("  METRICAS GLOBALES")
        print(f"{'='*65}")
        print(df_summary[show_cols].to_string(index=False))
        print(f"{'='*65}\n")

    logger.info("Pipeline de evaluacion finalizado\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
