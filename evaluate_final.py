#!/usr/bin/env python3
"""
evaluate_final.py — Pipeline de Producción: Inferencia + Evaluación Granular.

Estructura de dos fases secuenciales:

    FASE 1 — Inferencia Visual y Generación
        • Carga Config_A (best_model_Config_A) en bfloat16
        • Itera el Test Set con MimicCXRDataset (data_loader.py)
        • Genera reportes con beam search estricto
        • Exporta predicciones_A.csv  (dicom_id, reference_report, generated_report)
        • Libera VRAM con gc.collect() + torch.cuda.empty_cache()
        • Repite para Config_B → predicciones_B.csv

    FASE 2 — Evaluación Granular Fila por Fila
        • Abre predicciones_A.csv y predicciones_B.csv
        • Calcula BLEU-4, ROUGE-L, RadGraph F1 y vector CheXpert (14 cols) por fila
        • Guarda resultados_granulares_Config_A.csv / resultados_granulares_Config_B.csv
        • Calcula promedios globales → metricas_globales_comparacion.csv

Uso:
    python evaluate_final.py [--output-dir DIRECTORIO] [--num-samples N]

Rutas por defecto de los modelos:
    final_results/checkpoints/best_model_Config_A
    final_results/checkpoints/best_model_Config_B
"""

# ──────────────────────────────────────────────────────────────────────────────
# 0. Patch de compatibilidad para radgraph (debe ir ANTES de cualquier import)
# ──────────────────────────────────────────────────────────────────────────────
from transformers import PreTrainedTokenizerBase

if not hasattr(PreTrainedTokenizerBase, "encode_plus"):
    def _encode_plus_compat(self, text, text_pair=None, add_special_tokens=True,
                            padding=False, truncation=False, max_length=None,
                            stride=0, return_tensors=None, **kwargs):
        return self.__call__(
            text, text_pair=text_pair,
            add_special_tokens=add_special_tokens,
            padding=padding, truncation=truncation,
            max_length=max_length, stride=stride,
            return_tensors=return_tensors, **kwargs
        )
    PreTrainedTokenizerBase.encode_plus = _encode_plus_compat

if not hasattr(PreTrainedTokenizerBase, "build_inputs_with_special_tokens"):
    def _build_inputs(self, token_ids_0, token_ids_1=None):
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        return ([self.cls_token_id] + token_ids_0 + [self.sep_token_id]
                + token_ids_1 + [self.sep_token_id])
    PreTrainedTokenizerBase.build_inputs_with_special_tokens = _build_inputs

# ──────────────────────────────────────────────────────────────────────────────
# 1. Imports estándar
# ──────────────────────────────────────────────────────────────────────────────
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

# NLTK
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# ROUGE
from rouge_score import rouge_scorer as rouge_scorer_module

# RadGraph
from radgraph import F1RadGraph

# ──────────────────────────────────────────────────────────────────────────────
# 2. Módulos del proyecto (arquitectura modular — REGLA CERO)
# ──────────────────────────────────────────────────────────────────────────────
from config import config
from model_utils import cargar_modelo_entrenado, cargar_procesador
from data_loader import MimicCXRDataset
from preprocessing import estandarizar_imagen
from chexpert_labeler_lite import CheXpertLabelerLite

# ──────────────────────────────────────────────────────────────────────────────
# 3. Configuración de logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# 4. Constantes del pipeline de evaluación
# ──────────────────────────────────────────────────────────────────────────────
CONFIGS = [
    {
        "name":      "Config_A",
        "model_dir": config.paths.base_dir / "final_results" / "checkpoints" / "best_model_Config_A",
    },
    {
        "name":      "Config_B",
        "model_dir": config.paths.base_dir / "final_results" / "checkpoints" / "best_model_Config_B",
    },
]

# Parámetros de generación — ESTRICTOS (no modificar sin justificación)
GEN_PARAMS = dict(
    num_beams         = 5,
    do_sample         = False,
    repetition_penalty= 1.0,
    max_new_tokens    = 400,
    use_cache         = True,
)

# NLTK data
nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)

SEP = "─" * 65


# ──────────────────────────────────────────────────────────────────────────────
# 5. Helpers de métricas
# ──────────────────────────────────────────────────────────────────────────────

def _bleu4(reference: str, hypothesis: str) -> float:
    """BLEU-4 con smoothing method 1 (NLTK)."""
    ref_tok  = nltk.word_tokenize(reference.lower())
    hyp_tok  = nltk.word_tokenize(hypothesis.lower())
    smoother = SmoothingFunction().method1
    return sentence_bleu(
        [ref_tok], hyp_tok,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smoother
    )


_rouge_scorer = rouge_scorer_module.RougeScorer(["rougeL"], use_stemmer=True)

def _rouge_l(reference: str, hypothesis: str) -> float:
    """ROUGE-L F1 (rouge_score)."""
    scores = _rouge_scorer.score(reference, hypothesis)
    return scores["rougeL"].fmeasure


def _radgraph_f1(f1rg_model: F1RadGraph, reference: str, hypothesis: str) -> float:
    """
    F1-RadGraph: retorna el valor RG_ER̄ (reward_level='all' → índice 2).
    En caso de fallo retorna NaN.
    """
    try:
        mean_reward, _, _, _ = f1rg_model(refs=[reference], hyps=[hypothesis])
        if isinstance(mean_reward, (tuple, list)):
            return float(mean_reward[2])   # RG_ER̄
        return float(mean_reward)
    except Exception as exc:
        logger.warning(f"      RadGraph falló: {exc}")
        return float("nan")


_chexpert = CheXpertLabelerLite()

def _chexpert_cols(report: str) -> Dict[str, float]:
    """
    Etiqueta el reporte con las 14 categorías CheXpert.
    Prefija las columnas con 'chexpert_' para el CSV.
    """
    raw = _chexpert.label(report)
    return {f"chexpert_{k.lower().replace(' ', '_')}": v for k, v in raw.items()}


# ──────────────────────────────────────────────────────────────────────────────
# 6. FASE 1 — Inferencia Visual y Generación
# ──────────────────────────────────────────────────────────────────────────────

def fase1_inferencia(
    config_dict: Dict,
    output_dir: Path,
    num_samples: Optional[int] = None,
) -> Path:
    """
    Genera reportes para todo el Test Set y los guarda en predicciones_X.csv.

    Args:
        config_dict : Diccionario con 'name' y 'model_dir'.
        output_dir  : Directorio donde guardar el CSV de predicciones.
        num_samples : Si se indica, limita el número de muestras evaluadas.

    Returns:
        Ruta al CSV generado (predicciones_X.csv).
    """
    cfg_name  = config_dict["name"]
    model_dir = Path(config_dict["model_dir"])

    logger.info(f"\n{'═'*65}")
    logger.info(f"  FASE 1 — Inferencia: {cfg_name}")
    logger.info(f"  Checkpoint : {model_dir}")
    logger.info(f"{'═'*65}")

    if not model_dir.exists():
        logger.error(f"  ❌ No se encontró el directorio del modelo: {model_dir}")
        logger.error("     Asegúrate de haber completado el entrenamiento Gran Final.")
        raise FileNotFoundError(f"Modelo no encontrado: {model_dir}")

    # ── Cargar procesador y modelo ────────────────────────────────────────────
    logger.info("  Cargando procesador...")
    processor = cargar_procesador()

    logger.info("  Cargando modelo con adaptadores LoRA en bfloat16...")
    model, _ = cargar_modelo_entrenado(
        model_dir=model_dir,
        model_name=config.model.model_name,
        use_quantization=config.model.load_in_8bit,
    )
    model.eval()
    device = next(model.parameters()).device
    logger.info(f"  Modelo en: {device} | dtype: {next(model.parameters()).dtype}")

    # ── Construir Test Dataset (solo para metadata + búsqueda de imágenes) ───
    logger.info("  Cargando Test Set...")
    test_dataset = MimicCXRDataset(
        csv_path   = config.paths.test_csv,
        images_dir = config.paths.images_dir,
        processor  = processor,
        image_size = config.data.image_size,
        clahe_clip_limit    = config.data.clahe_clip_limit,
        clahe_tile_grid_size= config.data.clahe_tile_grid_size,
        max_length = config.data.max_length,
        padding    = config.data.padding,
    )

    total = len(test_dataset)
    n_eval = total if num_samples is None else min(num_samples, total)
    logger.info(f"  Muestras en Test Set: {total} | A evaluar: {n_eval}")

    # ── Prompt fijo para inferencia (sin etiqueta de reporte) ────────────────
    # Formato igual al usado en entrenamiento (data_loader._mapear_vista):
    # "[Context: {vista} view] {default_prompt}"
    # Para inferencia usamos el prompt sin vista (Unknown) porque no procesamos
    # ViewPosition aquí — el modelo igualmente genera el reporte completo.
    DEFAULT_PROMPT = (
        f"[Context: Unknown view] {config.inference.default_prompt}"
    )

    # ── Bucle de inferencia ───────────────────────────────────────────────────
    records = []
    errores = 0

    for idx in tqdm(range(n_eval), desc=f"  Inferencia {cfg_name}", unit="img"):
        row      = test_dataset.data.iloc[idx]
        dicom_id = str(row["dicom_id"])
        ref_text = str(row["report_text"]).strip()

        # Construir prompt con vista real si está disponible
        view_position = row.get("ViewPosition", None)
        if view_position is not None and not (isinstance(view_position, float) and math.isnan(view_position)):
            vista = test_dataset._mapear_vista(str(view_position))
        else:
            vista = "Unknown"
        prompt = f"[Context: {vista} view] {config.inference.default_prompt}"

        # Buscar y preprocesar imagen
        imagen_path = test_dataset._buscar_imagen(dicom_id)
        if imagen_path is None:
            logger.warning(f"  ⚠  Imagen no encontrada para {dicom_id} — saltando")
            errores += 1
            continue

        imagen_pil = estandarizar_imagen(
            imagen_path,
            target_size          = config.data.image_size,
            clahe_clip_limit     = config.data.clahe_clip_limit,
            clahe_tile_grid_size = config.data.clahe_tile_grid_size,
        )
        if imagen_pil is None:
            logger.warning(f"  ⚠  Error preprocesando {imagen_path.name} — saltando")
            errores += 1
            continue

        try:
            inputs = processor(
                images=imagen_pil,
                text=prompt,
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                generated_ids = model.generate(**inputs, **GEN_PARAMS)

            generated_report = processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0].strip()

        except Exception as exc:
            logger.error(f"  ❌ Error en inferencia para {dicom_id}: {exc}")
            errores += 1
            continue

        records.append({
            "dicom_id":         dicom_id,
            "reference_report": ref_text,
            "generated_report": generated_report,
        })

    logger.info(f"  ✅ Inferencia completada: {len(records)} reportes | {errores} errores")

    # ── Guardar CSV de predicciones ───────────────────────────────────────────
    preds_path = output_dir / f"predicciones_{cfg_name}.csv"
    df_preds = pd.DataFrame(records, columns=["dicom_id", "reference_report", "generated_report"])
    df_preds.to_csv(preds_path, index=False)
    logger.info(f"  💾 Predicciones guardadas en: {preds_path}")

    # ── Liberar VRAM ──────────────────────────────────────────────────────────
    del model, processor, inputs
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("  🧹 VRAM liberada (gc.collect + cuda.empty_cache)")

    return preds_path


# ──────────────────────────────────────────────────────────────────────────────
# 7. FASE 2 — Evaluación Granular Fila por Fila
# ──────────────────────────────────────────────────────────────────────────────

def fase2_evaluacion(
    preds_path: Path,
    cfg_name: str,
    output_dir: Path,
    f1rg_model: F1RadGraph,
) -> pd.DataFrame:
    """
    Calcula las 4 métricas fila por fila sobre el CSV de predicciones.

    Args:
        preds_path  : Ruta al CSV generado por fase1_inferencia().
        cfg_name    : Nombre de la config (ej. 'Config_A').
        output_dir  : Directorio donde guardar los resultados granulares.
        f1rg_model  : Instancia de F1RadGraph ya inicializada.

    Returns:
        DataFrame con columnas métricas añadidas.
    """
    logger.info(f"\n{'═'*65}")
    logger.info(f"  FASE 2 — Evaluación Granular: {cfg_name}")
    logger.info(f"  CSV entrada: {preds_path}")
    logger.info(f"{'═'*65}")

    df = pd.read_csv(preds_path)
    logger.info(f"  Filas a evaluar: {len(df)}")

    # Columnas de resultado
    bleu4_scores   : List[float] = []
    rougel_scores  : List[float] = []
    radgraph_scores: List[float] = []
    chexpert_rows  : List[Dict]  = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"  Métricas {cfg_name}", unit="row"):
        ref  = str(row["reference_report"]).strip()
        hyp  = str(row["generated_report"]).strip()

        # Fallbacks ante textos vacíos
        if not ref:
            ref = "no findings"
        if not hyp:
            hyp = "no findings"

        # ── BLEU-4 ────────────────────────────────────────────────────────────
        try:
            b4 = _bleu4(ref, hyp)
        except Exception as exc:
            logger.warning(f"  [row {idx}] BLEU-4 falló: {exc}")
            b4 = float("nan")
        bleu4_scores.append(b4)

        # ── ROUGE-L ───────────────────────────────────────────────────────────
        try:
            rl = _rouge_l(ref, hyp)
        except Exception as exc:
            logger.warning(f"  [row {idx}] ROUGE-L falló: {exc}")
            rl = float("nan")
        rougel_scores.append(rl)

        # ── F1-RadGraph ───────────────────────────────────────────────────────
        rg = _radgraph_f1(f1rg_model, ref, hyp)
        radgraph_scores.append(rg)

        # ── CheXpert (14 etiquetas) ───────────────────────────────────────────
        try:
            chex = _chexpert_cols(hyp)
        except Exception as exc:
            logger.warning(f"  [row {idx}] CheXpert falló: {exc}")
            chex = {f"chexpert_{lbl.lower().replace(' ', '_')}": float("nan")
                    for lbl in _chexpert.LABELS}
        chexpert_rows.append(chex)

    # Añadir columnas métricas al DataFrame
    df["bleu_4"]      = bleu4_scores
    df["rouge_l"]     = rougel_scores
    df["radgraph_f1"] = radgraph_scores

    chex_df = pd.DataFrame(chexpert_rows, index=df.index)
    df = pd.concat([df, chex_df], axis=1)

    # ── Guardar CSV granular ───────────────────────────────────────────────────
    granular_path = output_dir / f"resultados_granulares_{cfg_name}.csv"
    df.to_csv(granular_path, index=False)
    logger.info(f"  ✅ Resultados granulares: {granular_path}")
    logger.info(f"     Columnas finales ({len(df.columns)}): {list(df.columns)}")

    return df


# ──────────────────────────────────────────────────────────────────────────────
# 8. Tabla resumen global
# ──────────────────────────────────────────────────────────────────────────────

def calcular_resumen_global(
    dfs: Dict[str, pd.DataFrame],
    output_dir: Path,
) -> pd.DataFrame:
    """
    Calcula los promedios globales de todas las métricas y los guarda en
    metricas_globales_comparacion.csv.

    Args:
        dfs       : Diccionario {cfg_name: DataFrame granular}.
        output_dir: Directorio de salida.

    Returns:
        DataFrame de resumen con una fila por config.
    """
    metric_cols = ["bleu_4", "rouge_l", "radgraph_f1"] + [
        f"chexpert_{lbl.lower().replace(' ', '_')}"
        for lbl in CheXpertLabelerLite.LABELS
    ]

    rows = []
    for cfg_name, df in dfs.items():
        row = {"config": cfg_name, "n_samples": len(df)}
        for col in metric_cols:
            if col in df.columns:
                row[col] = df[col].mean(skipna=True)
            else:
                row[col] = float("nan")
        rows.append(row)

    df_summary = pd.DataFrame(rows)
    summary_path = output_dir / "metricas_globales_comparacion.csv"
    df_summary.to_csv(summary_path, index=False)
    logger.info(f"  📊 Resumen global guardado en: {summary_path}")

    return df_summary


# ──────────────────────────────────────────────────────────────────────────────
# 9. CLI
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline final de evaluación BLIP-2 (Config_A y Config_B).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(config.paths.base_dir / "evaluation_results"),
        help="Directorio donde se guardarán todos los CSV de salida.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Limitar evaluación a N muestras del Test Set (None = todas).",
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help=(
            "Saltar Fase 1 y usar predicciones_X.csv existentes en --output-dir. "
            "Útil para re-evaluar métricas sin re-inferir."
        ),
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=["Config_A", "Config_B"],
        default=["Config_A", "Config_B"],
        help="Qué configs evaluar.",
    )
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# 10. Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> int:
    args = _parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Directorio de salida: {output_dir}")

    # Filtrar configs solicitadas
    configs_a_evaluar = [c for c in CONFIGS if c["name"] in args.configs]

    # ── Pre-cargar F1RadGraph UNA SOLA VEZ (es costoso en RAM) ───────────────
    logger.info("\nCargando F1RadGraph (reward_level='all', model_type='radgraph-xl')...")
    f1rg_model = F1RadGraph(reward_level="all", model_type="radgraph-xl")
    logger.info("  ✅ F1RadGraph listo\n")

    # ── Diccionario acumulador de DataFrames granulares ────────────────────────
    granular_dfs: Dict[str, pd.DataFrame] = {}

    for cfg_dict in configs_a_evaluar:
        cfg_name = cfg_dict["name"]

        # ── FASE 1 ─────────────────────────────────────────────────────────────
        preds_path = output_dir / f"predicciones_{cfg_name}.csv"

        if args.skip_inference and preds_path.exists():
            logger.info(f"  [Fase 1] --skip-inference: usando {preds_path}")
        else:
            preds_path = fase1_inferencia(
                config_dict = cfg_dict,
                output_dir  = output_dir,
                num_samples = args.num_samples,
            )

        # ── FASE 2 ─────────────────────────────────────────────────────────────
        df_granular = fase2_evaluacion(
            preds_path  = preds_path,
            cfg_name    = cfg_name,
            output_dir  = output_dir,
            f1rg_model  = f1rg_model,
        )
        granular_dfs[cfg_name] = df_granular

    # ── Resumen global ─────────────────────────────────────────────────────────
    if len(granular_dfs) > 0:
        logger.info(f"\n{'═'*65}")
        logger.info("  Calculando resumen global de métricas...")
        logger.info(f"{'═'*65}")

        df_summary = calcular_resumen_global(granular_dfs, output_dir)

        print(f"\n{'═'*65}")
        print("  MÉTRICAS GLOBALES")
        print(f"{'═'*65}")
        print(df_summary[["config", "n_samples", "bleu_4", "rouge_l", "radgraph_f1"]].to_string(index=False))
        print(f"{'═'*65}\n")

    logger.info("✅ Pipeline de evaluación finalizado\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
