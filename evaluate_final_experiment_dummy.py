#!/usr/bin/env python3
"""
evaluate_final_experiment_dummy.py — EXPERIMENTO FINAL con PROMPT DUMMY
==========================================================================
Variante del evaluador del **Experimento FINAL** (RadDINO + Asclepius-R +
Puente entrenado) que ejecuta la inferencia sobre el MISMO Test Set 50%, pero
usando un **PROMPT DUMMY** (genérico, no médico) en lugar del PROMPT DISEÑADO
con el que el modelo fue entrenado.

    ⚠ DIFERENCIA ÚNICA respecto a evaluate_final_experiment.py:
        • NO se usa `config.inference.default_prompt` (el prompt diseñado).
        • NO se antepone el prefijo de vista `[Context: {vista} view]`.
        • Se usa un prompt DUMMY constante para TODAS las imágenes.
          Por defecto: "a photo of"  (prompt genérico canónico de BLIP/CLIP).
          Configurable con  --dummy-prompt "..."  (incluido "" = incondicional).

Sirve como ABLACIÓN del prompt sobre el modelo YA ENTRENADO: aísla cuánto
depende el Experimento FINAL del prompt DISEÑADO frente a un prompt trivial,
manteniendo TODO lo demás idéntico (pesos entrenados, arquitectura, GEN_PARAMS,
Test Set, preprocesado y las 5 métricas).

MÉTRICAS (idénticas al resto del proyecto → comparables directamente):
    • CheXpert personalizado (14 patologías, GT oficial vs generado: P/R/F1)
    • BLEU-4
    • ROUGE-L
    • F1-RadGraph (radgraph-xl, reward_level="all")
    • Clinical BERTScore (Bio_ClinicalBERT)

REUTILIZACIÓN ESTRICTA
────────────────────────────────────────────────────────────────────────────
  • MODELO   : `cargar_modelo_final_experiment` de evaluate_final_experiment
               (reconstruye RadDINO + Asclepius-R + Puente y carga final_state.pt).
  • TOKENIZER: `preparar_procesador_exp3` (intercambia OPT → Asclepius/LLaMA;
               OBLIGATORIO porque el LM del FINAL es Asclepius, vocab 32000).
  • MÉTRICAS : `fase2_evaluacion` + `calcular_resumen_global` de
               evaluate_exp3_asclepius (SIN cambios; las métricas NO dependen
               del prompt). Se REDIRIGEN sus globals de módulo a las rutas del
               dummy (misma técnica de composición del resto del proyecto).
  • Lo ÚNICO propio es `fase1_inferencia_dummy` (réplica de fase1_inferencia
    cambiando SOLO el prompt: dummy constante, sin prefijo de vista).

Salidas:
    final_experiment_dummy_results-50pct/evaluation/
        predicciones_Exp_Final_Dummy.csv           (inferencias crudas)
        resultados_granulares_Exp_Final_Dummy.csv  (granular: 14 patologías + métricas)
        metricas_globales_Exp_Final_Dummy.csv      (global: promedios)

Uso:
    python evaluate_final_experiment_dummy.py                       # dummy "a photo of"
    python evaluate_final_experiment_dummy.py --dummy-prompt ""     # incondicional
    python evaluate_final_experiment_dummy.py --num-samples 40      # smoke
    python evaluate_final_experiment_dummy.py --skip-inference      # solo re-evaluar

NOTA: NO usa entornos virtuales — corre sobre el Python del sistema.

Autor : Evelyn Silva Rozas
Fecha : Julio 2026
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0. Variables de entorno (ANTES de imports de transformers/torch)
# ─────────────────────────────────────────────────────────────────────────────
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import gc
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
cv2.setNumThreads(0)

import pandas as pd
import torch
from tqdm import tqdm
from radgraph import F1RadGraph

from config import config
from model_utils import cargar_procesador
from data_loader import MimicCXRDataset
from preprocessing import estandarizar_imagen
from chexbert_official import CheXbertLabeler

# ── Métricas REUTILIZADAS SIN CAMBIOS (no dependen del prompt) ───────────────
import evaluate_exp3_asclepius as E3
from evaluate_exp3_asclepius import (
    fase2_evaluacion,
    calcular_resumen_global,
    GEN_PARAMS,
)

# ── Cargador del modelo Experimento FINAL (RadDINO + Asclepius-R + Puente) ────
from evaluate_final_experiment import (
    cargar_modelo_final_experiment,
    FINAL_STATE,
)

# ── Constructor / tokenizer-swap / rutas del Experimento FINAL ───────────────
from trainer_utils_final_experiment import (
    preparar_procesador_exp3,
    BASE_MODEL_NAME,
    ASCLEPIUS_R_DIR,
    ASCLEPIUS_VARIANT,
    EXP2_STATE_PT,
    _asclepius_disponible,
)

# ── Clinical BERTScore (los mismos helpers de Fase 2/3/Exp2/Exp3/FINAL) ──────
from reevaluar_sucio_vs_limpio import (
    cargar_bertscorer,
    CLINICAL_BERT_MODEL,
    CLINICAL_BERT_LAYERS,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Constantes — rutas, nombre y PROMPT DUMMY
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR: Path = config.paths.base_dir

CONFIG_NAME: str = "Exp_Final_Dummy"

# PROMPT DUMMY por defecto (genérico, NO médico, NO el prompt diseñado).
# "a photo of" es el prompt dummy canónico de la literatura BLIP/CLIP.
DEFAULT_DUMMY_PROMPT: str = "a photo of"

OUTPUT_DIR:      Path = BASE_DIR / "final_experiment_dummy_results-50pct"
EVAL_DIR:        Path = OUTPUT_DIR / "evaluation"
PREDICTIONS_CSV: Path = EVAL_DIR / f"predicciones_{CONFIG_NAME}.csv"
GRANULAR_CSV:    Path = EVAL_DIR / f"resultados_granulares_{CONFIG_NAME}.csv"
GLOBAL_CSV:      Path = EVAL_DIR / f"metricas_globales_{CONFIG_NAME}.csv"

SEP = "─" * 70


# ─────────────────────────────────────────────────────────────────────────────
# 3. REDIRECCIÓN de los globals reutilizados (sin editar evaluate_exp3_asclepius)
# ─────────────────────────────────────────────────────────────────────────────
def _redirigir_globals_evaluador() -> None:
    """
    Apunta los globals de módulo de `evaluate_exp3_asclepius` a las rutas/nombre
    del Experimento FINAL DUMMY, para que las funciones de MÉTRICAS reutilizadas
    (fase2_evaluacion / calcular_resumen_global) escriban en
    final_experiment_dummy_results-50pct/ con CONFIG_NAME="Exp_Final_Dummy".

    NO se modifica el archivo fuente (composición): solo se reasignan atributos
    del módulo ya importado. El Test Set / imágenes son IDÉNTICOS.
    """
    E3.CONFIG_NAME      = CONFIG_NAME
    E3.EVAL_DIR         = EVAL_DIR
    E3.PREDICTIONS_CSV  = PREDICTIONS_CSV
    E3.GRANULAR_CSV     = GRANULAR_CSV
    E3.GLOBAL_CSV       = GLOBAL_CSV


# ─────────────────────────────────────────────────────────────────────────────
# 4. FASE 1 — Inferencia con PROMPT DUMMY (réplica de fase1_inferencia; solo
#    cambia el prompt: dummy constante, SIN prefijo de vista, SIN prompt diseñado)
# ─────────────────────────────────────────────────────────────────────────────
def fase1_inferencia_dummy(
    model,
    processor,
    device,
    output_csv:   Path,
    dummy_prompt: str,
    num_samples:  Optional[int] = None,
    batch_size:   int           = 8,
    resume:       bool          = False,
) -> Path:
    """
    Itera el Test Set 50% y genera reportes con un PROMPT DUMMY CONSTANTE,
    exportando un CSV incremental (idéntico esquema al del resto del proyecto:
    dicom_id, study_id, reference_report, generated_report).

    Es una réplica EXACTA de evaluate_exp3_asclepius.fase1_inferencia salvo por
    la construcción del prompt:
        Diseñado : f"[Context: {vista} view] {config.inference.default_prompt}"
        DUMMY    : dummy_prompt   (constante para todas las imágenes, sin vista)
    """
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"  Cargando Test Set 50%...")
    test_dataset = MimicCXRDataset(
        csv_path             = E3.TEST_CSV_50PCT,
        images_dir           = E3.IMAGES_DIR_50PCT,
        processor            = processor,
        image_size           = config.data.image_size,
        clahe_clip_limit     = config.data.clahe_clip_limit,
        clahe_tile_grid_size = config.data.clahe_tile_grid_size,
        max_length           = config.data.max_length,
        padding              = config.data.padding,
        prompt               = dummy_prompt,   # irrelevante para inferencia (no se tokeniza aquí)
        augment              = False,
    )
    n_total = len(test_dataset)
    n_eval  = n_total if num_samples is None else min(num_samples, n_total)
    logger.info(f"  Test Set: {n_total:,} muestras | A evaluar: {n_eval:,}")
    logger.info(f"  PROMPT DUMMY (constante): {dummy_prompt!r}")
    logger.info(f"  GEN_PARAMS : {GEN_PARAMS}")

    COLS = ["dicom_id", "study_id", "reference_report", "generated_report"]

    # ── Modo resume: detectar dicom_id ya generados ──────────────────────────
    procesados: set = set()
    if resume and output_csv.exists():
        try:
            _prev = pd.read_csv(output_csv, usecols=["dicom_id"])
            procesados = set(_prev["dicom_id"].astype(str).tolist())
        except Exception as exc:
            logger.warning(
                f"  [resume] no se pudo leer {output_csv.name}: {exc}; "
                f"se empieza desde cero."
            )
            procesados = set()

    if procesados:
        logger.info(
            f"  [RESUME] {len(procesados):,} reportes ya guardados → "
            f"se reanuda en modo append (faltan ~{max(0, n_eval - len(procesados)):,})."
        )
    else:
        pd.DataFrame(columns=COLS).to_csv(output_csv, index=False)
        logger.info(f"  CSV incremental iniciado: {output_csv}")

    batch_images:  List = []
    batch_prompts: List[str] = []
    batch_meta:    List[Dict] = []
    errores       = 0
    n_guardadas   = 0
    t0            = time.time()

    for idx in tqdm(range(n_eval), desc=f"  Inferencia {CONFIG_NAME}", unit="img"):
        row      = test_dataset.data.iloc[idx]
        dicom_id = str(row["dicom_id"])

        if procesados and dicom_id in procesados:
            continue

        study_id = str(row.get("study_id", ""))
        ref_text = str(row["report_text"]).strip()

        # ⚠ PROMPT DUMMY CONSTANTE (sin prefijo de vista, sin prompt diseñado).
        prompt = dummy_prompt

        # Imagen (mismo preprocesado que el resto del proyecto)
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

        batch_images.append(imagen_pil)
        batch_prompts.append(prompt)
        batch_meta.append({
            "dicom_id": dicom_id,
            "study_id": study_id,
            "ref_text": ref_text,
        })

        if len(batch_images) == batch_size or idx == n_eval - 1:
            if not batch_images:
                continue
            try:
                inputs = processor(
                    images        = batch_images,
                    text          = batch_prompts,
                    return_tensors= "pt",
                    padding       = True,
                ).to(device)

                with torch.inference_mode():
                    generated_ids = model.generate(**inputs, **GEN_PARAMS)

                decoded = processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )

                rows = [
                    {
                        "dicom_id":         m["dicom_id"],
                        "study_id":         m["study_id"],
                        "reference_report": m["ref_text"],
                        "generated_report": rep.strip(),
                    }
                    for m, rep in zip(batch_meta, decoded)
                ]
                pd.DataFrame(rows, columns=COLS).to_csv(
                    output_csv, mode="a", header=False, index=False
                )
                n_guardadas += len(rows)

            except Exception as exc:
                logger.exception(f"  Error inferencia lote idx={idx}: {exc}")
                errores += len(batch_images)

            batch_images.clear()
            batch_prompts.clear()
            batch_meta.clear()

    dt = time.time() - t0
    logger.info(
        f"  Inferencia {CONFIG_NAME} OK: {n_guardadas:,} reportes nuevos | "
        f"{errores} errores | {dt/60:.1f} min"
    )
    logger.info(f"  → {output_csv}")

    del test_dataset
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("  VRAM liberada\n")
    return output_csv


# ─────────────────────────────────────────────────────────────────────────────
# 5. CLI / main
# ─────────────────────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=("Evaluación del Experimento FINAL (RadDINO + Asclepius-R + "
                     "Puente) con PROMPT DUMMY (ablación del prompt) sobre el "
                     "Test Set 50%: CheXpert(14) + BLEU-4 + ROUGE-L + RadGraph + "
                     "Clinical BERTScore."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dummy-prompt", type=str, default=DEFAULT_DUMMY_PROMPT,
                   help="Prompt DUMMY constante (usa \"\" para incondicional).")
    p.add_argument("--num-samples", type=int, default=None,
                   help="Limitar a N muestras del Test Set (None = todas).")
    p.add_argument("--batch-size", type=int, default=4,
                   help="Tamaño de lote de inferencia.")
    p.add_argument("--num-beams", type=int, default=GEN_PARAMS["num_beams"],
                   help="num_beams para generate.")
    p.add_argument("--max-new-tokens", type=int, default=GEN_PARAMS["max_new_tokens"],
                   help="max_new_tokens para generate.")
    p.add_argument("--chexpert-csv", type=str,
                   default=str(BASE_DIR / "mimic-cxr-2.0.0-chexpert.csv"),
                   help="CSV oficial de etiquetas CheXpert (.csv o .csv.gz).")
    p.add_argument("--skip-inference", action="store_true",
                   help="Reutilizar predicciones_Exp_Final_Dummy.csv y solo (re)evaluar.")
    p.add_argument("--resume", action="store_true",
                   help="Reanudar la inferencia (conserva el CSV y genera lo faltante).")
    p.add_argument("--final-state", type=str, default=str(FINAL_STATE),
                   help="final_state.pt entrenado en el Experimento FINAL.")
    p.add_argument("--bert-batch", type=int, default=64,
                   help="Tamaño de lote de Bio_ClinicalBERT (BERTScore).")
    p.add_argument("--no-bertscore", action="store_true",
                   help="Saltar Clinical BERTScore (smoke tests rápidos).")
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    # Redirigir las salidas de las funciones de métricas reutilizadas al dir dummy
    _redirigir_globals_evaluador()

    # Override de generación desde CLI (misma copia de dict compartida con E3)
    if args.num_beams != GEN_PARAMS["num_beams"] or args.max_new_tokens != GEN_PARAMS["max_new_tokens"]:
        GEN_PARAMS["num_beams"]      = args.num_beams
        GEN_PARAMS["max_new_tokens"] = args.max_new_tokens
        E3.GEN_PARAMS["num_beams"]      = args.num_beams
        E3.GEN_PARAMS["max_new_tokens"] = args.max_new_tokens

    print("\n" + "=" * 80)
    print(f"  EVALUACIÓN — EXPERIMENTO FINAL (RadDINO + Asclepius-R + Puente) · PROMPT DUMMY (ablación)")
    print("=" * 80)
    print(f"  Config         : {CONFIG_NAME}")
    print(f"  base_model     : {BASE_MODEL_NAME}")
    print(f"  Asclepius-R    : {ASCLEPIUS_R_DIR}  (variante {ASCLEPIUS_VARIANT})")
    print(f"  RadDINO weights: {EXP2_STATE_PT}")
    print(f"  final_state.pt : {args.final_state}")
    print(f"  Test set       : {E3.TEST_CSV_50PCT}")
    print(f"  Imágenes       : {E3.IMAGES_DIR_50PCT}")
    print(f"  PROMPT DUMMY   : {args.dummy_prompt!r}   (NO se usa el prompt diseñado)")
    print(f"  Outputs        : {EVAL_DIR}")
    print(f"  GEN_PARAMS     : beam={GEN_PARAMS['num_beams']} max_new={GEN_PARAMS['max_new_tokens']}")
    print(f"  GPU            : {'disponible ✓' if torch.cuda.is_available() else 'NO disponible ✗'}")
    print("=" * 80 + "\n")

    if not torch.cuda.is_available():
        logger.warning("CUDA no disponible: la inferencia será MUY lenta en CPU.")

    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    chexpert_csv = Path(args.chexpert_csv) if args.chexpert_csv else None
    if chexpert_csv is not None and not chexpert_csv.exists():
        gz_alt = Path(str(chexpert_csv) + ".gz")
        if gz_alt.exists():
            chexpert_csv = gz_alt

    # ── Procesador BLIP-2 (image_processor de BLIP-2; tokenizer se intercambia) ─
    logger.info("Cargando Blip2Processor...")
    processor = cargar_procesador(model_name=config.model.model_name)

    # ── FASE 1 — Inferencia con PROMPT DUMMY ─────────────────────────────────
    if args.skip_inference and PREDICTIONS_CSV.exists():
        logger.info(f"--skip-inference: reutilizando {PREDICTIONS_CSV}")
    else:
        if not _asclepius_disponible(ASCLEPIUS_R_DIR):
            logger.error(
                "Asclepius-R no disponible. Ejecuta "
                "`python run_final_experiment.py --print-accion-obligatoria`."
            )
            return 2
        # ⚠ Intercambiar tokenizer OPT → Asclepius/LLaMA ANTES de generar.
        processor = preparar_procesador_exp3(processor, ASCLEPIUS_R_DIR)

        model, device = cargar_modelo_final_experiment(
            ruta_final_state = Path(args.final_state),
            base_model_name  = config.model.model_name,
            asclepius_dir    = ASCLEPIUS_R_DIR,
            exp2_state_pt    = EXP2_STATE_PT,
        )
        fase1_inferencia_dummy(
            model        = model,
            processor    = processor,
            device       = device,
            output_csv   = PREDICTIONS_CSV,
            dummy_prompt = args.dummy_prompt,
            num_samples  = args.num_samples,
            batch_size   = args.batch_size,
            resume       = args.resume,
        )
        del model
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("  VRAM del modelo Experimento FINAL liberada.\n")

    # ── FASE 2 — Métricas (RadGraph + CheXbert + Clinical BERTScore) ─────────
    logger.info("Cargando F1RadGraph (reward_level='all', model_type='radgraph-xl')...")
    f1rg = F1RadGraph(reward_level="all", model_type="radgraph-xl")
    logger.info("  F1RadGraph listo")

    logger.info("Cargando CheXbert oficial (Stanford AIMI)...")
    chexbert = CheXbertLabeler()
    logger.info("  CheXbert listo")

    bert_scorer    = None
    bert_tokenizer = None
    if not args.no_bertscore:
        logger.info(
            f"Cargando Clinical BERTScore (model='{CLINICAL_BERT_MODEL}', "
            f"num_layers={CLINICAL_BERT_LAYERS})..."
        )
        bert_scorer = cargar_bertscorer(
            model=CLINICAL_BERT_MODEL,
            num_layers=CLINICAL_BERT_LAYERS,
            device=None,
        )
        from transformers import AutoTokenizer
        bert_tokenizer = AutoTokenizer.from_pretrained(CLINICAL_BERT_MODEL)
        logger.info("  Clinical BERTScore listo\n")
    else:
        logger.info("  Clinical BERTScore: OMITIDO (--no-bertscore)\n")

    # fase2_evaluacion (reutilizada) escribe en E3.GRANULAR_CSV = nuestro GRANULAR_CSV
    df_gran = fase2_evaluacion(
        preds_path     = PREDICTIONS_CSV,
        f1rg           = f1rg,
        chexbert       = chexbert,
        chexpert_csv   = chexpert_csv,
        bert_scorer    = bert_scorer,
        bert_tokenizer = bert_tokenizer,
        bert_batch     = args.bert_batch,
    )

    # ── Resumen global (reutilizado; escribe en E3.GLOBAL_CSV = nuestro GLOBAL_CSV) ─
    logger.info(f"\n{SEP}")
    logger.info("  Calculando resumen global de métricas...")
    df_summary = calcular_resumen_global(df_gran)

    show_cols = [
        "config", "n_samples", "bleu_4", "rouge_l", "radgraph_f1",
        "bertscore_p", "bertscore_r", "bertscore_f1",
        "chexpert_precision", "chexpert_recall", "chexpert_f1",
    ]
    show_cols = [c for c in show_cols if c in df_summary.columns]
    print(f"\n{'=' * 70}")
    print(f"  MÉTRICAS GLOBALES — {CONFIG_NAME} (EXPERIMENTO FINAL · PROMPT DUMMY)")
    print(f"{'=' * 70}")
    print(df_summary[show_cols].to_string(index=False))
    print(f"{'=' * 70}\n")

    print("=" * 80)
    print("  EVALUACIÓN EXPERIMENTO FINAL (prompt DUMMY) COMPLETADA")
    print("=" * 80)
    print(f"  Predicciones    : {PREDICTIONS_CSV}")
    print(f"  Granular        : {GRANULAR_CSV}")
    print(f"  Resumen global  : {GLOBAL_CSV}")
    print(f"\n  Comparativo CheXpert: python actualizar_comparativos_final_dummy.py")
    print("=" * 80 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
