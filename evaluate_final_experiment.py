#!/usr/bin/env python3
"""
evaluate_final_experiment.py — RAMA 1 (Prompt DISEÑADO) · EXPERIMENTO FINAL
===========================================================================
Evaluación del modelo del Experimento FINAL (RadDINO + Asclepius-R + Puente) con
el PROMPT DISEÑADO (Default_promt), calculando las 5 métricas requeridas:

    • CheXpert personalizado (14 patologías, GT oficial vs generado: matches + P/R/F1)
    • BLEU-4
    • ROUGE-L
    • F1-RadGraph (radgraph-xl, reward_level="all")
    • Clinical BERTScore (Bio_ClinicalBERT)

REUTILIZACIÓN ESTRICTA (no se altera la lógica de los evaluadores existentes)
────────────────────────────────────────────────────────────────────────────
Este script NO reimplementa la inferencia ni las métricas: IMPORTA y reutiliza
TAL CUAL las funciones ya validadas de `evaluate_exp3_asclepius`:

    fase1_inferencia(...)        → generación batched con el prompt diseñado
    fase2_evaluacion(...)        → métricas granulares fila por fila
    calcular_resumen_global(...) → promedios globales

Lo ÚNICO específico del Experimento FINAL es:
    1) `cargar_modelo_final_experiment()` → construye la arquitectura RadDINO +
       Asclepius-R (build_final_experiment_model) y carga `final_state.pt`.
    2) Los nombres/rutas de salida (CONFIG_NAME = "Exp_Final", carpeta
       final_experiment_results-50pct/evaluation/).

Para que las funciones reutilizadas escriban en las rutas del Experimento FINAL,
se REDIRIGEN (sin editar el archivo fuente) los globals de módulo de
`evaluate_exp3_asclepius` ANTES de invocarlas. Es la misma técnica de
composición que usa el resto del proyecto (las métricas se importan de
evaluate_phase3_puente / reevaluar_sucio_vs_limpio sin modificarlas).

Salidas:
    final_experiment_results-50pct/evaluation/
        predicciones_Exp_Final.csv              (inferencias crudas)
        resultados_granulares_Exp_Final.csv     (reporte GRANULAR: 14 patologías + métricas)
        metricas_globales_Exp_Final.csv         (reporte GLOBAL: promedios)

Uso:
    python evaluate_final_experiment.py                    # inferencia + métricas
    python evaluate_final_experiment.py --skip-inference   # solo re-evaluar métricas
    python evaluate_final_experiment.py --num-samples 40   # smoke

Autor : Evelyn Silva Rozas
Fecha : Junio 2026
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
from pathlib import Path
from typing import Dict, Optional

import cv2
cv2.setNumThreads(0)

import torch
from radgraph import F1RadGraph

from config import config
from model_utils import cargar_procesador
from chexbert_official import CheXbertLabeler

# ── Funciones de inferencia + métricas REUTILIZADAS SIN CAMBIOS ──────────────
#   (evaluate_exp3_asclepius ya re-exporta las métricas de evaluate_phase3_puente)
import evaluate_exp3_asclepius as E3
from evaluate_exp3_asclepius import (
    fase1_inferencia,
    fase2_evaluacion,
    calcular_resumen_global,
    GEN_PARAMS,
)

# ── Clinical BERTScore (los mismos helpers de Fase 2/3/Exp2/Exp3) ────────────
from reevaluar_sucio_vs_limpio import (
    cargar_bertscorer,
    CLINICAL_BERT_MODEL,
    CLINICAL_BERT_LAYERS,
)

# ── Constructor + rutas del Experimento FINAL (RadDINO + Asclepius-R) ─────────
from trainer_utils_final_experiment import (
    build_final_experiment_model,
    preparar_procesador_exp3,
    BASE_MODEL_NAME,
    ASCLEPIUS_R_DIR,
    ASCLEPIUS_VARIANT,
    EXP2_STATE_PT,
    OUTPUT_DIR,
    CHECKPOINTS_DIR,
    TEST_CSV as TEST_CSV_50PCT,
    IMAGES_DIR as IMAGES_DIR_50PCT,
    _asclepius_disponible,
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
# 2. Constantes — rutas y nombres del Experimento FINAL
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR: Path = config.paths.base_dir

CONFIG_NAME: str = "Exp_Final"

FINAL_BEST_DIR: Path = CHECKPOINTS_DIR / f"best_model_{CONFIG_NAME}"
FINAL_STATE:    Path = FINAL_BEST_DIR / "final_state.pt"
FINAL_META:     Path = FINAL_BEST_DIR / "final_meta.json"

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
    del Experimento FINAL, de modo que las funciones reutilizadas
    (fase1_inferencia / fase2_evaluacion / calcular_resumen_global) escriban en
    final_experiment_results-50pct/ con CONFIG_NAME="Exp_Final".

    NO se modifica el archivo fuente: solo se reasignan atributos del módulo ya
    importado (composición), técnica idéntica a la usada en el resto del proyecto.
    """
    E3.CONFIG_NAME      = CONFIG_NAME
    E3.EVAL_DIR         = EVAL_DIR
    E3.PREDICTIONS_CSV  = PREDICTIONS_CSV
    E3.GRANULAR_CSV     = GRANULAR_CSV
    E3.GLOBAL_CSV       = GLOBAL_CSV
    # Test set / imágenes / GEN_PARAMS son idénticos (mismo split 50% y prompt).


# ─────────────────────────────────────────────────────────────────────────────
# 4. CARGA DEL MODELO Experimento FINAL (única función arch-específica)
# ─────────────────────────────────────────────────────────────────────────────
def cargar_modelo_final_experiment(
    ruta_final_state: Path = FINAL_STATE,
    base_model_name:  str  = BASE_MODEL_NAME,
    asclepius_dir:    Path = ASCLEPIUS_R_DIR,
    exp2_state_pt:    Path = EXP2_STATE_PT,
    max_memory_gib:   int  = 38,
):
    """
    Reconstruye el modelo del Experimento FINAL (RadDINO + Asclepius-R + Puente)
    para inferencia. Es el INVERSO LÓGICO del entrenamiento: la arquitectura se
    reconstruye con build_final_experiment_model() y se cargan los pesos
    entrenados (`final_state.pt`, solo params entrenables) con strict=False.

    Returns:
        (model, device)
    """
    ruta_final_state = Path(ruta_final_state)
    if not ruta_final_state.exists():
        raise FileNotFoundError(
            f"  ✗ Falta final_state.pt del Experimento FINAL: {ruta_final_state}\n"
            f"      Se genera al terminar run_final_experiment.py "
            f"(best_model_{CONFIG_NAME}/)."
        )

    logger.info(f"\n{SEP}")
    logger.info(f"  CARGA MODELO EXPERIMENTO FINAL (RadDINO + Asclepius-R + Puente)  ·  {CONFIG_NAME}")
    logger.info(f"  base_model     : {base_model_name}")
    logger.info(f"  Asclepius-R    : {asclepius_dir}  (variante {ASCLEPIUS_VARIANT})")
    logger.info(f"  RadDINO weights: {exp2_state_pt}")
    logger.info(f"  final_state.pt : {ruta_final_state}")
    logger.info(f"{SEP}")

    # ── 1. Reconstruir la arquitectura EXACTA (GC=False → KV-cache en generate) ─
    model = build_final_experiment_model(
        base_model_name        = base_model_name,
        asclepius_dir          = asclepius_dir,
        exp2_state_pt          = exp2_state_pt,
        train_query_tokens     = True,
        max_memory_gib         = max_memory_gib,
        gradient_checkpointing = False,
    )

    _ref = next(model.qformer.parameters())
    target_device, target_dtype = _ref.device, _ref.dtype
    logger.info(f"  device={target_device}  dtype={target_dtype}")

    # ── 2. Cargar los pesos ENTRENADOS del puente (strict=False) ─────────────
    logger.info(f"  Cargando final_state.pt (solo params entrenables) strict=False...")
    state = torch.load(str(ruta_final_state), map_location="cpu", weights_only=True)
    state = {k: v.to(device=target_device, dtype=target_dtype) for k, v in state.items()}

    missing, unexpected = model.load_state_dict(state, strict=False)
    n_loaded = sum(t.numel() for t in state.values())
    logger.info(f"  ✓ final_state cargado: {len(state)} keys, {n_loaded:,} params")
    if unexpected:
        logger.warning(f"  ⚠ {len(unexpected)} keys INESPERADAS en final_state.pt:")
        for k in unexpected[:5]:
            logger.warning(f"      {k}")

    loaded_prefixes = {k.split(".", 1)[0] for k in state.keys()}
    logger.info(f"  Bloques entrenados cargados: {sorted(loaded_prefixes)}")
    for pref_root in ("qformer", "language_projection", "query_tokens"):
        if pref_root not in loaded_prefixes:
            logger.warning(f"  ⚠ FALTA bloque entrenado: {pref_root}")

    # ── 3. Modo eval + congelar TODO (inferencia pura) ───────────────────────
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    del state
    gc.collect()
    torch.cuda.empty_cache()

    logger.info(f"  ✓ Modelo Experimento FINAL listo en {target_device} | {target_dtype}\n")
    return model, target_device


# ─────────────────────────────────────────────────────────────────────────────
# 5. CLI / main
# ─────────────────────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=("Evaluación del Experimento FINAL (RadDINO + Asclepius-R + "
                     "Puente) con el prompt DISEÑADO: CheXpert(14) + BLEU-4 + "
                     "ROUGE-L + RadGraph + Clinical BERTScore."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
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
                   help="Reutilizar predicciones_Exp_Final.csv y solo (re)evaluar.")
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

    # Redirigir las salidas de las funciones reutilizadas al dir del Exp FINAL
    _redirigir_globals_evaluador()

    # Override de generación desde CLI (sobre la copia local + la de E3 reusada)
    if args.num_beams != GEN_PARAMS["num_beams"] or args.max_new_tokens != GEN_PARAMS["max_new_tokens"]:
        GEN_PARAMS["num_beams"]      = args.num_beams
        GEN_PARAMS["max_new_tokens"] = args.max_new_tokens
        E3.GEN_PARAMS["num_beams"]      = args.num_beams
        E3.GEN_PARAMS["max_new_tokens"] = args.max_new_tokens

    print("\n" + "=" * 80)
    print(f"  EVALUACIÓN — EXPERIMENTO FINAL (RadDINO + Asclepius-R + Puente)  ·  prompt diseñado")
    print("=" * 80)
    print(f"  Config         : {CONFIG_NAME}")
    print(f"  Test set       : {TEST_CSV_50PCT}")
    print(f"  Imágenes       : {IMAGES_DIR_50PCT}")
    print(f"  Asclepius-R    : {ASCLEPIUS_R_DIR}  (variante {ASCLEPIUS_VARIANT})")
    print(f"  RadDINO weights: {EXP2_STATE_PT}")
    print(f"  final_state.pt : {args.final_state}")
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

    # ── Procesador BLIP-2 (image_processor de BLIP-2; tokenizer Asclepius/LLaMA) ─
    logger.info("Cargando Blip2Processor...")
    processor = cargar_procesador(model_name=config.model.model_name)

    # ── FASE 1 — Inferencia (reutiliza fase1_inferencia de evaluate_exp3) ────
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
        fase1_inferencia(
            model       = model,
            processor   = processor,
            device      = device,
            output_csv  = PREDICTIONS_CSV,
            num_samples = args.num_samples,
            batch_size  = args.batch_size,
            resume      = args.resume,
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
    print(f"  MÉTRICAS GLOBALES — {CONFIG_NAME}")
    print(f"{'=' * 70}")
    print(df_summary[show_cols].to_string(index=False))
    print(f"{'=' * 70}\n")

    print("=" * 80)
    print("  EVALUACIÓN EXPERIMENTO FINAL (prompt diseñado) COMPLETADA")
    print("=" * 80)
    print(f"  Predicciones    : {PREDICTIONS_CSV}")
    print(f"  Granular        : {GRANULAR_CSV}")
    print(f"  Resumen global  : {GLOBAL_CSV}")
    print("\n  Siguiente (RAMA 2): python vllm_formatter.py --tasks disenado_final")
    print("=" * 80 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
