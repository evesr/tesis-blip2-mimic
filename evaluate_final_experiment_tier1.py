#!/usr/bin/env python3
"""
evaluate_final_experiment_tier1.py — RAMA 1 (Prompt DISEÑADO) · EXPERIMENTO FINAL **TIER 1**
============================================================================================
Evaluación del modelo del Experimento FINAL **TIER 1** (RadDINO + Asclepius-R +
Puente Full-FT + cabeza clínica auxiliar CheXpert entrenada con
``L_total = L_LM + λ·L_clin``) con el PROMPT DISEÑADO, calculando las 5 métricas:

    • CheXpert personalizado (14 patologías, GT oficial vs generado: matches + P/R/F1)
    • BLEU-4
    • ROUGE-L
    • F1-RadGraph (radgraph-xl, reward_level="all")
    • Clinical BERTScore (Bio_ClinicalBERT)

DISEÑO — REUTILIZACIÓN ESTRICTA (no se altera ni un evaluador ni el original FINAL)
──────────────────────────────────────────────────────────────────────────────────
Este script NO reimplementa inferencia ni métricas: IMPORTA y reutiliza TAL CUAL
las funciones ya validadas de ``evaluate_exp3_asclepius`` (que a su vez re-exporta
las de ``evaluate_phase3_puente``):

    fase1_inferencia(...)        → generación batched con el prompt diseñado
    fase2_evaluacion(...)        → métricas granulares fila por fila (5 métricas + CheXpert 14)
    calcular_resumen_global(...) → promedios globales

Lo ÚNICO específico de TIER 1 es:
    1) La arquitectura se reconstruye con ``build_final_experiment_model`` del
       módulo **tier1** y se cargan los pesos entrenados ``final_state.pt`` del
       checkpoint TIER 1 (``best_model_Exp_Final_Tier1``). La cabeza auxiliar
       ``chexpert_aux_head.*`` NO se usa en inferencia (solo fue señal de
       entrenamiento) → se carga con ``strict=False`` y sus claves se ignoran.
    2) Nombres/rutas de salida bajo ``final_experiment_tier1_results-50pct/``
       (CONFIG_NAME = "Exp_Final_Tier1"), sin tocar la corrida FINAL original.

OPTIMIZACIÓN GPU (mismo output, menor tiempo)
─────────────────────────────────────────────
La generación conserva EXACTAMENTE los GEN_PARAMS del proyecto (beam=5,
max_new_tokens=400, rep_pen=1.1, no_repeat_ngram=3) para que las métricas sean
comparables con el resto de experimentos — cambiarlos alteraría los reportes.
La única palanca que NO altera la salida es el **tamaño de lote**: se sube de 4
a ``--batch-size 16`` (por defecto) para saturar la L40S (7B + beam5 ≈ 34 GB de
45 GB). La escritura del CSV es incremental y ``--resume`` reanuda sin recomputar.

MODO 2 — Evaluación de reportes FORMATEADOS por el LLM externo (``--eval-formatted-csv``)
────────────────────────────────────────────────────────────────────────────────────────
Tras el formateo con vLLM (``vllm_formatter_tier1.py``), este mismo script puede
re-evaluar las 5 métricas + CheXpert(14) sobre la columna ``formatted_report``:
construye un CSV de predicciones derivado (generated_report := formatted_report)
y corre ``fase2_evaluacion`` + ``calcular_resumen_global``, escribiendo:
    resultados_granulares_Exp_Final_Tier1_formateado.csv
    metricas_globales_Exp_Final_Tier1_formateado.csv
Así obtenemos CheXpert por las 14 patologías TAMBIÉN sobre el reporte formateado.

Salidas (en final_experiment_tier1_results-50pct/evaluation/):
    predicciones_Exp_Final_Tier1.csv                         (inferencias crudas)
    resultados_granulares_Exp_Final_Tier1.csv                (GRANULAR crudo: 14 patologías + 5 métricas)
    metricas_globales_Exp_Final_Tier1.csv                    (GLOBAL crudo: promedios)
    resultados_granulares_Exp_Final_Tier1_formateado.csv     (GRANULAR formateado, con --eval-formatted-csv)
    metricas_globales_Exp_Final_Tier1_formateado.csv         (GLOBAL formateado, con --eval-formatted-csv)

Uso:
    # RAMA 1 — inferencia cruda + métricas (batch grande, resume):
    python evaluate_final_experiment_tier1.py --batch-size 16 --resume

    # Solo re-evaluar métricas de un CSV de predicciones crudo existente:
    python evaluate_final_experiment_tier1.py --skip-inference

    # MODO 2 — evaluar métricas + CheXpert sobre reportes formateados por el LLM:
    python evaluate_final_experiment_tier1.py \
        --eval-formatted-csv final_experiment_tier1_results-50pct/evaluation/predicciones_Exp_Final_Tier1_llmexterno_llama8b.csv

    # Smoke:
    python evaluate_final_experiment_tier1.py --num-samples 32 --batch-size 16

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
from pathlib import Path
from typing import Optional

import cv2
cv2.setNumThreads(0)

import pandas as pd
import torch
from radgraph import F1RadGraph

from config import config
from model_utils import cargar_procesador
from chexbert_official import CheXbertLabeler

# ── Funciones de inferencia + métricas REUTILIZADAS SIN CAMBIOS ──────────────
import evaluate_exp3_asclepius as E3
from evaluate_exp3_asclepius import (
    fase1_inferencia,
    fase2_evaluacion,
    calcular_resumen_global,
    GEN_PARAMS,
)

# ── Clinical BERTScore (los mismos helpers de Fase 2/3/Exp2/Exp3/FINAL) ──────
from reevaluar_sucio_vs_limpio import (
    cargar_bertscorer,
    CLINICAL_BERT_MODEL,
    CLINICAL_BERT_LAYERS,
)

# ── Constructor + rutas del Experimento FINAL **TIER 1** ─────────────────────
from trainer_utils_final_experiment_tier1 import (
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
# 2. Constantes — rutas y nombres del Experimento FINAL TIER 1
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR: Path = config.paths.base_dir

CONFIG_NAME: str = "Exp_Final_Tier1"

FINAL_BEST_DIR: Path = CHECKPOINTS_DIR / f"best_model_{CONFIG_NAME}"
FINAL_STATE:    Path = FINAL_BEST_DIR / "final_state.pt"
FINAL_META:     Path = FINAL_BEST_DIR / "final_meta.json"

EVAL_DIR:        Path = OUTPUT_DIR / "evaluation"
PREDICTIONS_CSV: Path = EVAL_DIR / f"predicciones_{CONFIG_NAME}.csv"
GRANULAR_CSV:    Path = EVAL_DIR / f"resultados_granulares_{CONFIG_NAME}.csv"
GLOBAL_CSV:      Path = EVAL_DIR / f"metricas_globales_{CONFIG_NAME}.csv"

# ── Salidas del MODO 2 (reportes formateados por el LLM externo) ─────────────
GRANULAR_FMT_CSV: Path = EVAL_DIR / f"resultados_granulares_{CONFIG_NAME}_formateado.csv"
GLOBAL_FMT_CSV:   Path = EVAL_DIR / f"metricas_globales_{CONFIG_NAME}_formateado.csv"
# CSV de predicciones DERIVADO (generated_report := formatted_report) para fase2.
DERIVED_FMT_PREDS: Path = EVAL_DIR / f"predicciones_{CONFIG_NAME}_formateado_as_preds.csv"

SEP = "─" * 70


# ─────────────────────────────────────────────────────────────────────────────
# 3. REDIRECCIÓN de los globals reutilizados (sin editar evaluate_exp3_asclepius)
# ─────────────────────────────────────────────────────────────────────────────
def _redirigir_globals_evaluador(
    granular: Path = GRANULAR_CSV,
    global_csv: Path = GLOBAL_CSV,
    predicciones: Path = PREDICTIONS_CSV,
) -> None:
    """
    Apunta los globals de módulo de `evaluate_exp3_asclepius` a las rutas/nombre
    del Experimento FINAL TIER 1 para que las funciones reutilizadas
    (fase1_inferencia / fase2_evaluacion / calcular_resumen_global) escriban en
    final_experiment_tier1_results-50pct/ con CONFIG_NAME="Exp_Final_Tier1".

    NO se modifica el archivo fuente: solo se reasignan atributos del módulo ya
    importado (composición), técnica idéntica a la de evaluate_final_experiment.
    El Test Set 50% y GEN_PARAMS son idénticos (mismo split y prompt diseñado).
    """
    E3.CONFIG_NAME     = CONFIG_NAME
    E3.EVAL_DIR        = EVAL_DIR
    E3.PREDICTIONS_CSV = predicciones
    E3.GRANULAR_CSV    = granular
    E3.GLOBAL_CSV      = global_csv


# ─────────────────────────────────────────────────────────────────────────────
# 4. CARGA DEL MODELO Experimento FINAL TIER 1 (única función arch-específica)
# ─────────────────────────────────────────────────────────────────────────────
def cargar_modelo_final_tier1(
    ruta_final_state: Path = FINAL_STATE,
    base_model_name:  str  = BASE_MODEL_NAME,
    asclepius_dir:    Path = ASCLEPIUS_R_DIR,
    exp2_state_pt:    Path = EXP2_STATE_PT,
    max_memory_gib:   int  = 40,
):
    """
    Reconstruye el modelo del Experimento FINAL TIER 1 (RadDINO + Asclepius-R +
    Puente) para inferencia y carga los pesos entrenados (`final_state.pt`).

    La cabeza auxiliar CheXpert (`chexpert_aux_head.*`) se usó SOLO para la
    pérdida clínica durante el entrenamiento; ``build_final_experiment_model`` NO
    la reconstruye, de modo que sus claves aparecerán como "inesperadas" al
    cargar con strict=False y se ignoran limpiamente. La generación (Q-Former +
    language_projection + query_tokens + LLM + ViT) es idéntica a la del FINAL.

    Returns:
        (model, device)
    """
    ruta_final_state = Path(ruta_final_state)
    if not ruta_final_state.exists():
        raise FileNotFoundError(
            f"  ✗ Falta final_state.pt del Experimento FINAL TIER 1: {ruta_final_state}\n"
            f"      Se genera al terminar run_final_experiment_tier1.py "
            f"(best_model_{CONFIG_NAME}/)."
        )

    logger.info(f"\n{SEP}")
    logger.info(f"  CARGA MODELO FINAL TIER 1 (RadDINO + Asclepius-R + Puente + aux CheXpert)  ·  {CONFIG_NAME}")
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

    # ── 2. Cargar los pesos ENTRENADOS (strict=False → aux head ignorada) ────
    logger.info("  Cargando final_state.pt (params entrenables) strict=False...")
    state = torch.load(str(ruta_final_state), map_location="cpu", weights_only=True)
    state = {k: v.to(device=target_device, dtype=target_dtype) for k, v in state.items()}

    missing, unexpected = model.load_state_dict(state, strict=False)
    n_loaded = sum(t.numel() for t in state.values())
    logger.info(f"  ✓ final_state cargado: {len(state)} keys, {n_loaded:,} params")

    # La(s) clave(s) de la cabeza auxiliar son ESPERADAS como 'unexpected' aquí.
    aux_keys = [k for k in unexpected if k.startswith("chexpert_aux_head")]
    otras    = [k for k in unexpected if not k.startswith("chexpert_aux_head")]
    if aux_keys:
        logger.info(f"  (i) cabeza auxiliar CheXpert ignorada en inferencia: {aux_keys}")
    if otras:
        logger.warning(f"  ⚠ {len(otras)} keys INESPERADAS no-auxiliares en final_state.pt:")
        for k in otras[:5]:
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

    logger.info(f"  ✓ Modelo FINAL TIER 1 listo en {target_device} | {target_dtype}\n")
    return model, target_device


# ─────────────────────────────────────────────────────────────────────────────
# 5. Carga de los evaluadores de métricas (RadGraph + CheXbert + Clinical BERT)
# ─────────────────────────────────────────────────────────────────────────────
def _cargar_evaluadores(no_bertscore: bool):
    logger.info("Cargando F1RadGraph (reward_level='all', model_type='radgraph-xl')...")
    f1rg = F1RadGraph(reward_level="all", model_type="radgraph-xl")
    logger.info("  F1RadGraph listo")

    logger.info("Cargando CheXbert oficial (Stanford AIMI)...")
    chexbert = CheXbertLabeler()
    logger.info("  CheXbert listo")

    bert_scorer = bert_tokenizer = None
    if not no_bertscore:
        logger.info(
            f"Cargando Clinical BERTScore (model='{CLINICAL_BERT_MODEL}', "
            f"num_layers={CLINICAL_BERT_LAYERS})..."
        )
        bert_scorer = cargar_bertscorer(
            model=CLINICAL_BERT_MODEL, num_layers=CLINICAL_BERT_LAYERS, device=None,
        )
        from transformers import AutoTokenizer
        bert_tokenizer = AutoTokenizer.from_pretrained(CLINICAL_BERT_MODEL)
        logger.info("  Clinical BERTScore listo\n")
    else:
        logger.info("  Clinical BERTScore: OMITIDO (--no-bertscore)\n")
    return f1rg, chexbert, bert_scorer, bert_tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# 6. MODO 2 — Evaluación de reportes FORMATEADOS por el LLM externo
# ─────────────────────────────────────────────────────────────────────────────
def evaluar_formateados(
    formatted_csv: Path,
    chexpert_csv: Optional[Path],
    bert_batch: int,
    no_bertscore: bool,
    col_formatted: str = "formatted_report",
    out_granular: Path = GRANULAR_FMT_CSV,
    out_global: Path = GLOBAL_FMT_CSV,
    derived_preds: Path = DERIVED_FMT_PREDS,
) -> int:
    """
    Re-evalúa las 5 métricas + CheXpert(14) sobre la columna ``formatted_report``
    del CSV producido por el LLM externo, construyendo un CSV de predicciones
    derivado (generated_report := formatted_report) y reutilizando fase2 + global.

    Las rutas de salida (``out_granular`` / ``out_global`` / ``derived_preds``)
    son parametrizables → el MISMO evaluador sirve para el prompt DISEÑADO
    (por defecto) y para el prompt NUEVO/CoT (pasando rutas exp1_cot/).
    """
    formatted_csv = Path(formatted_csv)
    if not formatted_csv.exists():
        logger.error(f"  ✗ No existe el CSV formateado: {formatted_csv}")
        return 2

    logger.info(f"\n{SEP}")
    logger.info("  MODO 2 — Evaluación de reportes FORMATEADOS (LLM externo)")
    logger.info(f"  Entrada: {formatted_csv}")
    logger.info(f"  Granular salida: {out_granular}")
    logger.info(f"{SEP}")

    df = pd.read_csv(formatted_csv)
    if col_formatted not in df.columns:
        logger.error(
            f"  ✗ Falta la columna '{col_formatted}' en {formatted_csv.name}. "
            f"Cols: {list(df.columns)[:10]}"
        )
        return 2

    # Construir CSV de predicciones DERIVADO para fase2_evaluacion:
    #   generated_report := formatted_report (el reporte pulido por el LLM).
    derived = pd.DataFrame({
        "dicom_id":         df.get("dicom_id", pd.Series([""] * len(df))),
        "study_id":         df.get("study_id", pd.Series([""] * len(df))),
        "reference_report": df.get("reference_report", pd.Series([""] * len(df))),
        "generated_report": df[col_formatted].fillna("").astype(str),
    })
    Path(derived_preds).parent.mkdir(parents=True, exist_ok=True)
    derived.to_csv(derived_preds, index=False)
    logger.info(
        f"  CSV derivado (formatted→generated): {derived_preds}  ({len(derived):,} filas)"
    )

    # Redirigir salidas de fase2/global a los CSV *_formateado indicados.
    _redirigir_globals_evaluador(
        granular=out_granular, global_csv=out_global, predicciones=derived_preds,
    )

    f1rg, chexbert, bert_scorer, bert_tokenizer = _cargar_evaluadores(no_bertscore)

    df_gran = fase2_evaluacion(
        preds_path     = derived_preds,
        f1rg           = f1rg,
        chexbert       = chexbert,
        chexpert_csv   = chexpert_csv,
        bert_scorer    = bert_scorer,
        bert_tokenizer = bert_tokenizer,
        bert_batch     = bert_batch,
    )
    df_summary = calcular_resumen_global(df_gran)

    _imprimir_global(df_summary, titulo=f"{CONFIG_NAME} · FORMATEADO (LLM externo)")

    print("=" * 80)
    print("  EVALUACIÓN FORMATEADO (LLM externo) TIER 1 COMPLETADA")
    print("=" * 80)
    print(f"  Granular formateado : {out_granular}")
    print(f"  Global formateado   : {out_global}")
    print("=" * 80 + "\n")
    return 0


def _imprimir_global(df_summary: pd.DataFrame, titulo: str) -> None:
    show_cols = [
        "config", "n_samples", "bleu_4", "rouge_l", "radgraph_f1",
        "bertscore_p", "bertscore_r", "bertscore_f1",
        "chexpert_precision", "chexpert_recall", "chexpert_f1",
    ]
    show_cols = [c for c in show_cols if c in df_summary.columns]
    print(f"\n{'=' * 70}")
    print(f"  MÉTRICAS GLOBALES — {titulo}")
    print(f"{'=' * 70}")
    print(df_summary[show_cols].to_string(index=False))
    print(f"{'=' * 70}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 7. CLI / main
# ─────────────────────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=("Evaluación del Experimento FINAL TIER 1 (RadDINO + "
                     "Asclepius-R + Puente + aux CheXpert) con el prompt DISEÑADO: "
                     "CheXpert(14) + BLEU-4 + ROUGE-L + RadGraph + Clinical BERTScore."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--num-samples", type=int, default=None,
                   help="Limitar a N muestras del Test Set (None = todas).")
    p.add_argument("--batch-size", type=int, default=16,
                   help="Tamaño de lote de inferencia (beam=5, max_new=400). "
                        "16 satura la L40S sin OOM (~34/45 GB).")
    p.add_argument("--num-beams", type=int, default=GEN_PARAMS["num_beams"],
                   help="num_beams para generate (NO cambiar: parte de la metodología).")
    p.add_argument("--max-new-tokens", type=int, default=GEN_PARAMS["max_new_tokens"],
                   help="max_new_tokens para generate (NO cambiar: metodología).")
    p.add_argument("--chexpert-csv", type=str,
                   default=str(BASE_DIR / "mimic-cxr-2.0.0-chexpert.csv"),
                   help="CSV oficial de etiquetas CheXpert (.csv o .csv.gz).")
    p.add_argument("--skip-inference", action="store_true",
                   help="Reutilizar predicciones_Exp_Final_Tier1.csv y solo (re)evaluar.")
    p.add_argument("--resume", action="store_true",
                   help="Reanudar la inferencia (conserva el CSV y genera lo faltante).")
    p.add_argument("--final-state", type=str, default=str(FINAL_STATE),
                   help="final_state.pt entrenado en el Experimento FINAL TIER 1.")
    p.add_argument("--bert-batch", type=int, default=64,
                   help="Tamaño de lote de Bio_ClinicalBERT (BERTScore).")
    p.add_argument("--no-bertscore", action="store_true",
                   help="Saltar Clinical BERTScore (smoke tests rápidos).")
    p.add_argument("--eval-formatted-csv", type=str, default=None,
                   help="MODO 2: evaluar CheXpert(14)+5 métricas sobre la columna "
                        "formatted_report del CSV indicado (salida _formateado).")
    p.add_argument("--out-granular", type=str, default=None,
                   help="MODO 2: ruta del CSV granular de salida (default: "
                        "resultados_granulares_Exp_Final_Tier1_formateado.csv). "
                        "Usar rutas exp1_cot/ para el prompt NUEVO/CoT.")
    p.add_argument("--out-global", type=str, default=None,
                   help="MODO 2: ruta del CSV global de salida (default: "
                        "metricas_globales_Exp_Final_Tier1_formateado.csv).")
    p.add_argument("--derived-preds", type=str, default=None,
                   help="MODO 2: ruta del CSV de predicciones derivado "
                        "(generated_report:=formatted_report) intermedio.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    chexpert_csv = Path(args.chexpert_csv) if args.chexpert_csv else None
    if chexpert_csv is not None and not chexpert_csv.exists():
        gz_alt = Path(str(chexpert_csv) + ".gz")
        if gz_alt.exists():
            chexpert_csv = gz_alt

    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # ═════════════════════════════════════════════════════════════════════════
    # MODO 2 — evaluación de reportes formateados (no requiere el modelo BLIP-2)
    # ═════════════════════════════════════════════════════════════════════════
    if args.eval_formatted_csv:
        return evaluar_formateados(
            formatted_csv = Path(args.eval_formatted_csv),
            chexpert_csv  = chexpert_csv,
            bert_batch    = args.bert_batch,
            no_bertscore  = args.no_bertscore,
            out_granular  = Path(args.out_granular) if args.out_granular else GRANULAR_FMT_CSV,
            out_global    = Path(args.out_global) if args.out_global else GLOBAL_FMT_CSV,
            derived_preds = Path(args.derived_preds) if args.derived_preds else DERIVED_FMT_PREDS,
        )

    # ═════════════════════════════════════════════════════════════════════════
    # RAMA 1 — inferencia cruda (prompt diseñado) + métricas granulares/global
    # ═════════════════════════════════════════════════════════════════════════
    _redirigir_globals_evaluador()  # → CSVs crudos Tier1

    # Override de generación desde CLI (sobre la copia local + la de E3 reusada)
    if args.num_beams != GEN_PARAMS["num_beams"] or args.max_new_tokens != GEN_PARAMS["max_new_tokens"]:
        GEN_PARAMS["num_beams"]      = args.num_beams
        GEN_PARAMS["max_new_tokens"] = args.max_new_tokens
        E3.GEN_PARAMS["num_beams"]      = args.num_beams
        E3.GEN_PARAMS["max_new_tokens"] = args.max_new_tokens

    print("\n" + "=" * 80)
    print("  EVALUACIÓN — EXPERIMENTO FINAL TIER 1 (RadDINO + Asclepius-R + Puente)  ·  prompt diseñado")
    print("=" * 80)
    print(f"  Config         : {CONFIG_NAME}")
    print(f"  Test set       : {TEST_CSV_50PCT}")
    print(f"  Imágenes       : {IMAGES_DIR_50PCT}")
    print(f"  Asclepius-R    : {ASCLEPIUS_R_DIR}  (variante {ASCLEPIUS_VARIANT})")
    print(f"  RadDINO weights: {EXP2_STATE_PT}")
    print(f"  final_state.pt : {args.final_state}")
    print(f"  Outputs        : {EVAL_DIR}")
    print(f"  Batch size     : {args.batch_size}  (optimización GPU, mismo output)")
    print(f"  GEN_PARAMS     : beam={GEN_PARAMS['num_beams']} max_new={GEN_PARAMS['max_new_tokens']}")
    print(f"  GPU            : {'disponible ✓' if torch.cuda.is_available() else 'NO disponible ✗'}")
    print("=" * 80 + "\n")

    if not torch.cuda.is_available():
        logger.warning("CUDA no disponible: la inferencia será MUY lenta en CPU.")

    # ── Procesador BLIP-2 (image_processor de BLIP-2; tokenizer Asclepius/LLaMA) ─
    logger.info("Cargando Blip2Processor...")
    processor = cargar_procesador(model_name=config.model.model_name)

    # ── FASE 1 — Inferencia (reutiliza fase1_inferencia de E3) ───────────────
    if args.skip_inference and PREDICTIONS_CSV.exists():
        logger.info(f"--skip-inference: reutilizando {PREDICTIONS_CSV}")
    else:
        if not _asclepius_disponible(ASCLEPIUS_R_DIR):
            logger.error(
                "Asclepius-R no disponible. Ejecuta "
                "`python run_final_experiment_tier1.py --print-accion-obligatoria`."
            )
            return 2
        # ⚠ Intercambiar tokenizer OPT → Asclepius/LLaMA ANTES de generar.
        processor = preparar_procesador_exp3(processor, ASCLEPIUS_R_DIR)

        model, device = cargar_modelo_final_tier1(
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
        logger.info("  VRAM del modelo FINAL TIER 1 liberada.\n")

    # ── FASE 2 — Métricas granulares (RadGraph + CheXbert + Clinical BERTScore) ─
    f1rg, chexbert, bert_scorer, bert_tokenizer = _cargar_evaluadores(args.no_bertscore)

    df_gran = fase2_evaluacion(
        preds_path     = PREDICTIONS_CSV,
        f1rg           = f1rg,
        chexbert       = chexbert,
        chexpert_csv   = chexpert_csv,
        bert_scorer    = bert_scorer,
        bert_tokenizer = bert_tokenizer,
        bert_batch     = args.bert_batch,
    )

    logger.info(f"\n{SEP}")
    logger.info("  Calculando resumen global de métricas...")
    df_summary = calcular_resumen_global(df_gran)
    _imprimir_global(df_summary, titulo=f"{CONFIG_NAME} · CRUDO (prompt diseñado)")

    print("=" * 80)
    print("  EVALUACIÓN EXPERIMENTO FINAL TIER 1 (prompt diseñado, crudo) COMPLETADA")
    print("=" * 80)
    print(f"  Predicciones    : {PREDICTIONS_CSV}")
    print(f"  Granular        : {GRANULAR_CSV}")
    print(f"  Resumen global  : {GLOBAL_CSV}")
    print("\n  Siguiente: python vllm_formatter_tier1.py --resume")
    print("=" * 80 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
