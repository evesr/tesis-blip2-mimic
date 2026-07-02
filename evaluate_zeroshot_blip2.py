#!/usr/bin/env python3
"""
evaluate_zeroshot_blip2.py — EVALUACIÓN ZERO-SHOT del BLIP-2 BASE (sin fine-tuning)
==================================================================================
Evalúa el modelo BLIP-2 BASE de HuggingFace (`Salesforce/blip2-opt-2.7b`), tal
cual viene preentrenado y SIN NINGÚN ajuste de esta experimentación (zero-shot),
sobre el MISMO Test Set 50% y con el MISMO prompt diseñado que el resto de las
fases/experimentos, calculando las 5 métricas requeridas:

    • CheXpert personalizado (14 patologías, GT oficial vs generado: matches + P/R/F1)
    • BLEU-4
    • ROUGE-L
    • F1-RadGraph (radgraph-xl, reward_level="all")
    • Clinical BERTScore (Bio_ClinicalBERT)

El objetivo es establecer la LÍNEA BASE (baseline zero-shot): cuánto rinde el
BLIP-2 genérico sin haber visto MIMIC-CXR, para contrastarlo contra Fase 2,
Fase 3, los Experimentos 2/3 y el Experimento FINAL.

REUTILIZACIÓN ESTRICTA (no se altera la lógica de los evaluadores existentes)
────────────────────────────────────────────────────────────────────────────
Este script NO reimplementa la inferencia ni las métricas: IMPORTA y reutiliza
TAL CUAL las funciones ya validadas de `evaluate_exp3_asclepius` (que a su vez
reutiliza las métricas de `evaluate_phase3_puente` y `reevaluar_sucio_vs_limpio`):

    fase1_inferencia(...)        → generación batched con el prompt diseñado
    fase2_evaluacion(...)        → métricas granulares fila por fila
    calcular_resumen_global(...) → promedios globales

Lo ÚNICO específico del ZERO-SHOT es:
    1) `cargar_modelo_zeroshot()` → carga el BLIP-2 BASE (Blip2ForConditionalGeneration)
       en bf16, SIN cuantización, SIN LoRA, SIN Asclepius, SIN puente entrenado.
    2) El procesador es el `Blip2Processor` POR DEFECTO (tokenizer OPT del modelo
       base) → NO se intercambia el tokenizer (a diferencia del Exp3/FINAL, que
       usan Asclepius/LLaMA).
    3) Los nombres/rutas de salida (CONFIG_NAME = "Zeroshot_BLIP2", carpeta
       zeroshot_results-50pct/evaluation/).

Para que las funciones reutilizadas escriban en las rutas del zero-shot, se
REDIRIGEN (sin editar el archivo fuente) los globals de módulo de
`evaluate_exp3_asclepius` ANTES de invocarlas. Es la misma técnica de
composición que usa el resto del proyecto (evaluate_final_experiment.py, etc.).

Salidas:
    zeroshot_results-50pct/evaluation/
        predicciones_Zeroshot_BLIP2.csv           (inferencias crudas)
        resultados_granulares_Zeroshot_BLIP2.csv  (reporte GRANULAR: 14 patologías + métricas)
        metricas_globales_Zeroshot_BLIP2.csv      (reporte GLOBAL: promedios)

Uso:
    python evaluate_zeroshot_blip2.py                     # inferencia + métricas (Test Set completo)
    python evaluate_zeroshot_blip2.py --num-samples 40    # smoke (40 muestras)
    python evaluate_zeroshot_blip2.py --skip-inference    # solo re-evaluar métricas
    python evaluate_zeroshot_blip2.py --resume            # reanudar inferencia incompleta

NOTA: NO usa entornos virtuales — corre sobre el Python del sistema.

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
from typing import Optional

import cv2
cv2.setNumThreads(0)

import torch
from radgraph import F1RadGraph

from config import config
from model_utils import cargar_modelo_base, cargar_procesador
from chexbert_official import CheXbertLabeler

# ── Funciones de inferencia + métricas REUTILIZADAS SIN CAMBIOS ──────────────
#   (evaluate_exp3_asclepius ya re-exporta las métricas de evaluate_phase3_puente;
#    además aplica al importarse el patch de compatibilidad de radgraph.)
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

# ── Rutas del Test Set 50% (reutilizadas del Experimento 3; mismo split) ─────
#   E3.TEST_CSV_50PCT / E3.IMAGES_DIR_50PCT ya apuntan al Test Set 50% correcto,
#   por lo que NO necesitan redireccionarse.


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
# 2. Constantes — rutas y nombres del ZERO-SHOT
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR: Path = config.paths.base_dir

CONFIG_NAME: str = "Zeroshot_BLIP2"

OUTPUT_DIR:      Path = BASE_DIR / "zeroshot_results-50pct"
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
    del ZERO-SHOT, de modo que las funciones reutilizadas
    (fase1_inferencia / fase2_evaluacion / calcular_resumen_global) escriban en
    zeroshot_results-50pct/ con CONFIG_NAME="Zeroshot_BLIP2".

    NO se modifica el archivo fuente: solo se reasignan atributos del módulo ya
    importado (composición), técnica idéntica a la usada en el resto del proyecto
    (evaluate_final_experiment.py).

    El Test Set / imágenes / GEN_PARAMS son IDÉNTICOS (mismo split 50% y mismo
    prompt diseñado) → comparabilidad directa con el resto de las fases.
    """
    E3.CONFIG_NAME      = CONFIG_NAME
    E3.EVAL_DIR         = EVAL_DIR
    E3.PREDICTIONS_CSV  = PREDICTIONS_CSV
    E3.GRANULAR_CSV     = GRANULAR_CSV
    E3.GLOBAL_CSV       = GLOBAL_CSV


# ─────────────────────────────────────────────────────────────────────────────
# 4. CARGA DEL MODELO ZERO-SHOT (BLIP-2 BASE, sin fine-tuning)
# ─────────────────────────────────────────────────────────────────────────────
def cargar_modelo_zeroshot(
    base_model_name: str = config.model.model_name,
):
    """
    Carga el BLIP-2 BASE de HuggingFace TAL CUAL (zero-shot, sin ajuste alguno).

    A diferencia de Exp3/FINAL, aquí NO hay reconstrucción de arquitectura ni
    carga de pesos entrenados: se usa el checkpoint preentrenado público
    `Salesforce/blip2-opt-2.7b` directamente.

        • Precisión bf16 (igual que el resto de la experimentación).
        • SIN cuantización (use_quantization=False) → inferencia en precisión plena bf16.
        • device_map="auto" → todo en la GPU (L40S) en una sola pasada.
        • eval() + requires_grad=False (inferencia pura).

    Returns:
        (model, device)
    """
    logger.info(f"\n{SEP}")
    logger.info(f"  CARGA MODELO ZERO-SHOT (BLIP-2 BASE, sin fine-tuning)  ·  {CONFIG_NAME}")
    logger.info(f"  base_model : {base_model_name}")
    logger.info(f"  cuantización: NO (bf16 plena)")
    logger.info(f"{SEP}")

    model = cargar_modelo_base(
        model_name      = base_model_name,
        use_quantization= False,
        device_map      = "auto",
    )

    # Modo eval + congelar TODO (inferencia pura)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    device = next(model.parameters()).device
    target_dtype = next(model.parameters()).dtype

    gc.collect()
    torch.cuda.empty_cache()

    logger.info(f"  ✓ Modelo BLIP-2 BASE (zero-shot) listo en {device} | {target_dtype}\n")
    return model, device


# ─────────────────────────────────────────────────────────────────────────────
# 5. CLI / main
# ─────────────────────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=("Evaluación ZERO-SHOT del BLIP-2 BASE (Salesforce/blip2-opt-2.7b) "
                     "con el prompt DISEÑADO sobre el Test Set 50%: CheXpert(14) + "
                     "BLEU-4 + ROUGE-L + RadGraph + Clinical BERTScore."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--num-samples", type=int, default=None,
                   help="Limitar a N muestras del Test Set (None = todas).")
    p.add_argument("--batch-size", type=int, default=8,
                   help="Tamaño de lote de inferencia.")
    p.add_argument("--num-beams", type=int, default=GEN_PARAMS["num_beams"],
                   help="num_beams para generate.")
    p.add_argument("--max-new-tokens", type=int, default=GEN_PARAMS["max_new_tokens"],
                   help="max_new_tokens para generate.")
    p.add_argument("--chexpert-csv", type=str,
                   default=str(BASE_DIR / "mimic-cxr-2.0.0-chexpert.csv"),
                   help="CSV oficial de etiquetas CheXpert (.csv o .csv.gz).")
    p.add_argument("--skip-inference", action="store_true",
                   help="Reutilizar predicciones_Zeroshot_BLIP2.csv y solo (re)evaluar.")
    p.add_argument("--resume", action="store_true",
                   help="Reanudar la inferencia (conserva el CSV y genera lo faltante).")
    p.add_argument("--bert-batch", type=int, default=64,
                   help="Tamaño de lote de Bio_ClinicalBERT (BERTScore).")
    p.add_argument("--no-bertscore", action="store_true",
                   help="Saltar Clinical BERTScore (smoke tests rápidos).")
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    # Redirigir las salidas de las funciones reutilizadas al dir del ZERO-SHOT
    _redirigir_globals_evaluador()

    # Override de generación desde CLI (sobre la copia local + la de E3 reusada)
    if args.num_beams != GEN_PARAMS["num_beams"] or args.max_new_tokens != GEN_PARAMS["max_new_tokens"]:
        GEN_PARAMS["num_beams"]      = args.num_beams
        GEN_PARAMS["max_new_tokens"] = args.max_new_tokens
        E3.GEN_PARAMS["num_beams"]      = args.num_beams
        E3.GEN_PARAMS["max_new_tokens"] = args.max_new_tokens

    print("\n" + "=" * 80)
    print(f"  EVALUACIÓN ZERO-SHOT — BLIP-2 BASE (sin fine-tuning)  ·  prompt diseñado")
    print("=" * 80)
    print(f"  Config       : {CONFIG_NAME}")
    print(f"  Modelo base  : {config.model.model_name}")
    print(f"  Test set     : {E3.TEST_CSV_50PCT}")
    print(f"  Imágenes     : {E3.IMAGES_DIR_50PCT}")
    print(f"  Prompt       : {config.inference.default_prompt!r}")
    print(f"  Outputs      : {EVAL_DIR}")
    print(f"  GEN_PARAMS   : beam={GEN_PARAMS['num_beams']} max_new={GEN_PARAMS['max_new_tokens']}")
    print(f"  GPU          : {'disponible ✓' if torch.cuda.is_available() else 'NO disponible ✗'}")
    print("=" * 80 + "\n")

    if not torch.cuda.is_available():
        logger.warning("CUDA no disponible: la inferencia será MUY lenta en CPU.")

    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    chexpert_csv = Path(args.chexpert_csv) if args.chexpert_csv else None
    if chexpert_csv is not None and not chexpert_csv.exists():
        gz_alt = Path(str(chexpert_csv) + ".gz")
        if gz_alt.exists():
            chexpert_csv = gz_alt

    # ── Procesador BLIP-2 POR DEFECTO (tokenizer OPT del modelo base) ────────
    #   ⚠ A diferencia de Exp3/FINAL, NO se intercambia el tokenizer: el LLM del
    #     BLIP-2 base es OPT-2.7B, así que el processor por defecto es el correcto.
    logger.info("Cargando Blip2Processor por defecto (tokenizer OPT del modelo base)...")
    processor = cargar_procesador(model_name=config.model.model_name)

    # ── FASE 1 — Inferencia (reutiliza fase1_inferencia de evaluate_exp3) ────
    if args.skip_inference and PREDICTIONS_CSV.exists():
        logger.info(f"--skip-inference: reutilizando {PREDICTIONS_CSV}")
    else:
        model, device = cargar_modelo_zeroshot(
            base_model_name = config.model.model_name,
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
        logger.info("  VRAM del modelo BLIP-2 base liberada.\n")

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
    print(f"  MÉTRICAS GLOBALES — {CONFIG_NAME} (ZERO-SHOT)")
    print(f"{'=' * 70}")
    print(df_summary[show_cols].to_string(index=False))
    print(f"{'=' * 70}\n")

    print("=" * 80)
    print("  EVALUACIÓN ZERO-SHOT (BLIP-2 base, prompt diseñado) COMPLETADA")
    print("=" * 80)
    print(f"  Predicciones    : {PREDICTIONS_CSV}")
    print(f"  Granular        : {GRANULAR_CSV}")
    print(f"  Resumen global  : {GLOBAL_CSV}")
    print("=" * 80 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
