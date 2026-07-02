#!/usr/bin/env python3
"""
exp1_cot_inference_final_experiment.py — RAMA 3 (Prompt NUEVO / CoT) · EXP. FINAL
================================================================================
Experimento 1 (Chain-of-Thought, PROMPT NUEVO) ejecutado sobre el modelo del
EXPERIMENTO FINAL (RadDINO[LoRA Exp2, frozen] + Asclepius-R[frozen] + Puente
Full-FT, lr=1e-5). Es el análogo de exp1_cot_inference_exp3_asclepius.py pero
reconstruyendo la arquitectura FINAL con final_state.pt.

REUTILIZACIÓN ESTRICTA (no se altera la lógica de los evaluadores existentes)
────────────────────────────────────────────────────────────────────────────
NO se reimplementan ni la inferencia CoT ni las métricas: se IMPORTAN y reutilizan
TAL CUAL las funciones ya validadas de `exp1_cot_inference_exp3_asclepius`:

    fase_inferencia_cot_exp3_asclepius(...) → inferencia CoT batched con resume
    fase2_evaluacion_cot(...)               → 5 métricas granulares fila por fila
    calcular_resumen_global(...)            → promedios globales

Lo ÚNICO específico del Experimento FINAL es:
    1) El modelo se carga con `cargar_modelo_final_experiment()` (reutilizado de
       evaluate_final_experiment) → build_final_experiment_model + final_state.pt.
    2) Los nombres/rutas de salida (CONFIG_NAME="Exp_Final_CoT", carpeta
       final_experiment_results-50pct/exp1_cot/).

Para que las funciones reutilizadas escriban en las rutas del Experimento FINAL,
se REDIRIGEN (sin editar el archivo fuente) los globals de módulo de
`exp1_cot_inference_exp3_asclepius` ANTES de invocarlas (composición; misma
técnica que en evaluate_final_experiment.py).

Salidas:
    final_experiment_results-50pct/exp1_cot/
        exp1_raw_inference_final_experiment.csv      (inferencia CoT cruda)
        resultados_granulares_Exp_Final_CoT.csv      (GRANULAR: 14 patologías + métricas)
        metricas_globales_Exp_Final_CoT.csv          (GLOBAL: promedios)

⚠ El nombre EXACTO de RAW_CSV (exp1_raw_inference_final_experiment.csv) debe
  coincidir con vllm_formatter.TASKS["nuevo_final"].in_csv (RAMA 4).

Uso:
    python exp1_cot_inference_final_experiment.py
    python exp1_cot_inference_final_experiment.py --num-samples 200   # smoke
    python exp1_cot_inference_final_experiment.py --skip-inference    # reusa raw
    python exp1_cot_inference_final_experiment.py --skip-evaluation   # solo raw (→ Llama 8b)

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

import torch

from config import config
from model_utils import cargar_procesador

# ── Worker functions REUTILIZADAS SIN CAMBIOS (inferencia CoT + métricas) ────
#   (este módulo aplica además los parches de tokenizer/nltk al importarse)
import exp1_cot_inference_exp3_asclepius as E3CoT
from exp1_cot_inference_exp3_asclepius import (
    fase_inferencia_cot_exp3_asclepius,
    fase2_evaluacion_cot,
    calcular_resumen_global,
)
from exp1_cot_inference import EXP1_GEN_PARAMS, construir_prompt_cot

# ── Loader del modelo FINAL (reutilizado de la RAMA 1 → DRY, una sola arch) ──
from evaluate_final_experiment import cargar_modelo_final_experiment, FINAL_STATE

# ── Constructor + rutas del Experimento FINAL ────────────────────────────────
from trainer_utils_final_experiment import (
    preparar_procesador_exp3,
    BASE_MODEL_NAME,
    ASCLEPIUS_R_DIR,
    ASCLEPIUS_VARIANT,
    EXP2_STATE_PT,
    OUTPUT_DIR,
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
# 2. Constantes — rutas y nombres del Experimento FINAL (CoT)
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR: Path = config.paths.base_dir

CONFIG_NAME: str = "Exp_Final_CoT"

EXP1_DIR: Path = OUTPUT_DIR / "exp1_cot"
# ⚠ Nombre EXACTO requerido por vllm_formatter.TASKS["nuevo_final"].in_csv:
RAW_CSV:      Path = EXP1_DIR / "exp1_raw_inference_final_experiment.csv"
GRANULAR_CSV: Path = EXP1_DIR / f"resultados_granulares_{CONFIG_NAME}.csv"
GLOBAL_CSV:   Path = EXP1_DIR / f"metricas_globales_{CONFIG_NAME}.csv"

SEP = "─" * 70


# ─────────────────────────────────────────────────────────────────────────────
# 3. REDIRECCIÓN de los globals reutilizados (sin editar exp1_cot_inference_exp3)
# ─────────────────────────────────────────────────────────────────────────────
def _redirigir_globals_cot() -> None:
    """
    Apunta los globals de módulo de `exp1_cot_inference_exp3_asclepius` a las
    rutas/nombre del Experimento FINAL, de modo que las funciones reutilizadas
    (fase_inferencia_cot / fase2_evaluacion_cot / calcular_resumen_global)
    escriban en final_experiment_results-50pct/exp1_cot/ con
    CONFIG_NAME="Exp_Final_CoT". NO se modifica el archivo fuente.
    """
    E3CoT.CONFIG_NAME  = CONFIG_NAME
    E3CoT.EXP1_DIR     = EXP1_DIR
    E3CoT.RAW_CSV      = RAW_CSV
    E3CoT.GRANULAR_CSV = GRANULAR_CSV
    E3CoT.GLOBAL_CSV   = GLOBAL_CSV
    # TEST_CSV_50PCT / IMAGES_DIR_50PCT / EXP1_GEN_PARAMS son idénticos (mismo
    # split 50% y mismo prompt CoT) → no requieren redirección.


# ─────────────────────────────────────────────────────────────────────────────
# 4. CLI / main
# ─────────────────────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=("Experimento 1 (CoT, prompt NUEVO) sobre el Experimento "
                     "FINAL (RadDINO + Asclepius-R + Puente): inferencia + "
                     "evaluación granular (BLEU/ROUGE/RadGraph/CheXpert/ClinBERT)."),
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
    p.add_argument("--final-state", type=str, default=str(FINAL_STATE),
                   help="final_state.pt entrenado en el Experimento FINAL "
                        "(solo params entrenables; se carga con strict=False).")
    p.add_argument("--chexpert-csv", type=str,
                   default=str(BASE_DIR / "mimic-cxr-2.0.0-chexpert.csv"),
                   help="CSV oficial de etiquetas CheXpert (.csv o .csv.gz).")
    p.add_argument("--skip-inference", action="store_true",
                   help="Reutilizar exp1_raw_inference_final_experiment.csv y "
                        "solo (re)evaluar.")
    p.add_argument("--resume", action="store_true",
                   help="Reanudar la inferencia: conserva el CSV existente y "
                        "solo genera los reportes faltantes.")
    p.add_argument("--no-bertscore", action="store_true",
                   help="Saltar Clinical BERTScore (smoke tests rápidos).")
    p.add_argument("--skip-evaluation", action="store_true",
                   help="Solo realizar la inferencia (no evaluar). Útil cuando "
                        "se va a aplicar primero el formateo Llama 8b (RAMA 4).")
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    # Redirigir las salidas de las worker functions reutilizadas al dir FINAL
    _redirigir_globals_cot()

    print("\n" + "=" * 80)
    print("  EXPERIMENTO 1 — CHAIN-OF-THOUGHT  ·  EXPERIMENTO FINAL (RadDINO + Asclepius-R)")
    print("  Task 1: clasificación de 7 patologías  →  Task 2: generar reporte")
    print("=" * 80)
    print(f"  CONFIG_NAME    : {CONFIG_NAME}")
    print(f"  num_beams      : {EXP1_GEN_PARAMS['num_beams']}  "
          f"max_new_tokens : {EXP1_GEN_PARAMS['max_new_tokens']}")
    print(f"  Asclepius-R    : {ASCLEPIUS_R_DIR}  (variante {ASCLEPIUS_VARIANT})")
    print(f"  RadDINO weights: {EXP2_STATE_PT}")
    print(f"  final_state.pt : {args.final_state}")
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

    # ── FASE 1: Inferencia CoT ───────────────────────────────────────────────
    if args.skip_inference and RAW_CSV.exists():
        logger.info(f"--skip-inference: reutilizando {RAW_CSV}")
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
        # Worker REUTILIZADO sin cambios — escribe en E3CoT.RAW_CSV (redirigido).
        fase_inferencia_cot_exp3_asclepius(
            model, processor, device,
            num_samples = args.num_samples,
            batch_size  = args.batch_size,
            resume      = args.resume,
        )
        del model
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("  VRAM del modelo Experimento FINAL liberada.\n")

    # ── FASE 2: Evaluación (puede saltarse si se va a formatear primero) ─────
    if args.skip_evaluation:
        logger.info("--skip-evaluation: NO se realizará la evaluación granular.")
        logger.info(f"  RAW_CSV listo: {RAW_CSV}")
        logger.info("  Próximo paso (RAMA 4): "
                    "python vllm_formatter.py --tasks nuevo_final")
        return 0

    # Workers REUTILIZADOS sin cambios — escriben en E3CoT.GRANULAR_CSV/GLOBAL_CSV.
    df_granular = fase2_evaluacion_cot(
        RAW_CSV, chexpert_csv,
        chexbert_batch = args.chexbert_batch,
        bert_batch     = args.bert_batch,
        no_bertscore   = args.no_bertscore,
    )
    calcular_resumen_global(df_granular)

    print("\n" + "=" * 80)
    print("  EXP1 (CoT)  ·  EXPERIMENTO FINAL (RadDINO + Asclepius-R)  ·  COMPLETADO")
    print("=" * 80)
    print(f"  Raw          : {RAW_CSV}")
    print(f"  Granular     : {GRANULAR_CSV}")
    print(f"  Global       : {GLOBAL_CSV}")
    print("\n  Siguiente (RAMA 4): python vllm_formatter.py --tasks nuevo_final")
    print("=" * 80 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
