#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_phase1_final_10pct.py — Evaluacion FASE 1 (dos mejores modelos) sobre
el Test Set 10% COMPLETO.

Reutiliza SIN duplicar logica las funciones ya validadas de `evaluate_final.py`
(FASE 1 inferencia batched + FASE 2 metricas granulares + resumen global),
pero apuntando a:

    • Modelos   : phase1_final_results/checkpoints/best_model_Config_{A,B}
                  (LoRA sobre el language_model de BLIP-2)
    • Test CSV  : test_split_10pct_completo.csv  (3796 muestras, imagen garantizada)
    • Imagenes  : imagenes_50_porciento          (banco disponible)
    • Salida    : phase1_10pct_eval_results/

Metricas (FASE 1 — SIN Clinical BERTScore):
    • CheXpert (14 patologias, CheXbert oficial Stanford AIMI)  -> P / R / F1
    • F1-RadGraph (radgraph-xl, reward_level='all')
    • BLEU-4
    • ROUGE-L

Salidas:
    • predicciones_Config_A.csv / predicciones_Config_B.csv        (inferencia)
    • resultados_granulares_Config_A.csv / _Config_B.csv           (granular)
    • metricas_globales_comparacion.csv                            (comparativo global)

Uso:
    python evaluate_phase1_final_10pct.py                 # inferencia + metricas
    python evaluate_phase1_final_10pct.py --skip-inference  # solo re-evaluar
    python evaluate_phase1_final_10pct.py --configs Config_A

Autor : Evelyn Silva Rozas
Fecha : Julio 2026
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict

# ── Config (mutar rutas ANTES de construir datasets) ────────────────────────
from config import config

BASE = config.paths.base_dir

# Redirigir el pipeline al banco de imagenes y al test COMPLETO
config.paths.images_dir = BASE / "imagenes_50_porciento"
config.paths.test_csv   = BASE / "test_split_10pct_completo.csv"

# ── Reusar el pipeline validado ─────────────────────────────────────────────
import evaluate_final as ef  # noqa: E402  (import despues de mutar config)

logger = logging.getLogger("phase1_10pct")

# Modelos: los DOS MEJORES de FASE 1
CONFIGS = [
    {
        "name":      "Config_A",
        "model_dir": BASE / "phase1_final_results" / "checkpoints" / "best_model_Config_A",
    },
    {
        "name":      "Config_B",
        "model_dir": BASE / "phase1_final_results" / "checkpoints" / "best_model_Config_B",
    },
]

OUTPUT_DIR   = BASE / "phase1_10pct_eval_results"
CHEXPERT_CSV = BASE / "mimic-cxr-2.0.0-chexpert.csv"   # (.csv o .csv.gz, con fallback)

# Alinear las CONFIGS del modulo reutilizado (por consistencia interna)
ef.CONFIGS = CONFIGS


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluacion FASE 1 (dos mejores modelos) sobre test 10% completo.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    p.add_argument("--chexpert-csv", type=str, default=str(CHEXPERT_CSV))
    p.add_argument("--num-samples", type=int, default=None,
                   help="Limitar a N muestras (debug).")
    p.add_argument("--skip-inference", action="store_true",
                   help="Saltar FASE 1 y reutilizar predicciones_X.csv existentes.")
    p.add_argument("--configs", nargs="+",
                   choices=["Config_A", "Config_B"],
                   default=["Config_A", "Config_B"])
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 65)
    logger.info("  EVALUACION FASE 1 — Test 10%% COMPLETO (3796)")
    logger.info("  Test CSV : %s", config.paths.test_csv.name)
    logger.info("  Imagenes : %s", config.paths.images_dir.name)
    logger.info("  Salida   : %s", output_dir)
    logger.info("=" * 65)

    chexpert_csv_path = Path(args.chexpert_csv) if args.chexpert_csv else None
    configs_a_evaluar = [c for c in CONFIGS if c["name"] in args.configs]

    # Pre-cargar evaluadores pesados UNA sola vez
    logger.info("Cargando F1RadGraph (reward_level='all', model_type='radgraph-xl')...")
    from radgraph import F1RadGraph
    f1rg = F1RadGraph(reward_level="all", model_type="radgraph-xl")
    logger.info("  F1RadGraph listo")

    logger.info("Cargando CheXbert oficial (Stanford AIMI)...")
    from chexbert_official import CheXbertLabeler
    chexbert = CheXbertLabeler()
    logger.info("  CheXbert listo\n")

    granular_dfs: Dict[str, "object"] = {}

    for cfg_dict in configs_a_evaluar:
        cfg_name   = cfg_dict["name"]
        preds_path = output_dir / f"predicciones_{cfg_name}.csv"

        # ── FASE 1: inferencia (con resume simple) ──
        reuse = False
        if args.skip_inference and preds_path.exists():
            reuse = True
        elif preds_path.exists():
            # resume: si el CSV ya tiene las 3796 filas esperadas, no re-inferir
            try:
                import pandas as pd
                n_prev = sum(1 for _ in open(preds_path)) - 1
                n_test = sum(1 for _ in open(config.paths.test_csv)) - 1
                if n_prev >= n_test:
                    logger.info("[FASE 1] %s ya completo (%d filas) — se reutiliza.",
                                cfg_name, n_prev)
                    reuse = True
            except Exception:
                reuse = False

        if reuse:
            logger.info("[FASE 1] Reutilizando %s", preds_path.name)
        else:
            preds_path = ef.fase1_inferencia(
                config_dict = cfg_dict,
                output_dir  = output_dir,
                num_samples = args.num_samples,
            )

        # ── FASE 2: metricas granulares ──
        df_gran = ef.fase2_evaluacion(
            preds_path   = preds_path,
            cfg_name     = cfg_name,
            output_dir   = output_dir,
            f1rg         = f1rg,
            chexbert     = chexbert,
            chexpert_csv = chexpert_csv_path,
        )
        granular_dfs[cfg_name] = df_gran

    # ── Resumen global comparativo ──
    if granular_dfs:
        logger.info("\n" + "=" * 65)
        logger.info("  Resumen global comparativo...")
        df_summary = ef.calcular_resumen_global(granular_dfs, output_dir)

        show_cols = ["config", "n_samples", "bleu_4", "rouge_l", "radgraph_f1",
                     "chexpert_precision", "chexpert_recall", "chexpert_f1"]
        show_cols = [c for c in show_cols if c in df_summary.columns]
        print("\n" + "=" * 65)
        print("  METRICAS GLOBALES — FASE 1 (test 10% completo)")
        print("=" * 65)
        print(df_summary[show_cols].to_string(index=False))
        print("=" * 65 + "\n")

    logger.info("Evaluacion FASE 1 (test 10%% completo) finalizada.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
