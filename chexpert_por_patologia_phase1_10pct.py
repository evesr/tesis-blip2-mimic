#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
chexpert_por_patologia_phase1_10pct.py
======================================
Genera el CSV comparativo de CheXpert POR PATOLOGIA (14 clases + micro/macro)
para los dos mejores modelos de FASE 1 (Config_A, Config_B) evaluados sobre el
Test Set 10% COMPLETO.

Reutiliza SIN cambios los helpers ya validados de
`actualizar_comparativos_final.py`:
    • _metricas_pat_desde_granular(df_gran)  -> TP/FP/FN/TN/P/R/F1 por patologia
    • _micro_macro(df_pat)                   -> micro (pooled) + macro (nanmean)
    • _ren(df, pref)                         -> renombra P/R/F1 con prefijo

Fuentes (pre-computadas por evaluate_phase1_final_10pct.py):
    • phase1_10pct_eval_results/resultados_granulares_Config_A.csv
    • phase1_10pct_eval_results/resultados_granulares_Config_B.csv

Salida:
    • phase1_10pct_eval_results/chexpert_comparativo_phase1_10pct.csv

Autor : Evelyn Silva Rozas
Fecha : Julio 2026
"""

import os
import sys
from pathlib import Path

import pandas as pd

# Reutilizar helpers validados (import ligero: solo numpy/pandas)
from actualizar_comparativos_final import (
    CHEX_LABELS,
    ROUND,
    _metricas_pat_desde_granular,
    _micro_macro,
    _ren,
)

BASE = Path(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = BASE / "phase1_10pct_eval_results"

F_GRAN_A = EVAL_DIR / "resultados_granulares_Config_A.csv"
F_GRAN_B = EVAL_DIR / "resultados_granulares_Config_B.csv"
OUT_CSV  = EVAL_DIR / "chexpert_comparativo_phase1_10pct.csv"

SEP = "=" * 78


def _verificar(paths: dict) -> None:
    faltan = [(k, p) for k, p in paths.items() if not Path(p).exists()]
    if faltan:
        print("\nERROR: faltan archivos granulares (corre primero "
              "evaluate_phase1_final_10pct.py):")
        for k, p in faltan:
            print(f"  • {k:<12s} -> {p}")
        sys.exit(1)


def main() -> int:
    print(SEP)
    print(" CheXpert por patologia — FASE 1 (test 10% completo)")
    print(SEP)

    _verificar({"F_GRAN_A": F_GRAN_A, "F_GRAN_B": F_GRAN_B})

    print("  Reconstruyendo TP/FP/FN por patologia desde granulares "
          "(pred_chexpert_* + chexpert_match_*)...")
    pat_a = _metricas_pat_desde_granular(pd.read_csv(F_GRAN_A))
    pat_b = _metricas_pat_desde_granular(pd.read_csv(F_GRAN_B))

    # ── Seccion por patologia ──
    base = pat_a[["categoria"]].copy()
    merged = (
        base
        .merge(_ren(pat_a, "Config_A"), on="categoria", how="left")
        .merge(_ren(pat_b, "Config_B"), on="categoria", how="left")
    )
    merged.insert(0, "nivel", "por_patologia")

    # ── Seccion general (micro / macro) ──
    mm_a = _micro_macro(pat_a)
    mm_b = _micro_macro(pat_b)

    filas_gen = []
    for scope, etiqueta in [("micro", "MICRO (pooled 14 clases)"),
                            ("macro", "MACRO (nanmean 14 clases)")]:
        ap, ar, af = mm_a[scope]
        bp, br, bf = mm_b[scope]
        filas_gen.append({
            "nivel": "general",
            "categoria": etiqueta,
            "Config_A_P": ap, "Config_A_R": ar, "Config_A_F1": af,
            "Config_B_P": bp, "Config_B_R": br, "Config_B_F1": bf,
        })
    gen = pd.DataFrame(filas_gen)

    df = pd.concat([merged, gen], ignore_index=True)
    num_cols = [c for c in df.columns if c not in ("nivel", "categoria")]
    df[num_cols] = df[num_cols].astype(float).round(ROUND)

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    print(f"\n[OK] {OUT_CSV}")
    with pd.option_context("display.width", 240, "display.max_columns", 40):
        print(df.to_string(index=False))

    print("\n   [sanity — micro recomputado desde granular]")
    print(f"   Config_A micro:  P={mm_a['micro'][0]:.4f}  "
          f"R={mm_a['micro'][1]:.4f}  F1={mm_a['micro'][2]:.4f}")
    print(f"   Config_B micro:  P={mm_b['micro'][0]:.4f}  "
          f"R={mm_b['micro'][1]:.4f}  F1={mm_b['micro'][2]:.4f}")
    print("\n" + SEP + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
