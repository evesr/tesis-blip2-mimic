#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
actualizar_comparativos_final_tier1_cot.py
==========================================
Genera los CSVs comparativos del EXPERIMENTO FINAL **TIER 1** para el
PROMPT NUEVO / CoT (RadDINO + Asclepius-R + Puente Full-FT + cabeza clínica
auxiliar CheXpert), comparando la salida DIRECTA del modelo (cruda) contra la
versión FORMATEADA por el LLM externo (Llama 3.1 8B):

      · final_tier1_cot_crudo       (modelo TIER 1 raw, prompt NUEVO/CoT)
      · final_tier1_cot_formateado  (raw → Llama 3.1 8B)

Salidas (en final_experiment_tier1_results-50pct/):
    1) metricas_comparativas_inferencias_final_tier1_cot.csv
        Métricas de texto (BLEU-4, ROUGE-L, F1-RadGraph, Clinical BERTScore
        P/R/F1) CRUDO vs FORMATEADO, con delta medio pareado por reporte,
        % de mejora y conteos ↑mejoran/↓empeoran/=iguales.

    2) chexpert_comparativo_final_tier1_cot.csv
        CheXpert por las 14 patologías + general (micro/macro) para AMBAS
        versiones (crudo y formateado), con la MISMA agregación pooled
        (TP/FP/FN) validada en el resto del proyecto.

Fuentes (todo PRE-COMPUTADO por exp1_cot_inference_final_experiment_tier1.py +
evaluate_final_experiment_tier1.py --eval-formatted-csv; este script NO recalcula
métricas pesadas — solo numpy + pandas):
    REQUERIDAS (crudo):
      • exp1_cot/resultados_granulares_Exp_Final_Tier1_CoT.csv
    REQUERIDAS (formateado):
      • exp1_cot/resultados_granulares_Exp_Final_Tier1_CoT_formateado.csv

REUTILIZACIÓN: importa TAL CUAL la lógica de agregación CheXpert ya validada de
``actualizar_comparativos_final`` (misma definición TP/FP/FN, mismas 14
patologías, mismo micro/macro) → el comparativo CoT TIER 1 es DIRECTAMENTE
comparable con chexpert_comparativo_final_tier1.csv (prompt diseñado) y los demás.

NOTA: NO usa entornos virtuales — corre sobre el Python del sistema.

Autor : Evelyn Silva Rozas
Fecha : Julio 2026
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Lógica de agregación CheXpert REUTILIZADA SIN CAMBIOS del comparativo FINAL ─
from actualizar_comparativos_final import (
    CHEX_LABELS,
    ROUND,
    _micro_macro,
    _metricas_pat_desde_granular,
    _ren,
)

# ─────────────────────────────────────────────────────────────────────────────
# Rutas
# ─────────────────────────────────────────────────────────────────────────────
BASE = Path(os.path.dirname(os.path.abspath(__file__)))

F_EXP1_DIR = BASE / "final_experiment_tier1_results-50pct" / "exp1_cot"

F_GRAN_CRUDO      = F_EXP1_DIR / "resultados_granulares_Exp_Final_Tier1_CoT.csv"
F_GRAN_FORMATEADO = F_EXP1_DIR / "resultados_granulares_Exp_Final_Tier1_CoT_formateado.csv"

OUT_DIR = BASE / "final_experiment_tier1_results-50pct"
OUT_METRICAS = OUT_DIR / "metricas_comparativas_inferencias_final_tier1_cot.csv"
OUT_CHEXPERT = OUT_DIR / "chexpert_comparativo_final_tier1_cot.csv"

# Métricas de texto presentes en el CSV granular (nombre_columna → etiqueta).
TEXT_METRICS = [
    ("bleu_4",       "BLEU-4"),
    ("rouge_l",      "ROUGE-L"),
    ("radgraph_f1",  "F1-RadGraph"),
    ("bertscore_p",  "ClinicalBERT_P"),
    ("bertscore_r",  "ClinicalBERT_R"),
    ("bertscore_f1", "ClinicalBERT_F1"),
]


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════
def _verificar_archivos(paths_dict: dict) -> None:
    faltan = [(k, p) for k, p in paths_dict.items() if not Path(p).exists()]
    if faltan:
        print("\nERROR: faltan los siguientes archivos pre-requisito (FINAL TIER 1 · CoT):")
        for k, p in faltan:
            print(f"  • {k:<18s} → {p}")
        print("\nVerifica que la evaluación CoT TIER 1 haya terminado:")
        print("  1) python exp1_cot_inference_final_experiment_tier1.py --resume")
        print("  2) python vllm_formatter_tier1.py --tasks nuevo --resume")
        print("  3) python evaluate_final_experiment_tier1.py --eval-formatted-csv "
              ".../exp1_final_formatted_final_experiment_tier1_llama8b.csv \\")
        print("       --out-granular .../resultados_granulares_Exp_Final_Tier1_CoT_formateado.csv \\")
        print("       --out-global   .../metricas_globales_Exp_Final_Tier1_CoT_formateado.csv")
        sys.exit(1)


# ═════════════════════════════════════════════════════════════════════════════
# 1) metricas_comparativas_inferencias_final_tier1_cot.csv  (delta PAREADO)
# ═════════════════════════════════════════════════════════════════════════════
def construir_metricas() -> pd.DataFrame:
    print("\n" + "=" * 78)
    print(" CSV 1 — metricas_comparativas_inferencias_final_tier1_cot.csv")
    print("=" * 78)

    crudo = pd.read_csv(F_GRAN_CRUDO)
    fmt   = pd.read_csv(F_GRAN_FORMATEADO)

    # Unir por dicom_id para el análisis pareado (mismo reporte antes/después).
    key = "dicom_id" if ("dicom_id" in crudo.columns and "dicom_id" in fmt.columns) else None
    if key is not None:
        cols_c = [key] + [c for c, _ in TEXT_METRICS if c in crudo.columns]
        cols_f = [key] + [c for c, _ in TEXT_METRICS if c in fmt.columns]
        m = crudo[cols_c].merge(fmt[cols_f], on=key, suffixes=("_crudo", "_fmt"))
        n = len(m)
    else:
        n = min(len(crudo), len(fmt))
        m = pd.DataFrame(index=range(n))
        for c, _ in TEXT_METRICS:
            if c in crudo.columns:
                m[f"{c}_crudo"] = crudo[c].values[:n]
            if c in fmt.columns:
                m[f"{c}_fmt"] = fmt[c].values[:n]

    filas = []
    for col, etiqueta in TEXT_METRICS:
        cc, cf = f"{col}_crudo", f"{col}_fmt"
        if cc not in m.columns or cf not in m.columns:
            continue
        s = pd.to_numeric(m[cc], errors="coerce")
        l = pd.to_numeric(m[cf], errors="coerce")
        d = l - s
        media_c = float(np.nanmean(s))
        media_f = float(np.nanmean(l))
        delta   = float(np.nanmean(d))
        filas.append({
            "metrica": etiqueta,
            "final_tier1_cot_crudo":      round(media_c, ROUND),
            "final_tier1_cot_formateado": round(media_f, ROUND),
            "delta_medio":  round(delta, ROUND),
            "mejora_pct":   round(100.0 * delta / abs(media_c), 2) if media_c not in (0.0,) and not np.isnan(media_c) else np.nan,
            "n_mejoran":    int((d > 0).sum()),
            "n_empeoran":   int((d < 0).sum()),
            "n_iguales":    int((d == 0).sum()),
        })

    filas.append({
        "metrica": "n_muestras",
        "final_tier1_cot_crudo":      n,
        "final_tier1_cot_formateado": n,
        "delta_medio": "", "mejora_pct": "",
        "n_mejoran": "", "n_empeoran": "", "n_iguales": "",
    })

    df = pd.DataFrame(filas)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_METRICAS, index=False)
    print(f"\n[OK] {OUT_METRICAS}")
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(df.to_string(index=False))
    print(f"\n   * n muestras pareadas (crudo↔formateado) = {n:,}")
    return df


# ═════════════════════════════════════════════════════════════════════════════
# 2) chexpert_comparativo_final_tier1_cot.csv  (crudo + formateado)
# ═════════════════════════════════════════════════════════════════════════════
def construir_chexpert() -> pd.DataFrame:
    print("\n" + "=" * 78)
    print(" CSV 2 — chexpert_comparativo_final_tier1_cot.csv")
    print("=" * 78)

    print("  Calculando CheXpert desde granulares "
          "(chexpert_match_* + pred_chexpert_*)...")
    pat_crudo = _metricas_pat_desde_granular(pd.read_csv(F_GRAN_CRUDO))
    pat_fmt   = _metricas_pat_desde_granular(pd.read_csv(F_GRAN_FORMATEADO))

    # ── Sección por patología (merge sobre categoria) ──
    base = pat_crudo[["categoria"]].copy()
    merged = (
        base
        .merge(_ren(pat_crudo, "final_tier1_cot_crudo"),      on="categoria", how="left")
        .merge(_ren(pat_fmt,   "final_tier1_cot_formateado"), on="categoria", how="left")
    )
    merged.insert(0, "nivel", "por_patologia")

    # ── Sección general (micro / macro) ──
    mm_crudo = _micro_macro(pat_crudo)
    mm_fmt   = _micro_macro(pat_fmt)

    filas_gen = []
    for scope, etiqueta in [("micro", "MICRO (pooled 14 clases)"),
                            ("macro", "MACRO (nanmean 14 clases)")]:
        cp, cr, cf = mm_crudo[scope]
        fp, fr, ff = mm_fmt[scope]
        filas_gen.append({
            "nivel": "general",
            "categoria": etiqueta,
            "final_tier1_cot_crudo_P": cp, "final_tier1_cot_crudo_R": cr, "final_tier1_cot_crudo_F1": cf,
            "final_tier1_cot_formateado_P": fp, "final_tier1_cot_formateado_R": fr, "final_tier1_cot_formateado_F1": ff,
        })
    gen = pd.DataFrame(filas_gen)

    df = pd.concat([merged, gen], ignore_index=True)
    num_cols = [c for c in df.columns if c not in ("nivel", "categoria")]
    df[num_cols] = df[num_cols].astype(float).round(ROUND)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CHEXPERT, index=False)
    print(f"\n[OK] {OUT_CHEXPERT}")
    with pd.option_context("display.width", 240, "display.max_columns", 40):
        print(df.to_string(index=False))

    print("\n   [sanity FINAL TIER 1 · CoT — recomputado desde granular]")
    print(f"   crudo       micro:  P={mm_crudo['micro'][0]:.4f}  R={mm_crudo['micro'][1]:.4f}  F1={mm_crudo['micro'][2]:.4f}")
    print(f"   formateado  micro:  P={mm_fmt['micro'][0]:.4f}  R={mm_fmt['micro'][1]:.4f}  F1={mm_fmt['micro'][2]:.4f}")
    return df


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════
def main() -> int:
    _verificar_archivos({
        "F_GRAN_CRUDO":      F_GRAN_CRUDO,
        "F_GRAN_FORMATEADO": F_GRAN_FORMATEADO,
    })

    construir_metricas()
    construir_chexpert()

    print("\n" + "=" * 78)
    print(" Comparativos del Experimento FINAL TIER 1 · CoT generados:")
    print(f"   • {OUT_METRICAS}")
    print(f"   • {OUT_CHEXPERT}")
    print("=" * 78 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
