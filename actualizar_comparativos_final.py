#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
actualizar_comparativos_final.py
================================
Genera los CSVs comparativos del EXPERIMENTO FINAL (RadDINO + Asclepius-R +
Puente Full-FT), consolidando sus 4 pipelines de inferencia/evaluación:

      · final_disenado_sucio        (modelo final raw, prompt diseñado)
      · final_disenado_llm_externo  (raw → Llama 3.1 8B)
      · final_nuevo_sucio           (modelo final CoT raw, prompt nuevo)
      · final_nuevo_llm_externo     (CoT → Llama 3.1 8B)

Salidas (en final_experiment_results-50pct/):
    1) metricas_comparativas_inferencias_final.csv
        Métricas (BLEU-4, ROUGE-L, F1-RadGraph, Clinical BERTScore P/R/F1) para
        las 4 variantes del Experimento FINAL.

    2) chexpert_comparativo_final.csv
        CheXpert por las 14 patologías + general (micro/macro) para las dos
        variantes DIRECTAS (sucio):  final_disenado_sucio · final_nuevo_sucio
        (la metodología del proyecto computa CheXpert sobre la salida directa del
        modelo; el post-formateo con LLM externo se evalúa con métricas de texto).

Fuentes (todo PRE-COMPUTADO; este script NO recalcula métricas pesadas):
    REQUERIDAS:
      • evaluation/predicciones_Exp_Final_llmexterno_llama8b_reeval_resumen.csv
      • exp1_cot/exp1_final_formatted_final_experiment_llama8b_reeval_resumen.csv
      • evaluation/resultados_granulares_Exp_Final.csv
      • exp1_cot/resultados_granulares_Exp_Final_CoT.csv

NOTA: NO usa entornos virtuales — corre sobre el Python del sistema. Solo
requiere numpy + pandas (no carga torch/transformers).

Autor : Evelyn Silva Rozas
Fecha : Junio 2026
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Rutas
# ─────────────────────────────────────────────────────────────────────────────
BASE = Path(os.path.dirname(os.path.abspath(__file__)))

# ── Experimento FINAL (50%) — REQUERIDAS ──
F_EVAL_DIR = BASE / "final_experiment_results-50pct" / "evaluation"
F_EXP1_DIR = BASE / "final_experiment_results-50pct" / "exp1_cot"

F_REEVAL_DISENADO = F_EVAL_DIR / "predicciones_Exp_Final_llmexterno_llama8b_reeval_resumen.csv"
F_REEVAL_COT      = F_EXP1_DIR / "exp1_final_formatted_final_experiment_llama8b_reeval_resumen.csv"
F_GRAN_DISENADO   = F_EVAL_DIR / "resultados_granulares_Exp_Final.csv"
F_GRAN_COT        = F_EXP1_DIR / "resultados_granulares_Exp_Final_CoT.csv"

# ── Salidas ──
OUT_DIR = BASE / "final_experiment_results-50pct"
OUT_METRICAS = OUT_DIR / "metricas_comparativas_inferencias_final.csv"
OUT_CHEXPERT = OUT_DIR / "chexpert_comparativo_final.csv"

ROUND = 4

# 14 patologías oficiales CheXpert (mismo orden que CHEXBERT_LABELS).
CHEX_LABELS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices",
]


def _canon(lbl: str) -> str:
    """Normaliza nombre de patología para sufijo de columna (lower + '_')."""
    return lbl.lower().replace(" ", "_")


# ═════════════════════════════════════════════════════════════════════════════
# Helpers — verificación de archivos requeridos
# ═════════════════════════════════════════════════════════════════════════════

def _verificar_archivos(paths_dict: dict) -> None:
    """Aborta con mensaje claro si falta algún archivo crítico (requerido)."""
    faltan = [(k, p) for k, p in paths_dict.items() if not Path(p).exists()]
    if faltan:
        print("\nERROR: faltan los siguientes archivos pre-requisito (Experimento FINAL):")
        for k, p in faltan:
            print(f"  • {k:<20s} → {p}")
        print("\nVerifica que las 4 ramas de métricas hayan terminado:")
        print("  1) python evaluate_final_experiment.py --skip-inference")
        print("  2) python exp1_cot_inference_final_experiment.py --skip-inference")
        print("  3) python reevaluar_sucio_vs_limpio.py --csv "
              ".../evaluation/predicciones_Exp_Final_llmexterno_llama8b.csv "
              "--col-limpio formatted_report")
        print("  4) python reevaluar_sucio_vs_limpio.py --csv "
              ".../exp1_cot/exp1_final_formatted_final_experiment_llama8b.csv "
              "--col-limpio formatted_report")
        sys.exit(1)


# ═════════════════════════════════════════════════════════════════════════════
# 1) metricas_comparativas_inferencias_final.csv
# ═════════════════════════════════════════════════════════════════════════════

def _safe_n(df: pd.DataFrame, clave: str) -> int:
    """Suma n_mejoran + n_empeoran + n_iguales para una métrica del resumen."""
    cols = ["n_mejoran", "n_empeoran", "n_iguales"]
    if all(c in df.columns for c in cols):
        return int(df.loc[clave, cols].sum())
    return -1


def construir_metricas() -> pd.DataFrame:
    """Construye la tabla comparativa de las 4 variantes × 6 métricas."""
    print("\n" + "=" * 78)
    print(" CSV 1 — metricas_comparativas_inferencias_final.csv")
    print("=" * 78)

    # reeval: media_sucio = salida directa del modelo, media_limpio = post-Llama 8B
    f_disenado = pd.read_csv(F_REEVAL_DISENADO).set_index("metrica")
    f_cot      = pd.read_csv(F_REEVAL_COT).set_index("metrica")

    orden = [
        ("bleu4",         "BLEU-4"),
        ("rougeL",        "ROUGE-L"),
        ("radgraph",      "F1-RadGraph"),
        ("bertscore_p",   "ClinicalBERT_P"),
        ("bertscore_r",   "ClinicalBERT_R"),
        ("bertscore_f1",  "ClinicalBERT_F1"),
    ]

    filas = []
    for clave, etiqueta in orden:
        filas.append({
            "metrica": etiqueta,
            "final_disenado_sucio":        round(float(f_disenado.loc[clave, "media_sucio"]),  ROUND),
            "final_disenado_llm_externo":  round(float(f_disenado.loc[clave, "media_limpio"]), ROUND),
            "final_nuevo_sucio":           round(float(f_cot.loc[clave,      "media_sucio"]),  ROUND),
            "final_nuevo_llm_externo":     round(float(f_cot.loc[clave,      "media_limpio"]), ROUND),
        })

    # n de muestras (fila de transparencia)
    n_disenado = _safe_n(f_disenado, "bleu4")
    n_cot      = _safe_n(f_cot,      "bleu4")
    filas.append({
        "metrica": "n_muestras",
        "final_disenado_sucio":        n_disenado,
        "final_disenado_llm_externo":  n_disenado,
        "final_nuevo_sucio":           n_cot,
        "final_nuevo_llm_externo":     n_cot,
    })

    df = pd.DataFrame(filas)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_METRICAS, index=False)
    print(f"\n[OK] {OUT_METRICAS}")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(df.to_string(index=False))

    print(f"\n   * Final reeval disenado (sucio/llm) = {n_disenado} muestras")
    print(f"   * Final reeval CoT      (sucio/llm) = {n_cot} muestras")
    return df


# ═════════════════════════════════════════════════════════════════════════════
# 2) chexpert_comparativo_final.csv
# ═════════════════════════════════════════════════════════════════════════════

def _micro_macro(df_pat: pd.DataFrame) -> dict:
    """Calcula micro (pooled) y macro (nanmean) a partir de TP/FP/FN por patología."""
    tp, fp, fn = float(df_pat["TP"].sum()), float(df_pat["FP"].sum()), float(df_pat["FN"].sum())
    micro_p  = tp / (tp + fp) if (tp + fp) else np.nan
    micro_r  = tp / (tp + fn) if (tp + fn) else np.nan
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) else np.nan
    macro_p  = float(np.nanmean(df_pat["precision"]))
    macro_r  = float(np.nanmean(df_pat["recall"]))
    macro_f1 = float(np.nanmean(df_pat["f1"]))
    return {
        "micro": (micro_p, micro_r, micro_f1),
        "macro": (macro_p, macro_r, macro_f1),
    }


def _metricas_pat_desde_granular(
    df_gran: pd.DataFrame,
    labels: list = CHEX_LABELS,
) -> pd.DataFrame:
    """
    Reconstruye TP/FP/FN/TN/precision/recall/f1 por patología a partir de un CSV
    granular con 'pred_chexpert_*' y 'chexpert_match_*'.

    Lógica idéntica a evaluate_phase3_puente._chexpert_comparison_metrics
    (reutilizada SIN cambios por evaluate_exp3_asclepius / Experimento FINAL):
        match = 1, pred = 1 → TP   ·   match = 1, pred = 0 → TN
        match = 0, pred = 1 → FP   ·   match = 0, pred = 0 → FN
        (pred_bin = 1 si pred != 0 → un -1 incierto cuenta como positivo).
    """
    rows = []
    for lbl in labels:
        cn = _canon(lbl)
        col_pred  = f"pred_chexpert_{cn}"
        col_match = f"chexpert_match_{cn}"
        if col_pred not in df_gran.columns or col_match not in df_gran.columns:
            rows.append({"categoria": lbl, "TP": 0, "FP": 0, "FN": 0, "TN": 0,
                         "precision": np.nan, "recall": np.nan, "f1": np.nan})
            continue

        pred_raw  = pd.to_numeric(df_gran[col_pred],  errors="coerce")
        match_raw = pd.to_numeric(df_gran[col_match], errors="coerce")

        valid = match_raw.notna()
        pred  = pred_raw[valid].fillna(0)
        match = match_raw[valid]

        pred_bin = (pred != 0).astype(int)

        tp = int(((match == 1) & (pred_bin == 1)).sum())
        tn = int(((match == 1) & (pred_bin == 0)).sum())
        fp = int(((match == 0) & (pred_bin == 1)).sum())
        fn = int(((match == 0) & (pred_bin == 0)).sum())

        precision = tp / (tp + fp) if (tp + fp) else np.nan
        recall    = tp / (tp + fn) if (tp + fn) else np.nan
        f1 = (
            2 * precision * recall / (precision + recall)
            if (not np.isnan(precision) and not np.isnan(recall) and (precision + recall) > 0)
            else np.nan
        )
        rows.append({
            "categoria": lbl,
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision": precision, "recall": recall, "f1": f1,
        })
    return pd.DataFrame(rows)


def _ren(df: pd.DataFrame, pref: str) -> pd.DataFrame:
    """Renombra precision/recall/f1 con prefijo de variante (P/R/F1)."""
    return df.rename(columns={
        "precision": f"{pref}_P",
        "recall":    f"{pref}_R",
        "f1":        f"{pref}_F1",
    })[["categoria", f"{pref}_P", f"{pref}_R", f"{pref}_F1"]]


def construir_chexpert() -> pd.DataFrame:
    """Construye la tabla comparativa CheXpert (por patología + general)."""
    print("\n" + "=" * 78)
    print(" CSV 2 — chexpert_comparativo_final.csv")
    print("=" * 78)

    print("  Calculando Experimento FINAL desde granulares "
          "(chexpert_match_* + pred_chexpert_*)...")
    f_dsuc = _metricas_pat_desde_granular(pd.read_csv(F_GRAN_DISENADO))
    f_esuc = _metricas_pat_desde_granular(pd.read_csv(F_GRAN_COT))

    # ── Sección por patología (merge sobre categoria) ──
    base = f_dsuc[["categoria"]].copy()
    merged = (
        base
        .merge(_ren(f_dsuc, "final_disenado_sucio"), on="categoria", how="left")
        .merge(_ren(f_esuc, "final_nuevo_sucio"),    on="categoria", how="left")
    )
    merged.insert(0, "nivel", "por_patologia")

    # ── Sección general (micro / macro) ──
    mm_dsuc = _micro_macro(f_dsuc)
    mm_esuc = _micro_macro(f_esuc)

    filas_gen = []
    for scope, etiqueta in [("micro", "MICRO (pooled 14 clases)"),
                            ("macro", "MACRO (nanmean 14 clases)")]:
        dp, dr, df_ = mm_dsuc[scope]
        ep, er, ef = mm_esuc[scope]
        filas_gen.append({
            "nivel": "general",
            "categoria": etiqueta,
            "final_disenado_sucio_P": dp, "final_disenado_sucio_R": dr, "final_disenado_sucio_F1": df_,
            "final_nuevo_sucio_P":    ep, "final_nuevo_sucio_R":    er, "final_nuevo_sucio_F1":    ef,
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

    print("\n   [sanity Experimento FINAL — recomputado desde granular]")
    print(f"   Final disenado_sucio    micro:"
          f"  P={mm_dsuc['micro'][0]:.4f}  R={mm_dsuc['micro'][1]:.4f}"
          f"  F1={mm_dsuc['micro'][2]:.4f}")
    print(f"   Final nuevo_sucio (CoT) micro:"
          f"  P={mm_esuc['micro'][0]:.4f}  R={mm_esuc['micro'][1]:.4f}"
          f"  F1={mm_esuc['micro'][2]:.4f}")
    return df


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main() -> int:
    archivos_requeridos = {
        "F_REEVAL_DISENADO": F_REEVAL_DISENADO,
        "F_REEVAL_COT":      F_REEVAL_COT,
        "F_GRAN_DISENADO":   F_GRAN_DISENADO,
        "F_GRAN_COT":        F_GRAN_COT,
    }
    _verificar_archivos(archivos_requeridos)

    construir_metricas()
    construir_chexpert()

    print("\n" + "=" * 78)
    print(" Comparativos del Experimento FINAL (RadDINO + Asclepius-R) generados:")
    print(f"   • {OUT_METRICAS}")
    print(f"   • {OUT_CHEXPERT}")
    print("=" * 78 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
