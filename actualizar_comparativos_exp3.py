#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
actualizar_comparativos_exp3.py
===============================
Genera los CSVs comparativos del EXPERIMENTO 3 (ViT Fase 3 + Asclepius-R +
Puente), consolidando sus 4 pipelines de inferencia y — cuando estén
disponibles — los de FASE 3 PUENTE (OPT-2.7b) y EXPERIMENTO 2 (RadDINO) para
comparar DIRECTAMENTE el efecto de sustituir el LLM por Asclepius-R.

Salidas (en exp3_asclepius_results-50pct/):
    1) metricas_comparativas_inferencias_exp3.csv
        Métricas (BLEU-4, ROUGE-L, F1-RadGraph, Clinical BERTScore P/R/F1) para
        las 4 variantes del Experimento 3:
              · exp3_disenado_sucio          (Asclepius raw, prompt diseñado)
              · exp3_disenado_llm_externo    (raw → Llama 3.1 8B)
              · exp3_nuevo_sucio             (Asclepius CoT raw, prompt nuevo)
              · exp3_nuevo_llm_externo       (CoT → Llama 3.1 8B)
        + (si existen) las homólogas de Fase 3 PUENTE y/o Experimento 2.

    2) chexpert_comparativo_exp3.csv
        CheXpert por las 14 patologías + general (micro/macro) para:
              · exp3_disenado_sucio   · exp3_nuevo_sucio
        + (si existen) phase3_*_sucio y/o exp2_*_sucio.

Fuentes (todo PRE-COMPUTADO; este script NO recalcula métricas pesadas):
    Experimento 3 (REQUERIDAS):
      • exp3_asclepius_results-50pct/evaluation/predicciones_Exp3_Asclepius_llmexterno_llama8b_reeval_resumen.csv
      • exp3_asclepius_results-50pct/exp1_cot/exp1_final_formatted_exp3_asclepius_llama8b_reeval_resumen.csv
      • exp3_asclepius_results-50pct/evaluation/resultados_granulares_Exp3_Asclepius.csv
      • exp3_asclepius_results-50pct/exp1_cot/resultados_granulares_Exp3_Asclepius_CoT.csv

    Fase 3 PUENTE / Experimento 2 (OPCIONALES — se incluyen si están presentes).

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

# ── Experimento 3 (50%) — REQUERIDAS ──
E3_EVAL_DIR = BASE / "exp3_asclepius_results-50pct" / "evaluation"
E3_EXP1_DIR = BASE / "exp3_asclepius_results-50pct" / "exp1_cot"

E3_REEVAL_DISENADO = E3_EVAL_DIR / "predicciones_Exp3_Asclepius_llmexterno_llama8b_reeval_resumen.csv"
E3_REEVAL_COT      = E3_EXP1_DIR / "exp1_final_formatted_exp3_asclepius_llama8b_reeval_resumen.csv"
E3_GRAN_DISENADO   = E3_EVAL_DIR / "resultados_granulares_Exp3_Asclepius.csv"
E3_GRAN_COT        = E3_EXP1_DIR / "resultados_granulares_Exp3_Asclepius_CoT.csv"

# ── Fase 3 PUENTE (50%) — OPCIONALES (comparación) ──
P3_EVAL_DIR = BASE / "phase3_puente_results-50pct" / "evaluation"
P3_EXP1_DIR = BASE / "phase3_puente_results-50pct" / "exp1_cot"

P3_REEVAL_DISENADO = P3_EVAL_DIR / "predicciones_Phase3_Puente_llmexterno_llama8b_reeval_resumen.csv"
P3_REEVAL_COT      = P3_EXP1_DIR / "exp1_final_formatted_phase3_puente_llama8b_reeval_resumen.csv"
P3_GRAN_DISENADO   = P3_EVAL_DIR / "resultados_granulares_Phase3_Puente.csv"
P3_GRAN_COT        = P3_EXP1_DIR / "resultados_granulares_Phase3_Puente_CoT.csv"

# ── Experimento 2 RadDINO (50%) — OPCIONALES (comparación) ──
E2_EVAL_DIR = BASE / "exp2_raddino_results-50pct" / "evaluation"
E2_EXP1_DIR = BASE / "exp2_raddino_results-50pct" / "exp1_cot"

E2_REEVAL_DISENADO = E2_EVAL_DIR / "predicciones_Exp2_RadDINO_llmexterno_llama8b_reeval_resumen.csv"
E2_REEVAL_COT      = E2_EXP1_DIR / "exp1_final_formatted_exp2_raddino_llama8b_reeval_resumen.csv"
E2_GRAN_DISENADO   = E2_EVAL_DIR / "resultados_granulares_Exp2_RadDINO.csv"
E2_GRAN_COT        = E2_EXP1_DIR / "resultados_granulares_Exp2_RadDINO_CoT.csv"

# ── Salidas ──
OUT_DIR = BASE / "exp3_asclepius_results-50pct"
OUT_METRICAS = OUT_DIR / "metricas_comparativas_inferencias_exp3.csv"
OUT_CHEXPERT = OUT_DIR / "chexpert_comparativo_exp3.csv"

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
# Helpers — verificación de archivos requeridos / comparaciones opcionales
# ═════════════════════════════════════════════════════════════════════════════

def _verificar_archivos(paths_dict: dict) -> None:
    """Aborta con mensaje claro si falta algún archivo crítico (requerido)."""
    faltan = [(k, p) for k, p in paths_dict.items() if not Path(p).exists()]
    if faltan:
        print("\nERROR: faltan los siguientes archivos pre-requisito (Experimento 3):")
        for k, p in faltan:
            print(f"  • {k:<24s} → {p}")
        print("\nVerifica que las etapas previas del Experimento 3 hayan terminado:")
        print("  1) bash run_evaluate_exp3_asclepius_tmux.sh")
        print("  2) bash run_vllm_formatter_tmux.sh --tasks=disenado_exp3asclepius")
        print("  3) bash run_reevaluar_exp3_asclepius_disenado_tmux.sh")
        print("  4) bash run_exp1_cot_exp3_asclepius_tmux.sh")
        print("  5) bash run_vllm_formatter_tmux.sh --tasks=nuevo_exp3asclepius")
        print("  6) bash run_reevaluar_exp3_asclepius_cot_tmux.sh")
        sys.exit(1)


def _phase3_disponible() -> bool:
    return all(Path(p).exists() for p in (
        P3_REEVAL_DISENADO, P3_REEVAL_COT, P3_GRAN_DISENADO, P3_GRAN_COT
    ))


def _exp2_disponible() -> bool:
    return all(Path(p).exists() for p in (
        E2_REEVAL_DISENADO, E2_REEVAL_COT, E2_GRAN_DISENADO, E2_GRAN_COT
    ))


# ═════════════════════════════════════════════════════════════════════════════
# 1) metricas_comparativas_inferencias_exp3.csv
# ═════════════════════════════════════════════════════════════════════════════

def _safe_n(df: pd.DataFrame, clave: str) -> int:
    """Suma n_mejoran + n_empeoran + n_iguales para una métrica del resumen."""
    cols = ["n_mejoran", "n_empeoran", "n_iguales"]
    if all(c in df.columns for c in cols):
        return int(df.loc[clave, cols].sum())
    return -1


def construir_metricas(incluir_p3: bool, incluir_e2: bool) -> pd.DataFrame:
    """Construye la tabla comparativa de variantes × 6 métricas."""
    print("\n" + "=" * 78)
    print(" CSV 1 — metricas_comparativas_inferencias_exp3.csv")
    print("=" * 78)

    # reeval: media_sucio = Asclepius crudo, media_limpio = post-Llama 8B
    e3_disenado = pd.read_csv(E3_REEVAL_DISENADO).set_index("metrica")
    e3_cot      = pd.read_csv(E3_REEVAL_COT).set_index("metrica")

    if incluir_p3:
        p3_disenado = pd.read_csv(P3_REEVAL_DISENADO).set_index("metrica")
        p3_cot      = pd.read_csv(P3_REEVAL_COT).set_index("metrica")
    if incluir_e2:
        e2_disenado = pd.read_csv(E2_REEVAL_DISENADO).set_index("metrica")
        e2_cot      = pd.read_csv(E2_REEVAL_COT).set_index("metrica")

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
        fila = {
            "metrica": etiqueta,
            # ── Experimento 3 (Asclepius-R) ──
            "exp3_disenado_sucio":        round(float(e3_disenado.loc[clave, "media_sucio"]),  ROUND),
            "exp3_disenado_llm_externo":  round(float(e3_disenado.loc[clave, "media_limpio"]), ROUND),
            "exp3_nuevo_sucio":           round(float(e3_cot.loc[clave,      "media_sucio"]),  ROUND),
            "exp3_nuevo_llm_externo":     round(float(e3_cot.loc[clave,      "media_limpio"]), ROUND),
        }
        if incluir_p3:
            fila.update({
                "phase3_disenado_sucio":       round(float(p3_disenado.loc[clave, "media_sucio"]),  ROUND),
                "phase3_disenado_llm_externo": round(float(p3_disenado.loc[clave, "media_limpio"]), ROUND),
                "phase3_nuevo_sucio":          round(float(p3_cot.loc[clave,      "media_sucio"]),  ROUND),
                "phase3_nuevo_llm_externo":    round(float(p3_cot.loc[clave,      "media_limpio"]), ROUND),
            })
        if incluir_e2:
            fila.update({
                "exp2_disenado_sucio":       round(float(e2_disenado.loc[clave, "media_sucio"]),  ROUND),
                "exp2_disenado_llm_externo": round(float(e2_disenado.loc[clave, "media_limpio"]), ROUND),
                "exp2_nuevo_sucio":          round(float(e2_cot.loc[clave,      "media_sucio"]),  ROUND),
                "exp2_nuevo_llm_externo":    round(float(e2_cot.loc[clave,      "media_limpio"]), ROUND),
            })
        filas.append(fila)

    # n de muestras (fila de transparencia)
    n_e3_disenado = _safe_n(e3_disenado, "bleu4")
    n_e3_cot      = _safe_n(e3_cot,      "bleu4")
    fila_n = {
        "metrica": "n_muestras",
        "exp3_disenado_sucio":        n_e3_disenado,
        "exp3_disenado_llm_externo":  n_e3_disenado,
        "exp3_nuevo_sucio":           n_e3_cot,
        "exp3_nuevo_llm_externo":     n_e3_cot,
    }
    if incluir_p3:
        n_p3_disenado = _safe_n(p3_disenado, "bleu4")
        n_p3_cot      = _safe_n(p3_cot,      "bleu4")
        fila_n.update({
            "phase3_disenado_sucio":       n_p3_disenado,
            "phase3_disenado_llm_externo": n_p3_disenado,
            "phase3_nuevo_sucio":          n_p3_cot,
            "phase3_nuevo_llm_externo":    n_p3_cot,
        })
    if incluir_e2:
        n_e2_disenado = _safe_n(e2_disenado, "bleu4")
        n_e2_cot      = _safe_n(e2_cot,      "bleu4")
        fila_n.update({
            "exp2_disenado_sucio":       n_e2_disenado,
            "exp2_disenado_llm_externo": n_e2_disenado,
            "exp2_nuevo_sucio":          n_e2_cot,
            "exp2_nuevo_llm_externo":    n_e2_cot,
        })
    filas.append(fila_n)

    df = pd.DataFrame(filas)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_METRICAS, index=False)
    print(f"\n[OK] {OUT_METRICAS}")
    with pd.option_context("display.width", 280, "display.max_columns", 40):
        print(df.to_string(index=False))

    print(f"\n   * Exp3 reeval disenado (sucio/llm)  = {n_e3_disenado} muestras")
    print(f"   * Exp3 reeval CoT      (sucio/llm)  = {n_e3_cot} muestras")
    print(f"   * Fase 3 PUENTE comparación         = {'SÍ' if incluir_p3 else 'NO (faltan CSV)'}")
    print(f"   * Experimento 2 comparación         = {'SÍ' if incluir_e2 else 'NO (faltan CSV)'}")
    return df


# ═════════════════════════════════════════════════════════════════════════════
# 2) chexpert_comparativo_exp3.csv
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
    granular (Exp3 / Fase 3 / Exp2) con 'pred_chexpert_*' y 'chexpert_match_*'.

    Lógica idéntica a evaluate_phase3_puente._chexpert_comparison_metrics
    (reutilizada SIN cambios por evaluate_exp3_asclepius):
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


def construir_chexpert(incluir_p3: bool, incluir_e2: bool) -> pd.DataFrame:
    """Construye la tabla comparativa CheXpert (por patología + general)."""
    print("\n" + "=" * 78)
    print(" CSV 2 — chexpert_comparativo_exp3.csv")
    print("=" * 78)

    print("  Calculando Experimento 3 desde granulares (chexpert_match_* + pred_chexpert_*)...")
    e3_dsuc = _metricas_pat_desde_granular(pd.read_csv(E3_GRAN_DISENADO))
    e3_esuc = _metricas_pat_desde_granular(pd.read_csv(E3_GRAN_COT))

    if incluir_p3:
        print("  Calculando Fase 3 PUENTE desde granulares (comparación)...")
        p3_dsuc = _metricas_pat_desde_granular(pd.read_csv(P3_GRAN_DISENADO))
        p3_esuc = _metricas_pat_desde_granular(pd.read_csv(P3_GRAN_COT))
    if incluir_e2:
        print("  Calculando Experimento 2 desde granulares (comparación)...")
        e2_dsuc = _metricas_pat_desde_granular(pd.read_csv(E2_GRAN_DISENADO))
        e2_esuc = _metricas_pat_desde_granular(pd.read_csv(E2_GRAN_COT))

    # ── Sección por patología (merge sobre categoria) ──
    base = e3_dsuc[["categoria"]].copy()
    merged = (
        base
        .merge(_ren(e3_dsuc, "exp3_disenado_sucio"), on="categoria", how="left")
        .merge(_ren(e3_esuc, "exp3_nuevo_sucio"),    on="categoria", how="left")
    )
    if incluir_p3:
        merged = (
            merged
            .merge(_ren(p3_dsuc, "phase3_disenado_sucio"), on="categoria", how="left")
            .merge(_ren(p3_esuc, "phase3_nuevo_sucio"),    on="categoria", how="left")
        )
    if incluir_e2:
        merged = (
            merged
            .merge(_ren(e2_dsuc, "exp2_disenado_sucio"), on="categoria", how="left")
            .merge(_ren(e2_esuc, "exp2_nuevo_sucio"),    on="categoria", how="left")
        )
    merged.insert(0, "nivel", "por_patologia")

    # ── Sección general (micro / macro) ──
    mm_e3_dsuc = _micro_macro(e3_dsuc)
    mm_e3_esuc = _micro_macro(e3_esuc)
    if incluir_p3:
        mm_p3_dsuc = _micro_macro(p3_dsuc)
        mm_p3_esuc = _micro_macro(p3_esuc)
    if incluir_e2:
        mm_e2_dsuc = _micro_macro(e2_dsuc)
        mm_e2_esuc = _micro_macro(e2_esuc)

    filas_gen = []
    for scope, etiqueta in [("micro", "MICRO (pooled 14 clases)"),
                            ("macro", "MACRO (nanmean 14 clases)")]:
        e3dp, e3dr, e3df = mm_e3_dsuc[scope]
        e3ep, e3er, e3ef = mm_e3_esuc[scope]
        fila = {
            "nivel": "general",
            "categoria": etiqueta,
            "exp3_disenado_sucio_P": e3dp, "exp3_disenado_sucio_R": e3dr, "exp3_disenado_sucio_F1": e3df,
            "exp3_nuevo_sucio_P":    e3ep, "exp3_nuevo_sucio_R":    e3er, "exp3_nuevo_sucio_F1":    e3ef,
        }
        if incluir_p3:
            p3dp, p3dr, p3df = mm_p3_dsuc[scope]
            p3ep, p3er, p3ef = mm_p3_esuc[scope]
            fila.update({
                "phase3_disenado_sucio_P": p3dp, "phase3_disenado_sucio_R": p3dr, "phase3_disenado_sucio_F1": p3df,
                "phase3_nuevo_sucio_P":    p3ep, "phase3_nuevo_sucio_R":    p3er, "phase3_nuevo_sucio_F1":    p3ef,
            })
        if incluir_e2:
            e2dp, e2dr, e2df = mm_e2_dsuc[scope]
            e2ep, e2er, e2ef = mm_e2_esuc[scope]
            fila.update({
                "exp2_disenado_sucio_P": e2dp, "exp2_disenado_sucio_R": e2dr, "exp2_disenado_sucio_F1": e2df,
                "exp2_nuevo_sucio_P":    e2ep, "exp2_nuevo_sucio_R":    e2er, "exp2_nuevo_sucio_F1":    e2ef,
            })
        filas_gen.append(fila)
    gen = pd.DataFrame(filas_gen)

    df = pd.concat([merged, gen], ignore_index=True)

    num_cols = [c for c in df.columns if c not in ("nivel", "categoria")]
    df[num_cols] = df[num_cols].astype(float).round(ROUND)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CHEXPERT, index=False)
    print(f"\n[OK] {OUT_CHEXPERT}")
    with pd.option_context("display.width", 320, "display.max_columns", 60):
        print(df.to_string(index=False))

    print("\n   [sanity Experimento 3 — recomputado desde granular]")
    print(f"   Exp3 disenado_sucio    micro:"
          f"  P={mm_e3_dsuc['micro'][0]:.4f}  R={mm_e3_dsuc['micro'][1]:.4f}"
          f"  F1={mm_e3_dsuc['micro'][2]:.4f}")
    print(f"   Exp3 nuevo_sucio (CoT) micro:"
          f"  P={mm_e3_esuc['micro'][0]:.4f}  R={mm_e3_esuc['micro'][1]:.4f}"
          f"  F1={mm_e3_esuc['micro'][2]:.4f}")
    return df


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main() -> int:
    archivos_requeridos = {
        "E3_REEVAL_DISENADO": E3_REEVAL_DISENADO,
        "E3_REEVAL_COT":      E3_REEVAL_COT,
        "E3_GRAN_DISENADO":   E3_GRAN_DISENADO,
        "E3_GRAN_COT":        E3_GRAN_COT,
    }
    _verificar_archivos(archivos_requeridos)

    incluir_p3 = _phase3_disponible()
    incluir_e2 = _exp2_disponible()
    print(f"  Fase 3 PUENTE  : {'detectada → incluida' if incluir_p3 else 'NO detectada'}")
    print(f"  Experimento 2  : {'detectado → incluido' if incluir_e2 else 'NO detectado'}")

    construir_metricas(incluir_p3, incluir_e2)
    construir_chexpert(incluir_p3, incluir_e2)

    print("\n" + "=" * 78)
    print(" Comparativos del Experimento 3 (Asclepius-R) generados:")
    print(f"   • {OUT_METRICAS}")
    print(f"   • {OUT_CHEXPERT}")
    print("=" * 78 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
