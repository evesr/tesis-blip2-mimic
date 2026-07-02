#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
actualizar_comparativos_phase3.py
=================================
Genera los CSVs comparativos COMBINANDO Fase 2 (5 pipelines pre-existentes) con
Fase 3 PUENTE (4 pipelines nuevos), permitiendo comparar TODA la evolución del
proyecto en una sola tabla.

Salidas (en phase3_puente_results-50pct/):
    1) metricas_comparativas_inferencias_phase3.csv
        Métricas (BLEU-4, ROUGE-L, F1-RadGraph, Clinical BERTScore P/R/F1) para
        9 variantes:
            ── Fase 2 (50%) — modelo dual-LoRA Config_B1 ──
              · fase2_disenado_sucio          (raw BLIP-2)
              · fase2_disenado_limpio         (post-guillotina Tail Babbling)
              · fase2_disenado_llm_externo    (raw → Llama 3.1 8B)
              · fase2_nuevo_sucio             (CoT raw)
              · fase2_nuevo_limpio            (CoT → Llama 3.1 8B)
            ── Fase 3 PUENTE (50%) — qformer + lp + queries entrenados ──
              · phase3_disenado_sucio         (PUENTE raw)
              · phase3_disenado_llm_externo   (PUENTE raw → Llama 3.1 8B)
              · phase3_nuevo_sucio            (PUENTE CoT raw)
              · phase3_nuevo_llm_externo      (PUENTE CoT → Llama 3.1 8B)

        NOTA: Phase 3 PUENTE NO incluye una variante "limpio" post-guillotina:
        el entrenamiento del puente ya elimina el Tail Babbling estructural,
        por lo que no se aplica `limpiar_tail_babbling` (vs el modelo Fase 2
        dual-LoRA que sí lo necesita). El "limpio" en Phase 3 corresponde
        exclusivamente al pulido final del Llama 8B externo.

    2) chexpert_comparativo_phase3.csv
        CheXpert por patología + general (micro/macro) para 5 variantes:
            · fase2_disenado_sucio   · fase2_disenado_limpio   · fase2_nuevo_sucio
            · phase3_disenado_sucio  · phase3_nuevo_sucio
        (No se incluye CheXpert sobre los reportes formateados por Llama 8B,
        igual que el comparativo original de Fase 2.)

Fuentes (todo PRE-COMPUTADO; este script NO recalcula métricas pesadas):
    Fase 2:
      • phase2_results-50pct/evaluation/predicciones_Config_B1_clean_reeval_resumen.csv
      • phase2_results-50pct/evaluation/predicciones_Config_B1_llmexterno_metrics_resumen.csv
      • phase2_results-50pct/exp1_cot/exp1_comparison_resumen.csv
      • phase2_results-50pct/evaluation/predicciones_Config_B1_metricas_estrictas.csv
      • phase2_results-50pct/evaluation/predicciones_Config_B1_metricas_estrictas_SUCIO.csv
      • phase2_results-50pct/exp1_cot/exp1_chexpert_per_pathology.csv

    Fase 3 PUENTE:
      • phase3_puente_results-50pct/evaluation/predicciones_Phase3_Puente_llmexterno_llama8b_reeval_resumen.csv
      • phase3_puente_results-50pct/exp1_cot/exp1_final_formatted_phase3_puente_llama8b_reeval_resumen.csv
      • phase3_puente_results-50pct/evaluation/resultados_granulares_Phase3_Puente.csv
      • phase3_puente_results-50pct/exp1_cot/resultados_granulares_Phase3_Puente_CoT.csv

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

# ── Fase 2 (50%) ──
P2_EVAL_DIR    = BASE / "phase2_results-50pct" / "evaluation"
P2_EXP1_DIR    = BASE / "phase2_results-50pct" / "exp1_cot"

P2_REEVAL_RESUMEN     = P2_EVAL_DIR / "predicciones_Config_B1_clean_reeval_resumen.csv"
P2_LLMEXT_RESUMEN     = P2_EVAL_DIR / "predicciones_Config_B1_llmexterno_metrics_resumen.csv"
P2_EXP1_COMPARISON    = P2_EXP1_DIR / "exp1_comparison_resumen.csv"
P2_CHX_LIMPIO         = P2_EVAL_DIR / "predicciones_Config_B1_metricas_estrictas.csv"
P2_CHX_SUCIO          = P2_EVAL_DIR / "predicciones_Config_B1_metricas_estrictas_SUCIO.csv"
P2_EXP1_CHX_PATHOLOGY = P2_EXP1_DIR / "exp1_chexpert_per_pathology.csv"

# ── Fase 3 PUENTE (50%) ──
P3_EVAL_DIR     = BASE / "phase3_puente_results-50pct" / "evaluation"
P3_EXP1_DIR     = BASE / "phase3_puente_results-50pct" / "exp1_cot"

P3_REEVAL_DISENADO  = P3_EVAL_DIR / "predicciones_Phase3_Puente_llmexterno_llama8b_reeval_resumen.csv"
P3_REEVAL_COT       = P3_EXP1_DIR / "exp1_final_formatted_phase3_puente_llama8b_reeval_resumen.csv"
P3_GRAN_DISENADO    = P3_EVAL_DIR / "resultados_granulares_Phase3_Puente.csv"
P3_GRAN_COT         = P3_EXP1_DIR / "resultados_granulares_Phase3_Puente_CoT.csv"

# ── Salidas ──
OUT_DIR = BASE / "phase3_puente_results-50pct"
OUT_METRICAS = OUT_DIR / "metricas_comparativas_inferencias_phase3.csv"
OUT_CHEXPERT = OUT_DIR / "chexpert_comparativo_phase3.csv"

ROUND = 4

# 14 patologías oficiales CheXpert (mismo orden que CHEXBERT_LABELS).
# Se replica aquí para no depender de transformers/torch al ejecutar el script.
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
    """Aborta con mensaje claro si falta algún archivo crítico."""
    faltan = [(k, p) for k, p in paths_dict.items() if not Path(p).exists()]
    if faltan:
        print("\nERROR: faltan los siguientes archivos pre-requisito:")
        for k, p in faltan:
            print(f"  • {k:<30s} → {p}")
        print("\nVerifica que las etapas previas hayan terminado correctamente.")
        sys.exit(1)


# ═════════════════════════════════════════════════════════════════════════════
# 1) métricas_comparativas_inferencias_phase3.csv
# ═════════════════════════════════════════════════════════════════════════════

def _safe_n(df: pd.DataFrame, clave: str) -> int:
    """Suma n_mejoran + n_empeoran + n_iguales para una métrica del resumen."""
    cols = ["n_mejoran", "n_empeoran", "n_iguales"]
    if all(c in df.columns for c in cols):
        return int(df.loc[clave, cols].sum())
    return -1  # desconocido (no debería ocurrir si el resumen está bien)


def construir_metricas() -> pd.DataFrame:
    """Construye la tabla comparativa de 9 variantes × 6 métricas."""
    print("\n" + "=" * 78)
    print(" CSV 1 — metricas_comparativas_inferencias_phase3.csv")
    print("=" * 78)

    # ── Cargar resúmenes ya disponibles ──
    p2_clean    = pd.read_csv(P2_REEVAL_RESUMEN).set_index("metrica")
    p2_llmext   = pd.read_csv(P2_LLMEXT_RESUMEN).set_index("metrica")
    p2_exp1     = pd.read_csv(P2_EXP1_COMPARISON).set_index("metrica")
    p3_disenado = pd.read_csv(P3_REEVAL_DISENADO).set_index("metrica")
    p3_cot      = pd.read_csv(P3_REEVAL_COT).set_index("metrica")

    # Mapeo nombre interno → etiqueta legible
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
            # ── Fase 2 (referencia) ──
            "fase2_disenado_sucio":        round(float(p2_clean.loc[clave,    "media_sucio"]),     ROUND),
            "fase2_disenado_limpio":       round(float(p2_clean.loc[clave,    "media_limpio"]),    ROUND),
            "fase2_disenado_llm_externo":  round(float(p2_llmext.loc[clave,   "media_formateado"]), ROUND),
            "fase2_nuevo_sucio":           round(float(p2_exp1.loc[clave,     "media_original"]),   ROUND),
            "fase2_nuevo_limpio":          round(float(p2_exp1.loc[clave,     "media_formateado"]), ROUND),
            # ── Fase 3 PUENTE ──
            # En Fase 3 NO hay "limpio post-guillotina": el "limpio" del reeval
            # ES la salida del Llama 8B (mismo CSV: media_sucio = PUENTE crudo,
            # media_limpio = formatted_report = post-Llama). Por eso lo
            # mapeamos como `phase3_*_llm_externo` para coherencia semántica
            # con la columna homóloga de Fase 2.
            "phase3_disenado_sucio":       round(float(p3_disenado.loc[clave, "media_sucio"]),  ROUND),
            "phase3_disenado_llm_externo": round(float(p3_disenado.loc[clave, "media_limpio"]), ROUND),
            "phase3_nuevo_sucio":          round(float(p3_cot.loc[clave,      "media_sucio"]),  ROUND),
            "phase3_nuevo_llm_externo":    round(float(p3_cot.loc[clave,      "media_limpio"]), ROUND),
        })

    # n de muestras (fila de transparencia)
    n_p2_clean    = _safe_n(p2_clean,    "bleu4")
    n_p2_llmext   = _safe_n(p2_llmext,   "bleu4")
    n_p2_exp1     = _safe_n(p2_exp1,     "bleu4")
    n_p3_disenado = _safe_n(p3_disenado, "bleu4")
    n_p3_cot      = _safe_n(p3_cot,      "bleu4")

    filas.append({
        "metrica": "n_muestras",
        "fase2_disenado_sucio":        n_p2_clean,
        "fase2_disenado_limpio":       n_p2_clean,
        "fase2_disenado_llm_externo":  n_p2_llmext,
        "fase2_nuevo_sucio":           n_p2_exp1,
        "fase2_nuevo_limpio":          n_p2_exp1,
        "phase3_disenado_sucio":       n_p3_disenado,
        "phase3_disenado_llm_externo": n_p3_disenado,
        "phase3_nuevo_sucio":          n_p3_cot,
        "phase3_nuevo_llm_externo":    n_p3_cot,
    })

    df = pd.DataFrame(filas)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_METRICAS, index=False)
    print(f"\n[OK] {OUT_METRICAS}")
    with pd.option_context("display.width", 240, "display.max_columns", 30):
        print(df.to_string(index=False))

    print(f"\n   * Fase 2 reeval (sucio/limpio)      = {n_p2_clean} muestras")
    print(f"   * Fase 2 LLM externo (disenado)     = {n_p2_llmext} muestras")
    print(f"   * Fase 2 Exp1 CoT (sucio/limpio)    = {n_p2_exp1} muestras")
    print(f"   * Phase3 reeval disenado            = {n_p3_disenado} muestras")
    print(f"   * Phase3 reeval CoT                 = {n_p3_cot} muestras")

    # Sanity: el "sucio" del reeval Fase 2 debe coincidir con el "original"
    # del LLM externo Fase 2 (mismo punto de partida).
    diff = max(
        abs(round(float(p2_clean.loc[k, "media_sucio"]), ROUND) -
            round(float(p2_llmext.loc[k, "media_original"]), ROUND))
        for k, _ in orden
    )
    print(f"\n   [sanity Fase 2] |sucio_reeval - original_llmext| max = {diff:.5f}  (esperado ~0)")
    return df


# ═════════════════════════════════════════════════════════════════════════════
# 2) chexpert_comparativo_phase3.csv
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


def _metricas_pat_desde_granular_phase3(
    df_gran: pd.DataFrame,
    labels: list = CHEX_LABELS,
) -> pd.DataFrame:
    """
    Reconstruye TP/FP/FN/TN/precision/recall/f1 por patología a partir de un
    CSV granular de Fase 3 PUENTE que contiene SOLO 'pred_chexpert_*' y
    'chexpert_match_*' (sin las columnas GT crudas para el caso del prompt
    diseñado de Phase 3).

    Lógica (réplica de evaluate_phase3_puente._chexpert_comparison_metrics):
        match = NaN  → fila no contribuye (ambos NaN antes de imputación)
        match = 1, pred = 1 → TP   (gt=1, pred=1)
        match = 1, pred = 0 → TN   (gt=0, pred=0)
        match = 0, pred = 1 → FP   (gt=0, pred=1)
        match = 0, pred = 0 → FN   (gt=1, pred=0)

    NOTA: 'pred_bin = 1 si pred != 0 else 0' → un -1 (incierto) cuenta como
    positivo. Esto coincide con la regla `uncertain_as_positive=True` usada
    en evaluate_metricas_estrictas/Fase 2.

    Returns:
        DataFrame con columnas: categoria, TP, FP, FN, TN, precision, recall, f1
    """
    rows = []
    for lbl in labels:
        cn = _canon(lbl)
        col_pred  = f"pred_chexpert_{cn}"
        col_match = f"chexpert_match_{cn}"
        if col_pred not in df_gran.columns or col_match not in df_gran.columns:
            # Patología ausente → fila NaN
            rows.append({"categoria": lbl, "TP": 0, "FP": 0, "FN": 0, "TN": 0,
                         "precision": np.nan, "recall": np.nan, "f1": np.nan})
            continue

        pred_raw  = pd.to_numeric(df_gran[col_pred],  errors="coerce")
        match_raw = pd.to_numeric(df_gran[col_match], errors="coerce")

        # Solo cuentan filas con match definido (ambos no NaN tras la regla
        # de _chexpert_comparison_metrics: si solo uno es NaN, se imputa a 0
        # y match queda definido).
        valid = match_raw.notna()
        pred  = pred_raw[valid].fillna(0)
        match = match_raw[valid]

        # Binarización (1 si != 0, incluye -1 = incierto)
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


def construir_chexpert() -> pd.DataFrame:
    """Construye la tabla comparativa CheXpert (5 variantes × 14 patologías + general)."""
    print("\n" + "=" * 78)
    print(" CSV 2 — chexpert_comparativo_phase3.csv")
    print("=" * 78)

    cols = ["categoria", "TP", "FP", "FN", "precision", "recall", "f1"]

    # ── Fase 2 (archivos pre-computados con evaluate_final) ──
    f2_dsuc = pd.read_csv(P2_CHX_SUCIO)[cols].copy()
    f2_dlim = pd.read_csv(P2_CHX_LIMPIO)[cols].copy()
    f2_esuc = pd.read_csv(P2_EXP1_CHX_PATHOLOGY)[cols].copy()

    # ── Fase 3 PUENTE — calculado desde granular ──
    print("  Calculando Phase 3 PUENTE desde granulares (chexpert_match_* + pred_chexpert_*)...")
    p3_dsuc = _metricas_pat_desde_granular_phase3(pd.read_csv(P3_GRAN_DISENADO))
    p3_esuc = _metricas_pat_desde_granular_phase3(pd.read_csv(P3_GRAN_COT))

    # ── Sección por patología (merge sobre categoria) ──
    base = f2_dsuc[["categoria"]].copy()

    def _ren(df: pd.DataFrame, pref: str) -> pd.DataFrame:
        return df.rename(columns={
            "precision": f"{pref}_P",
            "recall":    f"{pref}_R",
            "f1":        f"{pref}_F1",
        })[["categoria", f"{pref}_P", f"{pref}_R", f"{pref}_F1"]]

    merged = (
        base
        .merge(_ren(f2_dsuc, "fase2_disenado_sucio"),  on="categoria", how="left")
        .merge(_ren(f2_dlim, "fase2_disenado_limpio"), on="categoria", how="left")
        .merge(_ren(f2_esuc, "fase2_nuevo_sucio"),     on="categoria", how="left")
        .merge(_ren(p3_dsuc, "phase3_disenado_sucio"), on="categoria", how="left")
        .merge(_ren(p3_esuc, "phase3_nuevo_sucio"),    on="categoria", how="left")
    )
    merged.insert(0, "nivel", "por_patologia")

    # ── Sección general (micro / macro) ──
    mm_f2_dsuc = _micro_macro(f2_dsuc)
    mm_f2_dlim = _micro_macro(f2_dlim)
    mm_f2_esuc = _micro_macro(f2_esuc)
    mm_p3_dsuc = _micro_macro(p3_dsuc)
    mm_p3_esuc = _micro_macro(p3_esuc)

    filas_gen = []
    for scope, etiqueta in [("micro", "MICRO (pooled 14 clases)"),
                            ("macro", "MACRO (nanmean 14 clases)")]:
        f2dp, f2dr, f2df = mm_f2_dsuc[scope]
        f2lp, f2lr, f2lf = mm_f2_dlim[scope]
        f2ep, f2er, f2ef = mm_f2_esuc[scope]
        p3dp, p3dr, p3df = mm_p3_dsuc[scope]
        p3ep, p3er, p3ef = mm_p3_esuc[scope]
        filas_gen.append({
            "nivel": "general",
            "categoria": etiqueta,
            "fase2_disenado_sucio_P":  f2dp, "fase2_disenado_sucio_R":  f2dr, "fase2_disenado_sucio_F1":  f2df,
            "fase2_disenado_limpio_P": f2lp, "fase2_disenado_limpio_R": f2lr, "fase2_disenado_limpio_F1": f2lf,
            "fase2_nuevo_sucio_P":     f2ep, "fase2_nuevo_sucio_R":     f2er, "fase2_nuevo_sucio_F1":     f2ef,
            "phase3_disenado_sucio_P": p3dp, "phase3_disenado_sucio_R": p3dr, "phase3_disenado_sucio_F1": p3df,
            "phase3_nuevo_sucio_P":    p3ep, "phase3_nuevo_sucio_R":    p3er, "phase3_nuevo_sucio_F1":    p3ef,
        })
    gen = pd.DataFrame(filas_gen)

    df = pd.concat([merged, gen], ignore_index=True)

    # Redondeo de columnas numéricas
    num_cols = [c for c in df.columns if c not in ("nivel", "categoria")]
    df[num_cols] = df[num_cols].astype(float).round(ROUND)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CHEXPERT, index=False)
    print(f"\n[OK] {OUT_CHEXPERT}")
    with pd.option_context("display.width", 280, "display.max_columns", 50):
        print(df.to_string(index=False))

    # ── Sanity check del cálculo Phase 3 ──
    print("\n   [sanity Phase 3 PUENTE — recomputado vs granular]")
    print(f"   Phase3 disenado_sucio  micro:"
          f"  P={mm_p3_dsuc['micro'][0]:.4f}  R={mm_p3_dsuc['micro'][1]:.4f}"
          f"  F1={mm_p3_dsuc['micro'][2]:.4f}")
    print(f"   Phase3 nuevo_sucio (CoT) micro:"
          f"  P={mm_p3_esuc['micro'][0]:.4f}  R={mm_p3_esuc['micro'][1]:.4f}"
          f"  F1={mm_p3_esuc['micro'][2]:.4f}")
    return df


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main() -> int:
    archivos_requeridos = {
        # Fase 2
        "P2_REEVAL_RESUMEN":     P2_REEVAL_RESUMEN,
        "P2_LLMEXT_RESUMEN":     P2_LLMEXT_RESUMEN,
        "P2_EXP1_COMPARISON":    P2_EXP1_COMPARISON,
        "P2_CHX_LIMPIO":         P2_CHX_LIMPIO,
        "P2_CHX_SUCIO":          P2_CHX_SUCIO,
        "P2_EXP1_CHX_PATHOLOGY": P2_EXP1_CHX_PATHOLOGY,
        # Fase 3 PUENTE
        "P3_REEVAL_DISENADO":    P3_REEVAL_DISENADO,
        "P3_REEVAL_COT":         P3_REEVAL_COT,        # ← lo genera la etapa 8 (en curso)
        "P3_GRAN_DISENADO":      P3_GRAN_DISENADO,
        "P3_GRAN_COT":           P3_GRAN_COT,
    }
    _verificar_archivos(archivos_requeridos)

    construir_metricas()
    construir_chexpert()

    print("\n" + "=" * 78)
    print(" Comparativos Fase 2 + Fase 3 PUENTE generados:")
    print(f"   • {OUT_METRICAS}")
    print(f"   • {OUT_CHEXPERT}")
    print("=" * 78 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
