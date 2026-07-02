#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
crear_csvs_comparativos.py
==========================
Genera dos CSV comparativos entre la inferencia con el PROMPT DISEÑADO (Fase 2,
prompt default) y la inferencia con el PROMPT NUEVO (Experimento 1, Chain-of-Thought).

Salidas (en phase2_results-50pct/):
  1) metricas_comparativas_inferencias.csv
       Metricas (BLEU-4, ROUGE-L, F1-RadGraph, ClinicalBERT P/R/F1) para 5 variantes:
         - prompt_disenado_sucio        (Fase 2, reporte crudo BLIP-2)
         - prompt_disenado_limpio       (Fase 2, reporte tras guillotina)
         - prompt_disenado_llm_externo  (Fase 2, reporte sucio post-LLM externo)
         - prompt_nuevo_sucio           (Exp1 CoT, reporte crudo)
         - prompt_nuevo_limpio          (Exp1 CoT, reporte post-LLM externo)

  2) chexpert_comparativo.csv
       CheXpert por patologia + general (micro/macro) para:
         - disenado_sucio   (Fase 2, reportes crudos)
         - disenado_limpio  (Fase 2, reportes limpios)
         - nuevo_sucio      (Exp1 CoT, reportes crudos)
       (nuevo_limpio NO esta disponible: CheXpert no se corrio sobre los
        reportes formateados del CoT.)

Fuentes (solo se usan resultados YA disponibles, no se recomputa nada pesado):
  - phase2_results-50pct/evaluation/predicciones_Config_B1_clean_reeval_resumen.csv
  - phase2_results-50pct/evaluation/predicciones_Config_B1_metricas_estrictas.csv        (LIMPIO)
  - phase2_results-50pct/evaluation/predicciones_Config_B1_metricas_estrictas_SUCIO.csv  (SUCIO)
  - phase2_results-50pct/exp1_cot/exp1_comparison_resumen.csv
  - phase2_results-50pct/exp1_cot/exp1_chexpert_per_pathology.csv
"""

import os
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Rutas
# --------------------------------------------------------------------------- #
BASE = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.join(BASE, "phase2_results-50pct", "evaluation")
EXP1_DIR = os.path.join(BASE, "phase2_results-50pct", "exp1_cot")
OUT_DIR = os.path.join(BASE, "phase2_results-50pct")

# Metricas (sucio/limpio para todas)
F_PHASE2_METRICS = os.path.join(EVAL_DIR, "predicciones_Config_B1_clean_reeval_resumen.csv")
F_EXP1_METRICS = os.path.join(EXP1_DIR, "exp1_comparison_resumen.csv")

# Metricas del LLM externo aplicado a los reportes SUCIOS del prompt disenado
# (mismo pipeline que Exp1 pero sobre la inferencia de Fase 2 con prompt default).
# Formato del resumen: media_original (=sucio Fase 2) / media_formateado (=salida LLM).
F_PHASE2_LLMEXT_METRICS = os.path.join(EVAL_DIR, "predicciones_Config_B1_llmexterno_metrics_resumen.csv")

# CheXpert por patologia
F_PHASE2_CHX_LIMPIO = os.path.join(EVAL_DIR, "predicciones_Config_B1_metricas_estrictas.csv")
F_PHASE2_CHX_SUCIO = os.path.join(EVAL_DIR, "predicciones_Config_B1_metricas_estrictas_SUCIO.csv")
F_EXP1_CHX = os.path.join(EXP1_DIR, "exp1_chexpert_per_pathology.csv")

ROUND = 4  # decimales en las tablas comparativas


# --------------------------------------------------------------------------- #
# 1) metricas_comparativas_inferencias.csv
# --------------------------------------------------------------------------- #
def construir_metricas():
    p2 = pd.read_csv(F_PHASE2_METRICS).set_index("metrica")
    e1 = pd.read_csv(F_EXP1_METRICS).set_index("metrica")
    p2x = pd.read_csv(F_PHASE2_LLMEXT_METRICS).set_index("metrica")

    # Mapeo nombre interno -> etiqueta legible
    orden = [
        ("bleu4", "BLEU-4"),
        ("rougeL", "ROUGE-L"),
        ("radgraph", "F1-RadGraph"),
        ("bertscore_p", "ClinicalBERT_P"),
        ("bertscore_r", "ClinicalBERT_R"),
        ("bertscore_f1", "ClinicalBERT_F1"),
    ]

    filas = []
    for clave, etiqueta in orden:
        filas.append({
            "metrica": etiqueta,
            # Fase 2 (prompt disenado) — sucio (raw BLIP-2) y limpio (post-guillotina)
            "prompt_disenado_sucio": round(float(p2.loc[clave, "media_sucio"]), ROUND),
            "prompt_disenado_limpio": round(float(p2.loc[clave, "media_limpio"]), ROUND),
            # Fase 2 + LLM externo: salida del LLM sobre el reporte SUCIO del prompt disenado
            "prompt_disenado_llm_externo": round(float(p2x.loc[clave, "media_formateado"]), ROUND),
            # Exp1 CoT: media_original (sucio) / media_formateado (LLM externo sobre CoT)
            "prompt_nuevo_sucio": round(float(e1.loc[clave, "media_original"]), ROUND),
            "prompt_nuevo_limpio": round(float(e1.loc[clave, "media_formateado"]), ROUND),
        })

    # n de muestras de cada fuente (transparencia)
    n_p2 = int(p2.loc["bleu4", ["n_mejoran", "n_empeoran", "n_iguales"]].sum())
    n_p2x = int(p2x.loc["bleu4", ["n_mejoran", "n_empeoran", "n_iguales"]].sum())
    n_e1 = int(e1.loc["bleu4", ["n_mejoran", "n_empeoran", "n_iguales"]].sum())
    filas.append({
        "metrica": "n_muestras",
        "prompt_disenado_sucio": n_p2,
        "prompt_disenado_limpio": n_p2,
        "prompt_disenado_llm_externo": n_p2x,
        "prompt_nuevo_sucio": n_e1,
        "prompt_nuevo_limpio": n_e1,
    })

    df = pd.DataFrame(filas)
    out = os.path.join(OUT_DIR, "metricas_comparativas_inferencias.csv")
    df.to_csv(out, index=False)
    print(f"[OK] {out}")
    print(df.to_string(index=False))
    print(f"\n   * Fase 2 (reeval sucio/limpio)            = {n_p2} muestras")
    print(f"   * Fase 2 (LLM externo sobre sucio diseñado) = {n_p2x} muestras")
    print(f"   * Exp1 CoT                                  = {n_e1} muestras")

    # Sanity check: el 'sucio' del reeval debe coincidir exactamente con el
    # 'media_original' del LLM externo (mismo punto de partida).
    diff = max(abs(round(float(p2.loc[k, "media_sucio"]), ROUND) -
                   round(float(p2x.loc[k, "media_original"]), ROUND))
               for k, _ in orden)
    print(f"   [check] |sucio_reeval - original_llmext| max = {diff:.5f} (debe ~0)")
    return df


# --------------------------------------------------------------------------- #
# 2) chexpert_comparativo.csv
# --------------------------------------------------------------------------- #
def _micro_macro(df):
    """Calcula micro (pooled) y macro (nanmean) a partir de TP/FP/FN por patologia."""
    tp, fp, fn = df["TP"].sum(), df["FP"].sum(), df["FN"].sum()
    micro_p = tp / (tp + fp) if (tp + fp) else np.nan
    micro_r = tp / (tp + fn) if (tp + fn) else np.nan
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) else np.nan
    macro_p = float(np.nanmean(df["precision"]))
    macro_r = float(np.nanmean(df["recall"]))
    macro_f1 = float(np.nanmean(df["f1"]))
    return {
        "micro": (micro_p, micro_r, micro_f1),
        "macro": (macro_p, macro_r, macro_f1),
    }


def construir_chexpert():
    cols = ["categoria", "TP", "FP", "FN", "precision", "recall", "f1"]
    d_suc = pd.read_csv(F_PHASE2_CHX_SUCIO)[cols].copy()
    d_lim = pd.read_csv(F_PHASE2_CHX_LIMPIO)[cols].copy()
    e_suc = pd.read_csv(F_EXP1_CHX)[cols].copy()

    # --- Seccion por patologia (merge sobre categoria) ---
    base = d_suc[["categoria"]].copy()

    def _ren(df, pref):
        return df.rename(columns={
            "precision": f"{pref}_P",
            "recall": f"{pref}_R",
            "f1": f"{pref}_F1",
        })[["categoria", f"{pref}_P", f"{pref}_R", f"{pref}_F1"]]

    merged = (base
              .merge(_ren(d_suc, "disenado_sucio"), on="categoria")
              .merge(_ren(d_lim, "disenado_limpio"), on="categoria")
              .merge(_ren(e_suc, "nuevo_sucio"), on="categoria"))
    merged.insert(0, "nivel", "por_patologia")

    # --- Seccion general (micro / macro) ---
    mm_dsuc = _micro_macro(d_suc)
    mm_dlim = _micro_macro(d_lim)
    mm_esuc = _micro_macro(e_suc)

    filas_gen = []
    for scope, etiqueta in [("micro", "MICRO (pooled 14 clases)"),
                            ("macro", "MACRO (nanmean 14 clases)")]:
        dp, dr, df1 = mm_dsuc[scope]
        lp, lr, lf1 = mm_dlim[scope]
        ep, er, ef1 = mm_esuc[scope]
        filas_gen.append({
            "nivel": "general",
            "categoria": etiqueta,
            "disenado_sucio_P": dp, "disenado_sucio_R": dr, "disenado_sucio_F1": df1,
            "disenado_limpio_P": lp, "disenado_limpio_R": lr, "disenado_limpio_F1": lf1,
            "nuevo_sucio_P": ep, "nuevo_sucio_R": er, "nuevo_sucio_F1": ef1,
        })
    gen = pd.DataFrame(filas_gen)

    df = pd.concat([merged, gen], ignore_index=True)

    # Redondeo de columnas numericas
    num_cols = [c for c in df.columns if c not in ("nivel", "categoria")]
    df[num_cols] = df[num_cols].astype(float).round(ROUND)

    out = os.path.join(OUT_DIR, "chexpert_comparativo.csv")
    df.to_csv(out, index=False)
    print(f"\n[OK] {out}")
    with pd.option_context("display.width", 240, "display.max_columns", 30):
        print(df.to_string(index=False))

    # Verificacion: el micro de Exp1 debe coincidir con exp1_chexpert_general.csv
    print(f"\n   [check] Exp1 micro recomputado: "
          f"P={mm_esuc['micro'][0]:.4f} R={mm_esuc['micro'][1]:.4f} F1={mm_esuc['micro'][2]:.4f}")
    print("   (esperado segun exp1_chexpert_general.csv: P=0.1486 R=0.3940 F1=0.2159)")
    return df


if __name__ == "__main__":
    print("=" * 70)
    print("CSV 1 — metricas_comparativas_inferencias.csv")
    print("=" * 70)
    construir_metricas()
    print("\n" + "=" * 70)
    print("CSV 2 — chexpert_comparativo.csv")
    print("=" * 70)
    construir_chexpert()
    print("\nListo.")
