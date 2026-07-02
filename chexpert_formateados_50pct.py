#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
chexpert_formateados_50pct.py
=============================
Calcula CheXpert (14 patologías) sobre los reportes FORMATEADOS por el LLM
externo (Llama-3.1-8B) — tanto los del PROMPT DISEÑADO como los del PROMPT
NUEVO (CoT) — para las 5 etapas con evaluación completa:

      · phase2_50%        (Config_B1)
      · phase3_puente_50% (Phase3_Puente)
      · exp2_raddino      (Exp2_RadDINO)
      · exp3_asclepius    (Exp3_Asclepius)
      · experimento final (Exp_Final)

(phase3_results-50pct — Q-Former LoRA — solo tiene datos de SMOKE (8 filas) y
 carece de evaluación del prompt diseñado, por lo que se OMITE.)

Metodología — IDÉNTICA a la del proyecto para el reporte "sucio"
---------------------------------------------------------------
  • GT  : mimic-cxr-2.0.0-chexpert.csv (oficial), merge por study_id normalizado.
  • Pred: CheXbert oficial de Stanford sobre el `formatted_report`.
  • Binarización (igual que evaluate_phase3_puente._chexpert_comparison_metrics):
        1.0 / -1.0 → positivo   ·   0.0 / NaN → negativo
  • match por patología, micro P/R/F1 por reporte (granular) y
    micro (pooled) + macro (nanmean) agregados (reconstruidos con la MISMA
    función que usan los comparativos: _metricas_pat_desde_granular).

Salidas (por etapa) — NO destructivas
-------------------------------------
  1) Métricas GRANULARES (archivos NUEVOS, paralelos a los del reporte sucio):
        evaluation/resultados_granulares_<NAME>_llmexterno.csv      (diseñado)
        exp1_cot/resultados_granulares_<NAME>_CoT_llmexterno.csv    (nuevo/CoT)

  2) CSV COMPARATIVO de CheXpert (se AÑADEN columnas; original respaldado .bak):
        <prefix>_disenado_limpio_{P,R,F1}
        <prefix>_nuevo_limpio_{P,R,F1}
     insertadas justo después de las columnas *_sucio_* de la propia etapa.

Validación incorporada
----------------------
  Antes de procesar, recalcula el CheXpert del reporte SUCIO (generated_report)
  del Experimento FINAL (prompt diseñado) y verifica que el micro P/R/F1 coincide
  con el ya almacenado en chexpert_comparativo_final.csv (0.1494 / 0.1745 / 0.161).
  Si coincide, la tubería es metodológicamente idéntica a la del "sucio" y los
  números "limpio" son confiables.

Uso
---
    python chexpert_formateados_50pct.py --validate     # solo el sanity-check
    python chexpert_formateados_50pct.py                # valida y procesa todo

No usa entornos virtuales (Python del sistema). Requiere: torch, pandas, numpy,
transformers (vía chexbert_official) — NO carga el modelo generador (BLIP-2).

Autor : Evelyn Silva Rozas
Fecha : Junio 2026
"""

import argparse
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Labeler oficial + utilidades de agregación YA usadas por los comparativos
from chexbert_official import CheXbertLabeler, CHEXBERT_LABELS
from actualizar_comparativos_final import (
    _metricas_pat_desde_granular,   # reconstruye TP/FP/FN/P/R/F1 por patología
    _micro_macro,                   # micro (pooled) + macro (nanmean)
)

BASE = Path(__file__).resolve().parent
GT_CSV = BASE / "mimic-cxr-2.0.0-chexpert.csv"
ROUND = 4
_PRED_PREFIX = "pred_chexpert_"
BATCH = 64

SEP = "═" * 80


def _canon(lbl: str) -> str:
    return lbl.lower().replace(" ", "_")


def _norm_sid(s) -> str:
    s = str(s).strip()
    return s.lstrip("s") if s.startswith("s") else s


# ─────────────────────────────────────────────────────────────────────────────
# Definición de las 5 etapas (rutas relativas a su carpeta base)
# ─────────────────────────────────────────────────────────────────────────────
STAGES: List[Dict] = [
    dict(
        key="phase2", name="Config_B1", base="phase2_results-50pct",
        designed_fmt="evaluation/predicciones_Config_B1_llmexterno_llama8b.csv",
        new_fmt="exp1_cot/exp1_final_formatted_llama8b.csv",
        gran_designed="evaluation/resultados_granulares_Config_B1_llmexterno.csv",
        gran_new="exp1_cot/resultados_granulares_Config_B1_CoT_llmexterno.csv",
        comparativo="chexpert_comparativo.csv",
        designed_limpio="disenado_limpio", new_limpio="nuevo_limpio",
        designed_anchor="disenado_sucio_F1", new_anchor="nuevo_sucio_F1",
    ),
    dict(
        key="phase3_puente", name="Phase3_Puente", base="phase3_puente_results-50pct",
        designed_fmt="evaluation/predicciones_Phase3_Puente_llmexterno_llama8b.csv",
        new_fmt="exp1_cot/exp1_final_formatted_phase3_puente_llama8b.csv",
        gran_designed="evaluation/resultados_granulares_Phase3_Puente_llmexterno.csv",
        gran_new="exp1_cot/resultados_granulares_Phase3_Puente_CoT_llmexterno.csv",
        comparativo="chexpert_comparativo_phase3.csv",
        designed_limpio="phase3_disenado_limpio", new_limpio="phase3_nuevo_limpio",
        designed_anchor="phase3_disenado_sucio_F1", new_anchor="phase3_nuevo_sucio_F1",
    ),
    dict(
        key="exp2", name="Exp2_RadDINO", base="exp2_raddino_results-50pct",
        designed_fmt="evaluation/predicciones_Exp2_RadDINO_llmexterno_llama8b.csv",
        new_fmt="exp1_cot/exp1_final_formatted_exp2_raddino_llama8b.csv",
        gran_designed="evaluation/resultados_granulares_Exp2_RadDINO_llmexterno.csv",
        gran_new="exp1_cot/resultados_granulares_Exp2_RadDINO_CoT_llmexterno.csv",
        comparativo="chexpert_comparativo_exp2.csv",
        designed_limpio="exp2_disenado_limpio", new_limpio="exp2_nuevo_limpio",
        designed_anchor="exp2_disenado_sucio_F1", new_anchor="exp2_nuevo_sucio_F1",
    ),
    dict(
        key="exp3", name="Exp3_Asclepius", base="exp3_asclepius_results-50pct",
        designed_fmt="evaluation/predicciones_Exp3_Asclepius_llmexterno_llama8b.csv",
        new_fmt="exp1_cot/exp1_final_formatted_exp3_asclepius_llama8b.csv",
        gran_designed="evaluation/resultados_granulares_Exp3_Asclepius_llmexterno.csv",
        gran_new="exp1_cot/resultados_granulares_Exp3_Asclepius_CoT_llmexterno.csv",
        comparativo="chexpert_comparativo_exp3.csv",
        designed_limpio="exp3_disenado_limpio", new_limpio="exp3_nuevo_limpio",
        designed_anchor="exp3_disenado_sucio_F1", new_anchor="exp3_nuevo_sucio_F1",
    ),
    dict(
        key="final", name="Exp_Final", base="final_experiment_results-50pct",
        designed_fmt="evaluation/predicciones_Exp_Final_llmexterno_llama8b.csv",
        new_fmt="exp1_cot/exp1_final_formatted_final_experiment_llama8b.csv",
        gran_designed="evaluation/resultados_granulares_Exp_Final_llmexterno.csv",
        gran_new="exp1_cot/resultados_granulares_Exp_Final_CoT_llmexterno.csv",
        comparativo="chexpert_comparativo_final.csv",
        designed_limpio="final_disenado_limpio", new_limpio="final_nuevo_limpio",
        designed_anchor="final_disenado_sucio_F1", new_anchor="final_nuevo_sucio_F1",
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# GT oficial → lookup {sid: {label: valor}}
# ─────────────────────────────────────────────────────────────────────────────

def cargar_gt() -> Dict[str, Dict[str, float]]:
    if not GT_CSV.exists():
        print(f"ERROR: no se encuentra el GT oficial: {GT_CSV}")
        sys.exit(1)
    gt = pd.read_csv(GT_CSV)
    gt["_sid"] = gt["study_id"].map(_norm_sid)
    gt = gt.drop_duplicates(subset=["_sid"]).set_index("_sid")
    # garantizar las 14 columnas
    faltan = [l for l in CHEXBERT_LABELS if l not in gt.columns]
    if faltan:
        print(f"ERROR: faltan columnas en el GT: {faltan}")
        sys.exit(1)
    lookup = gt[CHEXBERT_LABELS].to_dict("index")
    print(f"  GT oficial: {len(lookup):,} study_id únicos cargados.")
    return lookup


# ─────────────────────────────────────────────────────────────────────────────
# Comparación GT vs Pred por reporte — VERBATIM de
# evaluate_phase3_puente._chexpert_comparison_metrics
# ─────────────────────────────────────────────────────────────────────────────

def _comparison(gt_row: Dict[str, float], pred_dict: Dict[str, float]) -> Dict[str, float]:
    tp = fp = fn = 0
    metrics: Dict[str, float] = {}

    for lbl in CHEXBERT_LABELS:
        gt_val = gt_row.get(lbl, float("nan"))
        pred_val = pred_dict.get(f"{_PRED_PREFIX}{_canon(lbl)}", float("nan"))

        def _to_bin(v):
            if isinstance(v, float) and math.isnan(v):
                return None
            return 1 if float(v) != 0.0 else 0

        gt_bin = _to_bin(gt_val)
        pred_bin = _to_bin(pred_val)

        if gt_bin is None and pred_bin is None:
            metrics[f"chexpert_match_{_canon(lbl)}"] = float("nan")
            continue
        if gt_bin is None:
            gt_bin = 0
        if pred_bin is None:
            pred_bin = 0

        match = 1.0 if gt_bin == pred_bin else 0.0
        metrics[f"chexpert_match_{_canon(lbl)}"] = match

        if gt_bin == 1 and pred_bin == 1:
            tp += 1
        elif gt_bin == 0 and pred_bin == 1:
            fp += 1
        elif gt_bin == 1 and pred_bin == 0:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    if not (math.isnan(precision) or math.isnan(recall)) and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = float("nan")

    metrics["chexpert_precision"] = precision
    metrics["chexpert_recall"] = recall
    metrics["chexpert_f1"] = f1
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Procesa un CSV formateado → granular + agregados (micro/macro)
# ─────────────────────────────────────────────────────────────────────────────

def procesar_formateado(
    formatted_path: Path,
    report_col: str,
    chexbert: CheXbertLabeler,
    gt_lookup: Dict[str, Dict[str, float]],
    gran_out: Path = None,
    etiqueta: str = "",
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    df = pd.read_csv(formatted_path)
    n = len(df)
    print(f"    [{etiqueta}] {formatted_path.name}  ({n:,} filas)  col='{report_col}'")

    reports = (
        df[report_col].astype(str)
        .map(lambda s: s.strip() if s and s.strip() and s.strip().lower() != "nan" else "no findings")
        .tolist()
    )

    # CheXbert batched sobre el reporte (hipótesis)
    preds = chexbert.label_batch(reports, batch_size=BATCH)

    sids = df["study_id"].map(_norm_sid).tolist()
    dicom = df.get("dicom_id", pd.Series([""] * n)).tolist()
    study = df["study_id"].tolist()
    refs = df.get("reference_report", pd.Series([""] * n)).astype(str).tolist()

    nan_gt = {l: float("nan") for l in CHEXBERT_LABELS}
    rows_out: List[Dict] = []
    for i in range(n):
        pred_dict = {
            f"{_PRED_PREFIX}{_canon(lbl)}": preds[i].get(lbl, float("nan"))
            for lbl in CHEXBERT_LABELS
        }
        out: Dict = {
            "dicom_id": dicom[i],
            "study_id": study[i],
            "reference_report": refs[i],
            "formatted_report": reports[i],
        }
        out.update(pred_dict)
        gt_row = gt_lookup.get(sids[i], nan_gt)
        out.update(_comparison(gt_row, pred_dict))
        rows_out.append(out)

    df_out = pd.DataFrame(rows_out)

    if gran_out is not None:
        gran_out.parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(gran_out, index=False)
        print(f"        → granular: {gran_out}  ({len(df_out.columns)} cols)")

    per_pat = _metricas_pat_desde_granular(df_out)
    mm = _micro_macro(per_pat)
    return df_out, per_pat, mm


# ─────────────────────────────────────────────────────────────────────────────
# Validación: recomputar el SUCIO del Experimento FINAL (diseñado) y comparar
# ─────────────────────────────────────────────────────────────────────────────

_EXPECTED_FINAL_SUCIO = (0.1494, 0.1745, 0.1610)  # P, R, F1 (micro) de chexpert_comparativo_final.csv


def validar(chexbert, gt_lookup) -> bool:
    print("\n" + SEP)
    print(" VALIDACIÓN — recomputo del SUCIO (Exp_Final, prompt diseñado)")
    print(SEP)
    f = BASE / "final_experiment_results-50pct" / "evaluation" / \
        "predicciones_Exp_Final_llmexterno_llama8b.csv"
    _, _, mm = procesar_formateado(
        f, report_col="generated_report", chexbert=chexbert,
        gt_lookup=gt_lookup, gran_out=None, etiqueta="validate-sucio",
    )
    p, r, f1 = mm["micro"]
    ep, er, ef = _EXPECTED_FINAL_SUCIO
    ok = (abs(p - ep) < 1.5e-3 and abs(r - er) < 1.5e-3 and abs(f1 - ef) < 1.5e-3)
    print(f"\n    micro recomputado : P={p:.4f}  R={r:.4f}  F1={f1:.4f}")
    print(f"    micro esperado    : P={ep:.4f}  R={er:.4f}  F1={ef:.4f}")
    print(f"    → {'✓ COINCIDE (metodología idéntica)' if ok else '✗ NO coincide'}")
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Actualiza el CSV comparativo de CheXpert con columnas *_limpio
# ─────────────────────────────────────────────────────────────────────────────

def _perpat_map(per_pat: pd.DataFrame) -> Dict[str, Tuple[float, float, float]]:
    return {
        row["categoria"]: (row["precision"], row["recall"], row["f1"])
        for _, row in per_pat.iterrows()
    }


def _insert_after(cols: List[str], anchor: str, newcols: List[str]) -> List[str]:
    out: List[str] = []
    for c in cols:
        out.append(c)
        if c == anchor:
            out.extend(newcols)
    seen = set()
    return [x for x in out if not (x in seen or seen.add(x))]


def actualizar_comparativo(stage: Dict, res_d: dict, res_n: dict) -> None:
    path = BASE / stage["base"] / stage["comparativo"]
    if not path.exists():
        print(f"    ⚠ comparativo no encontrado, se omite: {path}")
        return

    df = pd.read_csv(path)
    shutil.copy(str(path), str(path) + ".bak_prelimpio")

    pm_d, mm_d = _perpat_map(res_d["per_pat"]), res_d["mm"]
    pm_n, mm_n = _perpat_map(res_n["per_pat"]), res_n["mm"]

    def _val(nivel, categoria, pm, mm):
        if nivel == "por_patologia":
            return pm.get(categoria, (np.nan, np.nan, np.nan))
        if "MICRO" in str(categoria).upper():
            return mm["micro"]
        if "MACRO" in str(categoria).upper():
            return mm["macro"]
        return (np.nan, np.nan, np.nan)

    for base_col, pm, mm in [
        (stage["designed_limpio"], pm_d, mm_d),
        (stage["new_limpio"],      pm_n, mm_n),
    ]:
        P, R, F = [], [], []
        for _, row in df.iterrows():
            p, r, f = _val(row["nivel"], row["categoria"], pm, mm)
            P.append(p); R.append(r); F.append(f)
        df[f"{base_col}_P"] = np.round(P, ROUND)
        df[f"{base_col}_R"] = np.round(R, ROUND)
        df[f"{base_col}_F1"] = np.round(F, ROUND)

    order = list(df.columns)
    order = _insert_after(order, stage["designed_anchor"],
                          [f"{stage['designed_limpio']}_{s}" for s in ("P", "R", "F1")])
    order = _insert_after(order, stage["new_anchor"],
                          [f"{stage['new_limpio']}_{s}" for s in ("P", "R", "F1")])
    df = df[order]
    df.to_csv(path, index=False)
    print(f"    → comparativo actualizado: {path}")
    print(f"        + {stage['designed_limpio']}_{{P,R,F1}}   micro F1={mm_d['micro'][2]:.4f}")
    print(f"        + {stage['new_limpio']}_{{P,R,F1}}   micro F1={mm_n['micro'][2]:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--validate", action="store_true",
                    help="Solo ejecuta el sanity-check (sucio del Exp_Final) y termina.")
    ap.add_argument("--stages", nargs="*", default=None,
                    help="Subconjunto de etapas por 'key' (phase2 phase3_puente exp2 exp3 final).")
    args = ap.parse_args()

    print(SEP)
    print(" CheXpert(14) sobre reportes FORMATEADOS por LLM externo (50%)")
    print(SEP)

    gt_lookup = cargar_gt()
    print("  Cargando CheXbert oficial (Stanford)...")
    chexbert = CheXbertLabeler()

    if not validar(chexbert, gt_lookup):
        print("\n✗ La validación NO coincide. Abortando para no escribir números "
              "inconsistentes. Revisa la metodología antes de continuar.")
        return 2
    if args.validate:
        print("\n[--validate] Sanity-check OK. No se procesa nada más.")
        return 0

    etapas = STAGES if not args.stages else [s for s in STAGES if s["key"] in args.stages]

    for stage in etapas:
        print("\n" + SEP)
        print(f" ETAPA: {stage['key']}  ({stage['name']})")
        print(SEP)
        base = BASE / stage["base"]

        # — Prompt DISEÑADO (formatted) —
        d_out, d_perpat, d_mm = procesar_formateado(
            base / stage["designed_fmt"], "formatted_report", chexbert, gt_lookup,
            gran_out=base / stage["gran_designed"], etiqueta="diseñado",
        )
        # — Prompt NUEVO / CoT (formatted) —
        n_out, n_perpat, n_mm = procesar_formateado(
            base / stage["new_fmt"], "formatted_report", chexbert, gt_lookup,
            gran_out=base / stage["gran_new"], etiqueta="nuevo/CoT",
        )

        actualizar_comparativo(
            stage,
            res_d={"per_pat": d_perpat, "mm": d_mm},
            res_n={"per_pat": n_perpat, "mm": n_mm},
        )

    print("\n" + SEP)
    print(" COMPLETADO — CheXpert de reportes formateados añadido a granulares y comparativos.")
    print(SEP)
    # sentinela de finalización
    (BASE / ".chexpert_formateados_done").write_text("ok\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
