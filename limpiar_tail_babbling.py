#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
limpiar_tail_babbling.py — Guillotina anti-alucinaciones + métricas CheXpert estrictas
======================================================================================

Problema
--------
Los reportes generados (`generated_report`) sufren "Tail Babbling": al final del
texto el modelo degenera y emite tokens administrativos o basura
(p. ej. ``NOTIFICATION:``, ``strutConnector``, ``guiActiveUnfocused``...).
Cuando CheXbert etiqueta ese ruido, inventa hallazgos → muchos Falsos Positivos
→ **Recall alto, Precisión baja**.

Solución (3 etapas)
-------------------
1.  **Guillotina (RegEx)** — corta el string JUSTO ANTES de la primera aparición
    de cualquier palabra clave de ruido (case-insensitive) →
    columna ``generated_report_clean``.

2.  **Re-etiquetado** — pasa ``generated_report_clean`` por el labeler CheXbert
    oficial del proyecto (``chexbert_official.CheXbertLabeler``) y crea columnas
    ``pred_clean_<categoria>``.  (Opcional: si ya las tienes, omite esta etapa.)

3.  **Métricas estrictas (respetando NaN)** — compara el Ground Truth oficial con
    ``pred_clean_<categoria>`` SIN rellenar NaN con ceros.  Un NaN = "no
    mencionado".  ``GT=NaN`` y ``Pred=NaN`` → coincidencia correcta (True
    Negative).  Reporta por cada una de las 14 categorías: Matches, Precisión,
    Recall y F1 (devolviendo NaN cuando el denominador es 0, y agregando con
    ``np.nanmean`` para no penalizar artificialmente).

Uso (CLI)
---------
    # Demo sintética (no requiere GPU ni CheXbert):
    python limpiar_tail_babbling.py

    # Sobre un CSV real, re-etiquetando con CheXbert y trayendo el GT oficial:
    python limpiar_tail_babbling.py \
        --csv evaluation_results/predicciones_Config_A.csv \
        --chexpert-csv mimic-cxr-2.0.0-chexpert.csv \
        --relabel \
        --out evaluation_results/predicciones_Config_A_clean.csv

    # Si ya tienes columnas pred_clean_* (re-etiquetado hecho aparte):
    python limpiar_tail_babbling.py --csv mi_df.csv --chexpert-csv mimic-cxr-2.0.0-chexpert.csv

Uso (como librería)
-------------------
    import pandas as pd
    from limpiar_tail_babbling import (
        limpiar_reportes_tail_babbling, reetiquetar_con_chexbert,
        merge_ground_truth, evaluar_metricas_estrictas,
    )

    df = limpiar_reportes_tail_babbling(df)                 # etapa 1
    df = reetiquetar_con_chexbert(df)                       # etapa 2 (real)
    df = merge_ground_truth(df, "mimic-cxr-2.0.0-chexpert.csv")
    res = evaluar_metricas_estrictas(df)                   # etapa 3 + reporte

Autor : Evelyn Silva Rozas
Fecha : Junio 2026
"""

from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# 0.  Etiquetas CheXpert (orden oficial del proyecto)
# ─────────────────────────────────────────────────────────────────────────────
try:
    # Reutiliza la lista canónica si chexbert_official está disponible.
    from chexbert_official import CHEXBERT_LABELS  # type: ignore
except Exception:  # pragma: no cover - fallback autocontenido
    CHEXBERT_LABELS = [
        "No Finding",
        "Enlarged Cardiomediastinum",
        "Cardiomegaly",
        "Lung Opacity",
        "Lung Lesion",
        "Edema",
        "Consolidation",
        "Pneumonia",
        "Atelectasis",
        "Pneumothorax",
        "Pleural Effusion",
        "Pleural Other",
        "Fracture",
        "Support Devices",
    ]


def _canon(label: str) -> str:
    """'Pleural Effusion' → 'pleural_effusion' (convención de columnas del proyecto)."""
    return label.lower().replace(" ", "_")


# ═════════════════════════════════════════════════════════════════════════════
# ETAPA 1 — GUILLOTINA RegEx
# ═════════════════════════════════════════════════════════════════════════════

# Palabras clave que marcan el inicio del "Tail Babbling".  El corte se hace
# JUSTO ANTES de la primera aparición de cualquiera de estas (case-insensitive).
TAIL_BABBLING_KEYWORDS: List[str] = [
    "NOTIFICATION:",
    "strutConnector",
    "attRot",
    "guiActiveUnfocused",
    "TheNitrome",
    "NOTICE:",
    "NOTIFY:",
    "NOTIFIED:",
    "NOTIFcation:",
    "NOTION:",
    "NOTATION:",
    "NOTIFIER:",
    "NOTIFICATIONS:",
    "RECOMMENDATION(S):",
    "RECOMMENDATION:",
    "guiIcon",
    "attRotitas",
]


def construir_regex_guillotina(
    keywords: Sequence[str] = TAIL_BABBLING_KEYWORDS,
) -> re.Pattern:
    """
    Compila un patrón que detecta la PRIMERA aparición de cualquier keyword.

    - ``re.escape`` neutraliza metacaracteres (los ':' y los '()' de
      'RECOMMENDATION(S):' se tratan literalmente).
    - Se ordena por longitud descendente para que, ante prefijos solapados
      (p. ej. 'NOTIFICATIONS:' vs 'NOTIFICATION:', o 'attRotitas' vs 'attRot'),
      la alternancia prefiera la variante más larga en la misma posición.
    - ``re.IGNORECASE`` → corte insensible a mayúsculas/minúsculas.
    """
    ordenadas = sorted(set(keywords), key=len, reverse=True)
    patron = "|".join(re.escape(k) for k in ordenadas)
    return re.compile(patron, flags=re.IGNORECASE)


# Patrón compilado una sola vez (reutilizable y eficiente sobre miles de filas).
_GUILLOTINA_RE = construir_regex_guillotina()


def cortar_tail_babbling(
    texto: object,
    patron: re.Pattern = _GUILLOTINA_RE,
) -> object:
    """
    Corta `texto` justo antes de la primera keyword de ruido.

    - Si no hay keyword → devuelve el texto original (sólo .strip()).
    - Si `texto` no es str (NaN/None) → se preserva tal cual (no se inventa nada).
    """
    if not isinstance(texto, str):
        return texto  # preserva NaN/None sin convertirlo a "0" ni a ""
    m = patron.search(texto)
    if m is None:
        return texto.strip()
    return texto[: m.start()].strip()


def limpiar_reportes_tail_babbling(
    df: pd.DataFrame,
    col_origen: str = "generated_report",
    col_destino: str = "generated_report_clean",
    patron: re.Pattern = _GUILLOTINA_RE,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Aplica la guillotina a `col_origen` y guarda el resultado en `col_destino`.

    Devuelve una COPIA del DataFrame (no muta el original).
    """
    if col_origen not in df.columns:
        raise KeyError(
            f"La columna de origen '{col_origen}' no existe. "
            f"Columnas disponibles: {list(df.columns)[:12]}..."
        )

    out = df.copy()
    out[col_destino] = out[col_origen].map(lambda t: cortar_tail_babbling(t, patron))

    if verbose:
        orig_len = out[col_origen].fillna("").astype(str).str.len()
        clean_len = out[col_destino].fillna("").astype(str).str.len()
        cortados = int((clean_len < orig_len).sum())
        chars_elim = int((orig_len - clean_len).clip(lower=0).sum())
        pct = 100.0 * cortados / max(len(out), 1)
        print("─" * 70)
        print("ETAPA 1 · GUILLOTINA (Tail Babbling)")
        print("─" * 70)
        print(f"  Filas totales           : {len(out):,}")
        print(f"  Reportes recortados     : {cortados:,}  ({pct:.1f}%)")
        print(f"  Caracteres eliminados   : {chars_elim:,}")
        print(f"  Columna creada          : '{col_destino}'")
        print(f"  Keywords vigiladas      : {len(set(TAIL_BABBLING_KEYWORDS))}")
        print()
    return out


# ═════════════════════════════════════════════════════════════════════════════
# ETAPA 2 — RE-ETIQUETADO con CheXbert (opcional, real)
# ═════════════════════════════════════════════════════════════════════════════

def reetiquetar_con_chexbert(
    df: pd.DataFrame,
    col_texto: str = "generated_report_clean",
    prefijo_pred: str = "pred_clean_",
    labeler: object = None,
    batch_size: int = 16,
    device: Optional[str] = None,
    labels: Sequence[str] = CHEXBERT_LABELS,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Pasa `col_texto` por el CheXbert oficial y crea columnas `prefijo_pred<cat>`.

    Reutiliza ``chexbert_official.CheXbertLabeler`` (mismos pesos de Stanford AIMI
    usados en evaluate_final.py).  Es la única etapa que requiere torch/GPU; por
    eso el import es perezoso y la etapa es OPCIONAL (puedes generar las columnas
    `pred_clean_*` por tu cuenta y saltarte esta función).
    """
    if col_texto not in df.columns:
        raise KeyError(f"No existe la columna de texto limpio '{col_texto}'.")

    try:
        from chexbert_official import CheXbertLabeler  # import perezoso
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "No se pudo importar chexbert_official.CheXbertLabeler. "
            "Genera las columnas pred_clean_* por separado o instala el entorno "
            f"con torch/transformers. Detalle: {exc}"
        )

    if labeler is None:
        labeler = CheXbertLabeler(device=device)

    out = df.copy()
    reportes = out[col_texto].fillna("").astype(str).tolist()

    if verbose:
        print("─" * 70)
        print("ETAPA 2 · RE-ETIQUETADO con CheXbert (texto limpio)")
        print("─" * 70)
        print(f"  Reportes a etiquetar    : {len(reportes):,}")
        print(f"  batch_size              : {batch_size}")

    resultados: List[Dict[str, float]] = labeler.label_batch(
        reportes, batch_size=batch_size
    )

    for lbl in labels:
        col = f"{prefijo_pred}{_canon(lbl)}"
        out[col] = [r.get(lbl, float("nan")) for r in resultados]

    if verbose:
        print(f"  Columnas creadas        : {len(labels)} ('{prefijo_pred}*')")
        print()
    return out


# ═════════════════════════════════════════════════════════════════════════════
# Helper — traer el Ground Truth oficial (mimic-cxr-2.0.0-chexpert.csv)
# ═════════════════════════════════════════════════════════════════════════════

def merge_ground_truth(
    df: pd.DataFrame,
    chexpert_csv: str | Path,
    claves: Sequence[str] = ("subject_id", "study_id"),
    labels: Sequence[str] = CHEXBERT_LABELS,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Une el GT oficial de CheXpert al DataFrame por las claves disponibles.

    Trae las columnas con los nombres CRUDOS ('No Finding', 'Cardiomegaly', ...),
    que son las que `evaluar_metricas_estrictas` reconoce como Ground Truth.
    Los valores conservan su semántica original (1.0/0.0/-1.0/NaN): NO se imputan.
    """
    gt = pd.read_csv(chexpert_csv)
    keys = [k for k in claves if k in df.columns and k in gt.columns]
    if not keys:
        raise KeyError(
            f"No hay claves comunes para el merge. df={list(df.columns)[:8]}... "
            f"gt={list(gt.columns)[:8]}..."
        )

    cols_gt = keys + [l for l in labels if l in gt.columns]
    merged = df.merge(gt[cols_gt].drop_duplicates(subset=keys), on=keys, how="left")

    if verbose:
        n_match = merged[labels[0]].notna().sum() if labels[0] in merged.columns else 0
        print("─" * 70)
        print("HELPER · MERGE Ground Truth oficial CheXpert")
        print("─" * 70)
        print(f"  Claves de unión         : {keys}")
        print(f"  Filas con GT tras merge : {n_match:,}/{len(merged):,}")
        print()
    return merged


# ═════════════════════════════════════════════════════════════════════════════
# ETAPA 3 — MÉTRICAS ESTRICTAS (respetando NaN)
# ═════════════════════════════════════════════════════════════════════════════

def _serie_present(series: pd.Series, uncertain_as_positive: bool = True) -> pd.Series:
    """
    Binariza a "presente" (hallazgo afirmativo) SIN rellenar NaN con ceros.

    Clave de la regla estricta: NaN NO se equipara a 1.0, por lo que NaN cae
    naturalmente en "no presente" sin ninguna imputación destructiva
    (no se hace fillna). Así, GT=NaN y Pred=NaN → ambos "no presente" → TN.

    - presente := (valor == 1.0)            [siempre]
    - presente := (valor == 1.0 o == -1.0)  [si uncertain_as_positive=True]
      (-1.0 = incierto = mencionado; coherente con _chexpert_comparison_metrics
       de evaluate_final.py)
    """
    vals = pd.to_numeric(series, errors="coerce")
    if uncertain_as_positive:
        present = (vals == 1.0) | (vals == -1.0)
    else:
        present = vals == 1.0
    return present.astype(bool)


def _resolver_columna_gt(
    df: pd.DataFrame,
    label: str,
    gt_prefijos: Sequence[str],
) -> Optional[str]:
    """Busca la columna de Ground Truth de `label` probando convenciones comunes."""
    canon = _canon(label)
    candidatos: List[str] = [label, canon]
    for p in gt_prefijos:
        candidatos.append(f"{p}{canon}")
        candidatos.append(f"{p}{label}")
    for c in candidatos:
        if c in df.columns:
            return c
    return None


def _safe_nanmean(valores: Sequence[float]) -> float:
    """np.nanmean robusto: devuelve NaN (sin warning) si TODO es NaN."""
    arr = np.asarray(list(valores), dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return float(np.nanmean(arr))


def evaluar_metricas_estrictas(
    df: pd.DataFrame,
    prefijo_pred: str = "pred_clean_",
    gt_prefijos: Sequence[str] = ("gt_chexpert_", "gt_", "chexpert_", "true_"),
    uncertain_as_positive: bool = True,
    labels: Sequence[str] = CHEXBERT_LABELS,
    verbose: bool = True,
    titulo: str = "MÉTRICAS ESTRICTAS (NaN-aware)",
) -> pd.DataFrame:
    """
    Compara GT vs predicciones limpias clase por clase con regla estricta de NaN.

    Definición de la matriz de confusión (clase positiva = "hallazgo presente"):
        TP : GT presente   y  Pred presente
        FP : GT no-presente y  Pred presente     (incluye GT=NaN & Pred=1.0)
        FN : GT presente    y  Pred no-presente
        TN : GT no-presente y  Pred no-presente  (incluye GT=NaN & Pred=NaN)

    Reglas pedidas:
        • NO se imputan NaN a 0: NaN entra como "no presente" de forma natural.
        • GT=NaN & Pred=NaN cuenta como coincidencia correcta (True Negative).
        • Matches = TP + TN (acuerdos presente/ausente, incluye NaN==NaN).
        • Precisión = TP/(TP+FP), Recall = TP/(TP+FN); si el denominador es 0 →
          NaN (no 0), para no penalizar artificialmente.
        • F1 por categoría = media armónica; el resumen macro usa np.nanmean.

    Returns:
        DataFrame (una fila por categoría) con columnas:
        categoria, n, gt_pos, pred_pos, matches, exact_matches,
        TP, FP, FN, TN, precision, recall, f1.
    """
    filas: List[Dict[str, float]] = []
    omitidas: List[str] = []

    for lbl in labels:
        canon = _canon(lbl)
        col_pred = f"{prefijo_pred}{canon}"
        col_gt = _resolver_columna_gt(df, lbl, gt_prefijos)

        if col_pred not in df.columns or col_gt is None:
            omitidas.append(lbl)
            filas.append(
                {
                    "categoria": lbl,
                    "n": 0, "gt_pos": np.nan, "pred_pos": np.nan,
                    "matches": np.nan, "exact_matches": np.nan,
                    "TP": np.nan, "FP": np.nan, "FN": np.nan, "TN": np.nan,
                    "precision": np.nan, "recall": np.nan, "f1": np.nan,
                }
            )
            continue

        gt_present = _serie_present(df[col_gt], uncertain_as_positive)
        pred_present = _serie_present(df[col_pred], uncertain_as_positive)

        tp = int((gt_present & pred_present).sum())
        fp = int((~gt_present & pred_present).sum())
        fn = int((gt_present & ~pred_present).sum())
        tn = int((~gt_present & ~pred_present).sum())
        n = tp + fp + fn + tn
        matches = tp + tn  # acuerdo presente/ausente (NaN==NaN cuenta como TN)

        # Coincidencias EXACTAS de valor crudo (1==1, 0==0, -1==-1, NaN==NaN)
        g_raw = pd.to_numeric(df[col_gt], errors="coerce")
        p_raw = pd.to_numeric(df[col_pred], errors="coerce")
        exact = int(((g_raw == p_raw) | (g_raw.isna() & p_raw.isna())).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        if not (np.isnan(precision) or np.isnan(recall)) and (precision + recall) > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = float("nan")

        filas.append(
            {
                "categoria": lbl,
                "n": n,
                "gt_pos": int(gt_present.sum()),
                "pred_pos": int(pred_present.sum()),
                "matches": matches,
                "exact_matches": exact,
                "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                "precision": precision, "recall": recall, "f1": f1,
            }
        )

    res = pd.DataFrame(filas)

    if verbose:
        _imprimir_reporte(res, titulo=titulo, omitidas=omitidas, prefijo_pred=prefijo_pred)

    return res


def _fmt(x: float) -> str:
    """Formatea floats; NaN → '  n/a '."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "  n/a "
    return f"{x:6.4f}"


def _celda_int(valor: object, ancho: int) -> str:
    """Formatea un entero a `ancho`; NaN/None → 'n/a' alineado a la derecha."""
    if valor is None or (isinstance(valor, float) and np.isnan(valor)):
        return f"{'n/a':>{ancho}}"
    return f"{int(valor):>{ancho}}"


def _imprimir_reporte(
    res: pd.DataFrame,
    titulo: str,
    omitidas: Sequence[str],
    prefijo_pred: str,
) -> None:
    """Imprime el reporte detallado por categoría + resúmenes macro y micro.

    Tabla compacta (≤72 col) con las 4 métricas pedidas: Matches, Precisión,
    Recall y F1.  Las columnas auxiliares (TP/FP/FN/TN/exact_matches) quedan en
    el DataFrame devuelto para análisis fino.
    """
    SEP = "═" * 76
    print(SEP)
    print(f"  ETAPA 3 · {titulo}")
    print(f"  (pred: '{prefijo_pred}*'  |  NaN = 'no mencionado', NO imputado a 0)")
    print(SEP)
    header = (
        f"  {'Categoría':<28}{'Matches':>8}{'GTpos':>7}"
        f"{'Prec.':>10}{'Recall':>10}{'F1':>10}"
    )
    print(header)
    print("  " + "─" * 73)

    for _, r in res.iterrows():
        print(
            f"  {r['categoria']:<28}{_celda_int(r['matches'], 8)}"
            f"{_celda_int(r['gt_pos'], 7)}"
            f"{_fmt(r['precision']):>10}{_fmt(r['recall']):>10}{_fmt(r['f1']):>10}"
        )

    print("  " + "─" * 73)

    # ── Resumen MACRO (promedio por categoría, ignorando NaN) ────────────────
    macro_p = _safe_nanmean(res["precision"])
    macro_r = _safe_nanmean(res["recall"])
    macro_f1 = _safe_nanmean(res["f1"])
    print(
        f"  {'MACRO (np.nanmean)':<28}{'':>8}{'':>7}"
        f"{_fmt(macro_p):>10}{_fmt(macro_r):>10}{_fmt(macro_f1):>10}"
    )

    # ── Resumen MICRO (pooled TP/FP/FN sobre las 14 categorías) ──────────────
    tp_t = float(np.nansum(res["TP"]))
    fp_t = float(np.nansum(res["FP"]))
    fn_t = float(np.nansum(res["FN"]))
    micro_p = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else float("nan")
    micro_r = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else float("nan")
    if not (np.isnan(micro_p) or np.isnan(micro_r)) and (micro_p + micro_r) > 0:
        micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r)
    else:
        micro_f1 = float("nan")
    print(
        f"  {'MICRO (pooled)':<28}{'':>8}{'':>7}"
        f"{_fmt(micro_p):>10}{_fmt(micro_r):>10}{_fmt(micro_f1):>10}"
    )
    print(SEP)

    if omitidas:
        faltan = ", ".join(omitidas)
        print(f"  ⚠ Categorías omitidas (sin columna GT o pred):")
        print(f"    {faltan}")
    print()


# ═════════════════════════════════════════════════════════════════════════════
# ORQUESTADOR — pipeline completo de 3 etapas
# ═════════════════════════════════════════════════════════════════════════════

def procesar_dataframe(
    df: pd.DataFrame,
    col_origen: str = "generated_report",
    col_destino: str = "generated_report_clean",
    relabel: bool = False,
    chexpert_csv: Optional[str | Path] = None,
    prefijo_pred: str = "pred_clean_",
    uncertain_as_positive: bool = True,
    device: Optional[str] = None,
    labeler: object = None,
    also_dirty: bool = False,
    prefijo_pred_dirty: str = "pred_chexpert_",
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Ejecuta las 3 etapas y devuelve (df_procesado, df_metricas_clean, df_metricas_dirty).

    Etapas:
        1. Guillotina   → crea `col_destino`.
        2. (opcional) Re-etiquetado CheXbert sobre `col_destino` → `prefijo_pred*`.
           Si `also_dirty=True` además etiqueta el texto SUCIO original
           (`col_origen`) con `prefijo_pred_dirty*` para comparación antes/después,
           reutilizando el MISMO `labeler` (carga el modelo una sola vez).
        3. (opcional GT) merge del CSV oficial + métricas estrictas (clean, y
           dirty si `also_dirty`).

    `df_metricas_dirty` es None salvo que `also_dirty=True` y `relabel=True`.
    """
    # ── Etapa 2 requiere un labeler; lo creamos UNA vez y lo reutilizamos ─────
    if relabel and labeler is None:
        try:
            from chexbert_official import CheXbertLabeler  # import perezoso
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "No se pudo importar chexbert_official.CheXbertLabeler. "
                f"Detalle: {exc}"
            )
        if verbose:
            print("Cargando modelo CheXbert (una sola vez)...\n")
        labeler = CheXbertLabeler(device=device)

    # Etapa 1
    df = limpiar_reportes_tail_babbling(
        df, col_origen=col_origen, col_destino=col_destino, verbose=verbose
    )

    # Etapa 2 (real, opcional) — texto LIMPIO → pred_clean_*
    if relabel:
        df = reetiquetar_con_chexbert(
            df, col_texto=col_destino, prefijo_pred=prefijo_pred,
            labeler=labeler, device=device, verbose=verbose,
        )
        # Texto SUCIO original → pred_chexpert_* (para el antes/después)
        if also_dirty:
            df = reetiquetar_con_chexbert(
                df, col_texto=col_origen, prefijo_pred=prefijo_pred_dirty,
                labeler=labeler, device=device, verbose=verbose,
            )

    # GT oficial (opcional pero necesario para la etapa 3)
    if chexpert_csv is not None:
        df = merge_ground_truth(df, chexpert_csv, verbose=verbose)

    # Etapa 3 — métricas sobre el texto LIMPIO
    res_clean = evaluar_metricas_estrictas(
        df, prefijo_pred=prefijo_pred,
        uncertain_as_positive=uncertain_as_positive, verbose=verbose,
        titulo="MÉTRICAS ESTRICTAS · texto LIMPIO (después)",
    )

    # Etapa 3 (bis) — métricas sobre el texto SUCIO, para comparar
    res_dirty: Optional[pd.DataFrame] = None
    if also_dirty and relabel:
        res_dirty = evaluar_metricas_estrictas(
            df, prefijo_pred=prefijo_pred_dirty,
            uncertain_as_positive=uncertain_as_positive, verbose=verbose,
            titulo="MÉTRICAS ESTRICTAS · texto SUCIO (antes)",
        )

    return df, res_clean, res_dirty


# ═════════════════════════════════════════════════════════════════════════════
# DEMO sintética — corre sin GPU ni CheXbert (valida etapas 1 y 3)
# ═════════════════════════════════════════════════════════════════════════════

def _demo() -> None:
    print("\n" + "#" * 78)
    print("#  DEMO sintética (sin GPU): guillotina + métricas estrictas NaN-aware")
    print("#" * 78 + "\n")

    NaN = float("nan")
    # Reportes con y sin Tail Babbling
    df = pd.DataFrame(
        {
            "generated_report": [
                "findings: lungs clear. no effusion. impression: normal chest. "
                "NOTIFICATION: dr. smith was paged at 3am regarding...",
                "findings: moderate cardiomegaly. mild edema. "
                "RECOMMENDATION(S): follow-up ct in 6 months. attRot guiIcon",
                "findings: small left pleural effusion. no pneumothorax. "
                "strutConnector guiActiveUnfocused TheNitrome attRotitas",
                "findings: no acute cardiopulmonary process.",  # sin babbling
            ],
            # ── Ground Truth oficial (nombres crudos; semántica 1/0/-1/NaN) ──
            "Cardiomegaly":     [0.0, 1.0, NaN, 0.0],
            "Edema":            [NaN, 1.0, NaN, NaN],
            "Pleural Effusion": [0.0, NaN, 1.0, 0.0],
            "Pneumothorax":     [0.0, NaN, 0.0, NaN],
            # ── pred_clean_* SIMULADAS (lo que daría CheXbert sobre el texto limpio) ──
            "pred_clean_cardiomegaly":     [0.0, 1.0, NaN, 0.0],   # 1 acierto pos
            "pred_clean_edema":            [NaN, 1.0, NaN, NaN],   # 1 acierto pos
            "pred_clean_pleural_effusion": [0.0, NaN, 1.0, 0.0],   # 1 acierto pos
            "pred_clean_pneumothorax":     [0.0, NaN, 0.0, NaN],   # todos negativos/NaN → TN
        }
    )

    # Etapa 1
    df = limpiar_reportes_tail_babbling(df)
    print("Comparativa original → limpio:")
    for i in range(len(df)):
        orig = df["generated_report"].iloc[i]
        clean = df["generated_report_clean"].iloc[i]
        print(f"  [{i}] orig ({len(orig):3d}c): ...{orig[-48:]!r}")
        print(f"      clean({len(clean):3d}c): {clean!r}")
    print()

    # Etapa 3 (sobre las 4 categorías con datos; el resto saldrá 'n/a')
    evaluar_metricas_estrictas(df, titulo="MÉTRICAS ESTRICTAS · DEMO")

    print(
        "Nota: en la DEMO sólo 4 de 14 categorías tienen columnas (GT + pred_clean);\n"
        "      las demás aparecen como 'n/a' por diseño (no se inventan métricas).\n"
        "      GT=NaN & Pred=NaN se contó como TN (ver columna Matches).\n"
    )


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Guillotina anti Tail-Babbling + métricas CheXpert estrictas (NaN-aware).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--csv", type=str, default=None,
                   help="CSV de entrada con la columna de reportes. Sin esto → DEMO.")
    p.add_argument("--col-origen", type=str, default="generated_report",
                   help="Columna con el reporte generado (default: generated_report).")
    p.add_argument("--col-destino", type=str, default="generated_report_clean",
                   help="Columna de salida limpia (default: generated_report_clean).")
    p.add_argument("--chexpert-csv", type=str, default=None,
                   help="CSV oficial de CheXpert para traer el Ground Truth (merge por study_id).")
    p.add_argument("--relabel", action="store_true",
                   help="Re-etiquetar el texto limpio con CheXbert (requiere torch/GPU).")
    p.add_argument("--also-dirty", action="store_true",
                   help="Además etiqueta el texto SUCIO original (pred_chexpert_*) "
                        "y calcula sus métricas, para comparar antes/después "
                        "reutilizando el mismo modelo CheXbert. Requiere --relabel.")
    p.add_argument("--device", type=str, default=None,
                   help="Dispositivo para CheXbert: 'cuda', 'cpu' o None (autodetecta).")
    p.add_argument("--prefijo-pred", type=str, default="pred_clean_",
                   help="Prefijo de las columnas de predicción limpia (default: pred_clean_).")
    p.add_argument("--no-uncertain-pos", action="store_true",
                   help="Tratar incierto (-1.0) como negativo en vez de positivo.")
    p.add_argument("--out", type=str, default=None,
                   help="CSV de salida con el DataFrame procesado. "
                        "Si se omite → '<input>_clean.csv' junto al de entrada.")
    p.add_argument("--out-metricas", type=str, default=None,
                   help="CSV de salida con las métricas por categoría (texto limpio). "
                        "Si se omite → '<input>_metricas_estrictas.csv'.")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)

    if args.csv is None:
        _demo()
        return 0

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: no existe el CSV de entrada: {csv_path}", file=sys.stderr)
        return 1

    print(f"Cargando: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  Filas: {len(df):,}  |  Columnas: {len(df.columns)}\n")

    if args.also_dirty and not args.relabel:
        print("AVISO: --also-dirty requiere --relabel; se ignora el antes/después.\n",
              file=sys.stderr)

    df, res_clean, res_dirty = procesar_dataframe(
        df,
        col_origen=args.col_origen,
        col_destino=args.col_destino,
        relabel=args.relabel,
        chexpert_csv=args.chexpert_csv,
        prefijo_pred=args.prefijo_pred,
        uncertain_as_positive=not args.no_uncertain_pos,
        device=args.device,
        also_dirty=args.also_dirty,
        verbose=True,
    )

    # ── Rutas de salida: si no se especifican, se AUTO-DERIVAN del input ──────
    #    Así "todo queda guardado en CSV" aunque no pases --out / --out-metricas.
    stem = csv_path.with_suffix("")            # .../predicciones_Config_B1
    out_df = Path(args.out) if args.out else Path(f"{stem}_clean.csv")
    out_met = (Path(args.out_metricas) if args.out_metricas
               else Path(f"{stem}_metricas_estrictas.csv"))

    print("─" * 70)
    print("GUARDADO DE RESULTADOS (CSV)")
    print("─" * 70)

    # 1) DataFrame procesado completo (incluye generated_report_clean,
    #    pred_clean_* y, si aplica, pred_chexpert_* + GT del merge).
    out_df.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_df, index=False)
    print(f"  ✓ DataFrame procesado     → {out_df}  ({len(df):,} filas, {len(df.columns)} cols)")

    # 2) Métricas por categoría del texto LIMPIO (siempre).
    out_met.parent.mkdir(parents=True, exist_ok=True)
    res_clean.to_csv(out_met, index=False)
    print(f"  ✓ Métricas (limpio)       → {out_met}  ({len(res_clean)} categorías)")

    # 3) Métricas del texto SUCIO (solo con --also-dirty) para el antes/después.
    if res_dirty is not None:
        out_met_dirty = Path(f"{stem}_metricas_estrictas_SUCIO.csv")
        res_dirty.to_csv(out_met_dirty, index=False)
        print(f"  ✓ Métricas (sucio/antes)  → {out_met_dirty}  ({len(res_dirty)} categorías)")

        # Tabla comparativa compacta antes/después (macro nanmean).
        comp = res_clean[["categoria", "precision", "recall", "f1"]].merge(
            res_dirty[["categoria", "precision", "recall", "f1"]],
            on="categoria", suffixes=("_limpio", "_sucio"),
        )
        out_comp = Path(f"{stem}_comparativa_antes_despues.csv")
        comp.to_csv(out_comp, index=False)
        print(f"  ✓ Comparativa antes/desp. → {out_comp}")
        print()
        print("  Resumen MACRO (np.nanmean)   Precisión    Recall        F1")
        for etiqueta, r in (("SUCIO (antes)", res_dirty), ("LIMPIO (después)", res_clean)):
            mp, mr, mf = (_safe_nanmean(r["precision"]),
                          _safe_nanmean(r["recall"]), _safe_nanmean(r["f1"]))
            print(f"    {etiqueta:<22}{_fmt(mp):>10}{_fmt(mr):>10}{_fmt(mf):>10}")

    print("─" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
