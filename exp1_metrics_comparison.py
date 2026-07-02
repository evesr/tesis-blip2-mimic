#!/usr/bin/env python3
"""
exp1_metrics_comparison.py — Métricas comparativas: BLIP-2 crudo vs LLM-formateado
==================================================================================
Tercera etapa del Experimento 1.

Lee  exp1_final_formatted_results.csv  (con reference_report, generated_report y
formatted_report) y calcula las MISMAS métricas usadas en la evaluación de
Fase 3, sobre DOS versiones de la hipótesis:

    • ORIGINAL  → generated_report   (salida cruda del modelo BLIP-2, prompt CoT)
    • FORMATEADO→ formatted_report   (reescrito por el LLM externo gpt-4o-mini)

Métricas (idénticas a evaluate_final.py / reevaluar_sucio_vs_limpio.py)
-----------------------------------------------------------------------
    1. BLEU-4             (n-gramas, smoothing method1)
    2. ROUGE-L            (F-measure con stemmer)
    3. F1-RadGraph        (RG_ERbar, radgraph-xl, reward_level='all', índice 2)
    4. Clinical BERTScore (P / R / F1 con emilyalsentzer/Bio_ClinicalBERT)

TODO el cómputo pesado (RadGraph + Clinical BERTScore) corre en GPU. Se reutilizan
las funciones de reevaluar_sucio_vs_limpio.py para garantizar EXACTAMENTE la
misma configuración (mismos pesos BLEU, mismo RG_ERbar, mismo Bio_ClinicalBERT
sin baseline de rescalado).

Salidas (en phase3_results-50pct/exp1_cot/)
-------------------------------------------
    • exp1_comparison_detalle.csv  → por reporte: *_original, *_formateado, *_delta
    • exp1_comparison_resumen.csv  → agregado: media/std/delta/mejora% por métrica

Uso
---
    python exp1_metrics_comparison.py                 # Test Set completo
    python exp1_metrics_comparison.py --limit 200     # smoke test
    python exp1_metrics_comparison.py --no-radgraph   # omitir RadGraph (rápido)

Autor : Evelyn Silva Rozas
Fecha : Junio 2026
"""

# ─────────────────────────────────────────────────────────────────────────────
# Variables de entorno (ANTES de imports de transformers/tokenizers)
# ─────────────────────────────────────────────────────────────────────────────
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

# Reutiliza la maquinaria de métricas EXACTA de la re-evaluación de Fase 3.
# (importar este módulo aplica el patch de tokenizer y prepara nltk).
# Se importan las funciones INDIVIDUALES (no evaluar_version) para poder truncar
# ÚNICAMENTE las entradas de Clinical BERTScore (Bio_ClinicalBERT tope=512 tokens)
# sin afectar BLEU/ROUGE/RadGraph, que usan el texto completo.
from reevaluar_sucio_vs_limpio import (
    bleu_rouge_por_fila,
    radgraph_por_fila,
    bertscore_clinico_por_fila,
    cargar_radgraph,
    cargar_bertscorer,
    CLINICAL_BERT_MODEL,
    CLINICAL_BERT_LAYERS,
    _METRICAS_BASE,
)

# Directorio del experimento (coincide con exp1_cot_inference.py)
# CAMBIO Fase 3 → Fase 2: las salidas viven bajo phase2_results-50pct/exp1_cot.
try:
    from config import config
    EXP1_DIR = config.paths.base_dir / "phase2_results-50pct" / "exp1_cot"
except Exception:
    EXP1_DIR = Path("phase2_results-50pct") / "exp1_cot"

IN_CSV: Path       = EXP1_DIR / "exp1_final_formatted_results.csv"
OUT_DETALLE: Path  = EXP1_DIR / "exp1_comparison_detalle.csv"
OUT_RESUMEN: Path  = EXP1_DIR / "exp1_comparison_resumen.csv"

# Nombres legibles de cada métrica para los reportes en consola
_NOMBRES = {
    "bleu4": "BLEU-4", "rougeL": "ROUGE-L", "radgraph": "F1-RadGraph",
    "bertscore_p": "ClinBERT-P", "bertscore_r": "ClinBERT-R",
    "bertscore_f1": "ClinBERT-F1",
}


# ─────────────────────────────────────────────────────────────────────────────
# 1.  EVALUACIÓN POR VERSIÓN (con truncación segura SOLO para BERTScore)
# ─────────────────────────────────────────────────────────────────────────────

def _truncar_para_bert(textos: Sequence[str], tokenizer, max_tokens: int = 510) -> list:
    """
    Trunca cada texto a <=max_tokens TOKENS del tokenizador BERT clínico.

    Necesario porque Bio_ClinicalBERT (BERT-base) tiene un máximo de 512 tokens
    de posición; los reportes Chain-of-Thought (que re-emiten el prompt + ambas
    tareas) pueden superarlo. La truncación se aplica SOLO a la entrada de
    BERTScore; BLEU/ROUGE/RadGraph siguen usando el texto completo.
    """
    out = []
    for t in textos:
        t = t if t else "."
        ids = tokenizer.encode(t, add_special_tokens=False,
                               truncation=True, max_length=max_tokens)
        dec = tokenizer.decode(ids, skip_special_tokens=True)
        out.append(dec if dec.strip() else ".")
    return out


def evaluar_version_segura(
    refs: Sequence[str],
    hyps: Sequence[str],
    etiqueta: str,
    f1rg=None,
    bert_scorer=None,
    bert_tokenizer=None,
    radgraph_chunk: int = 256,
    bert_batch: int = 64,
) -> Dict[str, np.ndarray]:
    """
    Calcula las 4 métricas por fila para una columna de hipótesis.

    Réplica de reevaluar_sucio_vs_limpio.evaluar_version PERO truncando las
    entradas de Clinical BERTScore a 510 tokens (evita el RuntimeError de
    Bio_ClinicalBERT cuando un reporte supera los 512 tokens de posición).
    """
    print(f"\n{'─'*70}\n  Evaluando versión: {etiqueta.upper()}  ({len(hyps):,} reportes)\n{'─'*70}")
    res: Dict[str, np.ndarray] = {}

    # BLEU-4 + ROUGE-L (texto completo)
    b, r = bleu_rouge_por_fila(refs, hyps, etiqueta=f"[{etiqueta}]")
    res["bleu4"] = b
    res["rougeL"] = r

    # F1-RadGraph (texto completo; radgraph-xl trunca internamente si hace falta)
    if f1rg is not None:
        res["radgraph"] = radgraph_por_fila(
            f1rg, refs, hyps, chunk_size=radgraph_chunk, etiqueta=f"[{etiqueta}]"
        )

    # Clinical BERTScore (entradas truncadas a 510 tokens)
    if bert_scorer is not None:
        refs_t = _truncar_para_bert(refs, bert_tokenizer) if bert_tokenizer is not None else list(refs)
        hyps_t = _truncar_para_bert(hyps, bert_tokenizer) if bert_tokenizer is not None else list(hyps)
        bs = bertscore_clinico_por_fila(
            bert_scorer, refs_t, hyps_t, batch_size=bert_batch, etiqueta=f"[{etiqueta}]"
        )
        res["bertscore_p"] = bs["p"]
        res["bertscore_r"] = bs["r"]
        res["bertscore_f1"] = bs["f1"]

    return res


# ─────────────────────────────────────────────────────────────────────────────
# 2.  CONSTRUCCIÓN DE DETALLE Y RESUMEN (ORIGINAL vs FORMATEADO)
# ─────────────────────────────────────────────────────────────────────────────

def construir_detalle(
    df_in: pd.DataFrame,
    res_orig: Dict[str, np.ndarray],
    res_fmt: Dict[str, np.ndarray],
    col_orig: str,
    col_fmt: str,
) -> pd.DataFrame:
    """DataFrame por reporte: longitudes + métricas original/formateado/delta."""
    det = pd.DataFrame()
    for key in ("dicom_id", "subject_id", "study_id"):
        if key in df_in.columns:
            det[key] = df_in[key].values

    len_orig = df_in[col_orig].fillna("").astype(str).str.len().values
    len_fmt  = df_in[col_fmt].fillna("").astype(str).str.len().values
    det["len_original"]   = len_orig
    det["len_formateado"] = len_fmt
    det["chars_delta"]    = len_fmt - len_orig

    for m in _METRICAS_BASE:
        if m in res_orig and m in res_fmt:
            o = res_orig[m]
            f = res_fmt[m]
            det[f"{m}_original"]   = o
            det[f"{m}_formateado"] = f
            det[f"{m}_delta"]      = f - o
    return det


def construir_resumen(det: pd.DataFrame) -> pd.DataFrame:
    """Agrega media/std original vs formateado + delta y % de mejora por métrica."""
    filas = []
    for m in _METRICAS_BASE:
        co, cf, cd = f"{m}_original", f"{m}_formateado", f"{m}_delta"
        if co not in det.columns:
            continue
        o = pd.to_numeric(det[co], errors="coerce")
        f = pd.to_numeric(det[cf], errors="coerce")
        d = pd.to_numeric(det[cd], errors="coerce")
        media_o = float(np.nanmean(o))
        filas.append({
            "metrica":        m,
            "media_original": media_o,
            "media_formateado": float(np.nanmean(f)),
            "delta_medio":    float(np.nanmean(d)),
            "std_original":   float(np.nanstd(o)),
            "std_formateado": float(np.nanstd(f)),
            "mejora_pct":     (100.0 * float(np.nanmean(d)) / abs(media_o)
                               if media_o not in (0.0,) and not np.isnan(media_o) and media_o != 0
                               else np.nan),
            "n_mejoran":      int((d > 0).sum()),
            "n_empeoran":     int((d < 0).sum()),
            "n_iguales":      int((d == 0).sum()),
        })
    return pd.DataFrame(filas)


def _fmt(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "  n/a "
    return f"{x:7.4f}"


def imprimir_resumen(resumen: pd.DataFrame) -> None:
    SEP = "═" * 82
    print("\n" + SEP)
    print("  RESUMEN · EXP1 — BLIP-2 CRUDO  →  LLM-FORMATEADO (gpt-4o-mini)")
    print(SEP)
    print(f"  {'Métrica':<14}{'Original':>11}{'Formateado':>12}{'Δ medio':>10}"
          f"{'Mejora%':>10}{'↑mejoran':>10}{'↓empeoran':>11}")
    print("  " + "─" * 79)
    for _, r in resumen.iterrows():
        nombre = _NOMBRES.get(r["metrica"], r["metrica"])
        print(f"  {nombre:<14}{_fmt(r['media_original']):>11}{_fmt(r['media_formateado']):>12}"
              f"{_fmt(r['delta_medio']):>10}{_fmt(r['mejora_pct']):>10}"
              f"{int(r['n_mejoran']):>10}{int(r['n_empeoran']):>11}")
    print(SEP)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  CLI / MAIN
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Comparación de métricas (BLEU/ROUGE/RadGraph/ClinicalBERT): "
                    "reporte BLIP-2 crudo vs reporte formateado por LLM externo.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--in-csv", type=str, default=str(IN_CSV),
                   help="CSV con reference_report, generated_report y formatted_report.")
    p.add_argument("--col-ref", type=str, default="reference_report")
    p.add_argument("--col-original", type=str, default="generated_report")
    p.add_argument("--col-formateado", type=str, default="formatted_report")
    p.add_argument("--limit", type=int, default=0, help="Evaluar solo N filas (0 = todas).")
    p.add_argument("--no-radgraph", action="store_true", help="Omitir F1-RadGraph.")
    p.add_argument("--no-bertscore", action="store_true", help="Omitir Clinical BERTScore.")
    p.add_argument("--bert-model", type=str, default=CLINICAL_BERT_MODEL)
    p.add_argument("--bert-layers", type=int, default=CLINICAL_BERT_LAYERS)
    p.add_argument("--bert-batch", type=int, default=64)
    p.add_argument("--radgraph-chunk", type=int, default=256)
    p.add_argument("--device", type=str, default=None, help="'cuda'/'cpu'/None (autodetecta).")
    p.add_argument("--out-detalle", type=str, default=str(OUT_DETALLE))
    p.add_argument("--out-resumen", type=str, default=str(OUT_RESUMEN))
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    t0 = time.time()

    in_csv = Path(args.in_csv)
    if not in_csv.exists():
        print(f"ERROR: no existe el CSV: {in_csv}\n"
              f"  Ejecuta antes: python llm_formatter_pipeline.py", file=sys.stderr)
        return 1

    print("\n" + "=" * 82)
    print("  EXPERIMENTO 1 — MÉTRICAS COMPARATIVAS (GPU)")
    print("=" * 82)

    df = pd.read_csv(in_csv)
    if args.limit and args.limit > 0:
        df = df.head(args.limit).copy()

    for col in (args.col_ref, args.col_original, args.col_formateado):
        if col not in df.columns:
            print(f"ERROR: falta la columna '{col}' en {in_csv}.", file=sys.stderr)
            print(f"  Columnas: {list(df.columns)}", file=sys.stderr)
            return 1
    print(f"  Filas a evaluar: {len(df):,}")

    refs       = df[args.col_ref].fillna("").astype(str).str.strip().tolist()
    hyps_orig  = df[args.col_original].fillna("").astype(str).str.strip().tolist()
    hyps_fmt   = df[args.col_formateado].fillna("").astype(str).str.strip().tolist()

    # ── Cargar modelos pesados UNA vez (GPU), reutilizados en ambas versiones ─
    device = args.device or ("cuda" if _cuda_disponible() else "cpu")
    print(f"  Device: {device}")
    f1rg = None if args.no_radgraph else cargar_radgraph()
    bert_scorer = None if args.no_bertscore else cargar_bertscorer(
        args.bert_model, args.bert_layers, device
    )
    # Tokenizer clínico para truncar las entradas de BERTScore a 510 tokens
    bert_tokenizer = None
    if bert_scorer is not None:
        from transformers import AutoTokenizer
        bert_tokenizer = AutoTokenizer.from_pretrained(args.bert_model)

    # ── Evaluar ambas versiones ──────────────────────────────────────────────
    res_orig = evaluar_version_segura(
        refs, hyps_orig, "original",
        f1rg=f1rg, bert_scorer=bert_scorer, bert_tokenizer=bert_tokenizer,
        radgraph_chunk=args.radgraph_chunk, bert_batch=args.bert_batch,
    )
    res_fmt = evaluar_version_segura(
        refs, hyps_fmt, "formateado",
        f1rg=f1rg, bert_scorer=bert_scorer, bert_tokenizer=bert_tokenizer,
        radgraph_chunk=args.radgraph_chunk, bert_batch=args.bert_batch,
    )

    # ── Detalle + resumen ────────────────────────────────────────────────────
    detalle = construir_detalle(df, res_orig, res_fmt,
                                args.col_original, args.col_formateado)
    resumen = construir_resumen(detalle)
    imprimir_resumen(resumen)

    out_det = Path(args.out_detalle)
    out_res = Path(args.out_resumen)
    out_det.parent.mkdir(parents=True, exist_ok=True)
    detalle.to_csv(out_det, index=False)
    resumen.to_csv(out_res, index=False)

    print(f"\n  ✓ Detalle por reporte → {out_det}  ({len(detalle):,} filas, {len(detalle.columns)} cols)")
    print(f"  ✓ Resumen comparativo → {out_res}  ({len(resumen)} métricas)")
    print(f"\n  Tiempo total: {(time.time() - t0)/60:.1f} min")
    print("=" * 82 + "\n")
    return 0


def _cuda_disponible() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


if __name__ == "__main__":
    sys.exit(main())
