#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reevaluar_sucio_vs_limpio.py — Re-evaluación textual SUCIO vs LIMPIO
====================================================================

Re-evalúa las predicciones del modelo ANTES (``generated_report``, con Tail
Babbling) y DESPUÉS de la guillotina (``generated_report_clean``) contra el
``reference_report``, calculando para CADA reporte cuatro métricas:

    1. BLEU-4            — n-gramas (idéntico a evaluate_final.py)
    2. ROUGE-L           — subsecuencia común más larga (F-measure)
    3. F1-RadGraph       — RG_ERbar (radgraph-xl, reward_level='all', índice 2)
    4. Clinical BERTScore— P / R / F1 con emilyalsentzer/Bio_ClinicalBERT

Salidas (todo en CSV, junto al archivo de entrada):
    • <input>_reeval_detalle.csv   → una fila por reporte con TODAS las métricas
                                     (sucio, limpio y delta = limpio − sucio).
    • <input>_reeval_resumen.csv   → agregados (media, std) sucio vs limpio + Δ.

Uso
---
    # Evaluación completa (GPU recomendada):
    python reevaluar_sucio_vs_limpio.py \
        --csv phase2_results-50pct/evaluation/predicciones_Config_B1_clean.csv

    # Prueba rápida sobre 200 reportes:
    python reevaluar_sucio_vs_limpio.py --csv ... --limit 200

    # Desactivar una métrica pesada:
    python reevaluar_sucio_vs_limpio.py --csv ... --no-radgraph
    python reevaluar_sucio_vs_limpio.py --csv ... --no-bertscore

Notas
-----
    • NO usa entornos virtuales: corre sobre el Python del sistema.
    • Reutiliza EXACTAMENTE la configuración de métricas de evaluate_final.py
      (mismos pesos BLEU, mismo ROUGE-L con stemmer, mismo RG_ERbar).
    • Clinical BERTScore usa Bio_ClinicalBERT SIN baseline de rescalado
      (no existe baseline oficial para modelos clínicos) → valores en escala
      cruda de cosine similarity; lo importante es el delta limpio−sucio.

Autor : Evelyn Silva Rozas
Fecha : Junio 2026
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# 0.  Patch de compatibilidad tokenizer — DEBE ir ANTES de importar radgraph
#     (idéntico a run_tournament.py / evaluate_final.py)
# ─────────────────────────────────────────────────────────────────────────────
from transformers import PreTrainedTokenizerBase

if not hasattr(PreTrainedTokenizerBase, "encode_plus"):
    def _ep(self, text, text_pair=None, add_special_tokens=True,
            padding=False, truncation=False, max_length=None,
            stride=0, return_tensors=None, **kwargs):
        return self.__call__(text, text_pair=text_pair,
                             add_special_tokens=add_special_tokens,
                             padding=padding, truncation=truncation,
                             max_length=max_length, stride=stride,
                             return_tensors=return_tensors, **kwargs)
    PreTrainedTokenizerBase.encode_plus = _ep

if not hasattr(PreTrainedTokenizerBase, "build_inputs_with_special_tokens"):
    def _bi(self, t0, t1=None):
        if t1 is None:
            return [self.cls_token_id] + t0 + [self.sep_token_id]
        return [self.cls_token_id] + t0 + [self.sep_token_id] + t1 + [self.sep_token_id]
    PreTrainedTokenizerBase.build_inputs_with_special_tokens = _bi

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Imports
# ─────────────────────────────────────────────────────────────────────────────
import argparse
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer as rouge_scorer_module

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# Modelo clínico por defecto para BERTScore (estándar de facto en RRG).
CLINICAL_BERT_MODEL = "emilyalsentzer/Bio_ClinicalBERT"
CLINICAL_BERT_LAYERS = 12  # Bio_ClinicalBERT = BERT-base (12 capas ocultas)


# ═════════════════════════════════════════════════════════════════════════════
# 2.  Helpers de métricas textuales (BLEU-4 / ROUGE-L)
# ═════════════════════════════════════════════════════════════════════════════

_rouge_scorer = rouge_scorer_module.RougeScorer(["rougeL"], use_stemmer=True)
_smooth = SmoothingFunction().method1


def bleu4(reference: str, hypothesis: str) -> float:
    """BLEU-4 con suavizado method1 (idéntico a evaluate_final._bleu4)."""
    if not reference or not hypothesis:
        return 0.0
    ref_tok = nltk.word_tokenize(reference.lower())
    hyp_tok = nltk.word_tokenize(hypothesis.lower())
    if not ref_tok or not hyp_tok:
        return 0.0
    return float(
        sentence_bleu(
            [ref_tok], hyp_tok,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=_smooth,
        )
    )


def rougeL(reference: str, hypothesis: str) -> float:
    """ROUGE-L F-measure con stemmer (idéntico a evaluate_final._rouge_l)."""
    if not reference or not hypothesis:
        return 0.0
    return float(_rouge_scorer.score(reference, hypothesis)["rougeL"].fmeasure)


def bleu_rouge_por_fila(
    refs: Sequence[str],
    hyps: Sequence[str],
    etiqueta: str = "",
) -> Tuple[np.ndarray, np.ndarray]:
    """Calcula BLEU-4 y ROUGE-L por fila. Devuelve (bleu[], rouge[])."""
    n = len(refs)
    b = np.empty(n, dtype=np.float64)
    r = np.empty(n, dtype=np.float64)
    desc = f"BLEU-4+ROUGE-L {etiqueta}".strip()
    for i in tqdm(range(n), desc=desc, unit="rep"):
        ref, hyp = refs[i], hyps[i]
        b[i] = bleu4(ref, hyp)
        r[i] = rougeL(ref, hyp)
    return b, r


# ═════════════════════════════════════════════════════════════════════════════
# 3.  F1-RadGraph (batched, por muestra → RG_ERbar)
# ═════════════════════════════════════════════════════════════════════════════

def radgraph_por_fila(
    f1rg,
    refs: Sequence[str],
    hyps: Sequence[str],
    chunk_size: int = 256,
    etiqueta: str = "",
) -> np.ndarray:
    """
    F1-RadGraph (RG_ERbar) por muestra, en lotes.

    Con reward_level='all', f1rg(...) devuelve:
        mean_reward, reward_list, hyp_anns, ref_anns
    donde reward_list = ([simple], [partial], [complete]); el índice 2
    (complete) es RG_ERbar — el mismo que usa evaluate_final.py.
    """
    n = len(refs)
    out = np.full(n, np.nan, dtype=np.float64)
    desc = f"F1-RadGraph {etiqueta}".strip()
    for start in tqdm(range(0, n, chunk_size), desc=desc, unit="chunk"):
        end = min(start + chunk_size, n)
        chunk_refs = [refs[i] if refs[i] else "." for i in range(start, end)]
        chunk_hyps = [hyps[i] if hyps[i] else "." for i in range(start, end)]
        try:
            _, reward_list, _, _ = f1rg(refs=chunk_refs, hyps=chunk_hyps)
            complete = reward_list[2]  # lista RG_ERbar por muestra
            out[start:end] = np.asarray(complete, dtype=np.float64)
        except Exception as exc:  # pragma: no cover
            print(f"  [RadGraph] error en chunk [{start}:{end}]: {exc}",
                  file=sys.stderr)
    return out


# ═════════════════════════════════════════════════════════════════════════════
# 4.  Clinical BERTScore (batched → P / R / F1 por muestra)
# ═════════════════════════════════════════════════════════════════════════════

def _truncar_para_bert(textos: Sequence[str], tokenizer, max_tokens: int = 510) -> List[str]:
    """
    Trunca cada texto a <= max_tokens TOKENS del tokenizador BERT clínico.

    Bio_ClinicalBERT (BERT-base) tiene un máximo de 512 tokens de posición
    (510 + [CLS] + [SEP]); los reportes generados/formateados pueden
    superarlo (p. ej. 647 tokens visto en producción). La truncación se
    aplica SOLO a la entrada de BERTScore; BLEU/ROUGE/RadGraph siguen
    usando el texto completo.

    Réplica EXACTA de evaluate_phase3_puente._truncar_para_bert.
    """
    out: List[str] = []
    for t in textos:
        t = t if t else "."
        ids = tokenizer.encode(
            t, add_special_tokens=False,
            truncation=True, max_length=max_tokens,
        )
        dec = tokenizer.decode(ids, skip_special_tokens=True)
        out.append(dec if dec.strip() else ".")
    return out


# ─── Versión ANTIGUA sin truncación (rompe con reportes >512 tokens) ─────────
# Se conserva comentada por trazabilidad histórica:
#
# def bertscore_clinico_por_fila(scorer, refs, hyps, batch_size=64, etiqueta=""):
#     cands = [h if h else "." for h in hyps]
#     references = [r if r else "." for r in refs]
#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore")
#         P, R, F1 = scorer.score(cands, references, batch_size=batch_size, verbose=True)
#     return {"p": P.cpu().numpy().astype(np.float64), ...}
# ─────────────────────────────────────────────────────────────────────────────

def bertscore_clinico_por_fila(
    scorer,
    refs: Sequence[str],
    hyps: Sequence[str],
    batch_size: int = 64,
    etiqueta: str = "",
    max_bert_tokens: int = 510,
) -> Dict[str, np.ndarray]:
    """
    Clinical BERTScore P/R/F1 por muestra usando un BERTScorer ya cargado.

    Las cadenas vacías se sustituyen por '.' para no romper el tokenizador;
    su score resultará bajo de forma natural (no se imputa artificialmente).

    Trunca AMBOS textos (ref e hyp) a max_bert_tokens tokens del tokenizer
    interno del scorer ANTES de pasarlos a BERTScore, evitando el error de
    posiciones (>512) que ocurre con reportes largos formateados por LLM.
    """
    # Truncar a <=510 tokens usando el tokenizer del propio scorer.
    tokenizer = getattr(scorer, "_tokenizer", None)
    if tokenizer is not None:
        refs_safe = _truncar_para_bert(refs, tokenizer, max_tokens=max_bert_tokens)
        hyps_safe = _truncar_para_bert(hyps, tokenizer, max_tokens=max_bert_tokens)
    else:
        refs_safe = [r if r else "." for r in refs]
        hyps_safe = [h if h else "." for h in hyps]

    desc = f"Clinical BERTScore {etiqueta}".strip()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        P, R, F1 = scorer.score(
            hyps_safe, refs_safe,
            batch_size=batch_size,
            verbose=True,
        )
    return {
        "p": P.cpu().numpy().astype(np.float64),
        "r": R.cpu().numpy().astype(np.float64),
        "f1": F1.cpu().numpy().astype(np.float64),
    }


# ═════════════════════════════════════════════════════════════════════════════
# 5.  Orquestación: evaluar una versión (sucio o limpio)
# ═════════════════════════════════════════════════════════════════════════════

def evaluar_version(
    refs: Sequence[str],
    hyps: Sequence[str],
    etiqueta: str,
    f1rg=None,
    bert_scorer=None,
    radgraph_chunk: int = 256,
    bert_batch: int = 64,
) -> Dict[str, np.ndarray]:
    """Calcula las 4 métricas por fila para una columna de hipótesis."""
    print(f"\n{'─'*70}\n  Evaluando versión: {etiqueta.upper()}  ({len(hyps):,} reportes)\n{'─'*70}")
    res: Dict[str, np.ndarray] = {}

    # BLEU-4 + ROUGE-L (CPU, por fila)
    b, r = bleu_rouge_por_fila(refs, hyps, etiqueta=f"[{etiqueta}]")
    res["bleu4"] = b
    res["rougeL"] = r

    # F1-RadGraph (GPU, batched)
    if f1rg is not None:
        res["radgraph"] = radgraph_por_fila(
            f1rg, refs, hyps, chunk_size=radgraph_chunk, etiqueta=f"[{etiqueta}]"
        )

    # Clinical BERTScore (GPU, batched)
    if bert_scorer is not None:
        bs = bertscore_clinico_por_fila(
            bert_scorer, refs, hyps, batch_size=bert_batch, etiqueta=f"[{etiqueta}]"
        )
        res["bertscore_p"] = bs["p"]
        res["bertscore_r"] = bs["r"]
        res["bertscore_f1"] = bs["f1"]

    return res


# ═════════════════════════════════════════════════════════════════════════════
# 6.  Construcción del detalle por reporte + resumen
# ═════════════════════════════════════════════════════════════════════════════

_METRICAS_BASE = ["bleu4", "rougeL", "radgraph",
                  "bertscore_p", "bertscore_r", "bertscore_f1"]


def construir_detalle(
    df_in: pd.DataFrame,
    res_dirty: Dict[str, np.ndarray],
    res_clean: Dict[str, np.ndarray],
    col_dirty: str,
    col_clean: str,
) -> pd.DataFrame:
    """Arma el DataFrame de detalle por reporte (sucio, limpio y delta)."""
    det = pd.DataFrame()
    for key in ("dicom_id", "study_id"):
        if key in df_in.columns:
            det[key] = df_in[key].values

    # Longitudes y caracteres recortados (evidencia del Tail Babbling)
    len_dirty = df_in[col_dirty].fillna("").astype(str).str.len().values
    len_clean = df_in[col_clean].fillna("").astype(str).str.len().values
    det["len_sucio"] = len_dirty
    det["len_limpio"] = len_clean
    det["chars_recortados"] = np.clip(len_dirty - len_clean, 0, None)
    det["fue_recortado"] = det["chars_recortados"] > 0

    # Métricas por versión + delta
    for m in _METRICAS_BASE:
        if m in res_dirty and m in res_clean:
            d = res_dirty[m]
            c = res_clean[m]
            det[f"{m}_sucio"] = d
            det[f"{m}_limpio"] = c
            det[f"{m}_delta"] = c - d
    return det


def construir_resumen(det: pd.DataFrame) -> pd.DataFrame:
    """Agrega media/std sucio vs limpio + delta por métrica."""
    filas = []
    for m in _METRICAS_BASE:
        cs, cl, cd = f"{m}_sucio", f"{m}_limpio", f"{m}_delta"
        if cs not in det.columns:
            continue
        s = pd.to_numeric(det[cs], errors="coerce")
        l = pd.to_numeric(det[cl], errors="coerce")
        d = pd.to_numeric(det[cd], errors="coerce")
        filas.append({
            "metrica": m,
            "media_sucio": np.nanmean(s),
            "media_limpio": np.nanmean(l),
            "delta_medio": np.nanmean(d),
            "std_sucio": np.nanstd(s),
            "std_limpio": np.nanstd(l),
            "mejora_pct": (100.0 * np.nanmean(d) / abs(np.nanmean(s))
                           if np.nanmean(s) not in (0, np.nan) and not np.isnan(np.nanmean(s)) and np.nanmean(s) != 0
                           else np.nan),
            "n_mejoran": int((d > 0).sum()),
            "n_empeoran": int((d < 0).sum()),
            "n_iguales": int((d == 0).sum()),
        })
    return pd.DataFrame(filas)


def _fmt(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "  n/a "
    return f"{x:7.4f}"


def imprimir_resumen(resumen: pd.DataFrame) -> None:
    """Imprime la tabla comparativa sucio vs limpio en consola."""
    SEP = "═" * 78
    print("\n" + SEP)
    print("  RESUMEN · RE-EVALUACIÓN TEXTUAL  (SUCIO → LIMPIO)")
    print(SEP)
    print(f"  {'Métrica':<16}{'Sucio':>10}{'Limpio':>10}{'Δ medio':>10}"
          f"{'Mejora%':>10}{'↑mejoran':>10}{'↓empeoran':>11}")
    print("  " + "─" * 75)
    nombres = {
        "bleu4": "BLEU-4", "rougeL": "ROUGE-L", "radgraph": "F1-RadGraph",
        "bertscore_p": "ClinBERT-P", "bertscore_r": "ClinBERT-R",
        "bertscore_f1": "ClinBERT-F1",
    }
    for _, r in resumen.iterrows():
        nombre = nombres.get(r["metrica"], r["metrica"])
        print(f"  {nombre:<16}{_fmt(r['media_sucio']):>10}{_fmt(r['media_limpio']):>10}"
              f"{_fmt(r['delta_medio']):>10}{_fmt(r['mejora_pct']):>10}"
              f"{int(r['n_mejoran']):>10}{int(r['n_empeoran']):>11}")
    print(SEP)


# ═════════════════════════════════════════════════════════════════════════════
# 7.  Carga perezosa de modelos (RadGraph / Clinical BERTScore)
# ═════════════════════════════════════════════════════════════════════════════

def cargar_radgraph():
    """Instancia F1RadGraph con la MISMA config del proyecto."""
    print("Cargando F1RadGraph (reward_level='all', model_type='radgraph-xl')...")
    from radgraph import F1RadGraph
    return F1RadGraph(reward_level="all", model_type="radgraph-xl")


def cargar_bertscorer(model: str, num_layers: int, device: Optional[str]):
    """Instancia un BERTScorer clínico (carga el modelo una sola vez)."""
    print(f"Cargando Clinical BERTScore (model='{model}', num_layers={num_layers})...")
    import torch
    from bert_score import BERTScorer
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return BERTScorer(
        model_type=model,
        num_layers=num_layers,
        lang="en",
        rescale_with_baseline=False,  # sin baseline oficial para modelos clínicos
        device=device,
    )


# ═════════════════════════════════════════════════════════════════════════════
# 8.  CLI / main
# ═════════════════════════════════════════════════════════════════════════════

def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Re-evaluación textual SUCIO vs LIMPIO (BLEU-4, ROUGE-L, "
                    "F1-RadGraph, Clinical BERTScore) con detalle por reporte.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--csv", type=str, required=True,
                   help="CSV con reference_report, generated_report y "
                        "generated_report_clean (el *_clean.csv).")
    p.add_argument("--col-ref", type=str, default="reference_report",
                   help="Columna de referencia (default: reference_report).")
    p.add_argument("--col-sucio", type=str, default="generated_report",
                   help="Columna del reporte SUCIO (default: generated_report).")
    p.add_argument("--col-limpio", type=str, default="generated_report_clean",
                   help="Columna del reporte LIMPIO (default: generated_report_clean).")
    p.add_argument("--limit", type=int, default=0,
                   help="Evaluar solo las primeras N filas (0 = todas).")
    p.add_argument("--no-radgraph", action="store_true",
                   help="Omitir F1-RadGraph.")
    p.add_argument("--no-bertscore", action="store_true",
                   help="Omitir Clinical BERTScore.")
    p.add_argument("--bert-model", type=str, default=CLINICAL_BERT_MODEL,
                   help=f"Modelo para BERTScore (default: {CLINICAL_BERT_MODEL}).")
    p.add_argument("--bert-layers", type=int, default=CLINICAL_BERT_LAYERS,
                   help=f"Capa de representación para BERTScore (default: {CLINICAL_BERT_LAYERS}).")
    p.add_argument("--bert-batch", type=int, default=64,
                   help="Batch size de BERTScore (default: 64).")
    p.add_argument("--radgraph-chunk", type=int, default=256,
                   help="Tamaño de lote de RadGraph (default: 256).")
    p.add_argument("--device", type=str, default=None,
                   help="'cuda' / 'cpu' / None (autodetecta).")
    p.add_argument("--out-detalle", type=str, default=None,
                   help="CSV de detalle por reporte (default: <input>_reeval_detalle.csv).")
    p.add_argument("--out-resumen", type=str, default=None,
                   help="CSV de resumen agregado (default: <input>_reeval_resumen.csv).")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    t0 = time.time()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: no existe el CSV: {csv_path}", file=sys.stderr)
        return 1

    print(f"Cargando: {csv_path}")
    df = pd.read_csv(csv_path)
    if args.limit and args.limit > 0:
        df = df.head(args.limit).copy()
    print(f"  Filas a evaluar: {len(df):,}")

    # Validar columnas
    for col in (args.col_ref, args.col_sucio, args.col_limpio):
        if col not in df.columns:
            print(f"ERROR: falta la columna '{col}' en el CSV.", file=sys.stderr)
            print(f"  Columnas disponibles: {list(df.columns)[:10]}...", file=sys.stderr)
            return 1

    refs = df[args.col_ref].fillna("").astype(str).str.strip().tolist()
    hyps_dirty = df[args.col_sucio].fillna("").astype(str).str.strip().tolist()
    hyps_clean = df[args.col_limpio].fillna("").astype(str).str.strip().tolist()

    # ── Cargar modelos pesados UNA sola vez (reutilizados en ambas versiones) ─
    f1rg = None if args.no_radgraph else cargar_radgraph()
    bert_scorer = None if args.no_bertscore else cargar_bertscorer(
        args.bert_model, args.bert_layers, args.device
    )

    # ── Evaluar ambas versiones ──────────────────────────────────────────────
    res_dirty = evaluar_version(
        refs, hyps_dirty, "sucio",
        f1rg=f1rg, bert_scorer=bert_scorer,
        radgraph_chunk=args.radgraph_chunk, bert_batch=args.bert_batch,
    )
    res_clean = evaluar_version(
        refs, hyps_clean, "limpio",
        f1rg=f1rg, bert_scorer=bert_scorer,
        radgraph_chunk=args.radgraph_chunk, bert_batch=args.bert_batch,
    )

    # ── Detalle por reporte + resumen ────────────────────────────────────────
    detalle = construir_detalle(
        df, res_dirty, res_clean, args.col_sucio, args.col_limpio
    )
    resumen = construir_resumen(detalle)
    imprimir_resumen(resumen)

    # ── Guardar (rutas auto-derivadas si no se pasan) ────────────────────────
    stem = csv_path.with_suffix("")
    out_det = Path(args.out_detalle) if args.out_detalle else Path(f"{stem}_reeval_detalle.csv")
    out_res = Path(args.out_resumen) if args.out_resumen else Path(f"{stem}_reeval_resumen.csv")

    out_det.parent.mkdir(parents=True, exist_ok=True)
    detalle.to_csv(out_det, index=False)
    resumen.to_csv(out_res, index=False)

    print(f"\n  ✓ Detalle por reporte → {out_det}  ({len(detalle):,} filas, {len(detalle.columns)} cols)")
    print(f"  ✓ Resumen agregado    → {out_res}  ({len(resumen)} métricas)")
    print(f"\n  Tiempo total: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
