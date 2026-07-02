#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reevaluar_phase2_full.py — Re-evaluación COMPLETA SUCIO vs LIMPIO (Fase 2, 19 137)
==================================================================================

Igual que ``reevaluar_sucio_vs_limpio.py`` PERO:

  1. Trunca las entradas de Clinical BERTScore a 510 tokens (reutiliza
     ``evaluar_version_segura`` de exp1_metrics_comparison.py). Esto evita el
     ``RuntimeError: expanded size of the tensor (647) must match (512)`` que
     abortó la corrida original sobre los 19 137 reportes (Bio_ClinicalBERT
     tope = 512 posiciones; los reportes SUCIOS con Tail Babbling lo superan).

  2. Hace CHECKPOINT por versión: guarda las métricas por fila de "sucio" y de
     "limpio" en .npz apenas terminan, de modo que si el pod se pausa durante
     la versión limpia NO se pierden los ~37 min de RadGraph sucio. Al reanudar
     (mismo comando) detecta los checkpoints y continúa. Con --fresh los ignora.

El formato de salida es IDÉNTICO al de reevaluar_sucio_vs_limpio.py
(``media_sucio`` / ``media_limpio`` / ``delta_medio`` ...), por lo que
``crear_csvs_comparativos.py`` lo consume sin cambios.

Uso
---
    # Corrida completa (19 137, GPU) — RECOMENDADO en tmux:
    python reevaluar_phase2_full.py

    # Smoke test:
    python reevaluar_phase2_full.py --limit 200

    # Reanudar tras pausa: mismo comando (usa los checkpoints automáticamente)
    # Empezar de cero ignorando checkpoints:
    python reevaluar_phase2_full.py --fresh

Autor : Evelyn Silva Rozas
Fecha : Junio 2026
"""

from __future__ import annotations

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

# Maquinaria de métricas EXACTA del proyecto.
#  · evaluar_version_segura  → BLEU/ROUGE/RadGraph (texto completo) + BERTScore
#                              con truncado a 510 tokens (Bio_ClinicalBERT).
#  · construir_detalle / construir_resumen / imprimir_resumen → formato
#                              _sucio/_limpio idéntico al reeval de Fase 2.
from exp1_metrics_comparison import evaluar_version_segura
from reevaluar_sucio_vs_limpio import (
    cargar_radgraph,
    cargar_bertscorer,
    construir_detalle,
    construir_resumen,
    imprimir_resumen,
    CLINICAL_BERT_MODEL,
    CLINICAL_BERT_LAYERS,
    _METRICAS_BASE,
)

# ─────────────────────────────────────────────────────────────────────────────
# Rutas por defecto
# ─────────────────────────────────────────────────────────────────────────────
try:
    from config import config
    EVAL_DIR = config.paths.base_dir / "phase2_results-50pct" / "evaluation"
except Exception:
    EVAL_DIR = Path("phase2_results-50pct") / "evaluation"

IN_CSV_DEFAULT      = EVAL_DIR / "predicciones_Config_B1_clean.csv"
OUT_DETALLE_DEFAULT = EVAL_DIR / "predicciones_Config_B1_clean_reeval_detalle.csv"
OUT_RESUMEN_DEFAULT = EVAL_DIR / "predicciones_Config_B1_clean_reeval_resumen.csv"

# Checkpoints por versión (se borran al terminar con éxito).
CKPT_DIR = EVAL_DIR / ".reeval_ckpt"


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────
def _ckpt_path(etiqueta: str) -> Path:
    return CKPT_DIR / f"reeval_{etiqueta}.npz"


def _guardar_ckpt(etiqueta: str, res: Dict[str, np.ndarray], n_filas: int) -> None:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    p = _ckpt_path(etiqueta)
    np.savez(p, _n_filas=np.array([n_filas]), **res)
    print(f"  [ckpt] guardado {p.name}  ({', '.join(res.keys())})")


def _cargar_ckpt(etiqueta: str, n_filas: int, claves: Sequence[str]) -> Optional[Dict[str, np.ndarray]]:
    """Carga el checkpoint si existe, cubre el mismo nº de filas y tiene TODAS las claves."""
    p = _ckpt_path(etiqueta)
    if not p.exists():
        return None
    try:
        d = np.load(p, allow_pickle=False)
        if int(d["_n_filas"][0]) != n_filas:
            print(f"  [ckpt] {p.name} ignorado (n_filas {int(d['_n_filas'][0])} != {n_filas}).")
            return None
        faltan = [k for k in claves if k not in d.files]
        if faltan:
            print(f"  [ckpt] {p.name} ignorado (faltan métricas: {faltan}).")
            return None
        res = {k: d[k] for k in d.files if k != "_n_filas"}
        print(f"  [ckpt] REUTILIZANDO {p.name}  ({', '.join(res.keys())}) — se omite recálculo.")
        return res
    except Exception as exc:  # pragma: no cover
        print(f"  [ckpt] error leyendo {p.name}: {exc} → se recalcula.", file=sys.stderr)
        return None


def _limpiar_ckpts() -> None:
    for et in ("sucio", "limpio"):
        p = _ckpt_path(et)
        if p.exists():
            p.unlink()
    try:
        CKPT_DIR.rmdir()
    except OSError:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Re-evaluación COMPLETA SUCIO vs LIMPIO (Fase 2) con truncado "
                    "seguro de BERTScore y checkpoint por versión.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--csv", type=str, default=str(IN_CSV_DEFAULT),
                   help="CSV con reference_report, generated_report y generated_report_clean.")
    p.add_argument("--col-ref", type=str, default="reference_report")
    p.add_argument("--col-sucio", type=str, default="generated_report")
    p.add_argument("--col-limpio", type=str, default="generated_report_clean")
    p.add_argument("--limit", type=int, default=0, help="Evaluar solo N filas (0 = todas).")
    p.add_argument("--no-radgraph", action="store_true", help="Omitir F1-RadGraph.")
    p.add_argument("--no-bertscore", action="store_true", help="Omitir Clinical BERTScore.")
    p.add_argument("--bert-model", type=str, default=CLINICAL_BERT_MODEL)
    p.add_argument("--bert-layers", type=int, default=CLINICAL_BERT_LAYERS)
    p.add_argument("--bert-batch", type=int, default=64)
    p.add_argument("--radgraph-chunk", type=int, default=256)
    p.add_argument("--device", type=str, default=None, help="'cuda'/'cpu'/None (autodetecta).")
    p.add_argument("--out-detalle", type=str, default=str(OUT_DETALLE_DEFAULT))
    p.add_argument("--out-resumen", type=str, default=str(OUT_RESUMEN_DEFAULT))
    p.add_argument("--fresh", action="store_true",
                   help="Ignorar checkpoints existentes y recalcular todo.")
    return p.parse_args(argv)


def _cuda_disponible() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _claves_esperadas(no_radgraph: bool, no_bertscore: bool) -> list:
    claves = ["bleu4", "rougeL"]
    if not no_radgraph:
        claves.append("radgraph")
    if not no_bertscore:
        claves += ["bertscore_p", "bertscore_r", "bertscore_f1"]
    return claves


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────
def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    t0 = time.time()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: no existe el CSV: {csv_path}", file=sys.stderr)
        return 1

    print("\n" + "=" * 82)
    print("  RE-EVALUACIÓN COMPLETA · SUCIO vs LIMPIO (Fase 2) — truncado seguro + checkpoint")
    print("=" * 82)
    print(f"Cargando: {csv_path}")
    df = pd.read_csv(csv_path)
    if args.limit and args.limit > 0:
        df = df.head(args.limit).copy()
    n_filas = len(df)
    print(f"  Filas a evaluar: {n_filas:,}")

    for col in (args.col_ref, args.col_sucio, args.col_limpio):
        if col not in df.columns:
            print(f"ERROR: falta la columna '{col}' en el CSV.", file=sys.stderr)
            print(f"  Columnas disponibles: {list(df.columns)[:12]}...", file=sys.stderr)
            return 1

    refs       = df[args.col_ref].fillna("").astype(str).str.strip().tolist()
    hyps_dirty = df[args.col_sucio].fillna("").astype(str).str.strip().tolist()
    hyps_clean = df[args.col_limpio].fillna("").astype(str).str.strip().tolist()

    if args.fresh:
        print("  --fresh: se ignoran checkpoints previos.")
        _limpiar_ckpts()

    claves = _claves_esperadas(args.no_radgraph, args.no_bertscore)

    # ── Cargar modelos pesados UNA sola vez (solo si hace falta calcular algo) ─
    ck_dirty = None if args.fresh else _cargar_ckpt("sucio", n_filas, claves)
    ck_clean = None if args.fresh else _cargar_ckpt("limpio", n_filas, claves)
    necesita_modelos = (ck_dirty is None) or (ck_clean is None)

    device = args.device or ("cuda" if _cuda_disponible() else "cpu")
    print(f"  Device: {device}")

    f1rg = None
    bert_scorer = None
    bert_tokenizer = None
    if necesita_modelos:
        f1rg = None if args.no_radgraph else cargar_radgraph()
        bert_scorer = None if args.no_bertscore else cargar_bertscorer(
            args.bert_model, args.bert_layers, device
        )
        if bert_scorer is not None:
            from transformers import AutoTokenizer
            bert_tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
    else:
        print("  Ambas versiones presentes en checkpoint → no se cargan modelos.")

    # ── Versión SUCIO ─────────────────────────────────────────────────────────
    if ck_dirty is not None:
        res_dirty = ck_dirty
    else:
        res_dirty = evaluar_version_segura(
            refs, hyps_dirty, "sucio",
            f1rg=f1rg, bert_scorer=bert_scorer, bert_tokenizer=bert_tokenizer,
            radgraph_chunk=args.radgraph_chunk, bert_batch=args.bert_batch,
        )
        _guardar_ckpt("sucio", res_dirty, n_filas)

    # ── Versión LIMPIO ────────────────────────────────────────────────────────
    if ck_clean is not None:
        res_clean = ck_clean
    else:
        res_clean = evaluar_version_segura(
            refs, hyps_clean, "limpio",
            f1rg=f1rg, bert_scorer=bert_scorer, bert_tokenizer=bert_tokenizer,
            radgraph_chunk=args.radgraph_chunk, bert_batch=args.bert_batch,
        )
        _guardar_ckpt("limpio", res_clean, n_filas)

    # ── Detalle + resumen (formato _sucio/_limpio, idéntico a Fase 2) ─────────
    detalle = construir_detalle(df, res_dirty, res_clean, args.col_sucio, args.col_limpio)
    resumen = construir_resumen(detalle)
    imprimir_resumen(resumen)

    out_det = Path(args.out_detalle)
    out_res = Path(args.out_resumen)
    out_det.parent.mkdir(parents=True, exist_ok=True)
    detalle.to_csv(out_det, index=False)
    resumen.to_csv(out_res, index=False)

    print(f"\n  ✓ Detalle por reporte → {out_det}  ({len(detalle):,} filas, {len(detalle.columns)} cols)")
    print(f"  ✓ Resumen agregado    → {out_res}  ({len(resumen)} métricas)")

    # Éxito → limpiar checkpoints
    _limpiar_ckpts()
    print(f"\n  Tiempo total: {(time.time() - t0)/60:.1f} min")
    print("=" * 82 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
