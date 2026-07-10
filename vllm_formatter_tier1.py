#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vllm_formatter_tier1.py — Formateo LLM externo (vLLM + Llama 3.1 8B) · FINAL TIER 1
==================================================================================
Wrapper delgado que REUTILIZA sin modificar toda la maquinaria validada de
``vllm_formatter.py`` (carga del modelo, construcción de prompts con el chat
template de Llama 3.1, batching continuo de vLLM, prefix caching y modo resume)
para formatear los reportes crudos del Experimento **FINAL TIER 1**, tanto del
prompt DISEÑADO (RAMA 2) como del prompt NUEVO/CoT (RAMA 4).

Tareas (``--tasks``):
  • disenado (RAMA 2) — prompt diseñado:
      in : evaluation/predicciones_Exp_Final_Tier1.csv
      out: evaluation/predicciones_Exp_Final_Tier1_llmexterno_llama8b.csv
  • nuevo    (RAMA 4) — prompt NUEVO/CoT:
      in : exp1_cot/exp1_raw_inference_final_experiment_tier1.csv
      out: exp1_cot/exp1_final_formatted_final_experiment_tier1_llama8b.csv

Ambas salidas preservan TODAS las columnas del CSV de entrada y añaden
``formatted_report`` + ``formatter_provider``.

Se mantiene EXACTAMENTE el mismo SYSTEM_PROMPT y parámetros de muestreo
(temperature=0.0 determinista) que el resto del proyecto, para que las métricas
del reporte formateado sean comparables con Fase 2/3/Exp2/Exp3/FINAL.

Optimización GPU: idéntica al original — una sola carga de pesos, continuous
batching (PagedAttention), prefix caching del system prompt repetido y
gpu_memory_utilization=0.92 (~42 GB de la L40S). No se fija batch manual: vLLM
agrupa/libera dinámicamente para saturar la GPU.

Uso:
    python vllm_formatter_tier1.py --tasks disenado --resume    # RAMA 2
    python vllm_formatter_tier1.py --tasks nuevo --resume       # RAMA 4 (CoT)
    python vllm_formatter_tier1.py --tasks both --resume        # ambas
    python vllm_formatter_tier1.py --tasks nuevo --limit 20     # smoke

Autor : Evelyn Silva Rozas
Fecha : Julio 2026
"""

from __future__ import annotations

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

# ── Reutilización ESTRICTA de la maquinaria del formateador original ─────────
from vllm_formatter import (
    FormatTask,
    cargar_vllm,
    construir_prompts,
    procesar_tarea,
    DEFAULT_MODEL,
)

BASE = Path(__file__).resolve().parent

# ── Rutas del Experimento FINAL TIER 1 (carpeta SEPARADA) ────────────────────
FINAL_TIER1_EVAL_DIR = BASE / "final_experiment_tier1_results-50pct" / "evaluation"
FINAL_TIER1_EXP1_DIR = BASE / "final_experiment_tier1_results-50pct" / "exp1_cot"

# ── REGISTRO DE TAREAS ───────────────────────────────────────────────────────
#   disenado → RAMA 2 (prompt diseñado, crudo → Llama 3.1 8B)
#   nuevo    → RAMA 4 (prompt NUEVO/CoT, crudo → Llama 3.1 8B)
TASKS: Dict[str, FormatTask] = {
    "disenado": FormatTask(
        name="disenado_final_tier1",
        in_csv=FINAL_TIER1_EVAL_DIR / "predicciones_Exp_Final_Tier1.csv",
        out_csv=FINAL_TIER1_EVAL_DIR / "predicciones_Exp_Final_Tier1_llmexterno_llama8b.csv",
        col_input="generated_report",
        col_output="formatted_report",
    ),
    "nuevo": FormatTask(
        name="nuevo_final_tier1",
        in_csv=FINAL_TIER1_EXP1_DIR / "exp1_raw_inference_final_experiment_tier1.csv",
        out_csv=FINAL_TIER1_EXP1_DIR / "exp1_final_formatted_final_experiment_tier1_llama8b.csv",
        col_input="generated_report",
        col_output="formatted_report",
    ),
}


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Formateo LLM externo (vLLM + Llama 3.1 8B) para el FINAL TIER 1.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--tasks", nargs="+", default=["disenado"],
                   choices=list(TASKS.keys()) + ["both"],
                   help="Tareas a ejecutar: disenado (RAMA 2) y/o nuevo/CoT (RAMA 4).")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="ID HuggingFace del modelo (mirror ungated de Llama 3.1 8B).")
    p.add_argument("--dtype", default="float16",
                   choices=["float16", "bfloat16", "auto"],
                   help="Precisión de pesos (FP16 = más rápido en L40S).")
    p.add_argument("--max-model-len", type=int, default=2560,
                   help="Longitud total (input + salida).")
    p.add_argument("--max-tokens", type=int, default=512,
                   help="Tokens máximos a generar por reporte.")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="0.0 = determinista (reproducible).")
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--gpu-memory-util", type=float, default=0.92,
                   help="Fracción de VRAM que vLLM puede usar (L40S 46GB → ~42GB).")
    p.add_argument("--no-prefix-cache", action="store_true",
                   help="Desactivar prefix caching (no recomendado).")
    p.add_argument("--limit", type=int, default=0,
                   help="Procesar solo las primeras N filas (0 = todas).")
    p.add_argument("--resume", action="store_true",
                   help="Saltar reportes ya formateados (lee CSV de salida si existe).")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)

    # Expandir alias "both" preservando el orden disenado → nuevo.
    tareas: List[FormatTask] = []
    for t in args.tasks:
        if t == "both":
            tareas = [TASKS["disenado"], TASKS["nuevo"]]
            break
        tareas.append(TASKS[t])

    # Validar entradas antes de cargar el modelo (evita cargar vLLM en vano).
    faltantes = [t for t in tareas if not t.in_csv.exists()]
    if faltantes:
        for t in faltantes:
            print(f"ERROR: no existe el CSV de entrada de '{t.name}': {t.in_csv}",
                  file=sys.stderr)
        print("\n  Genera primero las inferencias crudas:", file=sys.stderr)
        print("    disenado → python evaluate_final_experiment_tier1.py --resume",
              file=sys.stderr)
        print("    nuevo    → python exp1_cot_inference_final_experiment_tier1.py "
              "--resume --skip-evaluation", file=sys.stderr)
        return 1

    print("=" * 80)
    print("  vLLM Formatter TIER 1 — Llama 3.1 8B (FP16)  ·  L40S")
    print("=" * 80)
    print(f"  Tareas  : {[t.name for t in tareas]}")
    print(f"  Resume  : {args.resume}")
    if args.limit:
        print(f"  Limit   : {args.limit}")
    t_global = time.time()

    # ── 1) Cargar modelo UNA sola vez (compartido por todas las tareas) ─────
    llm = cargar_vllm(args)
    from vllm import SamplingParams
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    # ── 2) Procesar cada tarea reutilizando procesar_tarea del original ──────
    for t in tareas:
        try:
            procesar_tarea(llm, sampling_params, tokenizer, t, args)
        except Exception as exc:  # pragma: no cover
            print(f"\n  [{t.name}] ERROR FATAL: {exc}", file=sys.stderr)
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print(f"  TIEMPO TOTAL: {(time.time() - t_global)/60:.1f} min")
    for t in tareas:
        print(f"  Salida [{t.name}]: {t.out_csv}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
