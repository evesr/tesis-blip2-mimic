#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vllm_formatter.py — Formateador de reportes con vLLM + Llama 3.1 8B (FP16)
==========================================================================

Reemplazo local de la Etapa A de Experimento 1 (LLM externo). Usa vLLM con
batching continuo (PagedAttention) en GPU para formatear los reportes crudos
del modelo BLIP-2 en el formato canónico FINDINGS / IMPRESSION, manteniendo
EXACTAMENTE el mismo system prompt que ``llm_formatter_pipeline.py`` para
que las métricas sean comparables con la corrida `simulate` previa.

Diseño clave
------------
* **Una sola carga del modelo** — se procesan TODAS las tareas (prompt diseñado
  + prompt nuevo) en una única invocación; vLLM solo carga pesos una vez.
* **Continuous batching** vía ``llm.generate(prompts, sampling_params)``:
  vLLM agrupa, rellena y libera memoria de forma dinámica → uso óptimo de la
  L40S sin fijar batch_size manualmente.
* **Prefix caching** (system prompt repetido en los 19 137 mensajes) ahorra
  ~30 % de cómputo en el prefill.
* **Resumable** — si el CSV de salida ya existe, se saltan los ``dicom_id``
  que ya tienen ``formatted_report`` no vacío.
* **Tareas configurables** vía ``TASKS`` (dict en cabecera) — basta editar el
  dict para añadir/cambiar fuentes; el lanzador tmux las activa con --tasks.

Uso
---
    # Ambas tareas, modelo + corrida completa (recomendado en tmux):
    python vllm_formatter.py --tasks disenado nuevo

    # Solo prompt diseñado:
    python vllm_formatter.py --tasks disenado

    # Smoke test:
    python vllm_formatter.py --tasks disenado --limit 20

    # Reanudar tras interrupción (mismo comando):
    python vllm_formatter.py --tasks disenado nuevo --resume

Salidas
-------
* ``predicciones_Config_B1_llmexterno_llama8b.csv``  (prompt diseñado)
* ``exp1_final_formatted_llama8b.csv``               (prompt nuevo / CoT)

Ambos CSV preservan TODAS las columnas del CSV de entrada y añaden:
    formatted_report     — salida del LLM (vacío si fallo)
    formatter_provider   — "vllm-llama-3.1-8b-fp16"

Autor : Evelyn Silva Rozas
Fecha : Junio 2026
"""

from __future__ import annotations

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Algunos repos HF se descargan más rápido si está habilitado el transferencia rápida.
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN — System prompt EXACTO al de llm_formatter_pipeline.py
# (mantener idéntico para que las comparaciones de métricas sean válidas).
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT: str = (
    "You are an expert radiologist. Extract the clinical findings from the "
    "provided raw text and format them into a standard radiology report with "
    "'FINDINGS' and 'IMPRESSION' sections. Do NOT invent or deduce any pathology "
    "not explicitly mentioned in the raw text. Remove any administrative noise, "
    "HTML tags, or non-medical babble. Output strictly the formatted report."
)

# Modelo (mirror ungated del oficial meta-llama/Meta-Llama-3.1-8B-Instruct;
# pesos idénticos, no requiere HF_TOKEN).
DEFAULT_MODEL: str = "NousResearch/Meta-Llama-3.1-8B-Instruct"

# Identificador que se escribe en la columna formatter_provider del CSV.
PROVIDER_TAG: str = "vllm-llama-3.1-8b-fp16"

BASE = Path(__file__).resolve().parent
EVAL_DIR = BASE / "phase2_results-50pct" / "evaluation"
EXP1_DIR = BASE / "phase2_results-50pct" / "exp1_cot"

# Phase 3 PUENTE: misma estructura de carpetas pero bajo phase3_puente_results-50pct/
PHASE3_EVAL_DIR = BASE / "phase3_puente_results-50pct" / "evaluation"
PHASE3_EXP1_DIR = BASE / "phase3_puente_results-50pct" / "exp1_cot"

# Experimento 2 (RadDINO + Puente): misma estructura bajo exp2_raddino_results-50pct/
EXP2_EVAL_DIR = BASE / "exp2_raddino_results-50pct" / "evaluation"
EXP2_EXP1_DIR = BASE / "exp2_raddino_results-50pct" / "exp1_cot"

# Experimento 3 (ViT F3 + Asclepius-R + Puente): misma estructura bajo exp3_asclepius_results-50pct/
EXP3_EVAL_DIR = BASE / "exp3_asclepius_results-50pct" / "evaluation"
EXP3_EXP1_DIR = BASE / "exp3_asclepius_results-50pct" / "exp1_cot"

# Experimento FINAL (RadDINO[LoRA Exp2] + Asclepius-R + Puente Full-FT, lr=1e-5):
# misma estructura bajo final_experiment_results-50pct/
FINAL_EVAL_DIR = BASE / "final_experiment_results-50pct" / "evaluation"
FINAL_EXP1_DIR = BASE / "final_experiment_results-50pct" / "exp1_cot"


# ─────────────────────────────────────────────────────────────────────────────
# REGISTRO DE TAREAS — añadir/quitar fuentes editando solo este dict.
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class FormatTask:
    name: str
    in_csv: Path
    out_csv: Path
    col_input: str = "generated_report"   # columna con el reporte crudo
    col_output: str = "formatted_report"  # columna a escribir


TASKS: Dict[str, FormatTask] = {
    # ── FASE 2 (50%) — ya completadas ─────────────────────────────────────────
    "disenado": FormatTask(
        name="disenado",
        in_csv=EVAL_DIR / "predicciones_Config_B1.csv",
        out_csv=EVAL_DIR / "predicciones_Config_B1_llmexterno_llama8b.csv",
    ),
    "nuevo": FormatTask(
        name="nuevo",
        in_csv=EXP1_DIR / "exp1_raw_inference_results.csv",
        out_csv=EXP1_DIR / "exp1_final_formatted_llama8b.csv",
    ),
    # ── FASE 3 PUENTE (50%) — pendientes ──────────────────────────────────────
    # Prompt diseñado: usa el CSV crudo de evaluate_phase3_puente.py
    "disenado_phase3puente": FormatTask(
        name="disenado_phase3puente",
        in_csv=PHASE3_EVAL_DIR / "predicciones_Phase3_Puente.csv",
        out_csv=PHASE3_EVAL_DIR / "predicciones_Phase3_Puente_llmexterno_llama8b.csv",
    ),
    # Prompt CoT/nuevo: usa el CSV crudo de exp1_cot_inference_phase3_puente.py
    # (el archivo input se materializa cuando termine la inferencia CoT).
    "nuevo_phase3puente": FormatTask(
        name="nuevo_phase3puente",
        in_csv=PHASE3_EXP1_DIR / "exp1_raw_inference_phase3_puente.csv",
        out_csv=PHASE3_EXP1_DIR / "exp1_final_formatted_phase3_puente_llama8b.csv",
    ),
    # ── EXPERIMENTO 2 (RadDINO + Puente, 50%) — LLM externo Llama 3.1 8B ───────
    # Prompt diseñado: usa el CSV crudo de evaluate_exp2_raddino.py
    "disenado_exp2raddino": FormatTask(
        name="disenado_exp2raddino",
        in_csv=EXP2_EVAL_DIR / "predicciones_Exp2_RadDINO.csv",
        out_csv=EXP2_EVAL_DIR / "predicciones_Exp2_RadDINO_llmexterno_llama8b.csv",
    ),
    # Prompt CoT/nuevo: usa el CSV crudo de exp1_cot_inference_exp2_raddino.py
    # (el archivo input se materializa cuando termine la inferencia CoT del Exp2).
    "nuevo_exp2raddino": FormatTask(
        name="nuevo_exp2raddino",
        in_csv=EXP2_EXP1_DIR / "exp1_raw_inference_exp2_raddino.csv",
        out_csv=EXP2_EXP1_DIR / "exp1_final_formatted_exp2_raddino_llama8b.csv",
    ),
    # ── EXPERIMENTO 3 (ViT F3 + Asclepius-R + Puente, 50%) — LLM externo Llama 3.1 8B ─
    # Prompt diseñado: usa el CSV crudo de evaluate_exp3_asclepius.py
    "disenado_exp3asclepius": FormatTask(
        name="disenado_exp3asclepius",
        in_csv=EXP3_EVAL_DIR / "predicciones_Exp3_Asclepius.csv",
        out_csv=EXP3_EVAL_DIR / "predicciones_Exp3_Asclepius_llmexterno_llama8b.csv",
    ),
    # Prompt CoT/nuevo: usa el CSV crudo de exp1_cot_inference_exp3_asclepius.py
    # (el archivo input se materializa cuando termine la inferencia CoT del Exp3).
    "nuevo_exp3asclepius": FormatTask(
        name="nuevo_exp3asclepius",
        in_csv=EXP3_EXP1_DIR / "exp1_raw_inference_exp3_asclepius.csv",
        out_csv=EXP3_EXP1_DIR / "exp1_final_formatted_exp3_asclepius_llama8b.csv",
    ),
    # ── EXPERIMENTO FINAL (RadDINO + Asclepius-R + Puente, 50%) — LLM externo Llama 3.1 8B ─
    # RAMA 2 — Prompt diseñado post-procesado: usa el CSV crudo de evaluate_final_experiment.py
    "disenado_final": FormatTask(
        name="disenado_final",
        in_csv=FINAL_EVAL_DIR / "predicciones_Exp_Final.csv",
        out_csv=FINAL_EVAL_DIR / "predicciones_Exp_Final_llmexterno_llama8b.csv",
    ),
    # RAMA 4 — Prompt nuevo/CoT post-procesado: usa el CSV crudo de
    # exp1_cot_inference_final_experiment.py (input se materializa al terminar la inferencia CoT).
    "nuevo_final": FormatTask(
        name="nuevo_final",
        in_csv=FINAL_EXP1_DIR / "exp1_raw_inference_final_experiment.csv",
        out_csv=FINAL_EXP1_DIR / "exp1_final_formatted_final_experiment_llama8b.csv",
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# CARGA DEL MODELO (una sola vez para todas las tareas)
# ─────────────────────────────────────────────────────────────────────────────
def cargar_vllm(args: argparse.Namespace):
    """Construye una instancia de vllm.LLM optimizada para la L40S."""
    print(f"  Cargando modelo: {args.model}")
    print(f"    dtype                = {args.dtype}")
    print(f"    max_model_len        = {args.max_model_len}")
    print(f"    gpu_memory_utilization = {args.gpu_memory_util}")
    print(f"    enable_prefix_caching= {not args.no_prefix_cache}")
    t0 = time.time()
    from vllm import LLM
    llm = LLM(
        model=args.model,
        dtype=args.dtype,                                # float16 (FP16)
        max_model_len=args.max_model_len,                # input + salida
        gpu_memory_utilization=args.gpu_memory_util,     # 0.92 → ~42 GB
        enable_prefix_caching=not args.no_prefix_cache,  # cachea system prompt
        trust_remote_code=False,
        # disable_log_stats=False,  # mantener stats por minuto en stdout
    )
    print(f"  Modelo listo en {time.time() - t0:.1f}s")
    return llm


# ─────────────────────────────────────────────────────────────────────────────
# CONSTRUCCIÓN DE PROMPTS (formato chat de Llama 3.1)
# ─────────────────────────────────────────────────────────────────────────────
def construir_prompts(tokenizer, raws: Sequence[str]) -> List[str]:
    """
    Aplica el chat template de Llama 3.1 a cada reporte.

    Resultado por fila:
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>...
        <|start_header_id|>user<|end_header_id|>{raw}<|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
    """
    prompts: List[str] = []
    for raw in raws:
        raw = (raw or "").strip() or "."
        msg = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": raw},
        ]
        text = tokenizer.apply_chat_template(
            msg, tokenize=False, add_generation_prompt=True
        )
        prompts.append(text)
    return prompts


# ─────────────────────────────────────────────────────────────────────────────
# EJECUCIÓN DE UNA TAREA (un CSV de entrada → un CSV de salida)
# ─────────────────────────────────────────────────────────────────────────────
def procesar_tarea(
    llm,
    sampling_params,
    tokenizer,
    task: FormatTask,
    args: argparse.Namespace,
) -> None:
    if not task.in_csv.exists():
        print(f"  [{task.name}] ERROR: no existe {task.in_csv}", file=sys.stderr)
        return

    print("\n" + "=" * 80)
    print(f"  TAREA: {task.name}")
    print("=" * 80)
    print(f"  Entrada : {task.in_csv}")
    print(f"  Salida  : {task.out_csv}")

    df = pd.read_csv(task.in_csv)
    if args.limit and args.limit > 0:
        df = df.head(args.limit).copy()
    n_total = len(df)
    print(f"  Filas total: {n_total:,}")

    if task.col_input not in df.columns:
        print(f"  [{task.name}] ERROR: falta la columna '{task.col_input}'.", file=sys.stderr)
        return

    # ── Resume: precargar lo ya formateado y filtrar lo pendiente ────────────
    ya_hechos: Dict[str, str] = {}
    if args.resume and task.out_csv.exists():
        try:
            previo = pd.read_csv(task.out_csv)
            if {"dicom_id", task.col_output}.issubset(previo.columns):
                m = previo[task.col_output].fillna("").astype(str).str.strip().ne("")
                ya_hechos = dict(zip(
                    previo.loc[m, "dicom_id"].astype(str),
                    previo.loc[m, task.col_output].astype(str),
                ))
                print(f"  [resume] {len(ya_hechos):,} reportes ya formateados → se omiten")
        except Exception as exc:
            print(f"  [resume] error leyendo previo ({exc}); se ignora.", file=sys.stderr)

    if "dicom_id" not in df.columns:
        df["dicom_id"] = [f"row_{i}" for i in range(len(df))]
    pendientes_mask = ~df["dicom_id"].astype(str).isin(ya_hechos.keys())
    df_pend = df.loc[pendientes_mask].copy()
    n_pend = len(df_pend)
    print(f"  Pendientes : {n_pend:,}")

    # Inicializar columnas de salida con lo previo
    df[task.col_output] = df["dicom_id"].astype(str).map(ya_hechos).fillna("")
    df["formatter_provider"] = df[task.col_output].apply(
        lambda x: PROVIDER_TAG if x else ""
    )

    if n_pend == 0:
        print(f"  Nada pendiente → guardo y continúo.")
        task.out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(task.out_csv, index=False)
        return

    # ── Construir prompts y generar (vLLM hace batching dinámico) ────────────
    raws = df_pend[task.col_input].fillna("").astype(str).tolist()
    prompts = construir_prompts(tokenizer, raws)

    print(f"  Generando {n_pend:,} reportes con vLLM...")
    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params)  # vLLM continuous batching
    dt = time.time() - t0
    rps = n_pend / dt if dt > 0 else 0
    print(f"  ✓ Generación completada en {dt/60:.1f} min  ({rps:.1f} rep/s)")

    # vLLM devuelve los outputs en el MISMO orden que los prompts.
    formateados = [o.outputs[0].text.strip() if o.outputs else "" for o in outputs]

    # Re-inyectar en el DF completo respetando el orden original
    df.loc[pendientes_mask, task.col_output] = formateados
    df.loc[pendientes_mask, "formatter_provider"] = [
        PROVIDER_TAG if t else "" for t in formateados
    ]

    n_ok = int(df[task.col_output].astype(str).str.strip().ne("").sum())
    n_fail = n_total - n_ok
    task.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(task.out_csv, index=False)
    print(f"  ✓ Guardado en {task.out_csv}")
    print(f"     OK: {n_ok:,} | vacíos/fallidos: {n_fail:,}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Formateo local con vLLM + Llama 3.1 8B (FP16). "
                    "Reemplaza la Etapa A del LLM externo.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--tasks", nargs="+", default=["disenado", "nuevo"],
                   choices=list(TASKS.keys()) + ["both"],
                   help="Tareas a ejecutar (en este orden).")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="ID HuggingFace del modelo (default: mirror ungated de Llama 3.1 8B).")
    p.add_argument("--dtype", default="float16",
                   choices=["float16", "bfloat16", "auto"],
                   help="Precisión de pesos (FP16 = más rápido en L40S).")
    p.add_argument("--max-model-len", type=int, default=2560,
                   help="Longitud total (input + salida). Reportes ~1500 chars ≈ 600 tok.")
    p.add_argument("--max-tokens", type=int, default=512,
                   help="Tokens máximos a generar por reporte.")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="0.0 = determinista (reproducible).")
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--gpu-memory-util", type=float, default=0.92,
                   help="Fracción de VRAM que vLLM puede usar (L40S 46GB → ~42GB).")
    p.add_argument("--no-prefix-cache", action="store_true",
                   help="Desactivar prefix caching (no recomendado: el system prompt se repite).")
    p.add_argument("--limit", type=int, default=0,
                   help="Procesar solo las primeras N filas por tarea (0 = todas).")
    p.add_argument("--resume", action="store_true",
                   help="Saltar reportes ya formateados (lee CSV de salida si existe).")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)

    # Expandir alias "both"
    tareas: List[FormatTask] = []
    for t in args.tasks:
        if t == "both":
            tareas.extend(TASKS.values())
        else:
            tareas.append(TASKS[t])

    print("=" * 80)
    print("  vLLM Formatter — Llama 3.1 8B (FP16)  ·  L40S")
    print("=" * 80)
    print(f"  Tareas: {[t.name for t in tareas]}")
    print(f"  Resume: {args.resume}")
    if args.limit:
        print(f"  Limit : {args.limit}")
    t_global = time.time()

    # ── 1) Cargar modelo UNA sola vez ──────────────────────────────────────
    llm = cargar_vllm(args)
    from vllm import SamplingParams
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        # stop por <|eot_id|> está implícito en chat template + skip_special_tokens
    )

    # ── 2) Procesar cada tarea reutilizando el modelo en GPU ───────────────
    for t in tareas:
        try:
            procesar_tarea(llm, sampling_params, tokenizer, t, args)
        except Exception as exc:  # pragma: no cover
            print(f"\n  [{t.name}] ERROR FATAL: {exc}", file=sys.stderr)
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print(f"  TIEMPO TOTAL: {(time.time() - t_global)/60:.1f} min")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
