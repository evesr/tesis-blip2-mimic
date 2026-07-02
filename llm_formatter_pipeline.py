#!/usr/bin/env python3
"""
llm_formatter_pipeline.py — Formateo de reportes crudos vía LLM externo
=======================================================================
Segunda etapa del Experimento 1.

Lee  exp1_raw_inference_results.csv  (salida cruda del modelo BLIP-2 con el
prompt Chain-of-Thought) y pasa CADA generated_report por un LLM externo
(p. ej. gpt-4o-mini) que actúa como "radiólogo formateador": extrae los
hallazgos clínicos del texto crudo y los reescribe en un reporte estándar con
secciones FINDINGS e IMPRESSION, SIN inventar patología y eliminando el ruido
administrativo / HTML / "babble" (Tail Babbling).

Salida: exp1_final_formatted_results.csv  (= raw + columna `formatted_report`)

Proveedores
-----------
  • openai   : usa la librería `openai` (SDK >= 1.0) con OPENAI_API_KEY.
               Compatible con endpoints OpenAI-like vía --base-url (API genérica).
  • simulate : fallback determinista SIN red (limpieza por RegEx). Útil para
               validar el pipeline de extremo a extremo sin clave de API.
  • auto     : usa openai si hay clave disponible; si no, cae a simulate.

NOTA sobre GPU: el formateo lo ejecuta un LLM EXTERNO (servidores de OpenAI),
por lo que esta etapa es I/O de red, no cómputo local en GPU. Las etapas con
cómputo local pesado (inferencia BLIP-2, CheXbert, RadGraph, Clinical BERTScore)
SÍ corren en GPU (ver exp1_cot_inference.py y exp1_metrics_comparison.py).

Uso
---
    export OPENAI_API_KEY="sk-..."
    python llm_formatter_pipeline.py                         # auto (openai si hay clave)
    python llm_formatter_pipeline.py --provider openai --model gpt-4o-mini --workers 8
    python llm_formatter_pipeline.py --provider simulate     # sin red (testing)
    python llm_formatter_pipeline.py --limit 50              # solo primeras 50 filas

Autor : Evelyn Silva Rozas
Fecha : Junio 2026
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# 1.  CONSTANTES
# ─────────────────────────────────────────────────────────────────────────────

# Directorio del experimento (coincide con exp1_cot_inference.py)
# CAMBIO Fase 3 → Fase 2: las salidas viven bajo phase2_results-50pct/exp1_cot.
try:
    from config import config
    EXP1_DIR = config.paths.base_dir / "phase2_results-50pct" / "exp1_cot"
except Exception:
    # Fallback autocontenido si no se puede importar config (sin entorno, etc.)
    EXP1_DIR = Path("phase2_results-50pct") / "exp1_cot"

RAW_CSV: Path = EXP1_DIR / "exp1_raw_inference_results.csv"
OUT_CSV: Path = EXP1_DIR / "exp1_final_formatted_results.csv"

# System Prompt EXACTO solicitado para el LLM formateador.
SYSTEM_PROMPT: str = (
    "You are an expert radiologist. Extract the clinical findings from the "
    "provided raw text and format them into a standard radiology report with "
    "'FINDINGS' and 'IMPRESSION' sections. Do NOT invent or deduce any pathology "
    "not explicitly mentioned in the raw text. Remove any administrative noise, "
    "HTML tags, or non-medical babble. Output strictly the formatted report."
)

DEFAULT_MODEL: str = "gpt-4o-mini"


# ─────────────────────────────────────────────────────────────────────────────
# 2.  PROVEEDOR OPENAI (SDK >= 1.0)
# ─────────────────────────────────────────────────────────────────────────────

def crear_cliente_openai(api_key_env: str = "OPENAI_API_KEY",
                         base_url: Optional[str] = None):
    """
    Crea un cliente OpenAI (o compatible vía base_url). Devuelve None si la
    librería no está instalada o no hay clave disponible.
    """
    api_key = os.environ.get(api_key_env, "").strip()
    if not api_key:
        return None
    try:
        from openai import OpenAI
    except Exception as exc:  # pragma: no cover
        print(f"  [openai] librería no disponible ({exc}). Instala con: pip install openai",
              file=sys.stderr)
        return None
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def formatear_con_openai(
    client,
    raw_report: str,
    model: str,
    system_prompt: str = SYSTEM_PROMPT,
    max_tokens: int = 512,
    max_retries: int = 4,
) -> str:
    """
    Envía un reporte crudo al LLM externo y devuelve el reporte formateado.

    Reintenta con backoff exponencial ante errores transitorios (rate limit,
    timeouts). Si agota los reintentos, devuelve cadena vacía (se marcará como
    fallo y podrá reintentarse en otra corrida con --resume).
    """
    raw_report = (raw_report or "").strip()
    if not raw_report:
        return ""

    for intento in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": raw_report},
                ],
                temperature=0.0,          # determinista (reproducible)
                max_tokens=max_tokens,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as exc:  # pragma: no cover
            espera = 2.0 ** intento
            if intento == max_retries - 1:
                print(f"  [openai] fallo definitivo: {exc}", file=sys.stderr)
                return ""
            time.sleep(espera)
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# 3.  PROVEEDOR SIMULADO (fallback determinista, sin red)
# ─────────────────────────────────────────────────────────────────────────────
# Limpieza por RegEx que imita el comportamiento del system prompt SIN inventar
# patología: elimina HTML, ruido administrativo y "babble", y reorganiza el
# texto en secciones FINDINGS / IMPRESSION usando SOLO el contenido existente.

# Marcadores de ruido / Tail Babbling (reutiliza la lista del proyecto si existe)
try:
    from limpiar_tail_babbling import TAIL_BABBLING_KEYWORDS as _NOISE_KW
except Exception:  # pragma: no cover
    _NOISE_KW = [
        "NOTIFICATION:", "strutConnector", "attRot", "guiActiveUnfocused",
        "TheNitrome", "NOTICE:", "NOTIFY:", "NOTIFIED:", "RECOMMENDATION:",
        "guiIcon",
    ]

_RE_HTML       = re.compile(r"<[^>]+>")
_RE_CONTEXT    = re.compile(r"\[Context:.*?view\]", re.IGNORECASE)
_RE_TASK_TAGS  = re.compile(r"\bTask\s*\d+\s*:", re.IGNORECASE)
_RE_WS         = re.compile(r"\s+")
_RE_FINDINGS   = re.compile(r"\bFINDINGS?\b\s*:?", re.IGNORECASE)
_RE_IMPRESSION = re.compile(r"\bIMPRESSIONS?\b\s*:?", re.IGNORECASE)


def _cortar_en_ruido(texto: str) -> str:
    """Corta el texto JUSTO ANTES del primer marcador de Tail Babbling."""
    corte = len(texto)
    bajo = texto.lower()
    for kw in _NOISE_KW:
        i = bajo.find(kw.lower())
        if i != -1:
            corte = min(corte, i)
    return texto[:corte]


def _split_secciones(texto: str) -> Tuple[str, str]:
    """
    Separa el texto en (findings, impression) usando SOLO contenido existente.
    No inventa: si no hay IMPRESSION explícita, se deja una referencia neutra.
    """
    m_imp = _RE_IMPRESSION.search(texto)
    if m_imp:
        findings   = texto[:m_imp.start()]
        impression = texto[m_imp.end():]
    else:
        findings, impression = texto, ""

    # Quitar etiqueta FINDINGS si quedó embebida
    findings = _RE_FINDINGS.sub(" ", findings)
    findings   = _RE_WS.sub(" ", findings).strip(" .;:-")
    impression = _RE_WS.sub(" ", impression).strip(" .;:-")

    if not impression:
        # Extracción no-inventiva: última oración de findings como cierre, o
        # marcador neutro si no hay contenido.
        oraciones = [s.strip() for s in re.split(r"(?<=[.!?])\s+", findings) if s.strip()]
        impression = oraciones[-1] if oraciones else "See findings."
    return findings, impression


def formatear_simulado(raw_report: str) -> str:
    """Formateador determinista de respaldo (sin LLM externo)."""
    texto = raw_report or ""
    texto = _RE_HTML.sub(" ", texto)        # quitar HTML
    texto = _cortar_en_ruido(texto)         # quitar Tail Babbling
    texto = _RE_CONTEXT.sub(" ", texto)     # quitar "[Context: ... view]"
    texto = _RE_TASK_TAGS.sub(" ", texto)   # quitar "Task 1:" / "Task 2:"
    texto = _RE_WS.sub(" ", texto).strip()
    if not texto:
        return ""
    findings, impression = _split_secciones(texto)
    return f"FINDINGS: {findings}\nIMPRESSION: {impression}".strip()


# ─────────────────────────────────────────────────────────────────────────────
# 4.  ORQUESTACIÓN
# ─────────────────────────────────────────────────────────────────────────────

def formatear_lote(
    raws: List[str],
    provider: str,
    client,
    model: str,
    workers: int,
    max_tokens: int,
) -> List[str]:
    """
    Formatea una lista de reportes crudos. Para 'openai' usa un pool de hilos
    (las llamadas a la API son I/O-bound). Para 'simulate' es CPU puro.
    """
    n = len(raws)
    resultados: List[str] = [""] * n

    if provider == "simulate" or client is None:
        for i in tqdm(range(n), desc="  Formateando (simulate)", unit="rep"):
            resultados[i] = formatear_simulado(raws[i])
        return resultados

    # provider == "openai" (paralelo)
    def _tarea(i: int) -> Tuple[int, str]:
        return i, formatear_con_openai(client, raws[i], model, max_tokens=max_tokens)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futuros = [ex.submit(_tarea, i) for i in range(n)]
        for fut in tqdm(as_completed(futuros), total=n,
                        desc=f"  Formateando ({model})", unit="rep"):
            i, texto = fut.result()
            resultados[i] = texto
    return resultados


def run_pipeline(
    raw_csv: Path,
    out_csv: Path,
    provider: str,
    model: str,
    workers: int,
    limit: int,
    max_tokens: int,
    base_url: Optional[str],
    api_key_env: str,
    resume: bool,
) -> int:
    if not raw_csv.exists():
        print(f"ERROR: no existe el CSV crudo: {raw_csv}\n"
              f"  Ejecuta primero: python exp1_cot_inference.py", file=sys.stderr)
        return 1

    df = pd.read_csv(raw_csv)
    if "generated_report" not in df.columns:
        print("ERROR: el CSV crudo no tiene columna 'generated_report'.", file=sys.stderr)
        return 1
    if limit and limit > 0:
        df = df.head(limit).copy()
    print(f"  Filas a formatear: {len(df):,}")

    # ── Resolver proveedor efectivo ──────────────────────────────────────────
    client = None
    if provider in ("openai", "auto"):
        client = crear_cliente_openai(api_key_env=api_key_env, base_url=base_url)

    if provider == "auto":
        provider_efectivo = "openai" if client is not None else "simulate"
    else:
        provider_efectivo = provider
        if provider == "openai" and client is None:
            print("  [aviso] proveedor 'openai' solicitado pero sin clave/SDK; "
                  "usando 'simulate' como respaldo.", file=sys.stderr)
            provider_efectivo = "simulate"

    print(f"  Proveedor efectivo: {provider_efectivo}"
          + (f"  (modelo={model})" if provider_efectivo == "openai" else ""))

    # ── Reanudación: reutilizar formatted_report ya calculados ───────────────
    ya_hechos = {}
    if resume and out_csv.exists():
        prev = pd.read_csv(out_csv)
        if "dicom_id" in prev.columns and "formatted_report" in prev.columns:
            for _, r in prev.iterrows():
                fr = str(r.get("formatted_report", "") or "")
                if fr.strip():
                    ya_hechos[str(r["dicom_id"])] = fr
            print(f"  [resume] {len(ya_hechos):,} reportes ya formateados serán reutilizados.")

    pendientes_idx = [i for i, d in enumerate(df["dicom_id"].astype(str))
                      if d not in ya_hechos]
    print(f"  Pendientes: {len(pendientes_idx):,}")

    # ── Formatear pendientes ─────────────────────────────────────────────────
    raws_pend = df.iloc[pendientes_idx]["generated_report"].fillna("").astype(str).tolist()
    t0 = time.time()
    formateados_pend = formatear_lote(
        raws_pend, provider_efectivo, client, model, workers, max_tokens
    )

    # ── Ensamblar columna formatted_report en el orden original ──────────────
    formatted = [""] * len(df)
    for pos, idx in enumerate(pendientes_idx):
        formatted[idx] = formateados_pend[pos]
    for i, d in enumerate(df["dicom_id"].astype(str)):
        if d in ya_hechos:
            formatted[i] = ya_hechos[d]

    df_out = df.copy()
    df_out["formatted_report"] = formatted
    df_out["formatter_provider"] = provider_efectivo  # trazabilidad (columna extra)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_csv, index=False)

    n_vacios = int((df_out["formatted_report"].fillna("").str.strip() == "").sum())
    dt = time.time() - t0
    print(f"\n  ✓ {out_csv}  ({len(df_out):,} filas)")
    print(f"  Formateados OK : {len(df_out) - n_vacios:,} | vacíos/fallidos: {n_vacios:,}")
    print(f"  Tiempo         : {dt/60:.1f} min")
    if n_vacios and provider_efectivo == "openai":
        print("  (Reintenta los fallidos con: python llm_formatter_pipeline.py --resume)")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# 5.  CLI / MAIN
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Formatea reportes crudos (BLIP-2 CoT) vía LLM externo (gpt-4o-mini).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--raw-csv",  type=str, default=str(RAW_CSV),
                   help="CSV crudo de entrada (exp1_raw_inference_results.csv).")
    p.add_argument("--out-csv",  type=str, default=str(OUT_CSV),
                   help="CSV de salida (exp1_final_formatted_results.csv).")
    p.add_argument("--provider", type=str, default="auto",
                   choices=["auto", "openai", "simulate"],
                   help="Proveedor del formateador.")
    p.add_argument("--model",    type=str, default=DEFAULT_MODEL,
                   help="Modelo del LLM externo (provider=openai).")
    p.add_argument("--workers",  type=int, default=8,
                   help="Hilos paralelos para llamadas a la API (provider=openai).")
    p.add_argument("--max-tokens", type=int, default=512,
                   help="Máx. tokens de salida del LLM formateador.")
    p.add_argument("--limit",    type=int, default=0,
                   help="Formatear solo las primeras N filas (0 = todas).")
    p.add_argument("--base-url", type=str, default=None,
                   help="Endpoint OpenAI-compatible (API genérica). Opcional.")
    p.add_argument("--api-key-env", type=str, default="OPENAI_API_KEY",
                   help="Nombre de la variable de entorno con la API key.")
    p.add_argument("--resume",   action="store_true",
                   help="Reutilizar formatted_report ya calculados en el CSV de salida.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    print("\n" + "=" * 80)
    print("  EXPERIMENTO 1 — FORMATEO CON LLM EXTERNO (gpt-4o-mini)")
    print("=" * 80)
    print(f"  Entrada : {args.raw_csv}")
    print(f"  Salida  : {args.out_csv}")
    print(f"  System Prompt: {SYSTEM_PROMPT[:70]}...")
    print("=" * 80 + "\n")

    return run_pipeline(
        raw_csv     = Path(args.raw_csv),
        out_csv     = Path(args.out_csv),
        provider    = args.provider,
        model       = args.model,
        workers     = args.workers,
        limit       = args.limit,
        max_tokens  = args.max_tokens,
        base_url    = args.base_url,
        api_key_env = args.api_key_env,
        resume      = args.resume,
    )


if __name__ == "__main__":
    sys.exit(main())
