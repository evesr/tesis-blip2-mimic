#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
actualizar_comparativos_zeroshot.py
===================================
Replica el CSV COMPARATIVO de CheXpert (por patología + general) para la
inferencia ZERO-SHOT del BLIP-2 BASE (Salesforce/blip2-opt-2.7b, sin fine-tuning)
sobre el Test Set 50%, siguiendo EXACTAMENTE la misma metodología que el resto
de los experimentos (Fase 2/3, Exp2, Exp3, Experimento FINAL).

REUTILIZACIÓN ESTRICTA (misma lógica, sin reimplementar nada)
────────────────────────────────────────────────────────────────────────────
Importa TAL CUAL las funciones ya validadas de `actualizar_comparativos_final`:

    _metricas_pat_desde_granular(df_gran)  → reconstruye TP/FP/FN/TN/P/R/F1 por
                                             patología desde chexpert_match_* y
                                             pred_chexpert_* del CSV granular.
    _micro_macro(df_pat)                   → MICRO (pooled TP/FP/FN) y MACRO
                                             (nanmean de P/R/F1 por patología).
    _ren(df, pref)                         → renombra precision/recall/f1 →
                                             {pref}_P / {pref}_R / {pref}_F1.

Así el comparativo del zero-shot es DIRECTAMENTE comparable (misma agregación,
mismas 14 patologías, mismo orden) con chexpert_comparativo_final.csv y los demás.

VARIANTE
────────────────────────────────────────────────────────────────────────────
El zero-shot tiene UNA sola rama de inferencia: el PROMPT DISEÑADO sobre la
salida DIRECTA del modelo (no hay post-formateo con LLM externo, ni limpieza,
ni prompt "nuevo"/CoT). Por eso el comparativo tiene un único bloque de
columnas: `zeroshot_disenado_sucio_{P,R,F1}` (paralelo a `final_disenado_sucio_*`).

Entrada (PRE-COMPUTADA por evaluate_zeroshot_blip2.py):
    zeroshot_results-50pct/evaluation/resultados_granulares_Zeroshot_BLIP2.csv

Salida:
    zeroshot_results-50pct/chexpert_comparativo_zeroshot.csv
        nivel = por_patologia (14 filas) + general (MICRO / MACRO)
        columnas: zeroshot_disenado_sucio_P / _R / _F1

NOTA: NO usa entornos virtuales — corre sobre el Python del sistema. Solo
requiere numpy + pandas (no carga torch/transformers).

Autor : Evelyn Silva Rozas
Fecha : Junio 2026
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Lógica de agregación REUTILIZADA SIN CAMBIOS del comparativo FINAL ───────
#    (garantiza metodología idéntica: misma reconstrucción TP/FP/FN/TN y mismo
#     micro/macro que chexpert_comparativo_final.csv / _exp3 / _exp2 / phase2/3).
from actualizar_comparativos_final import (
    _metricas_pat_desde_granular,
    _micro_macro,
    _ren,
    CHEX_LABELS,
    ROUND,
)

# ─────────────────────────────────────────────────────────────────────────────
# Rutas
# ─────────────────────────────────────────────────────────────────────────────
BASE = Path(os.path.dirname(os.path.abspath(__file__)))

ZS_DIR   = BASE / "zeroshot_results-50pct"
ZS_EVAL  = ZS_DIR / "evaluation"
ZS_GRAN  = ZS_EVAL / "resultados_granulares_Zeroshot_BLIP2.csv"

OUT_CHEXPERT = ZS_DIR / "chexpert_comparativo_zeroshot.csv"

# Prefijo de la única variante del zero-shot (prompt diseñado, salida directa).
PREF = "zeroshot_disenado_sucio"


# ═════════════════════════════════════════════════════════════════════════════
# Verificación de archivos requeridos
# ═════════════════════════════════════════════════════════════════════════════
def _verificar() -> None:
    if not ZS_GRAN.exists():
        print("\nERROR: no se encontró el CSV granular del zero-shot:")
        print(f"  • {ZS_GRAN}")
        print("\nGenera primero la evaluación zero-shot:")
        print("  python evaluate_zeroshot_blip2.py            # inferencia + métricas")
        print("  python evaluate_zeroshot_blip2.py --skip-inference   # solo métricas")
        sys.exit(1)


# ═════════════════════════════════════════════════════════════════════════════
# Construcción del comparativo CheXpert del zero-shot
# ═════════════════════════════════════════════════════════════════════════════
def construir_chexpert() -> pd.DataFrame:
    """Construye la tabla comparativa CheXpert del zero-shot (por patología + general)."""
    print("\n" + "=" * 78)
    print(" chexpert_comparativo_zeroshot.csv  (BLIP-2 base · prompt diseñado · sucio)")
    print("=" * 78)

    print(f"  Leyendo granular: {ZS_GRAN.name}")
    df_gran = pd.read_csv(ZS_GRAN)
    print(f"  Filas (reportes): {len(df_gran):,}")

    print("  Reconstruyendo TP/FP/FN/TN → P/R/F1 por patología "
          "(chexpert_match_* + pred_chexpert_*)...")
    pat = _metricas_pat_desde_granular(df_gran, labels=CHEX_LABELS)

    # ── Sección por patología ────────────────────────────────────────────────
    merged = pat[["categoria"]].copy().merge(_ren(pat, PREF), on="categoria", how="left")
    merged.insert(0, "nivel", "por_patologia")

    # ── Sección general (micro / macro), misma agregación que el FINAL ───────
    mm = _micro_macro(pat)
    filas_gen = []
    for scope, etiqueta in [("micro", "MICRO (pooled 14 clases)"),
                            ("macro", "MACRO (nanmean 14 clases)")]:
        p, r, f = mm[scope]
        filas_gen.append({
            "nivel": "general",
            "categoria": etiqueta,
            f"{PREF}_P": p, f"{PREF}_R": r, f"{PREF}_F1": f,
        })
    gen = pd.DataFrame(filas_gen)

    df = pd.concat([merged, gen], ignore_index=True)

    num_cols = [c for c in df.columns if c not in ("nivel", "categoria")]
    df[num_cols] = df[num_cols].astype(float).round(ROUND)

    OUT_CHEXPERT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CHEXPERT, index=False)
    print(f"\n[OK] {OUT_CHEXPERT}")
    with pd.option_context("display.width", 240, "display.max_columns", 40):
        print(df.to_string(index=False))

    print("\n   [sanity zero-shot — recomputado desde granular]")
    print(f"   zeroshot disenado_sucio  MICRO:"
          f"  P={mm['micro'][0]:.4f}  R={mm['micro'][1]:.4f}  F1={mm['micro'][2]:.4f}")
    print(f"   zeroshot disenado_sucio  MACRO:"
          f"  P={mm['macro'][0]:.4f}  R={mm['macro'][1]:.4f}  F1={mm['macro'][2]:.4f}")
    return df


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════
def main() -> int:
    _verificar()
    construir_chexpert()
    print("\n" + "=" * 78)
    print(" Comparativo CheXpert del ZERO-SHOT (BLIP-2 base) generado:")
    print(f"   • {OUT_CHEXPERT}")
    print("=" * 78 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
