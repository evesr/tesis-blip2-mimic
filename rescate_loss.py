"""
rescate_loss.py — Rescate del historial Val_Loss desde trainer_state.json
=========================================================================
Lee el archivo de estado del Trainer de HuggingFace y reconstruye el CSV
de historial history_final_Config_A.csv con las columnas Step y Val_Loss.

Uso:
    python rescate_loss.py

Autor: Evelyn Silva Rozas
Fecha: Mayo 2026
"""

import json
import pandas as pd
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# RUTAS — ajustar si el checkpoint cambió de nombre
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR        = Path("/workspace/tesis-blip2-mimic")
TRAINER_STATE   = BASE_DIR / "final_results" / "Config_A" / "checkpoint-34500" / "trainer_state.json"
OUTPUT_CSV      = BASE_DIR / "final_results" / "history_final_Config_A.csv"

# ─────────────────────────────────────────────────────────────────────────────
# EXTRACCIÓN
# ─────────────────────────────────────────────────────────────────────────────
print(f"Leyendo: {TRAINER_STATE}")

if not TRAINER_STATE.exists():
    raise FileNotFoundError(
        f"No se encontró el archivo trainer_state.json en:\n  {TRAINER_STATE}\n"
        "Ajusta la variable TRAINER_STATE con la ruta correcta."
    )

with open(TRAINER_STATE, "r", encoding="utf-8") as f:
    state = json.load(f)

log_history = state.get("log_history", [])
print(f"Entradas totales en log_history: {len(log_history)}")

# Extraer solo los registros que contienen eval_loss
registros = [
    {"Step": entry["step"], "Val_Loss": entry["eval_loss"]}
    for entry in log_history
    if "eval_loss" in entry
]

print(f"Registros con eval_loss encontrados: {len(registros)}")

if not registros:
    raise ValueError(
        "No se encontraron entradas con 'eval_loss' en log_history. "
        "Verifica que el trainer_state.json corresponde a un checkpoint de evaluación."
    )

# ─────────────────────────────────────────────────────────────────────────────
# CONSTRUCCIÓN DEL DATAFRAME
# ─────────────────────────────────────────────────────────────────────────────
df = pd.DataFrame(registros).sort_values("Step", ascending=True).reset_index(drop=True)

print(f"\nHistorial reconstruido ({len(df)} puntos):")
print(df.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# EXPORTAR CSV
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_CSV, index=False)

print(f"\n✅ CSV guardado en: {OUTPUT_CSV}")
print(f"   Rango de steps: {df['Step'].min()} → {df['Step'].max()}")
print(f"   Val_Loss min:   {df['Val_Loss'].min():.6f}  @ step {df.loc[df['Val_Loss'].idxmin(), 'Step']}")
print(f"   Val_Loss max:   {df['Val_Loss'].max():.6f}  @ step {df.loc[df['Val_Loss'].idxmax(), 'Step']}")
