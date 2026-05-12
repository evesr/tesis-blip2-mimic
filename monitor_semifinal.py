#!/usr/bin/env python3
"""
Monitor Gran Final -- BLIP-2 + LoRA
=====================================
Pensado para correr con watch dentro del panel inferior de tmux:
    watch -n 30 -t python monitor_semifinal.py

Lee:
    final_results/history_final_Config_A.csv  -> Step, Val_Loss, BLEU-4, ROUGE-L
    final_results/history_final_Config_B.csv
    final_results/intermediate_results_final.csv
    final_results/gran_final.log              -> ultimas lineas
    final_results/gran_final.pid              -> estado del proceso

Autor: Evelyn Silva Rozas
Fecha: Mayo 2026
"""

import sys
from pathlib import Path
from datetime import datetime

import pandas as pd

RESULTS_DIR  = Path("/workspace/tesis-blip2-mimic/final_results")
CONFIG_NAMES = ["Config_A", "Config_B"]
CONFIG_META  = {
    "Config_A": {"r": 16, "alpha": 32},
    "Config_B": {"r": 32, "alpha": 64},
}
LOG_FILE  = RESULTS_DIR / "gran_final.log"
PID_FILE  = RESULTS_DIR / "gran_final.pid"
PATIENCE  = 10
LOG_TAIL  = 6


def _sparkline(values):
    """Minigrafic ASCII de tendencia (ultimos N valores)."""
    blocks = " ._-~=^*#"
    if len(values) < 2:
        return " " * max(len(values), 1)
    vmin, vmax = min(values), max(values)
    rng = vmax - vmin + 1e-9
    return "".join(blocks[int((v - vmin) / rng * 8)] for v in values)


def _pid_status():
    if not PID_FILE.exists():
        return "SIN PID FILE"
    pid = PID_FILE.read_text().strip()
    try:
        Path(f"/proc/{pid}").stat()
        return f"CORRIENDO  (PID {pid})"
    except (FileNotFoundError, PermissionError):
        return f"DETENIDO   (PID {pid} ya no existe)"


def _elapsed_since_log():
    """Segundos desde la ultima modificacion del log."""
    if not LOG_FILE.exists():
        return None
    import time
    return int(time.time() - LOG_FILE.stat().st_mtime)


def monitor():
    W   = 78
    SEP = "=" * W
    DIV = "-" * W
    now = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")

    print()
    print(SEP)
    print(f"  MONITOR GRAN FINAL - BLIP-2 + LoRA          [{now}]")
    print(SEP)

    proc  = _pid_status()
    elapsed = _elapsed_since_log()
    idle_str = f"  | sin actividad hace {elapsed}s" if elapsed is not None and elapsed > 120 else ""
    print(f"  Proceso : {proc}{idle_str}")
    print()

    if not RESULTS_DIR.exists():
        print("  Sin datos. Lanza el entrenamiento con:")
        print("    bash run_semifinal_tmux.sh")
        print(SEP)
        return

    # ── 1. Historial config por ────────
    any_data = False
    for name in CONFIG_NAMES:
        meta = CONFIG_META[name]
        hist_path = RESULTS_DIR / f"history_final_{name}.csv"

        if not hist_path.exists():
            print(f"  {name}  (r={meta['r']}, a={meta['alpha']})"
                  "  ->  esperando primera eval (step 500)...")
            print()
            continue

        df = pd.read_csv(hist_path)
        if df.empty:
            print(f"  {name}  ->  historial vacio...")
            continue

        any_data    = True
        latest      = df.iloc[-1]
        best_idx    = df["Val_Loss"].idxmin()
        best        = df.loc[best_idx]
        n_evals     = len(df)
        evals_since = (n_evals - 1) - best_idx

        if evals_since == 0:
            es_str = "  [mejorando]"
        elif evals_since >= PATIENCE:
            es_str = f"  [EARLY STOP alcanzado ({evals_since}/{PATIENCE})]"
        else:
            es_str = f"  [{evals_since}/{PATIENCE} evals sin mejora]"

        # Porcentaje completado (aprox, usando steps y max posible 50 epocas)
        print(f"  {name}  r={meta['r']} alpha={meta['alpha']}"
              f"  |  {n_evals} evals{es_str}")
        print(f"    Ultimo  step {int(latest['Step']):>7}"
              f"  val_loss={latest['Val_Loss']:.4f}"
              f"  BLEU-4={latest['BLEU-4']:.4f}"
              f"  ROUGE-L={latest['ROUGE-L']:.4f}")
        print(f"    Mejor   step {int(best['Step']):>7}"
              f"  val_loss={best['Val_Loss']:.4f}"
              f"  BLEU-4={best['BLEU-4']:.4f}"
              f"  ROUGE-L={best['ROUGE-L']:.4f}")

        recent = df["Val_Loss"].tail(20).tolist()
        spark  = _sparkline(recent)
        trend  = "bajando" if len(recent) > 1 and recent[-1] < recent[0] else "subiendo/estable"
        print(f"    Tendencia val_loss [{spark}]  {trend}  (ult. {len(recent)} evals)")
        print()

    if not any_data:
        print("  Sin evals completadas. Espera el primer checkpoint (step 500).")
        print()

    # ── 2. Configs terminadas ─────────────────────────────────────
    inter_path = RESULTS_DIR / "intermediate_results_final.csv"
    if inter_path.exists():
        df_i = pd.read_csv(inter_path)
        if not df_i.empty:
            print(DIV)
            print(f"  CONFIGS TERMINADAS: {len(df_i)} / {len(CONFIG_NAMES)}")
            cols = [c for c in
                    ["config_name", "r", "alpha", "total_steps",
                     "best_val_loss", "best_step", "final_train_loss", "timestamp"]
                    if c in df_i.columns]
            print(df_i[cols].to_string(index=False))
            print()

    # ── 3. Ultimas lineas del log ─────────────────────────────────────────────
    if LOG_FILE.exists():
        all_lines = LOG_FILE.read_text(errors="replace").splitlines()
        # Filtrar lineas vacias del final
        non_empty = [l.rstrip() for l in all_lines if l.strip()]
        tail = non_empty[-LOG_TAIL:]
        if tail:
            print(DIV)
            print(f"  ULTIMAS {LOG_TAIL} LINEAS  (gran_final.log):")
            for l in tail:
                print(f"    {l[:W-4]}")
            print()

    # ── 4. Graficos disponibles ───────────────────────────────
    plots_dir = RESULTS_DIR / "plots"
    if plots_dir.exists():
        pngs = sorted(plots_dir.glob("*.png"))
        if pngs:
            print(DIV)
            print("  GRAFICOS GENERADOS:")
            for p in pngs:
                print(f"    {p.name}")
            print()

    print(DIV)
    print("  Ctrl+b d -> desconectarse de tmux (sigue corriendo)")
    print("  tail -f final_results/gran_final.log   |   watch -n 3 nvidia-smi")
    print(SEP)
    print()


if __name__ == "__main__":
    try:
        monitor()
    except KeyboardInterrupt:
        print("  Monitor detenido.")
    except Exception as e:
        print(f"  Error en monitor: {e}")
        import traceback
        traceback.print_exc()
