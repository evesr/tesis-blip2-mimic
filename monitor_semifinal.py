#!/usr/bin/env python3
"""
Monitor Gran Final -- BLIP-2 + LoRA
=====================================
Lee los archivos generados por train_semifinal.py y muestra el estado.

Archivos leidos:
  final_results/history_final_Config_A.csv  -> Step, Val_Loss, BLEU-4, ROUGE-L
  final_results/history_final_Config_B.csv
  final_results/audit_final_Config_A.txt    -> ultimos reportes generados
  final_results/audit_final_Config_B.txt
  final_results/intermediate_results_final.csv
  final_results/gran_final.log

Uso:
    python monitor_semifinal.py              # snapshot
    watch -n 30 python monitor_semifinal.py  # refresco automatico

Autor: Evelyn Silva Rozas
Fecha: Mayo 2026
"""

import sys
from pathlib import Path
from datetime import datetime

import pandas as pd

RESULTS_DIR  = Path("final_results")
CONFIG_NAMES = ["Config_A", "Config_B"]
CONFIG_META  = {
    "Config_A": {"r": 16, "alpha": 32},
    "Config_B": {"r": 32, "alpha": 64},
}
LOG_FILE   = RESULTS_DIR / "gran_final.log"
PID_FILE   = RESULTS_DIR / "gran_final.pid"
PATIENCE   = 10
LOG_TAIL   = 8
AUDIT_SHOW = 2


def _sparkline(values):
    """Mini grafico ASCII de tendencia."""
    if len(values) < 2:
        return "-" * len(values)
    blocks = "._-~=^#@"
    vmin, vmax = min(values), max(values)
    rng = vmax - vmin + 1e-9
    return "".join(blocks[int((v - vmin) / rng * 7)] for v in values)


def _pid_status():
    if not PID_FILE.exists():
        return "desconocido (sin PID file)"
    pid = PID_FILE.read_text().strip()
    try:
        Path(f"/proc/{pid}").stat()
        return f"CORRIENDO (PID {pid})"
    except FileNotFoundError:
        return f"DETENIDO (PID {pid} ya no existe)"


def monitor():
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sep = "=" * 78
    line = "-" * 78
    print()
    print(sep)
    print(f"  MONITOR GRAN FINAL -- BLIP-2 + LoRA      [{now}]")
    print(sep)
    print(f"  Proceso : {_pid_status()}")

    if not RESULTS_DIR.exists():
        print()
        print("  Sin datos aun. Lanza el entrenamiento con:")
        print("    bash run_semifinal_tmux.sh")
        print(sep)
        print()
        return

    # -- 1. Historial por config ----------------------------------------------
    any_data = False
    for name in CONFIG_NAMES:
        meta = CONFIG_META[name]
        hist_path = RESULTS_DIR / f"history_final_{name}.csv"
        if not hist_path.exists():
            print()
            print(f"  * {name} (r={meta['r']}, alpha={meta['alpha']})"
                  "  -- esperando primera evaluacion (step 500)...")
            continue

        df = pd.read_csv(hist_path)
        if df.empty:
            print(f"  * {name} -- historia vacia...")
            continue

        any_data   = True
        latest     = df.iloc[-1]
        best_idx   = df["Val_Loss"].idxmin()
        best       = df.loc[best_idx]
        n_evals    = len(df)
        evals_since = n_evals - 1 - best_idx

        if evals_since == 0:
            es_status = "mejorando"
        elif evals_since >= PATIENCE:
            es_status = f"EARLY STOP ({evals_since}/{PATIENCE})"
        else:
            es_status = f"{evals_since}/{PATIENCE} sin mejora"

        print()
        print(f"  +-- {name}  (r={meta['r']}, alpha={meta['alpha']})"
              f"  {n_evals} evals  [{es_status}]")
        print(f"  |  Ahora   step {int(latest['Step']):>7}"
              f"  val_loss={latest['Val_Loss']:.4f}"
              f"  BLEU-4={latest['BLEU-4']:.4f}"
              f"  ROUGE-L={latest['ROUGE-L']:.4f}")
        print(f"  |  Mejor   step {int(best['Step']):>7}"
              f"  val_loss={best['Val_Loss']:.4f}"
              f"  BLEU-4={best['BLEU-4']:.4f}"
              f"  ROUGE-L={best['ROUGE-L']:.4f}")

        recent = df["Val_Loss"].tail(20).tolist()
        if len(recent) > 1:
            spark = _sparkline(recent)
            trend = "bajando" if recent[-1] < recent[0] else "subiendo"
            print(f"  |  Tendencia [{spark}] {trend}  (ultimos {len(recent)} evals)")
        print(f"  +{line[1:]}")

    if not any_data:
        print()
        print("  Sin evals aun. Espera ~500 steps para el primer checkpoint.")

    # -- 2. Configs terminadas ------------------------------------------------
    inter_path = RESULTS_DIR / "intermediate_results_final.csv"
    if inter_path.exists():
        df_i = pd.read_csv(inter_path)
        if not df_i.empty:
            print()
            print(line)
            print(f"  CONFIGS TERMINADAS: {len(df_i)}/{len(CONFIG_NAMES)}")
            cols = [c for c in
                    ["config_name", "r", "alpha", "total_steps",
                     "best_val_loss", "best_step", "final_train_loss", "timestamp"]
                    if c in df_i.columns]
            print(df_i[cols].to_string(index=False))

    # -- 3. Audit: ultimas entradas -------------------------------------------
    print()
    print(line)
    print("  ULTIMOS REPORTES GENERADOS (audit):")
    for name in CONFIG_NAMES:
        audit_path = RESULTS_DIR / f"audit_final_{name}.txt"
        if not audit_path.exists():
            continue
        raw = audit_path.read_text(errors="replace").splitlines()
        lines = [l.rstrip() for l in raw
                 if l.strip() and not l.startswith("#") and not l.startswith("-")]
        if not lines:
            continue
        print(f"  [{name}]")
        for l in lines[-AUDIT_SHOW:]:
            print(f"    {l[:160]}")

    # -- 4. Ultimas lineas del log --------------------------------------------
    if LOG_FILE.exists():
        print()
        print(line)
        print(f"  ULTIMAS {LOG_TAIL} LINEAS DE gran_final.log:")
        all_lines = LOG_FILE.read_text(errors="replace").splitlines()
        for l in all_lines[-LOG_TAIL:]:
            print(f"    {l}")

    # -- 5. Graficos disponibles ----------------------------------------------
    plots_dir = RESULTS_DIR / "plots"
    if plots_dir.exists():
        pngs = sorted(plots_dir.glob("*.png"))
        if pngs:
            print()
            print(line)
            print("  GRAFICOS DISPONIBLES:")
            for p in pngs:
                print(f"    {p}")

    print()
    print(sep)
    print("  Comandos utiles:")
    print("    tail -f final_results/gran_final.log      # log en tiempo real")
    print("    watch -n 30 python monitor_semifinal.py   # refresco auto 30s")
    print("    watch -n 3 nvidia-smi                     # estado GPU")
    print("    kill $(cat final_results/gran_final.pid)  # detener entrenamiento")
    print(sep)
    print()


if __name__ == "__main__":
    try:
        monitor()
    except KeyboardInterrupt:
        print("  Monitor detenido.")
        sys.exit(0)
    except Exception as e:
        print(f"  Error en monitor: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
