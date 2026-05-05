#!/usr/bin/env python
"""
Monitor de Entrenamiento Semifinal en Tiempo Real
==================================================
Lee los archivos generados por train_semifinal.py y muestra el estado actual.

Archivos leídos:
  semifinal_results/history_Config_X.csv   → Step, Val_Loss, BLEU, ROUGE-L
  semifinal_results/audit_Config_X.txt     → últimos reportes de muestra
  semifinal_results/intermediate_results.csv → resumen de configs terminadas
  semifinal_results/semifinal.log          → últimas líneas del log general

Uso:
    python monitor_semifinal.py                  # una sola vez
    watch -n 15 python monitor_semifinal.py      # actualización automática

Autor: Evelyn Silva Rozas
Fecha: Mayo 2026
"""

import sys
from pathlib import Path
from datetime import datetime

import pandas as pd


RESULTS_DIR   = Path("semifinal_results")
CONFIG_NAMES  = ["Config_4", "Config_5", "Config_6"]
LOG_TAIL_LINES = 6       # cuántas líneas del log general mostrar
AUDIT_TAIL    = 1        # cuántas entradas del audit mostrar por config
PATIENCE      = 10       # patience configurada en train_semifinal.py


def _bar(value: float, max_val: float = 1.0, width: int = 20) -> str:
    """Barra ASCII proporcional."""
    filled = int(round(value / max_val * width)) if max_val > 0 else 0
    return "█" * filled + "░" * (width - filled)


def monitor_semifinal():
    print("\n" + "=" * 80)
    print(f"📊  MONITOR SEMIFINAL — BLIP2 + LoRA   [{datetime.now().strftime('%H:%M:%S')}]")
    print("=" * 80)

    if not RESULTS_DIR.exists():
        print("\n⏳  semifinal_results/ aún no existe. El entrenamiento no ha iniciado.\n")
        return

    # ── 1. Historial por config ───────────────────────────────────────────────
    any_history = False
    for name in CONFIG_NAMES:
        hist_path = RESULTS_DIR / f"history_{name}.csv"
        if not hist_path.exists():
            continue
        any_history = True

        df = pd.read_csv(hist_path)
        if df.empty:
            print(f"\n⏳  {name}: historia vacía (primer eval en curso…)")
            continue

        latest     = df.iloc[-1]
        best_row   = df.loc[df["Val_Loss"].idxmin()]
        n_evals    = len(df)
        evals_since = n_evals - 1 - df["Val_Loss"].idxmin()   # evals sin mejora

        # Estado del early stopping
        if evals_since == 0:
            es_status = "✅ mejorando"
        elif evals_since >= PATIENCE:
            es_status = f"🛑 detenido ({evals_since}/{PATIENCE})"
        else:
            es_status = f"⏳ {evals_since}/{PATIENCE} sin mejora"

        print(f"\n┌─ {name}  (r={_config_r(name)}, α={_config_a(name)}) ── {n_evals} evals ── {es_status}")
        print(f"│  Step actual : {int(latest['Step']):<8}  "
              f"Val_Loss={latest['Val_Loss']:.4f}  "
              f"BLEU={latest['BLEU']:.4f}  "
              f"ROUGE-L={latest['ROUGE-L']:.4f}")
        print(f"│  Mejor step  : {int(best_row['Step']):<8}  "
              f"Val_Loss={best_row['Val_Loss']:.4f}  "
              f"BLEU={best_row['BLEU']:.4f}  "
              f"ROUGE-L={best_row['ROUGE-L']:.4f}")

        # Mini curva de val_loss (últimos 15 puntos)
        recent = df["Val_Loss"].tail(15).tolist()
        if len(recent) > 1:
            vmin, vmax = min(recent), max(recent)
            sparkline = "".join(
                "▁▂▃▄▅▆▇█"[int((v - vmin) / (vmax - vmin + 1e-9) * 7)]
                for v in recent
            )
            print(f"│  Loss trend  : {sparkline}  (últimos {len(recent)} evals)")
        print("└" + "─" * 78)

    if not any_history:
        print("\n⏳  Esperando primera evaluación (step 50)…\n")

    # ── 2. Configuraciones terminadas ─────────────────────────────────────────
    inter_path = RESULTS_DIR / "intermediate_results.csv"
    if inter_path.exists():
        df_inter = pd.read_csv(inter_path)
        if not df_inter.empty:
            print(f"\n{'─'*80}")
            print(f"✅  CONFIGS TERMINADAS: {len(df_inter)}/3")
            display_cols = [c for c in
                            ["config_name", "r", "alpha", "total_steps",
                             "best_val_loss", "best_step", "final_train_loss", "timestamp"]
                            if c in df_inter.columns]
            print(df_inter[display_cols].to_string(index=False))

    # ── 3. Última entrada del audit por config ────────────────────────────────
    print(f"\n{'─'*80}")
    print("📋  ÚLTIMAS ENTRADAS AUDIT (reporte generado vs real):")
    for name in CONFIG_NAMES:
        audit_path = RESULTS_DIR / f"audit_{name}.txt"
        if not audit_path.exists():
            continue
        # Leer las últimas líneas no vacías
        lines = [l.rstrip() for l in audit_path.read_text().splitlines()
                 if l.strip() and not l.startswith("#") and not l.startswith("─")]
        if lines:
            print(f"\n  [{name}]")
            for line in lines[-AUDIT_TAIL:]:
                # Truncar líneas muy largas para que quepan en el monitor
                print(f"    {line[:120]}")

    # ── 4. Últimas líneas del log general ─────────────────────────────────────
    log_path = RESULTS_DIR / "semifinal.log"
    if log_path.exists():
        print(f"\n{'─'*80}")
        print(f"📄  ÚLTIMAS {LOG_TAIL_LINES} LÍNEAS DE semifinal.log:")
        all_lines = log_path.read_text(errors="replace").splitlines()
        for line in all_lines[-LOG_TAIL_LINES:]:
            print(f"   {line}")

    # ── 5. Gráficos disponibles ────────────────────────────────────────────────
    plots_dir = RESULTS_DIR / "plots"
    if plots_dir.exists():
        pngs = sorted(plots_dir.glob("*.png"))
        if pngs:
            print(f"\n{'─'*80}")
            print("📈  GRÁFICOS DISPONIBLES:")
            for p in pngs:
                print(f"   {p.name}")

    print("\n" + "=" * 80)
    print("💡  watch -n 15 python monitor_semifinal.py  │  tail -f semifinal_results/semifinal.log")
    print("=" * 80 + "\n")


def _config_r(name: str) -> int:
    return {"Config_4": 16, "Config_5": 16, "Config_6": 32}.get(name, "?")


def _config_a(name: str) -> int:
    return {"Config_4": 32, "Config_5": 16, "Config_6": 64}.get(name, "?")


if __name__ == "__main__":
    try:
        monitor_semifinal()
    except KeyboardInterrupt:
        print("\n👋 Monitor detenido.\n")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error en monitor: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)

