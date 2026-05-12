#!/bin/bash
###############################################################################
# Lanzador Gran Final — BLIP-2 + LoRA  (nohup, persistente)
###############################################################################
#
# USO:
#   bash run_semifinal_tmux.sh
#
# MONITOREO (en cualquier terminal nueva):
#   python monitor_semifinal.py          # snapshot de métricas
#   watch -n 30 python monitor_semifinal.py   # refresco automático
#   tail -f final_results/gran_final.log      # log crudo en tiempo real
#   watch -n 3 nvidia-smi                     # GPU
#
# CONTROL:
#   cat  final_results/gran_final.pid         # ver PID
#   kill $(cat final_results/gran_final.pid)  # detener limpiamente
#
# Autor: Evelyn Silva Rozas
# Fecha: Mayo 2026
###############################################################################

WORKDIR="/workspace/tesis-blip2-mimic"
SCRIPT="$WORKDIR/train_semifinal.py"
RESULTS_DIR="$WORKDIR/final_results"
LOG_FILE="$RESULTS_DIR/gran_final.log"
PID_FILE="$RESULTS_DIR/gran_final.pid"

echo "======================================================================"
echo "  GRAN FINAL — BLIP-2 + LoRA  (nohup)"
echo "======================================================================"
echo ""

# ── previas Verificaciones ─────────────
if [ ! -f "$SCRIPT" ]; then
    echo "ERROR: no se encontró $SCRIPT"
    exit 1
fi

if [ ! -f "$WORKDIR/train_split_10pct_limpio.csv" ]; then
    echo "ERROR: no se encontró train_split_10pct_limpio.csv"
    echo "  Asegúrate de tener los splits del 10% en $WORKDIR"
    exit 1
fi

mkdir -p "$RESULTS_DIR"

 Verificar si ya hay un proceso corriendo ───────# ─
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "AVISO: ya hay un entrenamiento corriendo (PID=$OLD_PID)"
        echo ""
        read -p "  [K] Matar y relanzar  [C] Cancelar → " choice
        case "$choice" in
          k|K)
            echo "Deteniendo PID $OLD_PID..."
            kill "$OLD_PID" 2>/dev/null
            sleep 3
            ;;
          *)
            echo "Cancelado. Monitorea con:"
            echo "  tail -f $LOG_FILE"
            exit 0
            ;;
        esac
    else
        echo "INFO: PID $OLD_PID ya no está activo. Relanzando..."
        rm -f "$PID_FILE"
    fi
fi

# ── Lanzar con nohup ───────────────────────────
cd "$WORKDIR"

echo "Lanzando train_semifinal.py..."
echo "  Log → $LOG_FILE"
echo ""

# PYTHONUNBUFFERED=1 → los print() llegan al log sin buffering
# python -u          → fuerza modo unbuffered
# nohup + &          → desvincula de la terminal, manda al fondo
nohup env PYTHONUNBUFFERED=1 python -u "$SCRIPT" > "$LOG_FILE" 2>&1 &
TRAIN_PID=$!
echo "$TRAIN_PID" > "$PID_FILE"

# Esperar y confirmar que arrancó
sleep 3
if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
    echo "ERROR: el proceso terminó inesperadamente. Últimas líneas del log:"
"    echo "──────────────────
    tail -20 "$LOG_FILE"
"    echo "───────────────────────────────────────
    exit 1
fi

echo "======================================================================"
echo "  ENTRENAMIENTO CORRIENDO EN SEGUNDO PLANO"
echo "======================================================================"
echo ""
echo "  PID : $TRAIN_PID  (guardado en $PID_FILE)"
echo "  Log : $LOG_FILE"
echo ""
echo "COMANDOS DE MONITOREO:"
echo ""
echo "  # Log crudo en tiempo real:"
echo "  tail -f $LOG_FILE"
echo ""
echo "  # Monitor de mtricas (snapshot):"
echo "  python monitor_semifinal.py"
echo ""
echo "  # Monitor con refresco automático cada 30s:"
echo "  watch -n 30 python monitor_semifinal.py"
echo ""
echo "  # Estado GPU:"
echo "  watch -n 3 nvidia-smi"
echo ""
echo "  # Detener entrenamiento:"
echo "  kill \$(cat $PID_FILE)"
echo ""
echo "ARCHIVOS GENERADOS EN $RESULTS_DIR/:"
echo "  gran_final.log                → salida completa"
echo "  history_final_Config_A.csv    → Step, Val_Loss, BLEU-4, ROUGE-L"
echo "  history_final_Config_B.csv"
echo "  audit_final_Config_A.txt      → 10 reportes por eval (sin truncar)"
echo "  audit_final_Config_B.txt"
echo "  checkpoints/best_model_Config_A/  → mejores pesos"
echo "  checkpoints/best_model_Config_B/"
echo "  plots/01_val_loss.png         → gráficos al finalizar"
echo "  plots/02_bleu4.png"
echo "  plots/03_rougeL.png"
echo "  final_results_ranked.csv      → ranking final"
echo ""
echo "Tiempo estimado: 12-24 h (early stopping patience=10 × 500 steps)"
echo "======================================================================"
echo ""

# Mostrar primeras líneas del log para confirmar arranque
echo "Primeras leas del log:"
"echo "────────────────────────────
sleep 4
head -25 "$LOG_FILE" 2>/dev/null || echo "  (log aún vacío, espera unos segundos)"
"echo "─────────────────────────────────────
echo ""
echo "Puedes cerrar esta terminal. El proceso continuará en segundo plano."
