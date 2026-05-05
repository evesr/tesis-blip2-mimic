#!/bin/bash
###############################################################################
# Script para ejecutar Entrenamiento Semifinal con nohup (persistente)
###############################################################################
#
# DESCRIPCIÓN:
#   Lanza train_semifinal.py en segundo plano con nohup.
#   El proceso sigue corriendo aunque cierres VS Code o la terminal.
#
# USO:
#   bash run_semifinal_tmux.sh
#
# MONITOREO (en cualquier terminal):
#   python monitor_semifinal.py                  # una sola vez
#   watch -n 15 python monitor_semifinal.py      # actualización automática
#   tail -f semifinal_results/semifinal.log      # log en tiempo real
#   watch -n 2 nvidia-smi                        # GPU
#
# CONTROLAR EL PROCESO:
#   cat semifinal_results/semifinal.pid          # ver PID
#   kill $(cat semifinal_results/semifinal.pid)  # detener
#
# Autor: Evelyn Silva Rozas
# Fecha: Mayo 2026
###############################################################################

WORKDIR="/workspace/tesis-blip2-mimic/Tesis_blip2_local"
SCRIPT_PATH="$WORKDIR/train_semifinal.py"
RESULTS_DIR="$WORKDIR/semifinal_results"
LOG_FILE="$RESULTS_DIR/semifinal.log"
PID_FILE="$RESULTS_DIR/semifinal.pid"

echo "================================================================================"
echo "🚀 ENTRENAMIENTO SEMIFINAL — nohup"
echo "================================================================================"
echo ""

# ─── Verificaciones previas ───────────────────────────────────────────────────

if [ ! -f "$SCRIPT_PATH" ]; then
    echo "❌ No se encontró: $SCRIPT_PATH"
    exit 1
fi

mkdir -p "$RESULTS_DIR"

# ─── Verificar si ya hay un proceso corriendo ─────────────────────────────────

if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "⚠️  Ya hay un entrenamiento corriendo (PID=$OLD_PID)"
        echo ""
        read -p "   [K] Matar y relanzar  [C] Cancelar  → " choice
        case "$choice" in
          k|K)
            echo "💀 Deteniendo proceso $OLD_PID..."
            kill "$OLD_PID" 2>/dev/null
            sleep 2
            ;;
          *)
            echo "❌ Cancelado. Monitorea con: tail -f $LOG_FILE"
            exit 0
            ;;
        esac
    else
        echo "ℹ️  PID $OLD_PID encontrado pero ya no está corriendo. Relanzando..."
        rm -f "$PID_FILE"
    fi
fi

# ─── Lanzar con nohup ─────────────────────────────────────────────────────────

echo "✅ Lanzando train_semifinal.py con nohup..."
echo "   Log → $LOG_FILE"
echo ""

cd "$WORKDIR"

# PYTHONUNBUFFERED=1 garantiza que print() se escriba al log inmediatamente.
# nohup desvincula el proceso de la terminal actual.
# & lo manda al fondo y guarda el PID.
nohup env PYTHONUNBUFFERED=1 python -u train_semifinal.py > "$LOG_FILE" 2>&1 &
TRAIN_PID=$!

# Guardar PID para poder matar el proceso después si se necesita
echo "$TRAIN_PID" > "$PID_FILE"

# Esperar un segundo y verificar que el proceso arrancó correctamente
sleep 2
if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
    echo "❌ El proceso terminó inesperadamente. Revisa el log:"
    echo "   tail -20 $LOG_FILE"
    exit 1
fi

echo "================================================================================"
echo "✅ ENTRENAMIENTO CORRIENDO EN SEGUNDO PLANO"
echo "================================================================================"
echo ""
echo "   PID: $TRAIN_PID  (guardado en $PID_FILE)"
echo ""
echo "📋 Comandos de monitoreo:"
echo ""
echo "   Log en tiempo real:       tail -f $LOG_FILE"
echo "   Monitor con métricas:     watch -n 15 python monitor_semifinal.py"
echo "   Estado GPU:               watch -n 2 nvidia-smi"
echo "   Ver PID:                  cat $PID_FILE"
echo "   Detener entrenamiento:    kill \$(cat $PID_FILE)"
echo ""
echo "📁 Archivos generados en semifinal_results/:"
echo "   semifinal.log             → salida completa"
echo "   history_Config_X.csv      → Step, Val_Loss, BLEU, ROUGE-L"
echo "   audit_Config_X.txt        → reportes de muestra por eval"
echo "   checkpoints/best_model_*  → mejores pesos por config"
echo "   plots/01_val_loss.png     → gráficos al finalizar"
echo ""
echo "⏱️  Tiempo estimado: 6-18 h (early stopping patience=10 evals)"
echo "================================================================================"
echo ""
echo "💡 Puedes cerrar esta terminal. El proceso seguirá corriendo."
echo ""

# Mostrar las primeras líneas del log para confirmar que arrancó
echo "📄 Primeras líneas del log:"
echo "─────────────────────────────────────────"
sleep 3
head -20 "$LOG_FILE" 2>/dev/null || echo "   (log aún vacío, espera unos segundos)"
echo "─────────────────────────────────────────"
echo ""
