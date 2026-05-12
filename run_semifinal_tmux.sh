#!/bin/bash
###############################################################################
# Lanzador Gran Final — BLIP-2 + LoRA  (sesion tmux con monitor en tiempo real)
###############################################################################
#
# USO:
#   bash run_semifinal_tmux.sh            # lanza todo en tmux
#   bash run_semifinal_tmux.sh --reattach  # re-conectar a sesion existente
#
# DENTRO DE TMUX:
#   Panel superior  → entrenamiento (stdout directo)
#   Panel inferior  → monitor de metricas (watch cada 30s)
#
# NAVEGACION TMUX:
#   Ctrl+b + flecha   → cambiar de panel
#   Ctrl+b + d        → desconectarse (entrenamiento sigue corriendo)
#   tmux attach -t gran_final  → reconectarse desde otra terminal
#   Ctrl+b + &        → matar la sesion completa (detiene entrenamiento)
#
# MONITOREO ADICIONAL (desde cualquier terminal):
#   tail -f /workspace/tesis-blip2-mimic/final_results/gran_final.log
#   watch -n 3 nvidia-smi
#
# Autor: Evelyn Silva Rozas
# Fecha: Mayo 2026
###############################################################################

set -euo pipefail

WORKDIR="/workspace/tesis-blip2-mimic"
SCRIPT="$WORKDIR/train_semifinal.py"
MONITOR="$WORKDIR/monitor_semifinal.py"
RESULTS_DIR="$WORKDIR/final_results"
LOG_FILE="$RESULTS_DIR/gran_final.log"
PID_FILE="$RESULTS_DIR/gran_final.pid"
SESSION="gran_final"

# ── re-attach Modo ───────
if [[ "${1:-}" == "--reattach" ]]; then
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "Reconectando a sesion '$SESSION'..."
        tmux attach -t "$SESSION"
    else
        echo "ERROR: No existe la sesion '$SESSION'. Lanzala con:"
        echo "  bash run_semifinal_tmux.sh"
    fi
    exit 0
fi

# ── Verificaciones previas ───────────────────────────────────────────────
echo "======================================================================"
echo "  GRAN FINAL - BLIP-2 + LoRA"
echo "======================================================================"

if [ ! -f "$SCRIPT" ]; then
    echo "ERROR: no se encontro $SCRIPT"
    exit 1
fi

if [ ! -f "$MONITOR" ]; then
    echo "ERROR: no se encontro $MONITOR"
    exit 1
fi

for split in train_split_10pct_limpio.csv val_split_10pct_limpio.csv test_split_10pct_limpio.csv; do
    if [ ! -f "$WORKDIR/$split" ]; then
        echo "ERROR: no se encontro $split en $WORKDIR"
        exit 1
    fi
done

echo "  Splits 10pct:   OK"
echo "  train_semifinal.py: OK"
echo "  monitor_semifinal.py: OK"
echo ""

mkdir -p "$RESULTS_DIR"

# ── Matar sesion previa si existe ────────────────────────────────────────────
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "AVISO: ya existe la sesion tmux '$SESSION'."
    echo ""
    printf "  [K] Matar y relanzar   [R] Reconectarse   [C] Cancelar -> "
    read -r choice
    case "$choice" in
        k|K)
            tmux kill-session -t "$SESSION"
            echo "Sesion anterior eliminada."
            ;;
        r|R)
            echo "Reconectando..."
            tmux attach -t "$SESSION"
            exit 0
            ;;
        *)
            echo "Cancelado. Para reconectarte:"
            echo "  tmux attach -t $SESSION"
            echo "  bash run_semifinal_tmux.sh --reattach"
            exit 0
            ;;
    esac
fi

# ── Matar proceso previo si corre aun ───────────────────────────
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "Deteniendo entrenamiento previo (PID=$OLD_PID)..."
        kill "$OLD_PID" 2>/dev/null
        sleep 3
    fi
    rm -f "$PID_FILE"
fi

# ── Crear sesion tmux ──────────
echo ""
echo "Creando sesion tmux '$SESSION'..."

# Crear sesion con ventana principal
tmux new-session -d -s "$SESSION" -x 220 -y 50

# ── Panel superior (80% altura)entrenamiento : ────────────────────
# El entrenamiento corre directamente en el panel (stdout visible).
# PYTHONUNBUFFERED=1 garantiza que los print() aparecen en tiempo real.
# tee vuelca ademas al log para persistencia.
tmux send-keys -t "$SESSION" "
cd $WORKDIR && \\
echo 'Iniciando entrenamiento Gran Final...' && \\
PYTHONUNBUFFERED=1 python -u $SCRIPT 2>&1 | tee $LOG_FILE
echo ''
echo '===================================================================='
echo '  ENTRENAMIENTO FINALIZADO. Puedes cerrar esta sesion con Ctrl+b &'
echo '===================================================================='
" Enter

# ── Dividir horizontalmente: panel inferior (20% altura) para monitor ────────
tmux split-window -v -p 30 -t "$SESSION"

# Panel inferior: watch del monitor cada 30s
tmux send-keys -t "$SESSION" "
cd $WORKDIR && \\
sleep 5 && \\
watch -n 30 -t 'python monitor_semifinal.py'
" Enter

# Foco en panel superior (entrenamiento)
tmux select-pane -t "$SESSION:0.0"

# ── Conectarse a la sesion ────────────
echo ""
echo "======================================================================"
echo "  SESION TMUX '$SESSION' LISTA"
echo "======================================================================"
echo ""
echo "  Panel superior (70%): entrenamiento en tiempo real"
echo "  Panel inferior (30%): monitor de metricas (refresco 30s)"
echo ""
echo "NAVEGACION:"
echo "  Ctrl+b + flecha arriba/abajo  → cambiar panel"
echo "  Ctrl+b + d                    → desconectarse (sigue corriendo)"
echo "  tmux attach -t $SESSION       → reconectarse"
echo "  bash run_semifinal_tmux.sh --reattach"
echo ""
echo "MONITOREO ADICIONAL (otra terminal):"
echo "  tail -f $LOG_FILE"
echo "  watch -n 3 nvidia-smi"
echo ""
echo "DETENER ENTRENAMIENTO:"
echo "  kill \$(cat $PID_FILE) 2>/dev/null || tmux kill-session -t $SESSION"
echo ""
echo "Conectando en 2 segundos... (Ctrl+b d para desconectarte sin detener)"
echo "======================================================================"
sleep 2

tmux attach -t "$SESSION"
