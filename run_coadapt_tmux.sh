#!/bin/bash
###############################################################################
# Lanzador Fase 3 Co-adaptación — BLIP-2 Q-Former LoRA r=128 + ViT layers 37-38
# (sesion tmux con monitor en tiempo real)
###############################################################################
#
# USO:
#   bash run_coadapt_tmux.sh            # lanza el entrenamiento en tmux
#   bash run_coadapt_tmux.sh --reattach # re-conectar a sesion existente
#
# DENTRO DE TMUX:
#   Panel superior  → entrenamiento (stdout directo)
#   Panel inferior  → nvidia-smi + tail del log (refresco 30s)
#
# NAVEGACION TMUX:
#   Ctrl+b + flecha   → cambiar de panel
#   Ctrl+b + d        → desconectarse (entrenamiento sigue corriendo)
#   tmux attach -t coadapt  → reconectarse desde otra terminal
#   Ctrl+b + &        → matar la sesion completa (detiene entrenamiento)
#
# MONITOREO ADICIONAL (desde cualquier terminal):
#   tail -f /workspace/tesis-blip2-mimic/phase3_results-coadapt/log.txt
#   watch -n 3 nvidia-smi
#
# Autor: Evelyn Silva Rozas
# Fecha: Junio 2026
###############################################################################

set -euo pipefail

WORKDIR="/workspace/tesis-blip2-mimic"
SCRIPT="$WORKDIR/run_phase3_coadapt.py"
RESULTS_DIR="$WORKDIR/phase3_results-coadapt"
LOG_FILE="$RESULTS_DIR/log.txt"
SESSION="coadapt"

# ── re-attach Modo ───────────────────────────────────────────────────────────
if [[ "${1:-}" == "--reattach" ]]; then
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "Reconectando a sesion '$SESSION'..."
        tmux attach -t "$SESSION"
    else
        echo "ERROR: No existe la sesion '$SESSION'. Lanzala con:"
        echo "  bash run_coadapt_tmux.sh"
    fi
    exit 0
fi

# ── Verificaciones previas ───────────────────────────────────────────────────
echo "======================================================================"
echo "  FASE 3 CO-ADAPTACIÓN — Q-Former r=128 + ViT layers 37-38"
echo "======================================================================"

if [ ! -f "$SCRIPT" ]; then
    echo "ERROR: no se encontro $SCRIPT"
    exit 1
fi

for split in train_split_10pct_limpio.csv val_split_10pct_limpio.csv test_split_10pct_limpio.csv; do
    if [ ! -f "$WORKDIR/$split" ]; then
        echo "ERROR: no se encontro $split en $WORKDIR"
        exit 1
    fi
done

echo "  Splits 10pct:          OK"
echo "  run_phase3_coadapt.py: OK"
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
            echo "  bash run_coadapt_tmux.sh --reattach"
            exit 0
            ;;
    esac
fi

# ── Crear sesion tmux ─────────────────────────────────────────────────────────
echo ""
echo "Creando sesion tmux '$SESSION'..."

tmux new-session -d -s "$SESSION" -x 220 -y 50

# ── Panel superior (70%): entrenamiento ──────────────────────────────────────
# PYTHONUNBUFFERED=1 garantiza que los print() aparecen en tiempo real.
# tee vuelca ademas al log para persistencia.
tmux send-keys -t "$SESSION" "
cd $WORKDIR && \\
echo 'Iniciando Fase 3 Co-adaptacion...' && \\
PYTHONUNBUFFERED=1 python -u $SCRIPT 2>&1 | tee $LOG_FILE
echo ''
echo '===================================================================='
echo '  ENTRENAMIENTO FINALIZADO. Cierra con Ctrl+b & o desconectate con Ctrl+b d'
echo '===================================================================='
" Enter

# ── Panel inferior (30%): monitor GPU + tail del log ─────────────────────────
tmux split-window -v -p 30 -t "$SESSION"
tmux send-keys -t "$SESSION" "
cd $WORKDIR && \\
sleep 5 && \\
watch -n 30 -t 'nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits && echo \"\" && echo \"--- Ultimas lineas del log ---\" && tail -20 $LOG_FILE 2>/dev/null || echo \"(log aun no creado)\"'
" Enter

# Foco en panel superior (entrenamiento)
tmux select-pane -t "$SESSION:0.0"

# ── Conectarse a la sesion ────────────────────────────────────────────────────
echo ""
echo "======================================================================"
echo "  SESION TMUX '$SESSION' LISTA"
echo "======================================================================"
echo ""
echo "  Panel superior (70%): entrenamiento en tiempo real"
echo "  Panel inferior (30%): GPU + tail log (refresco 30s)"
echo ""
echo "NAVEGACION:"
echo "  Ctrl+b + flecha arriba/abajo  → cambiar panel"
echo "  Ctrl+b + d                    → desconectarse (sigue corriendo)"
echo "  tmux attach -t $SESSION       → reconectarse"
echo "  bash run_coadapt_tmux.sh --reattach"
echo ""
echo "MONITOREO ADICIONAL (otra terminal):"
echo "  tail -f $LOG_FILE"
echo "  watch -n 3 nvidia-smi"
echo ""
echo "DETENER ENTRENAMIENTO:"
echo "  tmux kill-session -t $SESSION"
echo ""
echo "Conectando en 2 segundos... (Ctrl+b d para desconectarte sin detener)"
echo "======================================================================"
sleep 2

tmux attach -t "$SESSION"
