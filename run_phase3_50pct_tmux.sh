#!/bin/bash
###############################################################################
# run_phase3_50pct_tmux.sh — Fase 3 (50%): Q-Former LoRA + language_projection
###############################################################################
#
# Lanza run_phase3_50pct.py en una sesión tmux con monitor de GPU.
# Entrena el Q-Former (LoRA) + language_projection con ViT+LLM congelados,
# enseñando al modelo a DETENER la generación (token EOS) para eliminar el
# "Tail Babbling" sobre el 50% de MIMIC-CXR (Config_Final_B).
#
# USO:
#   bash run_phase3_50pct_tmux.sh                 # lanza y se conecta
#   bash run_phase3_50pct_tmux.sh --no-attach     # lanza en segundo plano
#   bash run_phase3_50pct_tmux.sh --reattach      # reconectar a sesión viva
#   bash run_phase3_50pct_tmux.sh --skip-train    # (args extra → al script Python)
#   bash run_phase3_50pct_tmux.sh --num-samples 500
#
# DENTRO DE TMUX:
#   Panel superior → entrenamiento/eval en vivo (stdout + stderr, con tee al log)
#   Panel inferior → nvidia-smi cada 10 s
#   Ctrl+b + d     → desconectarse (sigue corriendo)
#
# LOG:
#   tail -f phase3_results-50pct/tournament_phase3_50pct.log
###############################################################################

set -euo pipefail

WORKDIR="/workspace/tesis-blip2-mimic"
SCRIPT="$WORKDIR/run_phase3_50pct.py"
RESULTS_DIR="$WORKDIR/phase3_results-50pct"
LOG_FILE="$RESULTS_DIR/tournament_phase3_50pct.log"
SESSION="fase3_50pct"

# ── Modo --reattach ───────────────────────────────────────────────────────────
if [[ "${1:-}" == "--reattach" ]]; then
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "Reconectando a sesion '$SESSION'..."
        tmux attach -t "$SESSION"
    else
        echo "ERROR: No existe la sesion '$SESSION'. Lanzala con: bash run_phase3_50pct_tmux.sh"
    fi
    exit 0
fi

# ── Parsear flags propios y separar args para Python ──────────────────────────
NO_ATTACH=0
PY_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --no-attach|--detach) NO_ATTACH=1 ;;
        *) PY_ARGS+=("$arg") ;;
    esac
done
EXTRA_ARGS="${PY_ARGS[*]:-}"

# ── Verificaciones previas ────────────────────────────────────────────────────
echo "======================================================================"
echo "  FASE 3 (50%) — Q-Former LoRA + language_projection (anti Tail-Babbling)"
echo "======================================================================"

if [ ! -f "$SCRIPT" ]; then
    echo "ERROR: no se encontro $SCRIPT"; exit 1
fi
mkdir -p "$RESULTS_DIR"

echo "  Script    : $SCRIPT"
echo "  Config    : Config_Final_B (base B: LLM r=32 + ViT r=8 de Fase 2/50%)"
echo "  Q-Former  : LoRA r=32, alpha=64  +  language_projection (full FT)"
echo "  Log       : $LOG_FILE"
echo "  Args extra: ${EXTRA_ARGS:-ninguno}"
echo ""

# ── Manejo de sesion previa ───────────────────────────────────────────────────
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "AVISO: ya existe la sesion tmux '$SESSION'."
    if [[ "$NO_ATTACH" -eq 1 ]]; then
        echo "Modo --no-attach: no se relanza. Reconectate: tmux attach -t $SESSION"
        exit 0
    fi
    printf "  [K] Matar y relanzar   [R] Reconectarse   [C] Cancelar -> "
    read -r choice
    case "$choice" in
        k|K) tmux kill-session -t "$SESSION"; echo "Sesion anterior eliminada." ;;
        r|R) tmux attach -t "$SESSION"; exit 0 ;;
        *)   echo "Cancelado. Reconectate con: tmux attach -t $SESSION"; exit 0 ;;
    esac
fi

# ── Crear sesion tmux ─────────────────────────────────────────────────────────
echo "Creando sesion tmux '$SESSION'..."
tmux new-session -d -s "$SESSION" -x 220 -y 55

# ── Panel superior: entrenamiento con tee al log ──────────────────────────────
# PYTHONUNBUFFERED=1 + python -u → salida en tiempo real; 2>&1 captura stderr;
# tee duplica al log. Al terminar imprime el código de salida.
tmux send-keys -t "$SESSION" \
"cd $WORKDIR && \
echo \"[\$(date +%H:%M:%S)] === Iniciando Fase 3 (50%) Q-Former + language_projection ===\" | tee -a $LOG_FILE && \
PYTHONUNBUFFERED=1 python -u $SCRIPT $EXTRA_ARGS 2>&1 | tee -a $LOG_FILE; \
EXIT_CODE=\${PIPESTATUS[0]}; \
echo \"\"; \
echo \"[\$(date +%H:%M:%S)] === FASE 3 (50%) FINALIZADA (exit \$EXIT_CODE) ===\" | tee -a $LOG_FILE; \
echo 'Ctrl+b d para desconectarte, o Ctrl+b & para cerrar la sesion.'" Enter

# ── Panel inferior: nvidia-smi ────────────────────────────────────────────────
tmux split-window -v -p 30 -t "$SESSION"
tmux send-keys -t "$SESSION" \
"watch -n 10 -t 'nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.free,temperature.gpu --format=csv,noheader,nounits'" Enter

tmux select-pane -t "$SESSION:0.0"

# ── Info al usuario ───────────────────────────────────────────────────────────
echo ""
echo "======================================================================"
echo "  SESION TMUX '$SESSION' LISTA"
echo "======================================================================"
echo "  Panel superior: entrenamiento Fase 3 en vivo"
echo "  Panel inferior: GPU (refresco 10s)"
echo ""
echo "  Reconectarse : tmux attach -t $SESSION   (o bash run_phase3_50pct_tmux.sh --reattach)"
echo "  Log en vivo  : tail -f $LOG_FILE"
echo "  Detener      : tmux kill-session -t $SESSION"
echo "======================================================================"

if [[ "$NO_ATTACH" -eq 1 ]]; then
    echo "Modo --no-attach: la Fase 3 corre en segundo plano."
else
    echo "Conectando en 2 segundos... (Ctrl+b d para desconectarte sin detener)"
    sleep 2
    tmux attach -t "$SESSION"
fi
