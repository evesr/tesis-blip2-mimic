#!/bin/bash
###############################################################################
# run_reeval_tmux.sh — Re-evaluación textual SUCIO vs LIMPIO (Config_B1)
###############################################################################
#
# Lanza reevaluar_sucio_vs_limpio.py en una sesión tmux con monitor de GPU.
# Calcula BLEU-4, ROUGE-L, F1-RadGraph y Clinical BERTScore por reporte,
# comparando generated_report (sucio) vs generated_report_clean (limpio).
#
# USO:
#   bash run_reeval_tmux.sh                 # lanza y se conecta
#   bash run_reeval_tmux.sh --no-attach     # lanza en segundo plano
#   bash run_reeval_tmux.sh --reattach      # reconectar a sesión viva
#   bash run_reeval_tmux.sh --no-radgraph   # (args extra → al script Python)
#
# DENTRO DE TMUX:
#   Panel superior → re-evaluación en vivo (stdout + stderr, con tee al log)
#   Panel inferior → nvidia-smi cada 10 s
#   Ctrl+b + d     → desconectarse (sigue corriendo)
#
# LOG:
#   tail -f phase2_results-50pct/evaluation/reeval_Config_B1.log
###############################################################################

set -euo pipefail

WORKDIR="/workspace/tesis-blip2-mimic"
SCRIPT="$WORKDIR/reevaluar_sucio_vs_limpio.py"
INPUT_CSV="$WORKDIR/phase2_results-50pct/evaluation/predicciones_Config_B1_clean.csv"
RESULTS_DIR="$WORKDIR/phase2_results-50pct/evaluation"
LOG_FILE="$RESULTS_DIR/reeval_Config_B1.log"
SESSION="reeval_b1"

# ── Modo --reattach ───────────────────────────────────────────────────────────
if [[ "${1:-}" == "--reattach" ]]; then
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "Reconectando a sesion '$SESSION'..."
        tmux attach -t "$SESSION"
    else
        echo "ERROR: No existe la sesion '$SESSION'. Lanzala con: bash run_reeval_tmux.sh"
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
echo "  RE-EVALUACIÓN TEXTUAL — Config_B1  (SUCIO vs LIMPIO)"
echo "======================================================================"

if [ ! -f "$SCRIPT" ]; then
    echo "ERROR: no se encontro $SCRIPT"; exit 1
fi
if [ ! -f "$INPUT_CSV" ]; then
    echo "ERROR: no se encontro el CSV de entrada $INPUT_CSV"; exit 1
fi
mkdir -p "$RESULTS_DIR"

echo "  Script    : $SCRIPT"
echo "  Input     : $INPUT_CSV"
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

# ── Panel superior: re-evaluación con tee al log ──────────────────────────────
# PYTHONUNBUFFERED=1 + python -u → salida en tiempo real; 2>&1 captura stderr;
# tee duplica al log. Al terminar imprime el código de salida.
tmux send-keys -t "$SESSION" \
"cd $WORKDIR && \
echo \"[\$(date +%H:%M:%S)] === Iniciando re-evaluacion SUCIO vs LIMPIO ===\" | tee -a $LOG_FILE && \
PYTHONUNBUFFERED=1 python -u $SCRIPT --csv $INPUT_CSV $EXTRA_ARGS 2>&1 | tee -a $LOG_FILE; \
EXIT_CODE=\${PIPESTATUS[0]}; \
echo \"\"; \
echo \"[\$(date +%H:%M:%S)] === RE-EVALUACION FINALIZADA (exit \$EXIT_CODE) ===\" | tee -a $LOG_FILE; \
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
echo "  Panel superior: re-evaluacion en vivo"
echo "  Panel inferior: GPU (refresco 10s)"
echo ""
echo "  Reconectarse : tmux attach -t $SESSION   (o bash run_reeval_tmux.sh --reattach)"
echo "  Log en vivo  : tail -f $LOG_FILE"
echo "  Detener      : tmux kill-session -t $SESSION"
echo "======================================================================"

if [[ "$NO_ATTACH" -eq 1 ]]; then
    echo "Modo --no-attach: la re-evaluacion corre en segundo plano."
else
    echo "Conectando en 2 segundos... (Ctrl+b d para desconectarte sin detener)"
    sleep 2
    tmux attach -t "$SESSION"
fi
