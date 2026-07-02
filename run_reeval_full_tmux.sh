#!/bin/bash
###############################################################################
# run_reeval_full_tmux.sh — Re-evaluación COMPLETA SUCIO vs LIMPIO (Fase 2)
###############################################################################
#
# Ejecuta reevaluar_phase2_full.py en una sesión tmux con monitor de GPU.
# Recalcula, sobre los 19,137 reportes del Test Set de Fase 2 (prompt diseñado),
# las 4 métricas (BLEU-4, ROUGE-L, F1-RadGraph, Clinical BERTScore P/R/F1) para:
#
#   • generated_report         (reporte SUCIO, con Tail Babbling)  → *_sucio
#   • generated_report_clean   (reporte LIMPIO, post-guillotina)   → *_limpio
#
# Diferencias clave vs el intento anterior (que abortó en 30 muestras):
#   - Trunca a 510 tokens SOLO la entrada de BERTScore (evita el RuntimeError
#     de Bio_ClinicalBERT con reportes >512 tokens).
#   - Checkpoint por versión: si el pod se pausa, al relanzar continúa.
#
# Salidas (phase2_results-50pct/evaluation/):
#   predicciones_Config_B1_clean_reeval_detalle.csv   (19,137 filas)
#   predicciones_Config_B1_clean_reeval_resumen.csv   (6 métricas, n=19,137)
#
# USO:
#   bash run_reeval_full_tmux.sh                 # corrida completa (19,137)
#   bash run_reeval_full_tmux.sh --no-attach     # en segundo plano
#   bash run_reeval_full_tmux.sh --reattach      # reconectar
#   bash run_reeval_full_tmux.sh --limit 500     # subconjunto (args → script)
#   bash run_reeval_full_tmux.sh --fresh         # ignorar checkpoints
###############################################################################

set -uo pipefail

WORKDIR="/workspace/tesis-blip2-mimic"
RESULTS_DIR="$WORKDIR/phase2_results-50pct/evaluation"
LOG_FILE="$RESULTS_DIR/reeval_full_Config_B1.log"
SESSION="reeval_full"

if [[ "${1:-}" == "--reattach" ]]; then
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        tmux attach -t "$SESSION"
    else
        echo "ERROR: no existe la sesion '$SESSION'."
    fi
    exit 0
fi

NO_ATTACH=0
PY_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --no-attach|--detach) NO_ATTACH=1 ;;
        *) PY_ARGS+=("$arg") ;;
    esac
done
PY_ARGS_STR="${PY_ARGS[*]:-}"

echo "======================================================================"
echo "  RE-EVALUACIÓN COMPLETA · SUCIO vs LIMPIO (Fase 2, 19,137)"
echo "======================================================================"
mkdir -p "$RESULTS_DIR"
echo "  Args  : ${PY_ARGS_STR:-ninguno}"
echo "  Log   : $LOG_FILE"
echo ""

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "AVISO: ya existe la sesion '$SESSION'."
    if [[ "$NO_ATTACH" -eq 1 ]]; then
        echo "Reconectate: tmux attach -t $SESSION"; exit 0
    fi
    printf "  [K] Matar y relanzar   [R] Reconectarse   [C] Cancelar -> "
    read -r choice
    case "$choice" in
        k|K) tmux kill-session -t "$SESSION"; echo "Sesion anterior eliminada." ;;
        r|R) tmux attach -t "$SESSION"; exit 0 ;;
        *)   echo "Cancelado."; exit 0 ;;
    esac
fi

tmux new-session -d -s "$SESSION" -x 220 -y 55

tmux send-keys -t "$SESSION" \
"cd $WORKDIR && \
echo \"[\$(date +%H:%M:%S)] === Re-evaluacion COMPLETA SUCIO vs LIMPIO (Fase 2) ===\" | tee -a $LOG_FILE && \
PYTHONUNBUFFERED=1 python -u reevaluar_phase2_full.py $PY_ARGS_STR 2>&1 | tee -a $LOG_FILE; \
EXIT_CODE=\${PIPESTATUS[0]}; \
echo \"\"; \
echo \"[\$(date +%H:%M:%S)] === RE-EVALUACION FINALIZADA (exit \$EXIT_CODE) ===\" | tee -a $LOG_FILE; \
echo 'Ctrl+b d para desconectarte.'" Enter

tmux split-window -v -p 28 -t "$SESSION"
tmux send-keys -t "$SESSION" \
"watch -n 10 -t 'nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.free,temperature.gpu --format=csv,noheader,nounits'" Enter

tmux select-pane -t "$SESSION:0.0"

echo "======================================================================"
echo "  SESION TMUX '$SESSION' LISTA"
echo "======================================================================"
echo "  Reconectarse : tmux attach -t $SESSION"
echo "  Log en vivo  : tail -f $LOG_FILE"
echo "  Detener      : tmux kill-session -t $SESSION"
echo "======================================================================"

if [[ "$NO_ATTACH" -eq 1 ]]; then
    echo "Modo --no-attach: la re-evaluacion corre en segundo plano."
else
    sleep 2
    tmux attach -t "$SESSION"
fi
