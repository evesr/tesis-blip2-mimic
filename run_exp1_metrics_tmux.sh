#!/bin/bash
###############################################################################
# run_exp1_metrics_tmux.sh — Solo Etapa 3 del Experimento 1 (métricas)
###############################################################################
#
# Ejecuta exp1_metrics_comparison.py en una sesión tmux con monitor de GPU.
# Genera, sobre los 19,137 reportes del Test Set, las MISMAS 4 métricas que la
# evaluación de Fase 2 (BLEU-4, ROUGE-L, F1-RadGraph y Clinical BERTScore P/R/F1):
#
#   • generated_report  (reporte CRUDO del nuevo prompt CoT)   → columnas *_original
#   • formatted_report  (reporte formateado por el LLM)        → columnas *_formateado
#
# Salidas (phase2_results-50pct/exp1_cot/):
#   exp1_comparison_detalle.csv   (por reporte, análogo a *_reeval_detalle.csv)
#   exp1_comparison_resumen.csv   (agregado,  análogo a *_reeval_resumen.csv)
#
# USO:
#   bash run_exp1_metrics_tmux.sh                 # corrida completa (19,137)
#   bash run_exp1_metrics_tmux.sh --no-attach     # en segundo plano
#   bash run_exp1_metrics_tmux.sh --reattach      # reconectar
#   bash run_exp1_metrics_tmux.sh --limit 500     # subconjunto (args → script Python)
###############################################################################

set -uo pipefail

WORKDIR="/workspace/tesis-blip2-mimic"
RESULTS_DIR="$WORKDIR/phase2_results-50pct/exp1_cot"
LOG_FILE="$RESULTS_DIR/exp1_metrics.log"
SESSION="exp1_metrics"

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
echo "  EXP1 · Etapa 3 — Métricas (BLEU/ROUGE/RadGraph/ClinicalBERT)"
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
echo \"[\$(date +%H:%M:%S)] === EXP1 · Etapa 3: Metricas comparativas ===\" | tee -a $LOG_FILE && \
PYTHONUNBUFFERED=1 python -u exp1_metrics_comparison.py $PY_ARGS_STR 2>&1 | tee -a $LOG_FILE; \
EXIT_CODE=\${PIPESTATUS[0]}; \
echo \"\"; \
echo \"[\$(date +%H:%M:%S)] === ETAPA 3 FINALIZADA (exit \$EXIT_CODE) ===\" | tee -a $LOG_FILE; \
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
    echo "Modo --no-attach: la Etapa 3 corre en segundo plano."
else
    sleep 2
    tmux attach -t "$SESSION"
fi
