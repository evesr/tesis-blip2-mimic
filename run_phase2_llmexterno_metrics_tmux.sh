#!/bin/bash
###############################################################################
# run_phase2_llmexterno_metrics_tmux.sh
#
# Etapa A) Formatea los reportes SUCIOS de la inferencia original (prompt
# diseñado, Config_B1) usando llm_formatter_pipeline.py.
# Etapa B) Evalúa métricas ORIGINAL vs FORMATEADO (BLEU/ROUGE/RadGraph/
#          ClinicalBERT) con truncado seguro para BERTScore.
#
# Entradas:
#   phase2_results-50pct/evaluation/predicciones_Config_B1.csv
# Salidas:
#   phase2_results-50pct/evaluation/predicciones_Config_B1_llmexterno.csv
#   phase2_results-50pct/evaluation/predicciones_Config_B1_llmexterno_metrics_detalle.csv
#   phase2_results-50pct/evaluation/predicciones_Config_B1_llmexterno_metrics_resumen.csv
###############################################################################

set -uo pipefail

WORKDIR="/workspace/tesis-blip2-mimic"
EVAL_DIR="$WORKDIR/phase2_results-50pct/evaluation"
SESSION="phase2_llmext"
LOG_FILE="$EVAL_DIR/phase2_llmexterno_pipeline.log"

IN_CSV="$EVAL_DIR/predicciones_Config_B1.csv"
FMT_CSV="$EVAL_DIR/predicciones_Config_B1_llmexterno.csv"
OUT_DET="$EVAL_DIR/predicciones_Config_B1_llmexterno_metrics_detalle.csv"
OUT_RES="$EVAL_DIR/predicciones_Config_B1_llmexterno_metrics_resumen.csv"

if [[ "${1:-}" == "--reattach" ]]; then
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        tmux attach -t "$SESSION"
    else
        echo "ERROR: no existe la sesion '$SESSION'."
    fi
    exit 0
fi

NO_ATTACH=0
FORCE_PROVIDER="auto"
for arg in "$@"; do
    case "$arg" in
        --no-attach|--detach) NO_ATTACH=1 ;;
        --provider=*) FORCE_PROVIDER="${arg#--provider=}" ;;
    esac
done

echo "======================================================================"
echo "  PHASE2 · Formateo LLM externo + métricas (original vs procesado)"
echo "======================================================================"
echo "  Entrada  : $IN_CSV"
echo "  Formato  : $FMT_CSV"
echo "  Detalle  : $OUT_DET"
echo "  Resumen  : $OUT_RES"
echo "  Provider : $FORCE_PROVIDER"
echo "  Log      : $LOG_FILE"
echo "======================================================================"

mkdir -p "$EVAL_DIR"

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
echo \"[\$(date +%H:%M:%S)] === ETAPA A: Formateo con LLM externo ===\" | tee -a $LOG_FILE && \
PYTHONUNBUFFERED=1 python -u llm_formatter_pipeline.py \
  --raw-csv $IN_CSV \
  --out-csv $FMT_CSV \
  --provider $FORCE_PROVIDER \
  --resume 2>&1 | tee -a $LOG_FILE; \
A=\${PIPESTATUS[0]}; \
if [[ \$A -ne 0 ]]; then \
  echo \"[\$(date +%H:%M:%S)] ETAPA A FALLÓ (exit \$A)\" | tee -a $LOG_FILE; \
  exit \$A; \
fi; \
echo \"[\$(date +%H:%M:%S)] === ETAPA B: Métricas original vs formateado ===\" | tee -a $LOG_FILE && \
PYTHONUNBUFFERED=1 python -u exp1_metrics_comparison.py \
  --in-csv $FMT_CSV \
  --col-ref reference_report \
  --col-original generated_report \
  --col-formateado formatted_report \
  --out-detalle $OUT_DET \
  --out-resumen $OUT_RES 2>&1 | tee -a $LOG_FILE; \
B=\${PIPESTATUS[0]}; \
echo \"\"; \
echo \"[\$(date +%H:%M:%S)] === PIPELINE FINALIZADO (exit \$B) ===\" | tee -a $LOG_FILE; \
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
    echo "Modo --no-attach: corriendo en segundo plano."
else
    sleep 2
    tmux attach -t "$SESSION"
fi
