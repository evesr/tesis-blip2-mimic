#!/bin/bash
###############################################################################
# run_metrics_llama8b_tmux.sh
#
# Corre exp1_metrics_comparison.py SECUENCIALMENTE sobre los 2 CSVs producidos
# por vllm_formatter.py (Llama 3.1 8B FP16):
#
#   ETAPA 1: prompt DISEÑADO  (Fase 2 sucio → llama8b)
#     in:   phase2_results-50pct/evaluation/predicciones_Config_B1_llmexterno_llama8b.csv
#     out:  predicciones_Config_B1_llmexterno_llama8b_metrics_detalle.csv
#           predicciones_Config_B1_llmexterno_llama8b_metrics_resumen.csv
#
#   ETAPA 2: prompt NUEVO     (CoT crudo → llama8b)
#     in:   phase2_results-50pct/exp1_cot/exp1_final_formatted_llama8b.csv
#     out:  exp1_comparison_llama8b_detalle.csv
#           exp1_comparison_llama8b_resumen.csv
#
# Métricas (ambas etapas): BLEU-4, ROUGE-L, F1-RadGraph, Clinical BERTScore P/R/F1
# Modelos pesados se cargan una vez por etapa (radgraph-xl + Bio_ClinicalBERT).
# La GPU se libera entre etapas porque cada llamada es un proceso Python distinto.
#
# USO:
#   bash run_metrics_llama8b_tmux.sh                 # ambas etapas (default)
#   bash run_metrics_llama8b_tmux.sh --no-attach     # en segundo plano
#   bash run_metrics_llama8b_tmux.sh --reattach      # reconectar
#   bash run_metrics_llama8b_tmux.sh --only=disenado # SOLO etapa 1
#   bash run_metrics_llama8b_tmux.sh --only=nuevo    # SOLO etapa 2
#   bash run_metrics_llama8b_tmux.sh --no-radgraph   # sin RadGraph (mucho mas rapido)
#   bash run_metrics_llama8b_tmux.sh --limit=200     # smoke test
###############################################################################

set -uo pipefail

WORKDIR="/workspace/tesis-blip2-mimic"
EVAL_DIR="$WORKDIR/phase2_results-50pct/evaluation"
EXP1_DIR="$WORKDIR/phase2_results-50pct/exp1_cot"
SESSION="metrics_llama8b"
LOG_FILE="$WORKDIR/phase2_results-50pct/metrics_llama8b.log"

# ── Etapa 1: DISEÑADO ────────────────────────────────────────────────────────
IN_DISENADO="$EVAL_DIR/predicciones_Config_B1_llmexterno_llama8b.csv"
DET_DISENADO="$EVAL_DIR/predicciones_Config_B1_llmexterno_llama8b_metrics_detalle.csv"
RES_DISENADO="$EVAL_DIR/predicciones_Config_B1_llmexterno_llama8b_metrics_resumen.csv"

# ── Etapa 2: NUEVO (CoT) ─────────────────────────────────────────────────────
IN_NUEVO="$EXP1_DIR/exp1_final_formatted_llama8b.csv"
DET_NUEVO="$EXP1_DIR/exp1_comparison_llama8b_detalle.csv"
RES_NUEVO="$EXP1_DIR/exp1_comparison_llama8b_resumen.csv"

# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────
if [[ "${1:-}" == "--reattach" ]]; then
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        tmux attach -t "$SESSION"
    else
        echo "ERROR: no existe la sesion '$SESSION'."
    fi
    exit 0
fi

NO_ATTACH=0
ONLY="both"
EXTRA_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --no-attach|--detach) NO_ATTACH=1 ;;
        --only=*)             ONLY="${arg#--only=}" ;;
        --no-radgraph)        EXTRA_ARGS+=("--no-radgraph") ;;
        --no-bertscore)       EXTRA_ARGS+=("--no-bertscore") ;;
        --limit=*)            EXTRA_ARGS+=("--limit" "${arg#--limit=}") ;;
        *)                    EXTRA_ARGS+=("$arg") ;;
    esac
done
EXTRA_STR="${EXTRA_ARGS[*]:-}"

case "$ONLY" in
    both|disenado|nuevo) ;;
    *) echo "ERROR: --only debe ser 'both' (default), 'disenado' o 'nuevo'."; exit 1 ;;
esac

# Validar entradas según etapas seleccionadas
if [[ "$ONLY" == "both" || "$ONLY" == "disenado" ]]; then
    if [[ ! -f "$IN_DISENADO" ]]; then
        echo "ERROR: no existe $IN_DISENADO"; exit 1
    fi
fi
if [[ "$ONLY" == "both" || "$ONLY" == "nuevo" ]]; then
    if [[ ! -f "$IN_NUEVO" ]]; then
        echo "ERROR: no existe $IN_NUEVO"; exit 1
    fi
fi

mkdir -p "$EVAL_DIR" "$EXP1_DIR"

echo "======================================================================"
echo "  METRICAS llama8b · BLEU/ROUGE/RadGraph/ClinicalBERT"
echo "======================================================================"
echo "  Modo       : $ONLY"
echo "  Args extra : ${EXTRA_STR:-ninguno}"
[[ "$ONLY" == "both" || "$ONLY" == "disenado" ]] && {
    echo "  --- ETAPA 1 (DISEÑADO) ---"
    echo "    in : $IN_DISENADO"
    echo "    det: $DET_DISENADO"
    echo "    res: $RES_DISENADO"
}
[[ "$ONLY" == "both" || "$ONLY" == "nuevo" ]] && {
    echo "  --- ETAPA 2 (NUEVO/CoT) ---"
    echo "    in : $IN_NUEVO"
    echo "    det: $DET_NUEVO"
    echo "    res: $RES_NUEVO"
}
echo "  Log        : $LOG_FILE"
echo "======================================================================"

# Manejar sesión existente
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

# ─────────────────────────────────────────────────────────────────────────────
# Construir el bloque de comandos según el modo
# ─────────────────────────────────────────────────────────────────────────────
CMD_DISENADO="echo \"[\$(date +%H:%M:%S)] === ETAPA 1: METRICAS · DISEÑADO (sucio→llama8b) ===\" | tee -a $LOG_FILE && \
PYTHONUNBUFFERED=1 python -u exp1_metrics_comparison.py \
  --in-csv $IN_DISENADO \
  --col-ref reference_report \
  --col-original generated_report \
  --col-formateado formatted_report \
  --out-detalle $DET_DISENADO \
  --out-resumen $RES_DISENADO $EXTRA_STR 2>&1 | tee -a $LOG_FILE; \
S1=\${PIPESTATUS[0]}; \
echo \"[\$(date +%H:%M:%S)] === ETAPA 1 FIN (exit \$S1) ===\" | tee -a $LOG_FILE"

CMD_NUEVO="echo \"[\$(date +%H:%M:%S)] === ETAPA 2: METRICAS · NUEVO (CoT→llama8b) ===\" | tee -a $LOG_FILE && \
PYTHONUNBUFFERED=1 python -u exp1_metrics_comparison.py \
  --in-csv $IN_NUEVO \
  --col-ref reference_report \
  --col-original generated_report \
  --col-formateado formatted_report \
  --out-detalle $DET_NUEVO \
  --out-resumen $RES_NUEVO $EXTRA_STR 2>&1 | tee -a $LOG_FILE; \
S2=\${PIPESTATUS[0]}; \
echo \"[\$(date +%H:%M:%S)] === ETAPA 2 FIN (exit \$S2) ===\" | tee -a $LOG_FILE"

case "$ONLY" in
    both)     PIPELINE="$CMD_DISENADO; $CMD_NUEVO" ;;
    disenado) PIPELINE="$CMD_DISENADO" ;;
    nuevo)    PIPELINE="$CMD_NUEVO" ;;
esac

# ─────────────────────────────────────────────────────────────────────────────
# Lanzar tmux
# ─────────────────────────────────────────────────────────────────────────────
tmux new-session -d -s "$SESSION" -x 220 -y 55

tmux send-keys -t "$SESSION" \
"cd $WORKDIR && \
echo \"[\$(date +%H:%M:%S)] === LANZANDO METRICAS LLAMA8B (modo: $ONLY) ===\" | tee -a $LOG_FILE && \
$PIPELINE; \
echo \"\"; \
echo \"[\$(date +%H:%M:%S)] === PIPELINE METRICAS LLAMA8B FINALIZADO ===\" | tee -a $LOG_FILE; \
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
