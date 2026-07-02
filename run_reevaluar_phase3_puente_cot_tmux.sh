#!/bin/bash
###############################################################################
# run_reevaluar_phase3_puente_cot_tmux.sh
###############################################################################
# Re-evaluación SUCIO vs LIMPIO sobre el CSV CoT de Phase 3 PUENTE tras
# Llama 8b (etapa 7 → etapa 8 del pipeline).
#
# Métricas calculadas (idénticas a etapa 4):
#   • BLEU-4 + ROUGE-L (CPU, fila por fila)
#   • F1-RadGraph (RG_ERbar, batched chunks=256)
#   • Clinical BERTScore (Bio_ClinicalBERT, batched 64, truncado a 510 tokens)
#
# Input  : phase3_puente_results-50pct/exp1_cot/exp1_final_formatted_phase3_puente_llama8b.csv
# Output :
#   • exp1_final_formatted_phase3_puente_llama8b_reeval_detalle.csv
#   • exp1_final_formatted_phase3_puente_llama8b_reeval_resumen.csv
#
# USO:
#   bash run_reevaluar_phase3_puente_cot_tmux.sh                   # ambas
#   bash run_reevaluar_phase3_puente_cot_tmux.sh --no-attach       # 2do plano
#   bash run_reevaluar_phase3_puente_cot_tmux.sh --reattach        # reconectar
#   bash run_reevaluar_phase3_puente_cot_tmux.sh --limit=200       # smoke
#
# Tiempo estimado: ~1h (etapa 4 con mismo flujo y volumen tomó 59 min)
###############################################################################

set -uo pipefail

WORKDIR="/workspace/tesis-blip2-mimic"
INPUT_CSV="$WORKDIR/phase3_puente_results-50pct/exp1_cot/exp1_final_formatted_phase3_puente_llama8b.csv"
LOG_FILE="$WORKDIR/phase3_puente_results-50pct/exp1_cot/reevaluar_cot_phase3puente.log"
SESSION="reeval_p3_cot"

# ── Modo --reattach ───────────────────────────────────────────────────────────
if [[ "${1:-}" == "--reattach" ]]; then
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        tmux attach -t "$SESSION"
    else
        echo "ERROR: no existe la sesion '$SESSION'."
    fi
    exit 0
fi

# ── Parsear flags ─────────────────────────────────────────────────────────────
NO_ATTACH=0
LIMIT_ARG=""
EXTRA_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --no-attach|--detach) NO_ATTACH=1 ;;
        --limit=*)            LIMIT_ARG="--limit ${arg#--limit=}" ;;
        --reattach)           ;;
        *)                    EXTRA_ARGS+=("$arg") ;;
    esac
done

mkdir -p "$(dirname "$LOG_FILE")"

echo "======================================================================"
echo "  RE-EVALUACIÓN SUCIO vs LIMPIO  ·  Phase 3 PUENTE prompt CoT"
echo "======================================================================"
echo "  Input    : $INPUT_CSV"
echo "  col-sucio : generated_report  (BLIP-2 CoT crudo)"
echo "  col-limpio: formatted_report  (Llama 3.1 8B)"
echo "  Métricas : BLEU-4 / ROUGE-L / F1-RadGraph / Clinical BERTScore"
echo "  Limit    : ${LIMIT_ARG:-(none)}"
echo "  Log      : $LOG_FILE"
echo "  Session  : $SESSION"
echo "======================================================================"

# Validaciones
if [ ! -f "$INPUT_CSV" ]; then
    echo "ERROR: no se encontró el CSV de entrada"
    echo "       $INPUT_CSV"
    echo "Ejecuta primero: bash run_vllm_formatter_tmux.sh --tasks=nuevo_phase3puente"
    exit 1
fi
if [ ! -f "$WORKDIR/reevaluar_sucio_vs_limpio.py" ]; then
    echo "ERROR: no se encontró $WORKDIR/reevaluar_sucio_vs_limpio.py"
    exit 1
fi

# ── Manejo de sesion previa ───────────────────────────────────────────────────
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

# ── Crear sesión tmux ─────────────────────────────────────────────────────────
echo "Creando sesion tmux '$SESSION'..."
tmux new-session -d -s "$SESSION" -x 220 -y 55

tmux send-keys -t "$SESSION" \
"cd $WORKDIR && \
clear && \
echo '╔════════════════════════════════════════════════════════════════════╗' && \
echo '║     ⚠️   PANEL DE RE-EVALUACIÓN  -  NO TIPEAR / PEGAR AQUÍ  ⚠️       ║' && \
echo '║                                                                    ║' && \
echo '║  Cualquier Ctrl+C / texto en este panel ABORTA la re-evaluación.   ║' && \
echo '║  Para monitorear usa OTRO terminal:                                ║' && \
echo '║    tail -f phase3_puente_results-50pct/exp1_cot/reevaluar_cot_phase3puente.log ║' && \
echo '║  Para desconectar (sin matar): Ctrl+b  d                           ║' && \
echo '╚════════════════════════════════════════════════════════════════════╝' && \
echo \"[\$(date +%H:%M:%S)] === Re-evaluacion SUCIO vs LIMPIO Phase 3 PUENTE prompt CoT ===\" | tee -a $LOG_FILE && \
PYTHONUNBUFFERED=1 \
TOKENIZERS_PARALLELISM=false \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -u reevaluar_sucio_vs_limpio.py \
  --csv $INPUT_CSV \
  --col-sucio generated_report \
  --col-limpio formatted_report \
  $LIMIT_ARG ${EXTRA_ARGS[*]:-} 2>&1 | tee -a $LOG_FILE; \
EXIT_CODE=\${PIPESTATUS[0]}; \
echo \"\"; \
echo \"[\$(date +%H:%M:%S)] === RE-EVALUACION FINALIZADA (exit \$EXIT_CODE) ===\" | tee -a $LOG_FILE; \
echo 'Ctrl+b d para desconectarte.'" Enter

# ── Panel inferior: nvidia-smi (cursor por defecto cae aqui = safe) ──────────
tmux split-window -v -p 30 -t "$SESSION"
tmux send-keys -t "$SESSION" \
"watch -n 10 -t 'nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.free,temperature.gpu --format=csv,noheader,nounits'" Enter

# Cursor por defecto en el panel inferior (safe pane).
tmux select-pane -t "$SESSION:0.1"

echo ""
echo "======================================================================"
echo "  SESION TMUX '$SESSION' LISTA"
echo "======================================================================"
echo "  Reconectarse : tmux attach -t $SESSION"
echo "  Log en vivo  : tail -f $LOG_FILE                     ← OTRO terminal"
echo "  Detener      : tmux kill-session -t $SESSION         (NO uses Ctrl+C)"
echo "======================================================================"

if [[ "$NO_ATTACH" -eq 1 ]]; then
    echo "Modo --no-attach: corriendo en segundo plano."
else
    echo "Conectando en 2 segundos..."
    sleep 2
    tmux attach -t "$SESSION"
fi
