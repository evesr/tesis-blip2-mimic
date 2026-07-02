#!/bin/bash
###############################################################################
# run_exp1_cot_phase3_puente_tmux.sh
###############################################################################
# Inferencia CoT (Chain-of-Thought) sobre Phase 3 PUENTE + evaluación granular
# (BLEU-4 / ROUGE-L / F1-RadGraph / CheXpert / Clinical BERTScore).
#
# Modelo
#   • Base BLIP-2 bf16
#   • ViT LoRA (Fase 2) → merge_and_unload()
#   • LLM LoRA (Fase 2) → merge_and_unload()
#   • puente_state.pt    → load_state_dict(strict=False)
#       (qformer + language_projection + query_tokens, ~107 M params)
#
# Generación
#   • num_beams=3, max_new_tokens=512, batch=16  (≈30 GB VRAM en L40S)
#
# Outputs (en phase3_puente_results-50pct/exp1_cot/)
#   • exp1_raw_inference_phase3_puente.csv          (inferencia cruda)
#   • resultados_granulares_Phase3_Puente_CoT.csv   (5 métricas granulares)
#   • metricas_globales_Phase3_Puente_CoT.csv       (resumen)
#
# USO típico
#   bash run_exp1_cot_phase3_puente_tmux.sh                    # full pipeline
#   bash run_exp1_cot_phase3_puente_tmux.sh --no-attach        # 2do plano
#   bash run_exp1_cot_phase3_puente_tmux.sh --reattach
#   bash run_exp1_cot_phase3_puente_tmux.sh --num-samples=200  # smoke
#   bash run_exp1_cot_phase3_puente_tmux.sh --skip-evaluation  # solo inferencia
#   bash run_exp1_cot_phase3_puente_tmux.sh --skip-inference   # solo evaluación
#   bash run_exp1_cot_phase3_puente_tmux.sh --resume           # reanudar
#   bash run_exp1_cot_phase3_puente_tmux.sh --no-bertscore     # debug rápido
#
# Tiempo estimado (full pipeline, Test Set 50% = 19,137):
#   • Inferencia (beam=3, max_new=512, bs=16) : ~9-13 h
#   • CheXbert + RadGraph + ClinBERT          : ~4-6 h (CPU+GPU)
#   • TOTAL                                    : ~13-19 h
###############################################################################

set -uo pipefail

WORKDIR="/workspace/tesis-blip2-mimic"
LOG_FILE="$WORKDIR/phase3_puente_results-50pct/exp1_cot/exp1_cot_phase3_puente.log"
SESSION="exp1_cot_p3"

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
EXTRA_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --no-attach|--detach)    NO_ATTACH=1 ;;
        --reattach)              ;;
        --num-samples=*)         EXTRA_ARGS+=("--num-samples" "${arg#--num-samples=}") ;;
        --batch-size=*)          EXTRA_ARGS+=("--batch-size" "${arg#--batch-size=}") ;;
        --chexbert-batch=*)      EXTRA_ARGS+=("--chexbert-batch" "${arg#--chexbert-batch=}") ;;
        --bert-batch=*)          EXTRA_ARGS+=("--bert-batch" "${arg#--bert-batch=}") ;;
        --skip-inference)        EXTRA_ARGS+=("--skip-inference") ;;
        --skip-evaluation)       EXTRA_ARGS+=("--skip-evaluation") ;;
        --resume)                EXTRA_ARGS+=("--resume") ;;
        --no-bertscore)          EXTRA_ARGS+=("--no-bertscore") ;;
        *)                       EXTRA_ARGS+=("$arg") ;;
    esac
done

mkdir -p "$(dirname "$LOG_FILE")"

echo "======================================================================"
echo "  EXP1 CoT  ·  FASE 3 PUENTE  ·  Inferencia + Evaluación"
echo "======================================================================"
echo "  Workdir : $WORKDIR"
echo "  Log     : $LOG_FILE"
echo "  Session : $SESSION"
echo "  Args    : ${EXTRA_ARGS[*]:-(none)}"
echo "======================================================================"

# Validaciones de pre-requisitos (paths del modelo)
PHASE2_BEST="$WORKDIR/phase2_results-50pct/checkpoints/best_model_Config_B1"
PUENTE_PT="$WORKDIR/phase3_puente_results-50pct/checkpoints/best_model_Phase3_Puente/puente_state.pt"

for p in "$PHASE2_BEST/llm/adapter_config.json" \
         "$PHASE2_BEST/vit/adapter_config.json" \
         "$PUENTE_PT" \
         "$WORKDIR/exp1_cot_inference_phase3_puente.py"; do
    if [ ! -e "$p" ]; then
        echo "ERROR: no se encontró el pre-requisito"
        echo "       $p"
        exit 1
    fi
done

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

# ── Crear sesion tmux ─────────────────────────────────────────────────────────
echo "Creando sesion tmux '$SESSION'..."
tmux new-session -d -s "$SESSION" -x 220 -y 55

tmux send-keys -t "$SESSION" \
"cd $WORKDIR && \
clear && \
echo '╔════════════════════════════════════════════════════════════════════╗' && \
echo '║     ⚠️   PANEL DE INFERENCIA CoT  -  NO TIPEAR / PEGAR AQUÍ  ⚠️       ║' && \
echo '║                                                                    ║' && \
echo '║  Cualquier Ctrl+C / texto en este panel ABORTA la inferencia.      ║' && \
echo '║  Para monitorear usa OTRO terminal:                                ║' && \
echo '║    tail -f phase3_puente_results-50pct/exp1_cot/exp1_cot_phase3_puente.log ║' && \
echo '║  Para desconectar (sin matar): Ctrl+b  d                           ║' && \
echo '╚════════════════════════════════════════════════════════════════════╝' && \
echo \"[\$(date +%H:%M:%S)] === EXP1 CoT Phase 3 PUENTE inicia ===\" | tee -a $LOG_FILE && \
PYTHONUNBUFFERED=1 \
TOKENIZERS_PARALLELISM=false \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -u exp1_cot_inference_phase3_puente.py ${EXTRA_ARGS[*]:-} 2>&1 | tee -a $LOG_FILE; \
EXIT_CODE=\${PIPESTATUS[0]}; \
echo \"\"; \
echo \"[\$(date +%H:%M:%S)] === EXP1 CoT FINALIZADO (exit \$EXIT_CODE) ===\" | tee -a $LOG_FILE; \
echo 'Ctrl+b d para desconectarte.'" Enter

# ── Panel inferior: nvidia-smi (cursor cae aqui = safe) ──────────────────────
tmux split-window -v -p 30 -t "$SESSION"
tmux send-keys -t "$SESSION" \
"watch -n 10 -t 'nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.free,temperature.gpu --format=csv,noheader,nounits'" Enter

# Cursor por defecto en el panel SAFE (inferior)
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
