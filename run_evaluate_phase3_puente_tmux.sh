#!/bin/bash
###############################################################################
# run_evaluate_phase3_puente_tmux.sh
###############################################################################
# Lanza evaluate_phase3_puente.py en una sesión tmux con monitor de GPU.
#
# Hace:
#   FASE 1 → Inferencia con prompt diseñado sobre Test Set 50% (19,137 imgs)
#            con num_beams=5, max_new_tokens=400 (igual que Fase 2).
#   FASE 2 → BLEU-4 + ROUGE-L + F1-RadGraph + CheXpert (CheXbert + GT oficial).
#
# El modelo se reconstruye en cada arranque (4 pasos):
#   1) Base BLIP-2 bf16 (Salesforce/blip2-opt-2.7b)
#   2) ViT LoRA Fase 2 + merge_and_unload()      [permanente]
#   3) LLM LoRA Fase 2 + merge_and_unload()      [permanente]
#   4) load_state_dict(puente_state.pt, strict=False)  [Phase 3 PUENTE]
#
# Salidas (en phase3_puente_results-50pct/evaluation/):
#   - predicciones_Phase3_Puente.csv         (raw)
#   - resultados_granulares_Phase3_Puente.csv
#   - metricas_globales_Phase3_Puente.csv
#
# USO:
#   bash run_evaluate_phase3_puente_tmux.sh                 # lanza y conecta
#   bash run_evaluate_phase3_puente_tmux.sh --no-attach     # lanza en bg
#   bash run_evaluate_phase3_puente_tmux.sh --reattach      # reconectar
#   bash run_evaluate_phase3_puente_tmux.sh --resume        # reanudar inferencia
#   bash run_evaluate_phase3_puente_tmux.sh --skip-inference # solo re-evaluar
#   bash run_evaluate_phase3_puente_tmux.sh --num-samples 200 # smoke test
#
# Ctrl+b + d = desconectarte (sigue corriendo)
# Ctrl+C en panel superior = ABORTA
###############################################################################

set -euo pipefail

WORKDIR="/workspace/tesis-blip2-mimic"
SCRIPT="$WORKDIR/evaluate_phase3_puente.py"
RESULTS_DIR="$WORKDIR/phase3_puente_results-50pct/evaluation"
LOG_FILE="$RESULTS_DIR/evaluate_phase3_puente.log"
SESSION="eval_fase3_puente"

# ── Modo --reattach ───────────────────────────────────────────────────────────
if [[ "${1:-}" == "--reattach" ]]; then
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "Reconectando a sesion '$SESSION'..."
        tmux attach -t "$SESSION"
    else
        echo "ERROR: No existe la sesion '$SESSION'."
    fi
    exit 0
fi

# ── Parsear flags propios + args para Python ──────────────────────────────────
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
echo "  EVALUACIÓN FASE 3 PUENTE — prompt diseñado · Test Set 50%"
echo "======================================================================"

if [ ! -f "$SCRIPT" ]; then
    echo "ERROR: no se encontro $SCRIPT"; exit 1
fi
mkdir -p "$RESULTS_DIR"

# Pre-check: deben existir los pesos LoRA Fase 2 + el puente_state.pt
LLM_LORA="$WORKDIR/phase2_results-50pct/checkpoints/best_model_Config_B1/llm/adapter_config.json"
VIT_LORA="$WORKDIR/phase2_results-50pct/checkpoints/best_model_Config_B1/vit/adapter_config.json"
PUENTE_PT="$WORKDIR/phase3_puente_results-50pct/checkpoints/best_model_Phase3_Puente/puente_state.pt"
TEST_CSV="$WORKDIR/test_split_50pct.csv"

missing=0
[ -f "$LLM_LORA" ]  || { echo "  ✗ FALTA LLM LoRA: $LLM_LORA"; missing=1; }
[ -f "$VIT_LORA" ]  || { echo "  ✗ FALTA ViT LoRA: $VIT_LORA"; missing=1; }
[ -f "$PUENTE_PT" ] || { echo "  ✗ FALTA puente Phase 3: $PUENTE_PT"; missing=1; }
[ -f "$TEST_CSV" ]  || { echo "  ✗ FALTA test split: $TEST_CSV"; missing=1; }
if [ "$missing" = "1" ]; then
    echo "Pre-requisitos no cumplidos."; exit 1
fi

echo "  Script    : $SCRIPT"
echo "  Test CSV  : $TEST_CSV"
echo "  ViT LoRA  : OK"
echo "  LLM LoRA  : OK"
echo "  Puente F3 : OK ($(stat -c %s "$PUENTE_PT" | numfmt --to=iec))"
echo "  Output    : $RESULTS_DIR"
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

# ── Panel superior: inferencia + métricas con tee al log ──────────────────────
tmux send-keys -t "$SESSION" \
"cd $WORKDIR && \
clear && \
echo '╔════════════════════════════════════════════════════════════════════╗' && \
echo '║     ⚠️   PANEL DE INFERENCIA  -  NO TIPEAR / PEGAR AQUÍ  ⚠️         ║' && \
echo '║                                                                    ║' && \
echo '║  Cualquier Ctrl+C / texto en este panel ABORTA la corrida.         ║' && \
echo '║  Para monitorear usa OTRO terminal:                                ║' && \
echo '║    tail -f phase3_puente_results-50pct/evaluation/evaluate_phase3_puente.log  ║' && \
echo '║    wc -l  phase3_puente_results-50pct/evaluation/predicciones_Phase3_Puente.csv  ║' && \
echo '║  Para desconectar (sin matar): Ctrl+b  d                           ║' && \
echo '╚════════════════════════════════════════════════════════════════════╝' && \
echo \"[\$(date +%H:%M:%S)] === Iniciando evaluacion FASE 3 PUENTE ===\" | tee -a $LOG_FILE && \
PYTHONUNBUFFERED=1 \
TOKENIZERS_PARALLELISM=false \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
OMP_NUM_THREADS=16 \
TRANSFORMERS_NO_ADVISORY_WARNINGS=1 \
python -u $SCRIPT $EXTRA_ARGS 2>&1 | tee -a $LOG_FILE; \
EXIT_CODE=\${PIPESTATUS[0]}; \
echo \"\"; \
echo \"[\$(date +%H:%M:%S)] === EVALUACION PHASE 3 PUENTE FINALIZADA (exit \$EXIT_CODE) ===\" | tee -a $LOG_FILE; \
echo 'Ctrl+b d para desconectarte, o Ctrl+b & para cerrar la sesion.'" Enter

# ── Panel inferior: nvidia-smi (cursor por defecto cae aqui = safe) ──────────
tmux split-window -v -p 30 -t "$SESSION"
tmux send-keys -t "$SESSION" \
"watch -n 10 -t 'nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.free,temperature.gpu --format=csv,noheader,nounits'" Enter

tmux select-pane -t "$SESSION:0.1"

# ── Info al usuario ───────────────────────────────────────────────────────────
echo ""
echo "======================================================================"
echo "  SESION TMUX '$SESSION' LISTA"
echo "======================================================================"
echo "  Panel superior: inferencia + metricas Fase 3 PUENTE  (⚠️  no escribas)"
echo "  Panel inferior: GPU (refresco 10s)  ← cursor por defecto cae aqui (safe)"
echo ""
echo "  Reconectarse : tmux attach -t $SESSION   (o bash run_evaluate_phase3_puente_tmux.sh --reattach)"
echo "  Log en vivo  : tail -f $LOG_FILE                       ← OTRO terminal"
echo "  Filas raw    : wc -l $RESULTS_DIR/predicciones_Phase3_Puente.csv  ← OTRO terminal"
echo "  Detener      : tmux kill-session -t $SESSION             (NO uses Ctrl+C)"
echo "======================================================================"

if [[ "$NO_ATTACH" -eq 1 ]]; then
    echo "Modo --no-attach: la evaluacion corre en segundo plano."
else
    echo "Conectando en 2 segundos... (Ctrl+b d para desconectarte sin detener)"
    sleep 2
    tmux attach -t "$SESSION"
fi
