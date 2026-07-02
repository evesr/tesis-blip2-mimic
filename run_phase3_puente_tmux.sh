#!/bin/bash
###############################################################################
# run_phase3_puente_tmux.sh — Fase 3 (PUENTE): Q-Former + language_projection FFT
###############################################################################
#
# Lanza train_phase3_puente.py en una sesión tmux con monitor de GPU.
# Re-entrena en Full Fine-Tuning el PUENTE (qformer + language_projection)
# sobre el modelo BLIP-2 con los LoRA Fase 2 (LLM r=32, ViT r=8) ya
# FUSIONADOS de forma permanente vía merge_and_unload(), todo sobre el 50%
# de MIMIC-CXR (mismos splits que Fase 2).
#
# La inferencia textual durante el entrenamiento está DESACTIVADA: el callback
# solo registra Train_Loss y Val_Loss en history_phase3_Phase3_Puente.csv tras
# cada eval_steps (la calidad textual se mide después en evaluate_phase3_*).
#
# USO:
#   bash run_phase3_puente_tmux.sh                 # lanza y se conecta
#   bash run_phase3_puente_tmux.sh --no-attach     # lanza en segundo plano
#   bash run_phase3_puente_tmux.sh --reattach      # reconectar a sesión viva
#   bash run_phase3_puente_tmux.sh --skip-train    # (args extra → al script Python)
#
# DENTRO DE TMUX:
#   Panel superior → entrenamiento/eval en vivo (stdout + stderr, con tee al log)
#   Panel inferior → nvidia-smi cada 10 s
#   Ctrl+b + d     → desconectarse (sigue corriendo)
#
# LOG:
#   tail -f phase3_puente_results-50pct/tournament_phase3_puente.log
###############################################################################

set -euo pipefail

WORKDIR="/workspace/tesis-blip2-mimic"
SCRIPT="$WORKDIR/train_phase3_puente.py"
RESULTS_DIR="$WORKDIR/phase3_puente_results-50pct"
LOG_FILE="$RESULTS_DIR/tournament_phase3_puente.log"
SESSION="fase3_puente"

# ── Modo --reattach ───────────────────────────────────────────────────────────
if [[ "${1:-}" == "--reattach" ]]; then
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "Reconectando a sesion '$SESSION'..."
        tmux attach -t "$SESSION"
    else
        echo "ERROR: No existe la sesion '$SESSION'. Lanzala con: bash run_phase3_puente_tmux.sh"
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
echo "  FASE 3 (PUENTE) — Q-Former + language_projection (Full FT, sin LoRA nueva)"
echo "======================================================================"

if [ ! -f "$SCRIPT" ]; then
    echo "ERROR: no se encontro $SCRIPT"; exit 1
fi
mkdir -p "$RESULTS_DIR"

# Pre-check: existir adaptadores LoRA Fase 2 (Config_B1) ya entrenados
LLM_LORA="$WORKDIR/phase2_results-50pct/checkpoints/best_model_Config_B1/llm/adapter_config.json"
VIT_LORA="$WORKDIR/phase2_results-50pct/checkpoints/best_model_Config_B1/vit/adapter_config.json"
if [ ! -f "$LLM_LORA" ] || [ ! -f "$VIT_LORA" ]; then
    echo "ERROR: faltan los adaptadores LoRA de Fase 2 (Config_B1):"
    echo "       LLM : $LLM_LORA  ($([ -f "$LLM_LORA" ] && echo OK || echo FALTA))"
    echo "       ViT : $VIT_LORA  ($([ -f "$VIT_LORA" ] && echo OK || echo FALTA))"
    echo "Verifica que la Fase 2 termino y produjo best_model_Config_B1."
    exit 1
fi

echo "  Script    : $SCRIPT"
echo "  Base      : Config_B1 Fase 2 (LLM r=32, ViT r=8) ya entrenado"
echo "  Estrategia: merge_and_unload(LoRA_LLM) + merge_and_unload(LoRA_ViT)"
echo "              + congelar ViT/LLM + Full FT de qformer + language_projection"
echo "  Inferencia: DESACTIVADA durante eval (solo Train/Val Loss en CSV)"
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
#
# IMPORTANTE: este panel NO debe recibir tipeado/pegado del usuario, porque
# cualquier Ctrl+C interrumpiría el entrenamiento. Por eso el cursor por
# defecto se posiciona en el panel inferior (nvidia-smi), véase «select-pane»
# más abajo. El banner ASCII grande es un recordatorio explícito.
tmux send-keys -t "$SESSION" \
"cd $WORKDIR && \
clear && \
echo '╔════════════════════════════════════════════════════════════════════╗' && \
echo '║     ⚠️   PANEL DE ENTRENAMIENTO  -  NO TIPEAR / PEGAR AQUÍ  ⚠️      ║' && \
echo '║                                                                    ║' && \
echo '║  Cualquier Ctrl+C / texto en este panel ABORTA el entrenamiento.   ║' && \
echo '║  Para monitorear usa OTRO terminal:                                ║' && \
echo '║    tail -f phase3_puente_results-50pct/tournament_phase3_puente.log ║' && \
echo '║    cat  phase3_puente_results-50pct/history_phase3_Phase3_Puente.csv║' && \
echo '║  Para desconectar (sin matar el proceso): Ctrl+b  d                ║' && \
echo '╚════════════════════════════════════════════════════════════════════╝' && \
echo \"[\$(date +%H:%M:%S)] === Iniciando Fase 3 PUENTE Q-Former + language_projection Full FT ===\" | tee -a $LOG_FILE && \
PYTHONUNBUFFERED=1 \
TOKENIZERS_PARALLELISM=false \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
OMP_NUM_THREADS=16 \
TRANSFORMERS_NO_ADVISORY_WARNINGS=1 \
python -u $SCRIPT $EXTRA_ARGS 2>&1 | tee -a $LOG_FILE; \
EXIT_CODE=\${PIPESTATUS[0]}; \
echo \"\"; \
echo \"[\$(date +%H:%M:%S)] === FASE 3 (PUENTE) FINALIZADA (exit \$EXIT_CODE) ===\" | tee -a $LOG_FILE; \
echo 'Ctrl+b d para desconectarte, o Ctrl+b & para cerrar la sesion.'" Enter

# ── Panel inferior: nvidia-smi ────────────────────────────────────────────────
tmux split-window -v -p 30 -t "$SESSION"
tmux send-keys -t "$SESSION" \
"watch -n 10 -t 'nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.free,temperature.gpu --format=csv,noheader,nounits'" Enter

# Cursor por defecto en el panel inferior («safe pane»): si el usuario
# tipea/pega en él, solo afecta a `watch`, NUNCA al entrenamiento. Para subir
# manualmente al panel del training: Ctrl+b + flecha-arriba.
tmux select-pane -t "$SESSION:0.1"

# ── Info al usuario ───────────────────────────────────────────────────────────
echo ""
echo "======================================================================"
echo "  SESION TMUX '$SESSION' LISTA"
echo "======================================================================"
echo "  Panel superior: entrenamiento Fase 3 (PUENTE) en vivo  (⚠️  no escribas en él)"
echo "  Panel inferior: GPU (refresco 10s)  ← cursor por defecto cae aquí (safe)"
echo ""
echo "  Reconectarse : tmux attach -t $SESSION   (o bash run_phase3_puente_tmux.sh --reattach)"
echo "  Log en vivo  : tail -f $LOG_FILE                       ← OTRO terminal"
echo "  CSV losses   : cat $RESULTS_DIR/history_phase3_Phase3_Puente.csv  ← OTRO terminal"
echo "  Detener      : tmux kill-session -t $SESSION             (NO uses Ctrl+C dentro)"
echo "======================================================================"

if [[ "$NO_ATTACH" -eq 1 ]]; then
    echo "Modo --no-attach: la Fase 3 (PUENTE) corre en segundo plano."
else
    echo "Conectando en 2 segundos... (Ctrl+b d para desconectarte sin detener)"
    sleep 2
    tmux attach -t "$SESSION"
fi
