#!/bin/bash
###############################################################################
# run_experiment2_raddino_tmux.sh — Experimento 2 (Fase 4): RadDINO + Puente FFT
###############################################################################
#
# Lanza train_experiment2_raddino.py en una sesión tmux con monitor de GPU.
# Entrena la arquitectura BLIP-2 personalizada donde el codificador visual
# genérico (EVA-CLIP) se SUSTITUYE por RadDINO (microsoft/rad-dino):
#   • RadDINO backbone   : CONGELADO, solo LoRA entrenable (query, value)
#   • proyector 768→1408 : NUEVO, Full FT (interfaz con el Q-Former)
#   • language_model OPT : LoRA Fase 2 (r=32) FUSIONADO + CONGELADO
#   • qformer + language_projection + query_tokens : Full FT (sin LoRA)
# Todo sobre el 50% de MIMIC-CXR (mismos splits que Fase 2/3).
#
# La inferencia textual durante el entrenamiento está DESACTIVADA: el callback
# solo registra Train_Loss y Val_Loss (+ LRs) en history_exp2_Exp2_RadDINO.csv
# tras cada eval_steps. Guarda checkpoints ligeros (solo params entrenables).
#
# USO:
#   bash run_experiment2_raddino_tmux.sh                 # lanza y se conecta
#   bash run_experiment2_raddino_tmux.sh --no-attach     # lanza en segundo plano
#   bash run_experiment2_raddino_tmux.sh --reattach      # reconectar a sesión viva
#   bash run_experiment2_raddino_tmux.sh --fresh         # (args extra → script Python)
#   bash run_experiment2_raddino_tmux.sh --smoke         # 1 paso, valida tubería
#
# DENTRO DE TMUX:
#   Panel superior → entrenamiento/eval en vivo (stdout + stderr, con tee al log)
#   Panel inferior → nvidia-smi cada 10 s
#   Ctrl+b + d     → desconectarse (sigue corriendo)
#
# LOG:
#   tail -f exp2_raddino_results-50pct/train_exp2_raddino.log
###############################################################################

set -euo pipefail

WORKDIR="/workspace/tesis-blip2-mimic"
SCRIPT="$WORKDIR/train_experiment2_raddino.py"
RESULTS_DIR="$WORKDIR/exp2_raddino_results-50pct"
LOG_FILE="$RESULTS_DIR/train_exp2_raddino.log"
SESSION="exp2_raddino"

# ── Modo --reattach ───────────────────────────────────────────────────────────
if [[ "${1:-}" == "--reattach" ]]; then
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "Reconectando a sesion '$SESSION'..."
        tmux attach -t "$SESSION"
    else
        echo "ERROR: No existe la sesion '$SESSION'. Lanzala con: bash run_experiment2_raddino_tmux.sh"
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
echo "  EXPERIMENTO 2 — RadDINO + Puente (Q-Former + language_projection FFT)"
echo "======================================================================"

if [ ! -f "$SCRIPT" ]; then
    echo "ERROR: no se encontro $SCRIPT"; exit 1
fi
mkdir -p "$RESULTS_DIR"

# Pre-check: existir el adaptador LoRA del LLM Fase 2 (Config_B1) ya entrenado.
# NOTA: el ViT de Fase 2 NO se usa (RadDINO reemplaza al codificador visual),
# por eso aquí SOLO se verifica el LoRA del LLM.
LLM_LORA="$WORKDIR/phase2_results-50pct/checkpoints/best_model_Config_B1/llm/adapter_config.json"
if [ ! -f "$LLM_LORA" ]; then
    echo "ERROR: falta el adaptador LoRA del LLM de Fase 2 (Config_B1):"
    echo "       LLM : $LLM_LORA  (FALTA)"
    echo "Verifica que la Fase 2 termino y produjo best_model_Config_B1/llm."
    exit 1
fi

echo "  Script    : $SCRIPT"
echo "  Visual    : RadDINO (microsoft/rad-dino) — reemplaza al EVA-CLIP"
echo "  Base LLM  : Config_B1 Fase 2 (LLM r=32) fusionado + congelado"
echo "  Estrategia: LoRA(RadDINO) + proyector 768->1408 + Full FT puente"
echo "  Inferencia: DESACTIVADA durante eval (solo Train/Val Loss + LRs en CSV)"
echo "  Checkpoint: solo params entrenables (~430 MB) en best_model_Exp2_RadDINO/"
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
echo '║    tail -f exp2_raddino_results-50pct/train_exp2_raddino.log       ║' && \
echo '║    cat  exp2_raddino_results-50pct/history_exp2_Exp2_RadDINO.csv   ║' && \
echo '║  Para desconectar (sin matar el proceso): Ctrl+b  d                ║' && \
echo '╚════════════════════════════════════════════════════════════════════╝' && \
echo \"[\$(date +%H:%M:%S)] === Iniciando Experimento 2: RadDINO + Puente Full FT ===\" | tee -a $LOG_FILE && \
PYTHONUNBUFFERED=1 \
TOKENIZERS_PARALLELISM=false \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
OMP_NUM_THREADS=16 \
TRANSFORMERS_NO_ADVISORY_WARNINGS=1 \
python -u $SCRIPT $EXTRA_ARGS 2>&1 | tee -a $LOG_FILE; \
EXIT_CODE=\${PIPESTATUS[0]}; \
echo \"\"; \
echo \"[\$(date +%H:%M:%S)] === EXPERIMENTO 2 FINALIZADO (exit \$EXIT_CODE) ===\" | tee -a $LOG_FILE; \
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
echo "  Panel superior: entrenamiento Exp. 2 en vivo  (⚠️  no escribas en él)"
echo "  Panel inferior: GPU (refresco 10s)  ← cursor por defecto cae aquí (safe)"
echo ""
echo "  Reconectarse : tmux attach -t $SESSION   (o bash run_experiment2_raddino_tmux.sh --reattach)"
echo "  Log en vivo  : tail -f $LOG_FILE                          ← OTRO terminal"
echo "  CSV losses   : cat $RESULTS_DIR/history_exp2_Exp2_RadDINO.csv  ← OTRO terminal"
echo "  Detener      : tmux kill-session -t $SESSION                (NO uses Ctrl+C dentro)"
echo "======================================================================"

if [[ "$NO_ATTACH" -eq 1 ]]; then
    echo "Modo --no-attach: el Experimento 2 corre en segundo plano."
else
    echo "Conectando en 2 segundos... (Ctrl+b d para desconectarte sin detener)"
    sleep 2
    tmux attach -t "$SESSION"
fi
