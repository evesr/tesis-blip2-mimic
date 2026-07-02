#!/bin/bash
###############################################################################
# run_vllm_formatter_tmux.sh — Formateo local con vLLM + Llama 3.1 8B (FP16)
###############################################################################
#
# Lanza vllm_formatter.py en una sesión tmux con monitor GPU.
# Reemplaza la Etapa A del LLM externo (gpt-4o-mini) por inferencia local
# en GPU L40S, sin costos ni rate limits.
#
# Modelo: NousResearch/Meta-Llama-3.1-8B-Instruct (mirror ungated)
# Tareas seleccionables (--tasks):
#   disenado : phase2_results-50pct/evaluation/predicciones_Config_B1.csv
#                 → predicciones_Config_B1_llmexterno_llama8b.csv
#   nuevo    : phase2_results-50pct/exp1_cot/exp1_raw_inference_results.csv
#                 → exp1_final_formatted_llama8b.csv
#   both     : ambas (default)
#
# USO:
#   bash run_vllm_formatter_tmux.sh                       # ambas tareas
#   bash run_vllm_formatter_tmux.sh --tasks=disenado      # solo diseñado
#   bash run_vllm_formatter_tmux.sh --tasks=nuevo         # solo nuevo
#   bash run_vllm_formatter_tmux.sh --limit=20            # smoke test
#   bash run_vllm_formatter_tmux.sh --no-attach           # 2do plano
#   bash run_vllm_formatter_tmux.sh --reattach            # reconectar
###############################################################################

set -uo pipefail

WORKDIR="/workspace/tesis-blip2-mimic"
LOG_FILE="$WORKDIR/phase2_results-50pct/vllm_formatter.log"
SESSION="vllm_fmt"

if [[ "${1:-}" == "--reattach" ]]; then
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        tmux attach -t "$SESSION"
    else
        echo "ERROR: no existe la sesion '$SESSION'."
    fi
    exit 0
fi

# Defaults
TASKS="disenado nuevo"
LIMIT_ARG=""
RESUME_ARG="--resume"
NO_ATTACH=0
EXTRA_ARGS=()

for arg in "$@"; do
    case "$arg" in
        --no-attach|--detach) NO_ATTACH=1 ;;
        --tasks=*)            TASKS="${arg#--tasks=}" ; TASKS="${TASKS//,/ }" ;;
        --limit=*)            LIMIT_ARG="--limit ${arg#--limit=}" ;;
        --no-resume)          RESUME_ARG="" ;;
        --reattach)           ;;  # ya manejado arriba
        *)                    EXTRA_ARGS+=("$arg") ;;
    esac
done

mkdir -p "$(dirname "$LOG_FILE")"

echo "======================================================================"
echo "  vLLM FORMATTER · Llama 3.1 8B (FP16) en L40S"
echo "======================================================================"
echo "  Tareas    : $TASKS"
echo "  Resume    : ${RESUME_ARG:-(no)}"
echo "  Limit     : ${LIMIT_ARG:-(none)}"
echo "  Extra     : ${EXTRA_ARGS[*]:-(none)}"
echo "  Log       : $LOG_FILE"
echo "  Session   : $SESSION"
echo "======================================================================"

# Validaciones rápidas
if ! command -v tmux >/dev/null 2>&1; then
    echo "ERROR: tmux no instalado. Instala con: apt-get install -y tmux"
    exit 1
fi
if ! python3 -c "import vllm" 2>/dev/null; then
    echo "ERROR: vllm no instalado. Instala con: pip install 'vllm>=0.6.0,<0.7.0'"
    exit 1
fi

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
clear && \
echo '╔════════════════════════════════════════════════════════════════════╗' && \
echo '║     ⚠️   PANEL DE FORMATEO  -  NO TIPEAR / PEGAR AQUÍ  ⚠️         ║' && \
echo '║                                                                    ║' && \
echo '║  Cualquier Ctrl+C / texto en este panel ABORTA el formateo.        ║' && \
echo '║  Para monitorear usa OTRO terminal:                                ║' && \
echo '║    tail -f phase2_results-50pct/vllm_formatter.log                  ║' && \
echo '║  Para desconectar (sin matar): Ctrl+b  d                           ║' && \
echo '╚════════════════════════════════════════════════════════════════════╝' && \
echo \"[\$(date +%H:%M:%S)] === vLLM Formatter (Llama 3.1 8B FP16) ===\" | tee -a $LOG_FILE && \
echo \"  Tareas: $TASKS\" | tee -a $LOG_FILE && \
PYTHONUNBUFFERED=1 python -u vllm_formatter.py \
  --tasks $TASKS $RESUME_ARG $LIMIT_ARG ${EXTRA_ARGS[*]:-} 2>&1 | tee -a $LOG_FILE; \
EXIT_CODE=\${PIPESTATUS[0]}; \
echo \"\"; \
echo \"[\$(date +%H:%M:%S)] === FORMATEO FINALIZADO (exit \$EXIT_CODE) ===\" | tee -a $LOG_FILE; \
echo 'Ctrl+b d para desconectarte.'" Enter

tmux split-window -v -p 28 -t "$SESSION"
tmux send-keys -t "$SESSION" \
"watch -n 5 -t 'nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.free,temperature.gpu,power.draw --format=csv,noheader,nounits'" Enter

# Cursor por defecto en el panel inferior («safe pane»): si el usuario tipea
# o pega allí, solo afecta a `watch`, NUNCA al proceso vLLM. Para subir
# manualmente al panel del formateo: Ctrl+b + flecha-arriba.
tmux select-pane -t "$SESSION:0.1"

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
