#!/bin/bash
###############################################################################
# run_exp1_tmux.sh — Experimento 1 (CoT) end-to-end en una sesión tmux
###############################################################################
#
# Ejecuta, EN SECUENCIA y sobre GPU, las 3 etapas del Experimento 1:
#   Etapa 1  exp1_cot_inference.py      → inferencia Chain-of-Thought + CheXpert
#   Etapa 2  llm_formatter_pipeline.py  → formateo con LLM externo (gpt-4o-mini)
#   Etapa 3  exp1_metrics_comparison.py → BLEU/ROUGE/RadGraph/ClinicalBERT comparativo
#
# La Etapa 2 usa OpenAI si OPENAI_API_KEY está definida; si no, cae a 'simulate'
# (limpieza determinista por RegEx) para que el pipeline corra sin clave.
#
# USO:
#   export OPENAI_API_KEY="sk-..."                # opcional (para gpt-4o-mini real)
#   bash run_exp1_tmux.sh                          # corrida completa (19,137 test)
#   bash run_exp1_tmux.sh --no-attach              # en segundo plano
#   bash run_exp1_tmux.sh --num-samples 500        # subconjunto (se pasa a Etapa 1)
#   bash run_exp1_tmux.sh --reattach               # reconectar a sesión viva
#
# DENTRO DE TMUX:
#   Panel superior → etapas en vivo (tee al log)
#   Panel inferior → nvidia-smi cada 10 s
#   Ctrl+b + d     → desconectarse (sigue corriendo)
###############################################################################

set -uo pipefail

WORKDIR="/workspace/tesis-blip2-mimic"
RESULTS_DIR="$WORKDIR/phase2_results-50pct/exp1_cot"
LOG_FILE="$RESULTS_DIR/exp1_pipeline.log"
SESSION="exp1_cot"

# ── Modo --reattach ───────────────────────────────────────────────────────────
if [[ "${1:-}" == "--reattach" ]]; then
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        tmux attach -t "$SESSION"
    else
        echo "ERROR: no existe la sesion '$SESSION'. Lanzala con: bash run_exp1_tmux.sh"
    fi
    exit 0
fi

# ── Separar flags propios de los args para Etapa 1 ────────────────────────────
NO_ATTACH=0
INF_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --no-attach|--detach) NO_ATTACH=1 ;;
        *) INF_ARGS+=("$arg") ;;
    esac
done
INF_ARGS_STR="${INF_ARGS[*]:-}"

echo "======================================================================"
echo "  EXPERIMENTO 1 (Chain-of-Thought) — pipeline end-to-end"
echo "======================================================================"
mkdir -p "$RESULTS_DIR"

# Proveedor de la Etapa 2 según disponibilidad de la clave
if [[ -n "${OPENAI_API_KEY:-}" ]]; then
    FMT_PROVIDER="openai"
    echo "  Etapa 2: OpenAI gpt-4o-mini (OPENAI_API_KEY detectada)"
else
    FMT_PROVIDER="simulate"
    echo "  Etapa 2: simulate (sin OPENAI_API_KEY → limpieza determinista)"
fi
echo "  Args Etapa 1: ${INF_ARGS_STR:-ninguno}"
echo "  Log         : $LOG_FILE"
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
        *)   echo "Cancelado."; exit 0 ;;
    esac
fi

# ── Crear sesion tmux ─────────────────────────────────────────────────────────
tmux new-session -d -s "$SESSION" -x 220 -y 55

# Panel superior: las 3 etapas en secuencia (cada una aborta si la anterior falla)
tmux send-keys -t "$SESSION" \
"cd $WORKDIR && \
echo \"[\$(date +%H:%M:%S)] === EXP1 · Etapa 1: Inferencia CoT + CheXpert ===\" | tee -a $LOG_FILE && \
PYTHONUNBUFFERED=1 python -u exp1_cot_inference.py $INF_ARGS_STR 2>&1 | tee -a $LOG_FILE && \
echo \"[\$(date +%H:%M:%S)] === EXP1 · Etapa 2: Formateo LLM ($FMT_PROVIDER) ===\" | tee -a $LOG_FILE && \
PYTHONUNBUFFERED=1 python -u llm_formatter_pipeline.py --provider $FMT_PROVIDER 2>&1 | tee -a $LOG_FILE && \
echo \"[\$(date +%H:%M:%S)] === EXP1 · Etapa 3: Metricas comparativas ===\" | tee -a $LOG_FILE && \
PYTHONUNBUFFERED=1 python -u exp1_metrics_comparison.py 2>&1 | tee -a $LOG_FILE; \
EXIT_CODE=\$?; \
echo \"\"; \
echo \"[\$(date +%H:%M:%S)] === EXP1 FINALIZADO (exit \$EXIT_CODE) ===\" | tee -a $LOG_FILE; \
echo 'Ctrl+b d para desconectarte.'" Enter

# Panel inferior: monitor GPU
tmux split-window -v -p 28 -t "$SESSION"
tmux send-keys -t "$SESSION" \
"watch -n 10 -t 'nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.free,temperature.gpu --format=csv,noheader,nounits'" Enter

tmux select-pane -t "$SESSION:0.0"

echo "======================================================================"
echo "  SESION TMUX '$SESSION' LISTA"
echo "======================================================================"
echo "  Reconectarse : tmux attach -t $SESSION   (o bash run_exp1_tmux.sh --reattach)"
echo "  Log en vivo  : tail -f $LOG_FILE"
echo "  Detener      : tmux kill-session -t $SESSION"
echo "======================================================================"

if [[ "$NO_ATTACH" -eq 1 ]]; then
    echo "Modo --no-attach: el Experimento 1 corre en segundo plano."
else
    echo "Conectando en 2s... (Ctrl+b d para desconectarte sin detener)"
    sleep 2
    tmux attach -t "$SESSION"
fi
