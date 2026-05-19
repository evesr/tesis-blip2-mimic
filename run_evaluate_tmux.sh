#!/bin/bash
###############################################################################
# Lanzador Evaluación Final — BLIP-2 + LoRA  (sesión tmux)
###############################################################################
#
# USO:
#   bash run_evaluate_tmux.sh                        # lanza evaluación completa
#   bash run_evaluate_tmux.sh --reattach             # reconectarse a sesión existente
#   bash run_evaluate_tmux.sh --skip-inference       # sólo métricas (sin re-inferir)
#   bash run_evaluate_tmux.sh --num-samples 50       # prueba rápida
#   bash run_evaluate_tmux.sh --config Config_A      # sólo una config
#
# DENTRO DE TMUX:
#   Panel superior       → evaluate_final.py (stdout en vivo)
#   Panel inferior izq.  → tail -f del log (cada línea al momento)
#   Panel inferior der.  → nvidia-smi (GPU VRAM, refresco 10s)
#
# NAVEGACIÓN:
#   Ctrl+b + flecha   → cambiar panel
#   Ctrl+b + d        → desconectarse (evaluación sigue corriendo)
#   tmux attach -t evaluacion_final  → reconectarse
#
# OPCIONES AVANZADAS:
#   --chexpert-csv RUTA   ruta al CSV oficial (default: mimic-cxr-2.0.0-chexpert.csv.gz)
#   --output-dir  RUTA    directorio de salida (default: evaluation_results/)
#
# Autor: Evelyn Silva Rozas
# Fecha: Mayo 2026
###############################################################################

set -euo pipefail

WORKDIR="/workspace/tesis-blip2-mimic"
SCRIPT="$WORKDIR/evaluate_final.py"
OUTPUT_DIR="$WORKDIR/evaluation_results"
LOG_FILE="$OUTPUT_DIR/evaluate_final.log"
SESSION="evaluacion_final"
CHEXPERT_CSV="$WORKDIR/mimic-cxr-2.0.0-chexpert.csv.gz"

# ── Parsear argumentos ────────────────────────────────────────────────────────
EXTRA_ARGS=""
SKIP_INFERENCE=0
NUM_SAMPLES=""
CONFIG_FILTER=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --reattach)
            if tmux has-session -t "$SESSION" 2>/dev/null; then
                echo "Reconectando a sesión '$SESSION'..."
                tmux attach -t "$SESSION"
            else
                echo "ERROR: No existe la sesión '$SESSION'."
                echo "  Lánzala con: bash run_evaluate_tmux.sh"
            fi
            exit 0
            ;;
        --skip-inference)
            SKIP_INFERENCE=1
            EXTRA_ARGS="$EXTRA_ARGS --skip-inference"
            shift
            ;;
        --num-samples)
            NUM_SAMPLES="$2"
            EXTRA_ARGS="$EXTRA_ARGS --num-samples $2"
            shift 2
            ;;
        --config)
            CONFIG_FILTER="$2"
            EXTRA_ARGS="$EXTRA_ARGS --configs $2"
            shift 2
            ;;
        --chexpert-csv)
            CHEXPERT_CSV="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            LOG_FILE="$OUTPUT_DIR/evaluate_final.log"
            shift 2
            ;;
        *)
            echo "Argumento desconocido: $1"
            echo "Uso: bash run_evaluate_tmux.sh [--reattach|--skip-inference|--num-samples N|--config Config_A|--chexpert-csv RUTA|--output-dir RUTA]"
            exit 1
            ;;
    esac
done

# Añadir siempre la ruta al CSV de CheXpert y al output-dir
EXTRA_ARGS="$EXTRA_ARGS --chexpert-csv $CHEXPERT_CSV --output-dir $OUTPUT_DIR"

# ── Verificaciones previas ────────────────────────────────────────────────────
echo "======================================================================"
echo "  EVALUACIÓN FINAL — BLIP-2 + LoRA"
echo "======================================================================"

if [ ! -f "$SCRIPT" ]; then
    echo "ERROR: no se encontró $SCRIPT"
    exit 1
fi

# Verificar modelos si no se usa --skip-inference
if [ "$SKIP_INFERENCE" -eq 0 ]; then
    CONFIGS_TO_CHECK=("Config_A" "Config_B")
    if [ -n "$CONFIG_FILTER" ]; then
        CONFIGS_TO_CHECK=("$CONFIG_FILTER")
    fi
    for cfg in "${CONFIGS_TO_CHECK[@]}"; do
        MODEL_DIR="$WORKDIR/final_results/checkpoints/best_model_$cfg"
        if [ ! -d "$MODEL_DIR" ]; then
            echo "AVISO: No se encontró $MODEL_DIR"
            echo "       Asegúrate de haber completado el entrenamiento Gran Final."
            echo "       Si tienes los CSVs de predicciones, usa --skip-inference."
            exit 1
        fi
        echo "  Modelo $cfg:  OK ($MODEL_DIR)"
    done
fi

# Verificar CSV CheXpert
if [ -f "$CHEXPERT_CSV" ]; then
    echo "  CheXpert CSV: OK ($CHEXPERT_CSV)"
else
    echo "  AVISO: CheXpert CSV no encontrado en $CHEXPERT_CSV"
    echo "         La evaluación CheXpert GT se omitirá (sólo Predicted Vector)."
fi

# Verificar test split
if [ ! -f "$WORKDIR/test_split_10pct_limpio.csv" ]; then
    echo "ERROR: no se encontró test_split_10pct_limpio.csv"
    exit 1
fi
echo "  Test split:   OK"

mkdir -p "$OUTPUT_DIR"
echo ""
echo "  Output dir:   $OUTPUT_DIR"
echo "  Log:          $LOG_FILE"
[ -n "$NUM_SAMPLES" ] && echo "  Num samples:  $NUM_SAMPLES (modo prueba)"
[ "$SKIP_INFERENCE" -eq 1 ] && echo "  Modo:         --skip-inference (sólo métricas)"
echo ""

# ── Matar sesión previa si existe ────────────────────────────────────────────
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "AVISO: ya existe la sesión tmux '$SESSION'."
    printf "  [K] Matar y relanzar   [R] Reconectarse   [C] Cancelar -> "
    read -r choice
    case "$choice" in
        k|K)
            tmux kill-session -t "$SESSION"
            echo "Sesión anterior eliminada."
            ;;
        r|R)
            tmux attach -t "$SESSION"
            exit 0
            ;;
        *)
            echo "Cancelado."
            exit 0
            ;;
    esac
fi

# ── Crear sesión tmux ─────────────────────────────────────────────────────────
echo "Creando sesión tmux '$SESSION'..."

tmux new-session -d -s "$SESSION" -x 220 -y 50

# Panel superior (70%): evaluate_final.py con stdout en vivo + tee al log
BANNER_OK="echo && echo ====================================================== && echo '  EVALUACION FINALIZADA.' && echo \"  Resultados en: $OUTPUT_DIR\" && echo ======================================================"
CMD_EVAL="cd $WORKDIR && PYTHONUNBUFFERED=1 python -u $SCRIPT $EXTRA_ARGS 2>&1 | tee $LOG_FILE; $BANNER_OK"

tmux send-keys -t "$SESSION" "$CMD_EVAL" Enter

# Panel inferior izquierdo (30%): tail -f del log en tiempo real
tmux split-window -v -p 30 -t "$SESSION"
tmux send-keys -t "$SESSION" "tail -f $LOG_FILE" Enter

# Panel inferior derecho (50% del inferior): nvidia-smi
tmux split-window -h -p 50 -t "$SESSION"
tmux send-keys -t "$SESSION" "watch -n 10 -t nvidia-smi" Enter

# Foco en panel superior
tmux select-pane -t "$SESSION:0.0"

# ── Resumen y conexión ────────────────────────────────────────────────────────
echo ""
echo "======================================================================"
echo "  SESIÓN TMUX '$SESSION' LISTA"
echo "======================================================================"
echo ""
echo "  Panel superior    (70%): evaluate_final.py en vivo"
echo "  Panel inf. izq.   (15%): tail -f del log en tiempo real"
echo "  Panel inf. der.   (15%): nvidia-smi (refresco 10s)"
echo ""
echo "NAVEGACIÓN:"
echo "  Ctrl+b + flecha arriba/abajo  → cambiar panel"
echo "  Ctrl+b + d                    → desconectarse (sigue corriendo)"
echo "  tmux attach -t $SESSION       → reconectarse"
echo "  bash run_evaluate_tmux.sh --reattach"
echo ""
echo "MONITOREO (otra terminal):"
echo "  tail -f $LOG_FILE"
echo "  watch -n 5 ls -lh $OUTPUT_DIR/*.csv 2>/dev/null"
echo ""
echo "Conectando en 2 segundos... (Ctrl+b d para desconectarte)"
echo "======================================================================"
sleep 2

tmux attach -t "$SESSION"
