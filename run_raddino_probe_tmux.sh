#!/bin/bash
###############################################################################
# run_raddino_probe_tmux.sh
###############################################################################
# Lanza la SONDA DIAGNÓSTICA AISLADA de RadDINO (eval_raddino_probe.py) en una
# única sesión tmux, ejecutando SECUENCIALMENTE:
#
#   ETAPA 1/2 → LINEAR PROBING   (RadDINO 100% congelado + cabezal lineal)
#               python eval_raddino_probe.py --eval-test
#               · cachea las features [CLS] una sola vez → cabezal casi instantáneo
#               · 40 épocas (def.), early-stopping patience=8 sobre val_macro_f1
#
#   ETAPA 2/2 → LoRA PROBING     (adaptadores query/value + cabezal, end-to-end)
#               python eval_raddino_probe.py --apply-lora --eval-test
#               · 8 épocas (def.), patience=4, autocast bf16
#
# Se ejecutan en serie (NO en paralelo) para no competir por la VRAM y dejar un
# solo log/panel de monitoreo. La etapa LoRA corre AUNQUE la linear falle
# (son benchmarks independientes); al final se imprime el resumen de exit codes.
#
# Métricas (14 patologías CheXpert) por etapa, en su propio sub-directorio:
#   exp2_raddino_results-50pct/raddino_probe/raddino_linear_probe/
#   exp2_raddino_results-50pct/raddino_probe/raddino_lora_probe/
#     ├─ history.csv               curva de Loss (train/val) + macro P/R/F1/AUROC
#     ├─ metrics_per_pathology.csv P/R/F1/AUROC + soporte por patología (val/test)
#     ├─ metrics_summary.json      macro/micro + config + mejor época + timing
#     ├─ loss_curve.png            curva de Loss + macro-F1
#     └─ best_state.pt             cabezal (+ LoRA si aplica)
#
# USO:
#   bash run_raddino_probe_tmux.sh                      # lanza y se conecta
#   bash run_raddino_probe_tmux.sh --no-attach          # lanza en segundo plano
#   bash run_raddino_probe_tmux.sh --reattach           # reconectarse
#   bash run_raddino_probe_tmux.sh --image-size 518     # extra-args → AMBAS etapas
#   bash run_raddino_probe_tmux.sh --num-train 2000 --num-val 1000   # smoke rápido
#
#   (cualquier flag no reconocido se pasa TAL CUAL a las dos corridas de Python)
#
# Ctrl+b + d = desconectarte (sigue corriendo).  Ctrl+C en panel superior = ABORTA.
###############################################################################

set -euo pipefail

WORKDIR="/workspace/tesis-blip2-mimic"
SCRIPT="$WORKDIR/eval_raddino_probe.py"
PROBE_DIR="$WORKDIR/exp2_raddino_results-50pct/raddino_probe"
LOG_FILE="$PROBE_DIR/raddino_probe.log"
SESSION="raddino_probe"

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

# ── Parsear flags propios + args extra para Python (se aplican a AMBAS etapas) ─
NO_ATTACH=0
PY_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --no-attach|--detach) NO_ATTACH=1 ;;
        *) PY_ARGS+=("$arg") ;;
    esac
done
EXTRA_ARGS="${PY_ARGS[*]:-}"

# ── Optimización GPU: alimentar la L40S con MUCHOS workers ────────────────────
# El cuello de botella NO es la GPU (es rapidísima) sino el preprocesado en CPU
# (CLAHE + decode de imagen). Con solo 16 workers la GPU se quedaba OCIOSA
# esperando datos (CPU full / GPU 0%). Con muchos cores disponibles usamos
# muchos más workers para SATURAR la GPU, y un batch mayor para mejor uso de VRAM.
# (El usuario puede sobreescribir cualquiera de estos flags; argparse toma el último.)
CORES="$(nproc)"
NUM_WORKERS=$(( CORES > 64 ? 64 : (CORES > 2 ? CORES - 1 : 1) ))
OPT_ARGS="--num-workers $NUM_WORKERS --batch-size 256"

# ── Verificaciones previas ────────────────────────────────────────────────────
echo "======================================================================"
echo "  SONDA DIAGNÓSTICA AISLADA · RadDINO · LINEAR → LoRA (secuencial)"
echo "======================================================================"

if [ ! -f "$SCRIPT" ]; then
    echo "ERROR: no se encontro $SCRIPT"; exit 1
fi
mkdir -p "$PROBE_DIR"

# Pre-check: deben existir los splits 50%, el GT CheXpert y el dir de imagenes.
TRAIN_CSV="$WORKDIR/train_split_50pct.csv"
VAL_CSV="$WORKDIR/val_split_50pct.csv"
TEST_CSV="$WORKDIR/test_split_50pct.csv"
CHEX_CSV="$WORKDIR/mimic-cxr-2.0.0-chexpert.csv"
IMAGES_DIR="$WORKDIR/imagenes_50_porciento"

missing=0
[ -f "$TRAIN_CSV" ] || { echo "  ✗ FALTA train split: $TRAIN_CSV"; missing=1; }
[ -f "$VAL_CSV" ]   || { echo "  ✗ FALTA val split:   $VAL_CSV"; missing=1; }
[ -f "$TEST_CSV" ]  || { echo "  ✗ FALTA test split:  $TEST_CSV"; missing=1; }
[ -f "$CHEX_CSV" ]  || { echo "  ✗ FALTA GT CheXpert: $CHEX_CSV"; missing=1; }
[ -d "$IMAGES_DIR" ] || { echo "  ✗ FALTA dir imagenes: $IMAGES_DIR"; missing=1; }
if [ "$missing" = "1" ]; then
    echo "Pre-requisitos no cumplidos."; exit 1
fi

echo "  Script    : $SCRIPT"
echo "  Train/Val/Test: $(wc -l < "$TRAIN_CSV")/$(wc -l < "$VAL_CSV")/$(wc -l < "$TEST_CSV") filas"
echo "  GT CheXpert: OK"
echo "  Imagenes  : OK ($IMAGES_DIR)"
echo "  Output    : $PROBE_DIR/raddino_{linear,lora}_probe/"
echo "  Log comun : $LOG_FILE"
echo "  GPU-feed  : $NUM_WORKERS workers (de $CORES cores) | batch 256  ← satura la L40S"
echo "  Args extra: ${EXTRA_ARGS:-ninguno}  (se aplican a las DOS etapas)"
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

# ── Panel superior: ejecuta el runner interno (ETAPA 1 linear → ETAPA 2 LoRA) ──
# Toda la lógica vive en run_raddino_probe_inner.sh para evitar el infierno de
# comillas/Unicode al meter un comando gigante en `send-keys` (eso rompía el
# quoting del panel y dejaba el shell colgado en un prompt de continuación '>').
INNER="$WORKDIR/run_raddino_probe_inner.sh"
tmux send-keys -t "$SESSION" "bash $INNER $OPT_ARGS $EXTRA_ARGS" Enter

# ── Panel inferior: nvidia-smi (cursor por defecto cae aqui = safe) ───────────
tmux split-window -v -p 30 -t "$SESSION"
tmux send-keys -t "$SESSION" \
"watch -n 10 -t 'nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.free,temperature.gpu --format=csv,noheader,nounits'" Enter

tmux select-pane -t "$SESSION:0.1"

# ── Info al usuario ───────────────────────────────────────────────────────────
echo ""
echo "======================================================================"
echo "  SESION TMUX '$SESSION' LISTA"
echo "======================================================================"
echo "  Panel superior: ETAPA 1 linear -> ETAPA 2 LoRA  (⚠️  no escribas aqui)"
echo "  Panel inferior: GPU (refresco 10s)  ← cursor por defecto cae aqui (safe)"
echo ""
echo "  Reconectarse : tmux attach -t $SESSION   (o bash run_raddino_probe_tmux.sh --reattach)"
echo "  Log en vivo  : tail -f $LOG_FILE                      ← OTRO terminal"
echo "  Metricas     : cat $PROBE_DIR/raddino_linear_probe/metrics_summary.json"
echo "               : cat $PROBE_DIR/raddino_lora_probe/metrics_summary.json"
echo "  Detener      : tmux kill-session -t $SESSION            (NO uses Ctrl+C)"
echo "======================================================================"

if [[ "$NO_ATTACH" -eq 1 ]]; then
    echo "Modo --no-attach: la sonda corre en segundo plano (linear -> lora)."
else
    echo "Conectando en 2 segundos... (Ctrl+b d para desconectarte sin detener)"
    sleep 2
    tmux attach -t "$SESSION"
fi
