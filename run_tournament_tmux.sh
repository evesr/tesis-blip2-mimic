#!/bin/bash
###############################################################################
# run_tournament_tmux.sh — Torneo Fase 2.2 Config_B1  (L40S, 48 GB VRAM)
###############################################################################
#
# USO:
#   bash run_tournament_tmux.sh                    # lanza entrenamiento
#   bash run_tournament_tmux.sh --skip-train       # solo inferencia + eval
#   bash run_tournament_tmux.sh --reattach         # reconectar a sesion viva
#
# DENTRO DE TMUX:
#   Panel superior  → entrenamiento en tiempo real (stdout + stderr)
#   Panel inferior  → nvidia-smi cada 10 s
#
# NAVEGACION:
#   Ctrl+b + flecha   → cambiar panel
#   Ctrl+b + d        → desconectarse (el entrenamiento sigue corriendo)
#   tmux attach -t tournament_b1 → reconectarse desde otra terminal
#
# LOG:
#   tail -f /workspace/tesis-blip2-mimic/phase2_results-50pct/tournament_Config_B1.log
#
###############################################################################

set -euo pipefail

WORKDIR="/workspace/tesis-blip2-mimic"
SCRIPT="$WORKDIR/run_tournament.py"
RESULTS_DIR="$WORKDIR/phase2_results-50pct"
LOG_FILE="$RESULTS_DIR/tournament_Config_B1.log"
SESSION="tournament_b1"

# ── Modo --reattach ───────────────────────────────────────────────────────────
if [[ "${1:-}" == "--reattach" ]]; then
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "Reconectando a sesion '$SESSION'..."
        tmux attach -t "$SESSION"
    else
        echo "ERROR: No existe la sesion '$SESSION'. Lanzala con:"
        echo "  bash run_tournament_tmux.sh"
    fi
    exit 0
fi

# ── Parsear flags propios del wrapper y separar los args para Python ─────────
# --no-attach / --detach : crea la sesion tmux y NO hace 'attach' al final
#   (util para lanzar el entrenamiento en segundo plano desde un script o
#    una terminal no interactiva).  El resto de args se reenvian a Python.
NO_ATTACH=0
PY_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --no-attach|--detach)
            NO_ATTACH=1
            ;;
        *)
            PY_ARGS+=("$arg")
            ;;
    esac
done
EXTRA_ARGS="${PY_ARGS[*]:-}"   # solo los args destinados al script Python

# ── Verificaciones previas ────────────────────────────────────────────────────
echo "======================================================================"
echo "  TORNEO FASE 2.2 — Config_B1  (LLM r=32 + ViT r=8)"
echo "======================================================================"

if [ ! -f "$SCRIPT" ]; then
    echo "ERROR: no se encontro $SCRIPT"
    exit 1
fi

for split in train_split_50pct.csv val_split_50pct.csv test_split_50pct.csv; do
    if [ ! -f "$WORKDIR/$split" ]; then
        echo "ERROR: no se encontro $split en $WORKDIR"
        exit 1
    fi
done

mkdir -p "$RESULTS_DIR"

echo "  Script    : $SCRIPT"
echo "  Log       : $LOG_FILE"
echo "  Args extra: ${EXTRA_ARGS:-ninguno}"
echo ""

# ── Matar sesion previa si existe ─────────────────────────────────────────────
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "AVISO: ya existe la sesion tmux '$SESSION'."
    echo ""
    # En modo --no-attach (segundo plano) no podemos leer stdin: reconectar info.
    if [[ "$NO_ATTACH" -eq 1 ]]; then
        echo "Modo --no-attach: la sesion ya esta corriendo, no se relanza."
        echo "  Reconectate con:  tmux attach -t $SESSION"
        exit 0
    fi
    printf "  [K] Matar y relanzar   [R] Reconectarse   [C] Cancelar -> "
    read -r choice
    case "$choice" in
        k|K)
            tmux kill-session -t "$SESSION"
            echo "Sesion anterior eliminada."
            ;;
        r|R)
            echo "Reconectando..."
            tmux attach -t "$SESSION"
            exit 0
            ;;
        *)
            echo "Cancelado."
            echo "  Para reconectarte: tmux attach -t $SESSION"
            exit 0
            ;;
    esac
fi

# ── Crear sesion tmux ─────────────────────────────────────────────────────────
echo "Creando sesion tmux '$SESSION'..."
tmux new-session -d -s "$SESSION" -x 220 -y 55

# ── Panel superior (70 %): entrenamiento con tee al log ───────────────────────
# PYTHONUNBUFFERED=1 + python -u garantizan que cada print/log aparece
# inmediatamente en tmux sin esperar el buffer de 4 KB.
# 2>&1 captura tanto stdout como stderr.
# tee duplica la salida al fichero de log sin interrumpir el stream en pantalla.
#
# AUTO-RESUME: el contenedor cloud puede matar el proceso (preemption /
# timeout / reclamo de GPU) sin dejar traceback.  El bucle while reinicia
# automáticamente run_tournament.py, que detecta el último checkpoint-* y
# reanuda con resume_from_checkpoint.  Si el script termina con éxito
# (exit 0) salimos del bucle; si muere por señal (exit != 0) reintenta
# tras 15 s, hasta MAX_RETRIES veces.
tmux send-keys -t "$SESSION" \
"cd $WORKDIR && \
MAX_RETRIES=20; RETRY=0; \
while [ \$RETRY -lt \$MAX_RETRIES ]; do \
  echo \"[\$(date +%H:%M:%S)] === Lanzando entrenamiento (intento \$((RETRY+1))/\$MAX_RETRIES) ===\" | tee -a $LOG_FILE; \
  PYTHONUNBUFFERED=1 python -u $SCRIPT $EXTRA_ARGS 2>&1 | tee -a $LOG_FILE; \
  EXIT_CODE=\${PIPESTATUS[0]}; \
  if [ \$EXIT_CODE -eq 0 ]; then \
    echo \"[\$(date +%H:%M:%S)] === Entrenamiento COMPLETADO (exit 0) ===\" | tee -a $LOG_FILE; \
    break; \
  fi; \
  RETRY=\$((RETRY+1)); \
  echo \"[\$(date +%H:%M:%S)] === Proceso muerto (exit \$EXIT_CODE). Reanudando desde checkpoint en 15s... ===\" | tee -a $LOG_FILE; \
  sleep 15; \
done
echo ''
echo '=================================================================='
echo '  TORNEO FINALIZADO. Ctrl+b d para desconectarte o Ctrl+b & para cerrar.'
echo '=================================================================='
" Enter

# ── Panel inferior (30 %): nvidia-smi para monitorear GPU ─────────────────────
tmux split-window -v -p 30 -t "$SESSION"
tmux send-keys -t "$SESSION" \
"watch -n 10 -t 'nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.free,temperature.gpu --format=csv,noheader,nounits | awk -v d=\"\$(date +\"%H:%M:%S\")\" \"{print d,\\\" | GPU:\\\",\$1\\\"%  Mem:\\\",\$2\\\"/47697 MB  Free:\\\",\$3\\\" MB  Temp:\\\",\$4\\\"C\\\"}\"'" \
Enter

# Foco de vuelta al panel de entrenamiento
tmux select-pane -t "$SESSION:0.0"

# ── Info al usuario ───────────────────────────────────────────────────────────
echo ""
echo "======================================================================"
echo "  SESION TMUX '$SESSION' LISTA"
echo "======================================================================"
echo ""
echo "  Panel superior (70%): entrenamiento en tiempo real"
echo "  Panel inferior (30%): GPU utilization (refresco 10s)"
echo ""
echo "NAVEGACION:"
echo "  Ctrl+b + flecha arriba/abajo  → cambiar panel"
echo "  Ctrl+b + d                    → desconectarse (sigue corriendo)"
echo "  tmux attach -t $SESSION       → reconectarse"
echo ""
echo "LOG EN VIVO (otra terminal):"
echo "  tail -f $LOG_FILE"
echo ""
echo "DETENER:"
echo "  tmux kill-session -t $SESSION"
echo ""
if [[ "$NO_ATTACH" -eq 1 ]]; then
    echo "Modo --no-attach: el entrenamiento corre en segundo plano."
    echo "  Reconectate con:  tmux attach -t $SESSION"
    echo "======================================================================"
else
    echo "Conectando en 2 segundos... (Ctrl+b d para desconectarte sin detener)"
    echo "======================================================================"
    sleep 2
    tmux attach -t "$SESSION"
fi
