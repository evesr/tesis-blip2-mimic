#!/bin/bash
###############################################################################
# run_raddino_probe_inner.sh
###############################################################################
# Runner INTERNO de la sonda RadDINO (etapa LINEAR -> etapa LoRA, secuencial).
# Lo invoca run_raddino_probe_tmux.sh DENTRO de la sesion tmux:
#       bash run_raddino_probe_inner.sh  <args-para-python...>
#
# Todos los argumentos que reciba se pasan TAL CUAL a las DOS etapas de python.
# Se separa del lanzador para evitar el infierno de comillas/escapes/Unicode al
# meter un comando gigante en `tmux send-keys` (eso rompia el quoting del panel).
#
# NO usa `set -e`: queremos que la etapa LoRA corra AUNQUE la linear falle
# (son benchmarks independientes).
###############################################################################

set -uo pipefail

WORKDIR="/workspace/tesis-blip2-mimic"
SCRIPT="$WORKDIR/eval_raddino_probe.py"
PROBE_DIR="$WORKDIR/exp2_raddino_results-50pct/raddino_probe"
LOG_FILE="$PROBE_DIR/raddino_probe.log"

EXTRA=("$@")              # args para AMBAS etapas (workers, batch, overrides...)

mkdir -p "$PROBE_DIR"
cd "$WORKDIR"

# ── Entorno de rendimiento ────────────────────────────────────────────────────
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=16
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

ts() { date +%H:%M:%S; }

# ── Aviso visual en el panel (no tocar este panel) ────────────────────────────
clear
echo '╔════════════════════════════════════════════════════════════════════╗'
echo '║   PANEL DE LA SONDA RadDINO  -  NO TIPEAR / PEGAR AQUI              ║'
echo '║   Cualquier Ctrl+C / texto en este panel ABORTA la corrida.        ║'
echo '║   Monitorea desde OTRO terminal:                                   ║'
echo "║     tail -f $LOG_FILE"
echo '║   Para desconectar (sin matar): Ctrl+b  d                          ║'
echo '╚════════════════════════════════════════════════════════════════════╝'

# ── ETAPA 1/2: LINEAR PROBING ─────────────────────────────────────────────────
{
  echo "[$(ts)] === INICIO sonda RadDINO (LINEAR -> LoRA) ==="
  echo ""
  echo ">>>>>>>>>>>>>>>>>>>>  ETAPA 1/2: LINEAR PROBING  <<<<<<<<<<<<<<<<<<<<"
  echo "[$(ts)] linear: RadDINO congelado + cabezal lineal | args: ${EXTRA[*]:-(defaults)}"
} | tee -a "$LOG_FILE"

python -u "$SCRIPT" --eval-test "${EXTRA[@]}" 2>&1 | tee -a "$LOG_FILE"
EXIT_LINEAR=${PIPESTATUS[0]}

{
  echo ""
  echo "[$(ts)] === ETAPA 1 (LINEAR) finalizada (exit $EXIT_LINEAR) ==="
} | tee -a "$LOG_FILE"

# ── ETAPA 2/2: LoRA PROBING ───────────────────────────────────────────────────
{
  echo ""
  echo ">>>>>>>>>>>>>>>>>>>>>  ETAPA 2/2: LoRA PROBING  <<<<<<<<<<<<<<<<<<<<<"
  echo "[$(ts)] lora: adaptadores query/value + cabezal (end-to-end) | args: ${EXTRA[*]:-(defaults)}"
} | tee -a "$LOG_FILE"

python -u "$SCRIPT" --apply-lora --eval-test "${EXTRA[@]}" 2>&1 | tee -a "$LOG_FILE"
EXIT_LORA=${PIPESTATUS[0]}

# ── Resumen ───────────────────────────────────────────────────────────────────
{
  echo ""
  echo "[$(ts)] === ETAPA 2 (LoRA) finalizada (exit $EXIT_LORA) ==="
  echo ""
  echo "==================== RESUMEN SONDA RadDINO ===================="
  echo "  LINEAR PROBING : exit $EXIT_LINEAR  -> raddino_linear_probe/metrics_summary.json"
  echo "  LoRA   PROBING : exit $EXIT_LORA  -> raddino_lora_probe/metrics_summary.json"
  echo "=============================================================="
} | tee -a "$LOG_FILE"

echo ""
echo "Terminado. Ctrl+b d para desconectarte, o Ctrl+b & para cerrar la sesion."
