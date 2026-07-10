#!/usr/bin/env bash
# =============================================================================
# run_evaluate_final_tier1_tmux.sh
# -----------------------------------------------------------------------------
# Orquesta el pipeline COMPLETO de evaluación del Experimento FINAL **TIER 1**
# (prompt diseñado) en una sesión tmux, maximizando el uso de la L40S y dejando
# todo en CSV. Cada etapa corre en su PROPIO proceso (torch vs vLLM no comparten
# GPU a la vez) y es RESUMIBLE.
#
#   ETAPA 1 · Inferencia cruda (prompt diseñado) + 5 métricas granulares/global
#             (CheXpert 14 + BLEU-4 + ROUGE-L + RadGraph + Clinical BERTScore).
#             batch=32 (beam=5 → 160 secuencias, ~34/45 GB, 0 OOM) para saturar GPU.
#   ETAPA 2 · Formateo de los reportes crudos con el LLM externo (vLLM + Llama 3.1 8B).
#   ETAPA 3 · Re-evaluación de las 5 métricas + CheXpert 14 sobre el reporte formateado.
#   ETAPA 4 · Comparativos globales (crudo vs formateado) + CheXpert 14 patologías.
#
# Uso:
#   bash run_evaluate_final_tier1_tmux.sh              # lanza en tmux 'tier1_eval'
#   tmux attach -t tier1_eval                          # ver progreso
#   tail -f final_experiment_tier1_results-50pct/tier1_eval_pipeline.log
# =============================================================================
set -uo pipefail

cd "$(dirname "$0")"

SESSION="tier1_eval"
BATCH="${BATCH:-32}"
OUT_DIR="final_experiment_tier1_results-50pct"
EVAL_DIR="${OUT_DIR}/evaluation"
FMT_CSV="${EVAL_DIR}/predicciones_Exp_Final_Tier1_llmexterno_llama8b.csv"
LOG="${OUT_DIR}/tier1_eval_pipeline.log"

mkdir -p "${EVAL_DIR}"

# Comando compuesto que corre las 4 etapas en secuencia (para dentro de tmux).
read -r -d '' PIPELINE <<EOF || true
set -e
echo "==============================================================================="
echo " FINAL TIER 1 · EVALUACIÓN COMPLETA (prompt diseñado)  ·  batch=${BATCH}"
echo " inicio: \$(date)"
echo "==============================================================================="

echo
echo ">>> ETAPA 1/4 — Inferencia cruda + métricas granulares/global (batch=${BATCH})"
python evaluate_final_experiment_tier1.py --batch-size ${BATCH} --resume

echo
echo ">>> ETAPA 2/4 — Formateo LLM externo (vLLM + Llama 3.1 8B)"
python vllm_formatter_tier1.py --resume

echo
echo ">>> ETAPA 3/4 — Re-evaluación métricas + CheXpert sobre reporte formateado"
python evaluate_final_experiment_tier1.py --eval-formatted-csv "${FMT_CSV}"

echo
echo ">>> ETAPA 4/4 — Comparativos globales + CheXpert 14 patologías"
python actualizar_comparativos_final_tier1.py

echo
echo "==============================================================================="
echo " PIPELINE FINAL TIER 1 COMPLETADO  ·  fin: \$(date)"
echo " Salidas en: ${OUT_DIR}/"
echo "==============================================================================="
EOF

# (Re)crear la sesión tmux limpia.
tmux kill-session -t "${SESSION}" 2>/dev/null || true
tmux new-session -d -s "${SESSION}" -x 220 -y 50

# Ejecutar el pipeline dentro de tmux, con tee al log.
tmux send-keys -t "${SESSION}" \
  "cd $(pwd) && { ${PIPELINE}; } 2>&1 | tee ${LOG}" C-m

echo "==============================================================================="
echo " Lanzado en tmux '${SESSION}' (batch=${BATCH})."
echo "   Ver en vivo : tmux attach -t ${SESSION}"
echo "   Log         : tail -f ${LOG}"
echo "   Salir attach: Ctrl-b d"
echo "==============================================================================="
