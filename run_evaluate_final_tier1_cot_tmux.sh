#!/usr/bin/env bash
# =============================================================================
# run_evaluate_final_tier1_cot_tmux.sh
# -----------------------------------------------------------------------------
# Orquesta el pipeline COMPLETO de evaluación del Experimento FINAL **TIER 1**
# con el PROMPT NUEVO / CoT (RAMA 3 + RAMA 4) en una sesión tmux, maximizando el
# uso de la L40S y dejando todo en CSV — registrado de la MISMA forma que el
# pipeline del prompt diseñado. Cada etapa es RESUMIBLE.
#
#   ETAPA 1 · Inferencia CoT cruda (prompt nuevo) + 5 métricas granulares/global
#             (CheXpert 14 + BLEU-4 + ROUGE-L + RadGraph + Clinical BERTScore).
#             batch=32 (beam=3 → 96 secuencias, KV ≤ diseñado@32, 0 OOM).
#   ETAPA 2 · Formateo de los reportes CoT crudos con el LLM externo (vLLM + Llama 3.1 8B).
#   ETAPA 3 · Re-evaluación de las 5 métricas + CheXpert 14 sobre el reporte formateado.
#   ETAPA 4 · Comparativos globales (crudo vs formateado) + CheXpert 14 patologías.
#
# Uso:
#   bash run_evaluate_final_tier1_cot_tmux.sh          # lanza en tmux 'tier1_cot'
#   tmux attach -t tier1_cot                           # ver progreso
#   tail -f final_experiment_tier1_results-50pct/tier1_cot_pipeline.log
# =============================================================================
set -uo pipefail

cd "$(dirname "$0")"

SESSION="tier1_cot"
BATCH="${BATCH:-32}"
OUT_DIR="final_experiment_tier1_results-50pct"
EXP1_DIR="${OUT_DIR}/exp1_cot"
FMT_CSV="${EXP1_DIR}/exp1_final_formatted_final_experiment_tier1_llama8b.csv"
FMT_GRAN="${EXP1_DIR}/resultados_granulares_Exp_Final_Tier1_CoT_formateado.csv"
FMT_GLOB="${EXP1_DIR}/metricas_globales_Exp_Final_Tier1_CoT_formateado.csv"
FMT_PREDS="${EXP1_DIR}/predicciones_Exp_Final_Tier1_CoT_formateado_as_preds.csv"
LOG="${OUT_DIR}/tier1_cot_pipeline.log"

mkdir -p "${EXP1_DIR}"

# Comando compuesto que corre las 4 etapas en secuencia (para dentro de tmux).
read -r -d '' PIPELINE <<EOF || true
set -e
echo "==============================================================================="
echo " FINAL TIER 1 · EVALUACIÓN COMPLETA (prompt NUEVO / CoT)  ·  batch=${BATCH}"
echo " inicio: \$(date)"
echo "==============================================================================="

echo
echo ">>> ETAPA 1/4 — Inferencia CoT cruda + métricas granulares/global (batch=${BATCH})"
python exp1_cot_inference_final_experiment_tier1.py --batch-size ${BATCH} --resume

echo
echo ">>> ETAPA 2/4 — Formateo LLM externo (vLLM + Llama 3.1 8B) · tarea nuevo/CoT"
python vllm_formatter_tier1.py --tasks nuevo --resume

echo
echo ">>> ETAPA 3/4 — Re-evaluación métricas + CheXpert sobre reporte formateado (CoT)"
python evaluate_final_experiment_tier1.py \
    --eval-formatted-csv "${FMT_CSV}" \
    --out-granular "${FMT_GRAN}" \
    --out-global "${FMT_GLOB}" \
    --derived-preds "${FMT_PREDS}"

echo
echo ">>> ETAPA 4/4 — Comparativos globales + CheXpert 14 patologías (CoT)"
python actualizar_comparativos_final_tier1_cot.py

echo
echo "==============================================================================="
echo " PIPELINE FINAL TIER 1 · CoT COMPLETADO  ·  fin: \$(date)"
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
