#!/usr/bin/env bash
# ============================================================================
# run_evaluate_phase1_10pct_tmux.sh
# ----------------------------------------------------------------------------
# Lanza la evaluacion de FASE 1 (dos mejores modelos) sobre el Test Set 10%
# COMPLETO dentro de una sesion tmux 'phase1_10pct', y al terminar la
# inferencia+metricas genera el CSV CheXpert por patologia.
#
# Uso:
#   bash run_evaluate_phase1_10pct_tmux.sh            # inferencia + metricas
#   bash run_evaluate_phase1_10pct_tmux.sh --skip-inference
#
# Monitorear:
#   tmux attach -t phase1_10pct       (Ctrl+B luego D para salir)
#   tail -f phase1_10pct_eval_results/run.log
# ============================================================================
set -euo pipefail

cd "$(dirname "$0")"

SESSION="phase1_10pct"
OUT_DIR="phase1_10pct_eval_results"
LOG="${OUT_DIR}/run.log"
EXTRA_ARGS="$*"

mkdir -p "${OUT_DIR}"

# Matar sesion previa si existe
tmux has-session -t "${SESSION}" 2>/dev/null && tmux kill-session -t "${SESSION}" || true

# 1) Asegurar el test completo (3796, imagen garantizada)
# 2) Inferencia + metricas (Config_A y Config_B)
# 3) CheXpert por patologia
CMD="set -e; \
echo '== [1/3] Construyendo test 10% completo (si falta) =='; \
[ -f test_split_10pct_completo.csv ] || python3 build_test_completo_10pct.py; \
echo '== [2/3] Inferencia + metricas FASE 1 =='; \
python3 evaluate_phase1_final_10pct.py ${EXTRA_ARGS}; \
echo '== [3/3] CheXpert por patologia =='; \
python3 chexpert_por_patologia_phase1_10pct.py; \
echo '== LISTO =='"

tmux new-session -d -s "${SESSION}" "stdbuf -oL -eL bash -lc \"${CMD}\" 2>&1 | tee ${LOG}"

echo "[OK] Sesion tmux '${SESSION}' lanzada."
echo "     Monitorear:  tmux attach -t ${SESSION}"
echo "     Log:         tail -f ${LOG}"
