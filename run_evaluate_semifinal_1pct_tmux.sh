#!/usr/bin/env bash
# ============================================================================
# run_evaluate_semifinal_1pct_tmux.sh
# ----------------------------------------------------------------------------
# Lanza la evaluacion SEMIFINAL (Config_4/5/6, best checkpoint) sobre el Test
# Set 1% COMPLETO dentro de una sesion tmux 'semifinal_1pct'. Solo BLEU-4 y
# ROUGE-L. Espera a que la GPU tenga memoria libre suficiente antes de iniciar
# la inferencia (por si hay otro job corriendo).
#
# Uso:
#   bash run_evaluate_semifinal_1pct_tmux.sh
#   bash run_evaluate_semifinal_1pct_tmux.sh --skip-inference
#
# Monitorear:
#   tmux attach -t semifinal_1pct       (Ctrl+B luego D para salir)
#   tail -f phase1_semifinal_1pct_eval_results/run.log
# ============================================================================
set -euo pipefail

cd "$(dirname "$0")"

SESSION="semifinal_1pct"
OUT_DIR="phase1_semifinal_1pct_eval_results"
LOG="${OUT_DIR}/run.log"
INNER="${OUT_DIR}/_run_inner.sh"
MIN_FREE_MIB=10000        # memoria GPU libre minima para arrancar inferencia
EXTRA_ARGS="$*"

mkdir -p "${OUT_DIR}"

tmux has-session -t "${SESSION}" 2>/dev/null && tmux kill-session -t "${SESSION}" || true

# Script interno (evita problemas de escaping dentro de tmux new-session)
cat > "${INNER}" <<INNEREOF
#!/usr/bin/env bash
set -e
cd "$(pwd)"

echo "== [1/3] Construyendo test 1% completo (si falta) =="
[ -f test_split_completo_1pct.csv ] || python3 build_test_completo_1pct.py

echo "== [2/3] Esperando GPU libre (>= ${MIN_FREE_MIB} MiB) =="
while true; do
  FREE=\$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
  echo "    GPU libre: \${FREE} MiB"
  if [ "\${FREE}" -ge ${MIN_FREE_MIB} ]; then
    break
  fi
  sleep 30
done

echo "== [3/3] Inferencia + BLEU/ROUGE (Config_4/5/6) =="
python3 evaluate_semifinal_1pct.py ${EXTRA_ARGS}

echo "== LISTO =="
INNEREOF

chmod +x "${INNER}"

tmux new-session -d -s "${SESSION}" "stdbuf -oL -eL bash '${INNER}' 2>&1 | tee '${LOG}'"

echo "[OK] Sesion tmux '${SESSION}' lanzada."
echo "     Monitorear:  tmux attach -t ${SESSION}"
echo "     Log:         tail -f ${LOG}"
