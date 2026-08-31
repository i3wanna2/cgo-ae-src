#!/bin/bash
# Figure-7: attention variants kernel @ seqlen=4096 (paper eva_atten_kernel).
# Resume-friendly: collect → check_missing (content success) → run only missing → plot.
#
# Layout (this directory):
#   run.sh / run_one.sh / check_missing.py / collect.py / plot.py
#   logs/     <model>_kernel_<system>_<seqlen>.log
#   results/  atten_kernel_<hw>_seq4096.json
#   eva_atten_kernel_4096.pdf
#
# Matrix (matches tilefusion-paper/script/kernel_atten_4096.py):
#   models:  attn corm h2o roco kf snapkv gemma2   (plot: kf→keyformer)
#   systems: torch dynamo tensorrt tvm flashtensor our
#   type:    op / kernel only
#
# Env routing (see run_one.sh / ENV.md):
#   our/torch       → base 2.9.1 TileFusion models
#   dynamo/tensorrt → base 2.9.1 (same baseline as DSA/KV) + FlashTensor-AE
#   flashtensor/tvm → flashtensor conda + FlashTensor-AE
#
# Success = log contains mean=/avg line (incomplete tee'd logs count as missing).
#
# Usage (inside tilefusion-ae-291):
#   CUDA_VISIBLE_DEVICES=0 bash /workspace/ae/figure-7/run.sh
#   SKIP_BENCH=1 bash ...              # collect + plot only
#   CHECK_ONLY=1 bash ...              # print missing and exit
#   MODELS="attn h2o" SYSTEMS="our torch" bash ...   # subset
#   FORCE=1 bash ...                   # ignore done logs, re-run matrix

set -uo pipefail

FIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AE_DIR="$(cd "${FIG_DIR}/.." && pwd)"
# shellcheck disable=SC1091
source "${AE_DIR}/env_tilefusion_291.sh"
PY="${AE_PY}"

SEQLEN="${SEQLEN:-4096}"
SKIP_BENCH="${SKIP_BENCH:-0}"
CHECK_ONLY="${CHECK_ONLY:-0}"
FORCE="${FORCE:-0}"
LOG_DIR="${FIG_DIR}/logs"
RESULTS_DIR="${FIG_DIR}/results"
mkdir -p "${LOG_DIR}" "${RESULTS_DIR}"

ensure_plot_deps() {
  "${PY}" - <<'PY' 2>/dev/null && return 0
import matplotlib, pandas
print("plot deps OK", matplotlib.__version__, pandas.__version__)
PY
  echo "Installing matplotlib + pandas into conda ..."
  "${PY}" -m pip install -q matplotlib pandas
}

echo "================================================"
echo "Figure-7 attention variants (resume / fill missing)"
echo "  SEQLEN=${SEQLEN}  SKIP_BENCH=${SKIP_BENCH}  CHECK_ONLY=${CHECK_ONLY}  FORCE=${FORCE}"
echo "  logs=${LOG_DIR}"
echo "  results=${RESULTS_DIR}"
echo "================================================"

echo
echo "==== [1] collect (from existing logs) ===="
"${PY}" "${FIG_DIR}/collect.py" --seqlen "${SEQLEN}" --log-dir "${LOG_DIR}" --out-dir "${RESULTS_DIR}"

echo
echo "==== [2] check_missing ===="
"${PY}" "${FIG_DIR}/check_missing.py" --seqlen "${SEQLEN}" --log-dir "${LOG_DIR}"
mapfile -t MISSING_LINES < <("${PY}" "${FIG_DIR}/check_missing.py" --seqlen "${SEQLEN}" --log-dir "${LOG_DIR}" --quiet-header)

if [[ "${CHECK_ONLY}" == "1" ]]; then
  echo "CHECK_ONLY=1 — not running benches."
  exit 0
fi

rc_bench=0
if [[ "${SKIP_BENCH}" == "1" ]]; then
  echo "SKIP_BENCH=1 — using existing logs"
else
  TASKS=()
  if [[ "${FORCE}" == "1" ]]; then
    MODELS_STR="${MODELS:-attn corm h2o roco kf snapkv gemma2}"
    SYSTEMS_STR="${SYSTEMS:-torch dynamo tensorrt tvm flashtensor our}"
    read -r -a MODELS_ARR <<< "${MODELS_STR}"
    read -r -a SYSTEMS_ARR <<< "${SYSTEMS_STR}"
    for m in "${MODELS_ARR[@]}"; do
      for s in "${SYSTEMS_ARR[@]}"; do
        TASKS+=("${m} ${s}")
      done
    done
    echo "FORCE=1 — will run ${#TASKS[@]} cells"
  else
    # Optional subset filter on missing list
    MODELS_FILTER="${MODELS:-}"
    SYSTEMS_FILTER="${SYSTEMS:-}"
    for line in "${MISSING_LINES[@]}"; do
      [[ -z "${line}" ]] && continue
      read -r m s <<< "${line}"
      if [[ -n "${MODELS_FILTER}" ]] && ! echo " ${MODELS_FILTER} " | grep -q " ${m} "; then
        continue
      fi
      if [[ -n "${SYSTEMS_FILTER}" ]] && ! echo " ${SYSTEMS_FILTER} " | grep -q " ${s} "; then
        continue
      fi
      TASKS+=("${m} ${s}")
    done
    echo "Will run ${#TASKS[@]} missing cells"
  fi

  if [[ "${#TASKS[@]}" -eq 0 ]]; then
    echo "Nothing missing — skip bench."
  else
    for pair in "${TASKS[@]}"; do
      read -r m s <<< "${pair}"
      echo
      echo ">>>> run ${m} / ${s}"
      if ! MODEL="${m}" SYSTEM="${s}" SEQLEN="${SEQLEN}" LOG_DIR="${LOG_DIR}" \
          bash "${FIG_DIR}/run_one.sh"; then
        echo "FAILED: ${m} ${s} (continue to next missing)"
        rc_bench=1
      fi
    done
  fi

  echo
  echo "==== [3] re-collect ===="
  "${PY}" "${FIG_DIR}/collect.py" --seqlen "${SEQLEN}" --log-dir "${LOG_DIR}" --out-dir "${RESULTS_DIR}"
  echo
  echo "==== remaining missing ===="
  "${PY}" "${FIG_DIR}/check_missing.py" --seqlen "${SEQLEN}" --log-dir "${LOG_DIR}" || true
fi

echo
echo "==== [4] plot ===="
ensure_plot_deps
"${PY}" "${FIG_DIR}/plot.py" --results-dir "${RESULTS_DIR}" --out-dir "${FIG_DIR}" --seqlen "${SEQLEN}"

echo
echo "Done figure-7."
echo "  logs:    ${LOG_DIR}/"
echo "  results: ${RESULTS_DIR}/"
echo "  pdf:     ${FIG_DIR}/eva_atten_kernel_4096.pdf"
exit "${rc_bench}"
