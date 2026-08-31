#!/bin/bash
# Figure-6: rope_quant_kvcache (paper eva_kvcache_kernel).
# Layout: scripts + logs/ + results/ live in this directory.
#
# Usage (inside tilefusion-ae-291):
#   CUDA_VISIBLE_DEVICES=0 bash /workspace/ae/figure-6/run.sh
#   SKIP_BENCH=1 bash ...         # collect + plot only

set -euo pipefail

FIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AE_DIR="$(cd "${FIG_DIR}/.." && pwd)"
# shellcheck disable=SC1091
source "${AE_DIR}/env_tilefusion_291.sh"
# Absolute reset then set KV-only path (do not append leftover FlashTensor-AE / empty FT path)
EXPECTED_PYTHONPATH="/workspace/triton/python:/workspace/triton/tilefusion/ops/csrc/mla_rope"
unset PYTHONPATH
export PYTHONPATH="${EXPECTED_PYTHONPATH}"
if [[ "${PYTHONPATH}" != "${EXPECTED_PYTHONPATH}" ]]; then
  echo "ERROR: expected PYTHONPATH=${EXPECTED_PYTHONPATH}, got: ${PYTHONPATH}"
  exit 1
fi

PY="${AE_PY}"
SEQLEN="${KVCACHE_SEQLEN:-${SEQLEN:-4096}}"
export KVCACHE_SEQLEN="${SEQLEN}"
export KVCACHE_WRITE_JSON="${KVCACHE_WRITE_JSON:-0}"
SKIP_BENCH="${SKIP_BENCH:-0}"
LOG_DIR="${FIG_DIR}/logs"
RESULTS_DIR="${FIG_DIR}/results"
mkdir -p "${LOG_DIR}" "${RESULTS_DIR}"
LOG="${LOG_DIR}/rope_quant_kvcache_${SEQLEN}.log"

ensure_plot_deps() {
  "${PY}" - <<'PY' 2>/dev/null && return 0
import matplotlib, pandas
print("plot deps OK", matplotlib.__version__, pandas.__version__)
PY
  echo "Installing matplotlib + pandas into conda ..."
  "${PY}" -m pip install -q matplotlib pandas
}

echo "================================================"
echo "Figure-6 rope_quant_kvcache (torch 2.9.1)"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "  PYTHONPATH=${PYTHONPATH}"
echo "  seqlen=${SEQLEN}  SKIP_BENCH=${SKIP_BENCH}"
echo "  methods=Torch,Dynamo,TensorRT,CUDA,Triton_fused"
echo "  log=${LOG}"
echo "  results=${RESULTS_DIR}"
echo "================================================"

if [[ "${SKIP_BENCH}" != "1" ]]; then
  "${PY}" - <<'PY' || exit 1
import torch, tensorrt as t, mla_rope_quant_cuda, triton
assert torch.__version__.startswith("2.9.1"), torch.__version__
assert t.__version__ == "10.13.2.6", t.__version__
assert "python/triton" in (triton.__file__ or ""), triton.__file__
print("preflight OK", torch.__version__, t.__version__, mla_rope_quant_cuda.__file__)
PY

  "${PY}" "${FIG_DIR}/run_kvcache.py" 2>&1 | tee "${LOG}"
  echo "OK: kvcache seqlen=${SEQLEN}"
else
  echo "SKIP_BENCH=1 — using existing log ${LOG}"
fi

echo
echo "==== collect JSON ===="
"${PY}" "${FIG_DIR}/collect.py" --seqlen "${SEQLEN}" --kv-log "${LOG}" --out-dir "${RESULTS_DIR}"

echo
echo "==== plot ===="
ensure_plot_deps
"${PY}" "${FIG_DIR}/plot.py" --results-dir "${RESULTS_DIR}" --out-dir "${FIG_DIR}" --seqlen "${SEQLEN}"

echo
echo "Done figure-6."
echo "  logs:    ${LOG_DIR}/"
echo "  results: ${RESULTS_DIR}/"
echo "  pdf:     ${FIG_DIR}/eva_rope_quant_kvcache_kernel_4096.pdf"
