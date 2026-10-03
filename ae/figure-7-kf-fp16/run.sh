#!/bin/bash
# KeyFormer only. Same Figure-7 routing, logs, collect, and plot.
# The one difference: FlashTensor keeps stock FP32 noise; every other system
# uses FP16 noise. TileFusion and PyTorch already do that in kf_model.py.
# TorchInductor, TensorRT, and TVM go through run_kernel_fp16.py.
#
# Usage (inside the AE container, one GPU, systems are sequential):
#   CUDA_VISIBLE_DEVICES=0 bash /workspace/ae/figure-7-kf-fp16/run.sh
#   FORCE=1 bash /workspace/ae/figure-7-kf-fp16/run.sh
#   SYSTEMS="our torch" bash /workspace/ae/figure-7-kf-fp16/run.sh

set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AE_DIR="$(cd "${DIR}/.." && pwd)"
FIG7="$(cd "${DIR}/../figure-7" && pwd)"

MODEL="kf"
SEQLEN="${SEQLEN:-4096}"
FORCE="${FORCE:-0}"
SYSTEMS="${SYSTEMS:-torch dynamo tensorrt tvm flashtensor our}"
LOG_DIR="${DIR}/logs"
RESULTS_DIR="${DIR}/results"
mkdir -p "${LOG_DIR}" "${RESULTS_DIR}"

log_path() {
  echo "${LOG_DIR}/${MODEL}_kernel_${1}_${SEQLEN}.log"
}

log_done() {
  local log="$1"
  [[ -f "${log}" ]] || return 1
  grep -Eq 'kernel:.*mean=[0-9]|\] avg [0-9]' "${log}"
}

detect_platform() {
  local py="$1"
  local gpu
  gpu="$("${py}" -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")')"
  if echo "${gpu}" | grep -q "A100"; then echo "A100"
  elif echo "${gpu}" | grep -q "H100"; then echo "H100"
  elif echo "${gpu}" | grep -q "H800"; then echo "H800"
  else echo "H800"
  fi
}

run_system() {
  local system="$1"
  local log
  log="$(log_path "${system}")"
  if [[ "${FORCE}" != "1" ]] && log_done "${log}"; then
    echo "skip ${system}: ${log}"
    return 0
  fi
  (
    set -euo pipefail
    case "${system}" in
      our|torch)
        # shellcheck disable=SC1091
        source "${AE_DIR}/env_tilefusion_291.sh"
        unset PYTHONPATH
        export PYTHONPATH="/workspace/triton/python"
        PLATFORM="${PLATFORM:-$(detect_platform "${AE_PY}")}"
        echo "================================================"
        echo "kf-fp16 [TileFusion 2.9.1] ${MODEL} ${system} S=${SEQLEN} noise=fp16"
        echo "  PY=${AE_PY} PYTHONPATH=${PYTHONPATH}"
        echo "  log=${log}"
        echo "================================================"
        cd /workspace/triton/tilefusion
        "${AE_PY}" "models/${MODEL}_model.py" \
          --system "${system}" \
          --platform "${PLATFORM}" \
          --seqlen "${SEQLEN}" \
          --mode kernel \
          2>&1 | tee "${log}"
        ;;
      dynamo|tensorrt)
        # shellcheck disable=SC1091
        source "${AE_DIR}/env_tilefusion_291.sh"
        unset PYTHONPATH
        export PYTHONPATH="/workspace/FlashTensor-AE"
        echo "================================================"
        echo "kf-fp16 [base 2.9.1] ${MODEL} ${system} S=${SEQLEN} noise=fp16"
        echo "  PY=${AE_PY} PYTHONPATH=${PYTHONPATH}"
        echo "  log=${log}"
        echo "================================================"
        cd /workspace/FlashTensor-AE
        "${AE_PY}" "${DIR}/run_kernel_fp16.py" \
          --model "${MODEL}" \
          --system "${system}" \
          --seqlen "${SEQLEN}" \
          2>&1 | tee "${log}"
        ;;
      tvm)
        # shellcheck disable=SC1091
        source "${AE_DIR}/env_flashtensor.sh"
        unset PYTHONPATH
        export PYTHONPATH=""
        echo "================================================"
        echo "kf-fp16 [flashtensor conda] ${MODEL} tvm S=${SEQLEN} noise=fp16"
        echo "  PY=${AE_PY} PYTHONPATH='${PYTHONPATH}'"
        echo "  log=${log}"
        echo "================================================"
        cd /workspace/FlashTensor-AE
        "${AE_PY}" "${DIR}/run_kernel_fp16.py" \
          --model "${MODEL}" \
          --system tvm \
          --seqlen "${SEQLEN}" \
          2>&1 | tee "${log}"
        ;;
      flashtensor)
        # shellcheck disable=SC1091
        source "${AE_DIR}/env_flashtensor.sh"
        unset PYTHONPATH
        export PYTHONPATH=""
        echo "================================================"
        echo "kf-fp16 [flashtensor conda] ${MODEL} flashtensor S=${SEQLEN} noise=fp32"
        echo "  PY=${AE_PY} PYTHONPATH='${PYTHONPATH}'"
        echo "  log=${log}"
        echo "================================================"
        cd /workspace/FlashTensor-AE
        "${AE_PY}" run_kernel.py \
          --model "${MODEL}" \
          --system our \
          --seqlen "${SEQLEN}" \
          2>&1 | tee "${log}"
        ;;
      *)
        echo "Unknown SYSTEM=${system}" >&2
        exit 1
        ;;
    esac
  )
}

echo "================================================"
echo "KeyFormer FP16 baselines, FlashTensor stays FP32"
echo "  SEQLEN=${SEQLEN} FORCE=${FORCE}"
echo "  SYSTEMS=${SYSTEMS}"
echo "  logs=${LOG_DIR}"
echo "================================================"

for system in ${SYSTEMS}; do
  run_system "${system}"
done

# shellcheck disable=SC1091
source "${AE_DIR}/env_tilefusion_291.sh"
unset PYTHONPATH
export PYTHONPATH="/workspace/triton/python"
"${AE_PY}" - <<'PY' 2>/dev/null || "${AE_PY}" -m pip install -q matplotlib pandas
import matplotlib, pandas
print("plot deps OK", matplotlib.__version__, pandas.__version__)
PY

echo
echo "==== collect ===="
"${AE_PY}" "${FIG7}/collect.py" \
  --seqlen "${SEQLEN}" \
  --log-dir "${LOG_DIR}" \
  --out-dir "${RESULTS_DIR}"

echo
echo "==== plot ===="
"${AE_PY}" "${DIR}/plot.py" \
  --seqlen "${SEQLEN}" \
  --results-dir "${RESULTS_DIR}" \
  --out-dir "${DIR}"

echo "Done. PDF: ${DIR}/eva_atten_kernel_4096.pdf"
corr=0
bash "${AE_DIR}/report_correctness.sh" "${LOG_DIR}" || corr=$?
if [[ "${corr}" -ne 0 ]]; then
  exit 1
fi
exit 0
