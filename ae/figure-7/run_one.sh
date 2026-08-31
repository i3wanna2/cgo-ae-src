#!/bin/bash
# Run one Figure-7 attention kernel cell.
# Usage: MODEL=attn SYSTEM=our SEQLEN=4096 bash run_one.sh
#
# Routing (match where packages are installed in this Docker / ENV.md):
#   our | torch
#       → TileFusion base conda 2.9.1  (models/<model>_model.py)
#         PYTHONPATH=/workspace/triton/python
#   tensorrt | dynamo
#       → same base conda as DSA/KV (torch 2.9.1; TRT only required for tensorrt)
#         + FlashTensor-AE/run_kernel.py
#         PYTHONPATH=/workspace/FlashTensor-AE  (overwrite; do NOT keep TileFusion triton)
#   flashtensor | tvm
#       → flashtensor conda (torch 2.2.2) + FlashTensor-AE/run_kernel.py
#         PYTHONPATH must be empty; flashtensor maps to FT --system our
#
# Log name (used by collect/check_missing):
#   logs/<model>_kernel_<system>_<seqlen>.log

set -euo pipefail

FIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AE_DIR="$(cd "${FIG_DIR}/.." && pwd)"

MODEL="${MODEL:?MODEL required (attn|corm|h2o|roco|kf|snapkv|gemma2)}"
SYSTEM="${SYSTEM:?SYSTEM required}"
SEQLEN="${SEQLEN:-4096}"
LOG_DIR="${LOG_DIR:-${FIG_DIR}/logs}"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/${MODEL}_kernel_${SYSTEM}_${SEQLEN}.log"

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

case "${SYSTEM}" in
  our|torch)
    # shellcheck disable=SC1091
    source "${AE_DIR}/env_tilefusion_291.sh"
    # Belt-and-suspenders: env script already unset+set; force exact path again.
    unset PYTHONPATH
    export PYTHONPATH="/workspace/triton/python"
    PLATFORM="${PLATFORM:-$(detect_platform "${AE_PY}")}"
    echo "================================================"
    echo "Figure-7 [TileFusion 2.9.1] ${MODEL} ${SYSTEM} S=${SEQLEN}"
    echo "  CONDA=${CONDA_DEFAULT_ENV:-?}  PY=${AE_PY}"
    echo "  PYTHONPATH=${PYTHONPATH}"
    echo "  log=${LOG}"
    echo "================================================"
    if [[ "${PYTHONPATH}" != "/workspace/triton/python" ]]; then
      echo "ERROR: expected PYTHONPATH=/workspace/triton/python, got: ${PYTHONPATH}"
      exit 1
    fi
    "${AE_PY}" - <<'PY' || exit 1
import torch, triton
assert torch.__version__.startswith("2.9.1"), torch.__version__
assert "python/triton" in (triton.__file__ or ""), triton.__file__
print("preflight OK torch", torch.__version__, "triton", triton.__file__)
PY
    cd /workspace/triton/tilefusion
    "${AE_PY}" "models/${MODEL}_model.py" \
      --system "${SYSTEM}" \
      --platform "${PLATFORM}" \
      --seqlen "${SEQLEN}" \
      --mode kernel \
      2>&1 | tee "${LOG}"
    ;;
  dynamo)
    # Baseline TorchInductor on base 2.9.1 (same as DSA/KV), graph via FlashTensor-AE.
    # shellcheck disable=SC1091
    source "${AE_DIR}/env_tilefusion_291.sh"
    # Drop TileFusion triton path completely — do not append.
    unset PYTHONPATH
    export PYTHONPATH="/workspace/FlashTensor-AE"
    echo "================================================"
    echo "Figure-7 [base 2.9.1 dynamo] ${MODEL} ${SYSTEM} S=${SEQLEN}"
    echo "  CONDA=${CONDA_DEFAULT_ENV:-?}  PY=${AE_PY}"
    echo "  PYTHONPATH=${PYTHONPATH}"
    echo "  log=${LOG}"
    echo "================================================"
    if [[ "${PYTHONPATH}" != "/workspace/FlashTensor-AE" ]]; then
      echo "ERROR: expected PYTHONPATH=/workspace/FlashTensor-AE, got: ${PYTHONPATH}"
      exit 1
    fi
    "${AE_PY}" - <<'PY' || exit 1
import torch
assert torch.__version__.startswith("2.9.1"), torch.__version__
from asuka_exp.cases.kernels import KERNEL_ZOO
print("preflight OK torch", torch.__version__, "models", sorted(KERNEL_ZOO.keys()))
PY
    cd /workspace/FlashTensor-AE
    "${AE_PY}" run_kernel.py \
      --model "${MODEL}" \
      --system dynamo \
      --seqlen "${SEQLEN}" \
      2>&1 | tee "${LOG}"
    ;;
  tensorrt)
    # Baseline TensorRT on base (10.13.2.6, same as DSA/KV), graph via FlashTensor-AE.
    # shellcheck disable=SC1091
    source "${AE_DIR}/env_tilefusion_291.sh"
    unset PYTHONPATH
    export PYTHONPATH="/workspace/FlashTensor-AE"
    echo "================================================"
    echo "Figure-7 [base 2.9.1 tensorrt] ${MODEL} ${SYSTEM} S=${SEQLEN}"
    echo "  CONDA=${CONDA_DEFAULT_ENV:-?}  PY=${AE_PY}"
    echo "  PYTHONPATH=${PYTHONPATH}"
    echo "  log=${LOG}"
    echo "================================================"
    if [[ "${PYTHONPATH}" != "/workspace/FlashTensor-AE" ]]; then
      echo "ERROR: expected PYTHONPATH=/workspace/FlashTensor-AE, got: ${PYTHONPATH}"
      exit 1
    fi
    "${AE_PY}" - <<'PY' || exit 1
import torch, tensorrt as t
assert torch.__version__.startswith("2.9.1"), torch.__version__
assert t.__version__ == "10.13.2.6", t.__version__
from asuka_exp.cases.kernels import KERNEL_ZOO
print("preflight OK torch", torch.__version__, "tensorrt", t.__version__,
      "models", sorted(KERNEL_ZOO.keys()))
PY
    cd /workspace/FlashTensor-AE
    "${AE_PY}" run_kernel.py \
      --model "${MODEL}" \
      --system tensorrt \
      --seqlen "${SEQLEN}" \
      2>&1 | tee "${LOG}"
    ;;
  flashtensor|tvm)
    # shellcheck disable=SC1091
    source "${AE_DIR}/env_flashtensor.sh"
    # Force-clear again (parent run.sh may have exported TileFusion PYTHONPATH).
    unset PYTHONPATH
    export PYTHONPATH=""
    echo "================================================"
    echo "Figure-7 [flashtensor conda] ${MODEL} ${SYSTEM} S=${SEQLEN}"
    echo "  CONDA=${CONDA_DEFAULT_ENV:-?}  PY=${AE_PY}"
    echo "  PYTHONPATH='${PYTHONPATH}' (must be empty)"
    echo "  log=${LOG}"
    echo "================================================"
    if [[ -n "${PYTHONPATH}" ]]; then
      echo "ERROR: PYTHONPATH must be empty for flashtensor env, got: ${PYTHONPATH}"
      exit 1
    fi
    "${AE_PY}" - <<'PY' || exit 1
import torch, asuka, tvm, triton
assert torch.__version__.startswith("2.2.2"), torch.__version__
assert "site-packages/triton" in (triton.__file__ or ""), triton.__file__
print("preflight OK torch", torch.__version__, "tvm", tvm.__version__, "triton", triton.__file__)
PY
    FT_SYS="${SYSTEM}"
    [[ "${SYSTEM}" == "flashtensor" ]] && FT_SYS="our"
    cd /workspace/FlashTensor-AE
    "${AE_PY}" run_kernel.py \
      --model "${MODEL}" \
      --system "${FT_SYS}" \
      --seqlen "${SEQLEN}" \
      2>&1 | tee "${LOG}"
    ;;
  *)
    echo "Unknown SYSTEM=${SYSTEM}. Use: our|torch|dynamo|tensorrt|flashtensor|tvm"
    exit 1
    ;;
esac

echo "OK: ${MODEL} ${SYSTEM} seqlen=${SEQLEN}"
