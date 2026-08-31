#!/bin/bash
# AE env for FlashTensor (asuka) + TVM in conda env `flashtensor` (torch 2.2.2).
# Clears TileFusion PYTHONPATH — /workspace/triton/python breaks triton-nightly.
#
#   source /workspace/ae/env_flashtensor.sh

export PATH="/workspace/miniconda3/bin:/usr/local/cuda/bin:${PATH}"
# shellcheck disable=SC1091
source /workspace/miniconda3/etc/profile.d/conda.sh

# Drop any prior env, then activate flashtensor
while [[ -n "${CONDA_DEFAULT_ENV:-}" && "${CONDA_DEFAULT_ENV}" != "base" ]]; do
  conda deactivate || break
done
conda activate flashtensor

# Explicitly clear TileFusion / leaked PYTHONPATH (absolute reset, never append)
unset PYTHONPATH
export PYTHONPATH=""

export PYTHONUNBUFFERED=1
export LLVM_DIR="${CONDA_PREFIX}/lib/cmake/llvm"
export MLIR_DIR="${CONDA_PREFIX}/lib/cmake/mlir"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${CONDA_PREFIX}/lib"
export PATH="${CONDA_PREFIX}/bin:/usr/local/cuda/bin:/workspace/miniconda3/bin:${PATH}"

AE_PY="${CONDA_PREFIX}/bin/python"
export AE_PY

# Ensure common FlashTensor-AE runtime deps (idempotent)
"${AE_PY}" -c "import tqdm, tabulate, yaml, pulp" 2>/dev/null || \
  "${AE_PY}" -m pip install -q -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --trusted-host pypi.tuna.tsinghua.edu.cn tqdm tabulate pyyaml pulp
