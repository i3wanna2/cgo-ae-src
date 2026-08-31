#!/bin/bash
# AE env for TileFusion / DSA / KV / attention(our,torch) on torch 2.9.1 *base* conda.
# Always resets conda + PYTHONPATH so a prior `flashtensor` activate cannot leak.
#
#   source /workspace/ae/env_tilefusion_291.sh

export PATH="/workspace/miniconda3/bin:/usr/local/cuda/bin:${PATH}"
# shellcheck disable=SC1091
source /workspace/miniconda3/etc/profile.d/conda.sh

# Leave flashtensor / other envs completely
while [[ -n "${CONDA_DEFAULT_ENV:-}" && "${CONDA_DEFAULT_ENV}" != "base" ]]; do
  conda deactivate || break
done
# Ensure base is active (2.9.1 lives here)
conda activate base 2>/dev/null || true

export PYTHONUNBUFFERED=1
# Absolute reset — never append a previous PYTHONPATH (e.g. FlashTensor-AE / mla_rope)
unset PYTHONPATH
export PYTHONPATH="/workspace/triton/python"
export LD_LIBRARY_PATH="/workspace/miniconda3/lib/python3.12/site-packages/torch/lib:/workspace/miniconda3/lib:/usr/local/cuda/lib64"
# Prefer base python on PATH after activate
export PATH="/workspace/miniconda3/bin:/usr/local/cuda/bin:${PATH}"

AE_PY="/workspace/miniconda3/bin/python"
export AE_PY
