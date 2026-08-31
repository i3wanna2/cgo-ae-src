#!/bin/bash
# Figure-5: DSA kernel (paper eva_dsa_kernel).
# Layout: scripts + logs/ + results/ live in this directory.
#
# Usage (inside tilefusion-ae-291):
#   CUDA_VISIBLE_DEVICES=0 bash /workspace/ae/figure-5/run.sh
#   SYSTEMS="tilelang dynamo" bash ...
#   SKIP_DONE=0 bash ...          # force re-run even if log has mean=
#   SKIP_BENCH=1 bash ...         # collect + plot only

set -uo pipefail

FIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AE_DIR="$(cd "${FIG_DIR}/.." && pwd)"
# shellcheck disable=SC1091
source "${AE_DIR}/env_tilefusion_291.sh"
# Absolute reset (env script already unset+set; force again so run_all / prior figure-7 cannot leak)
EXPECTED_PYTHONPATH="/workspace/triton/python"
unset PYTHONPATH
export PYTHONPATH="${EXPECTED_PYTHONPATH}"
if [[ "${PYTHONPATH}" != "${EXPECTED_PYTHONPATH}" ]]; then
  echo "ERROR: expected PYTHONPATH=${EXPECTED_PYTHONPATH}, got: ${PYTHONPATH}"
  exit 1
fi

PY="${AE_PY}"
TF_DIR=/workspace/triton/tilefusion
MODEL="${TF_DIR}/models/dsa_mla_model.py"
LOG_DIR="${FIG_DIR}/logs"
RESULTS_DIR="${FIG_DIR}/results"
mkdir -p "${LOG_DIR}" "${RESULTS_DIR}"
cd "${TF_DIR}"

SEQLEN="${SEQLEN:-4096}"
SKIP_DONE="${SKIP_DONE:-1}"
SKIP_BENCH="${SKIP_BENCH:-0}"
SYSTEMS_STR="${SYSTEMS:-our torch dynamo tensorrt tilelang tilelang-ws}"
read -r -a SYSTEMS <<< "${SYSTEMS_STR}"

ensure_plot_deps() {
  "${PY}" - <<'PY' 2>/dev/null && return 0
import matplotlib, pandas
print("plot deps OK", matplotlib.__version__, pandas.__version__)
PY
  echo "Installing matplotlib + pandas into conda ..."
  "${PY}" -m pip install -q matplotlib pandas
}

echo "================================================"
echo "Figure-5 DSA (torch 2.9.1)"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "  PYTHONPATH=${PYTHONPATH}"
echo "  seqlen=${SEQLEN}  SKIP_BENCH=${SKIP_BENCH}  SKIP_DONE=${SKIP_DONE}"
echo "  systems=${SYSTEMS[*]}"
echo "  logs=${LOG_DIR}"
echo "  results=${RESULTS_DIR}"
echo "================================================"

rc_bench=0
if [[ "${SKIP_BENCH}" != "1" ]]; then
  "${PY}" - <<'PY' || exit 1
import torch, triton, tilefusion, tilelang
assert torch.__version__.startswith("2.9.1"), torch.__version__
assert triton.__file__ and "python/triton" in triton.__file__, triton.__file__
print("preflight OK", "torch", torch.__version__, "cuda", torch.cuda.is_available(),
      "gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
try:
    import tensorrt as t
    print("tensorrt", t.__version__)
except Exception as e:
    print("tensorrt not importable:", e)
PY

  GPU_NAME="$("${PY}" -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")')"
  if echo "${GPU_NAME}" | grep -q "A100"; then
    PLATFORM="A100"
  elif echo "${GPU_NAME}" | grep -q "H100"; then
    PLATFORM="H100"
  elif echo "${GPU_NAME}" | grep -q "H800"; then
    PLATFORM="H800"
  else
    PLATFORM="H800"
    echo "WARNING: unrecognized GPU '${GPU_NAME}', defaulting platform=${PLATFORM}"
  fi
  echo "  GPU=${GPU_NAME}  PLATFORM=${PLATFORM}"

  # tilelang-ws is Hopper-only; A100 must use plain tilelang.
  FILTERED_SYSTEMS=()
  for sys in "${SYSTEMS[@]}"; do
    if [[ "${sys}" == "tilelang-ws" && "${PLATFORM}" == "A100" ]]; then
      echo "NOTE: skip tilelang-ws on A100 (unsupported); plot will use tilelang"
      continue
    fi
    FILTERED_SYSTEMS+=("${sys}")
  done
  SYSTEMS=("${FILTERED_SYSTEMS[@]}")
  echo "  systems (after GPU filter)=${SYSTEMS[*]}"

  FAILED=()
  for sys in "${SYSTEMS[@]}"; do
    log_file="${LOG_DIR}/dsa_mla_model_${sys}_${SEQLEN}.log"
    echo
    echo "---- [${sys}] -> ${log_file} ----"

    if [[ "${SKIP_DONE}" == "1" ]] && [[ -f "${log_file}" ]] && grep -qE 'kernel:.*mean=' "${log_file}"; then
      echo "SKIP (already have kernel mean)"
      grep -E 'kernel:.*mean=' "${log_file}" | tail -1
      continue
    fi

    if [[ "${sys}" == "tensorrt" ]]; then
      if ! "${PY}" -c "import tensorrt" 2>/dev/null; then
        echo "STOP: tensorrt not installed in conda 2.9.1. Ask before workaround."
        FAILED+=("tensorrt:missing")
        continue
      fi
    fi

    rc=0
    "${PY}" "${MODEL}" \
      --system "${sys}" \
      --platform "${PLATFORM}" \
      --seqlen "${SEQLEN}" \
      --mode kernel \
      2>&1 | tee "${log_file}" || rc=$?

    if [[ "${rc}" -ne 0 ]] || ! grep -qE 'kernel:.*mean=' "${log_file}"; then
      echo "FAILED: ${sys} (rc=${rc})"
      FAILED+=("${sys}")
      rc_bench=1
      if [[ "${sys}" == "tensorrt" || "${sys}" == "dynamo" ]]; then
        echo "STOPPING after ${sys} failure (env/tooling issue — ask user)."
        break
      fi
    else
      echo "OK: ${sys}"
      grep -E 'kernel:.*mean=' "${log_file}" | tail -1
    fi
  done

  echo
  echo "---- bench summary ----"
  for sys in "${SYSTEMS[@]}"; do
    log_file="${LOG_DIR}/dsa_mla_model_${sys}_${SEQLEN}.log"
    if [[ -f "${log_file}" ]] && grep -qE 'kernel:.*mean=' "${log_file}"; then
      grep -E 'kernel:.*mean=' "${log_file}" | tail -1 | sed "s/^/[${sys}] /"
    else
      echo "[${sys}] MISSING"
    fi
  done
else
  echo "SKIP_BENCH=1 — using existing logs"
fi

echo
echo "==== collect JSON ===="
"${PY}" "${FIG_DIR}/collect.py" --seqlen "${SEQLEN}" --log-dir "${LOG_DIR}" --out-dir "${RESULTS_DIR}"

echo
echo "==== plot ===="
ensure_plot_deps
"${PY}" "${FIG_DIR}/plot.py" --results-dir "${RESULTS_DIR}" --out-dir "${FIG_DIR}" --seqlen "${SEQLEN}"

echo
echo "Done figure-5."
echo "  logs:    ${LOG_DIR}/"
echo "  results: ${RESULTS_DIR}/"
echo "  pdf:     ${FIG_DIR}/eva_dsa_kernel_4096.pdf"
exit "${rc_bench}"
