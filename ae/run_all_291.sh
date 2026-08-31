#!/bin/bash
# One-shot AE pipeline for paper Figures 5–6 (DSA + rope_quant_kvcache).
#
#   CUDA_VISIBLE_DEVICES=0 bash /workspace/ae/run_all_291.sh
#   ONLY=dsa|kv|all  SKIP_BENCH=1  SEQLEN=4096

set -uo pipefail

AE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ONLY="${ONLY:-all}"
SEQLEN="${SEQLEN:-4096}"
SKIP_BENCH="${SKIP_BENCH:-0}"

echo "================================================"
echo "AE291 run_all (figure-5 DSA + figure-6 KV)"
echo "  ONLY=${ONLY}  SEQLEN=${SEQLEN}  SKIP_BENCH=${SKIP_BENCH}"
echo "================================================"

rc=0
if [[ "${ONLY}" == "all" || "${ONLY}" == "dsa" ]]; then
  echo
  echo "==== Figure-5 (DSA) ===="
  SEQLEN="${SEQLEN}" SKIP_BENCH="${SKIP_BENCH}" bash "${AE_DIR}/figure-5/run.sh" || rc=$?
fi
if [[ "${ONLY}" == "all" || "${ONLY}" == "kv" ]]; then
  echo
  echo "==== Figure-6 (KV) ===="
  SEQLEN="${SEQLEN}" SKIP_BENCH="${SKIP_BENCH}" bash "${AE_DIR}/figure-6/run.sh" || rc=$?
fi

echo
echo "Done."
echo "  figure-5: ${AE_DIR}/figure-5/{logs,results}/"
echo "  figure-6: ${AE_DIR}/figure-6/{logs,results}/"
exit "${rc}"
