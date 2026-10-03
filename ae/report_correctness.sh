#!/bin/bash
# Scan AE logs after all methods have run. Does not stop individual benches.
# Exit 1 if any log records a correctness mismatch, or a finished bench has no check.
set -uo pipefail

LOG_DIR="${1:?log directory}"
echo
echo "==== correctness summary (${LOG_DIR}) ===="

shopt -s nullglob
logs=("${LOG_DIR}"/*.log)
if [[ ${#logs[@]} -eq 0 ]]; then
  echo "No logs found."
  exit 1
fi

failed=0
missing=0
passed=0

is_finished() {
  grep -qE 'kernel:.*mean=|\] avg [0-9]|All Figure 6 methods passed|Correctness check (passed|failed)' "$1"
}

for log in "${logs[@]}"; do
  name="$(basename "${log}")"
  if grep -q "Correctness check failed" "${log}"; then
    echo "FAIL     ${name}"
    failed=1
  elif grep -qE "Correctness check passed|All Figure 6 methods passed" "${log}"; then
    echo "PASS     ${name}"
    passed=$((passed + 1))
  elif is_finished "${log}"; then
    echo "MISSING  ${name}"
    missing=1
  else
    echo "SKIP     ${name} (incomplete)"
  fi
done

if [[ "${failed}" -ne 0 || "${missing}" -ne 0 ]]; then
  echo "Correctness: FAIL"
  exit 1
fi
echo "Correctness: PASS (${passed} logs)"
exit 0
