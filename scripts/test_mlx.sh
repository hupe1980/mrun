#!/usr/bin/env bash
# Runs mrun-engine's MLX test suites and fails when any of them ran 0 tests (T072): a missing
# `mlx` feature compiles them to nothing, which cargo reports as a pass.
#
#   scripts/test_mlx.sh                       # tiny models (CPU and Metal)
#   MRUN_TEST_MODELS=qwen3,qwen3.5 scripts/test_mlx.sh -- --ignored   # real models too
#
# Extra arguments go to `cargo test` (after the package and feature flags).
set -euo pipefail
cd "$(dirname "$0")/.."

suites=(tiny_qwen3 tiny_qwen3_5 tiny_qwen3_5_moe tiny_qwen3_moe exactness failure load_refusal)
if [[ -n "${MRUN_TEST_MODELS:-}" ]]; then
  suites+=(conformance)
fi

log=$(mktemp)
trap 'rm -f "$log"' EXIT
MRUN_REQUIRE_MLX=1 cargo test -p mrun-engine --features mlx --release "$@" 2>&1 | tee "$log"
status=${PIPESTATUS[0]}
[[ $status -eq 0 ]] || exit "$status"

fail=0
for suite in "${suites[@]}"; do
  # "Running tests/<suite>.rs (...)" is followed by "running N tests".
  count=$(awk -v s="tests/$suite.rs" '
    index($0, "Running " s) { found = 1; next }
    found && /^running [0-9]+ tests?/ { print $2; exit }
  ' "$log")
  if [[ -z "$count" ]]; then
    echo "error: suite $suite did not run" >&2
    fail=1
  elif [[ "$count" -eq 0 ]]; then
    echo "error: suite $suite ran 0 tests" >&2
    fail=1
  fi
done
exit "$fail"
