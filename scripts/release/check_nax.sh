#!/usr/bin/env bash
# Checks an extracted mlx.metallib for MLX's NAX kernels (D044, 008 FR-019, SC-007).
# Usage: scripts/release/check_nax.sh <mlx.metallib> <deployment-target>
#
# With a deployment target of 26.2 or later the metallib must hold `*_nax` kernels; without them
# this prints a warning, and fails (exit 1) when MLX_SYS_REQUIRE_NAX=1 (release CI). Below 26.2
# NAX is not compiled and nothing is checked. The same rule as `check_nax` in
# crates/mrun-mlx-sys/build.rs, for the binary that was actually built.
set -euo pipefail

metallib="$1"
target="$2"

major="${target%%.*}"
minor="${target#*.}"
[[ "${minor}" == "${target}" ]] && minor=0
minor="${minor%%.*}"
if (( major < 26 || (major == 26 && minor < 2) )); then
  echo "deployment target ${target}: no NAX kernels expected"
  exit 0
fi
if LC_ALL=C grep -aq '_nax' "${metallib}"; then
  echo "deployment target ${target}: ${metallib} holds NAX kernels"
  exit 0
fi
message="${metallib} holds no *_nax kernels although the deployment target is ${target} (NAX needs the macOS 26.2 SDK or later)"
if [[ "${MLX_SYS_REQUIRE_NAX:-}" == "1" ]]; then
  echo "error: ${message}" >&2
  exit 1
fi
echo "warning: ${message}" >&2
