#!/usr/bin/env bash
# mlx-lm 0.32.0 reference speeds on the same weights and prompt ids as `bench_engine` (001 R16).
#
#   scripts/bench_reference.sh <model-dir-or-repo> [--depths 1024,16384] [--decode 128] [--runs 2]
#
# Needs Python with mlx-lm 0.32.0 on MLX 0.32.4 (`PYTHON`, default `python3`). Prints JSON; the
# matching engine run is
#   cargo run --release -p mrun-engine --features mlx --example bench_engine -- <model-dir> ...
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
[[ $# -ge 1 ]] || { sed -n '2,8p' "$0" >&2; exit 2; }
exec "${PYTHON:-python3}" -I "$here/bench_reference.py" "$@"
