#!/usr/bin/env bash
# Type-checks the experiment against Candle with patches/0001 applied, for aarch64-apple-darwin,
# in a temporary copy so the committed Cargo.lock keeps its pinned git sources.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
rev=c68b24997319b8d76e5a4e775dab197c7f26322a
git clone -q --filter=blob:none https://github.com/huggingface/candle.git "$work/candle"
git -C "$work/candle" checkout -q "$rev"
git -C "$work/candle" apply "$here/patches/0001-metal-allocation-gate.patch"
mkdir "$work/pkg"
cp -r "$here/Cargo.toml" "$here/Cargo.lock" "$here/rust-toolchain.toml" "$here/src" "$here/tests" "$work/pkg/"
cd "$work/pkg"
cargo check --tests --features metal-gate --target aarch64-apple-darwin \
  --config "patch.'https://github.com/huggingface/candle.git'.candle-core.path='$work/candle/candle-core'" \
  --config "patch.'https://github.com/huggingface/candle.git'.candle-metal-kernels.path='$work/candle/candle-metal-kernels'"
