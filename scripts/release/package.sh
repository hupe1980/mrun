#!/usr/bin/env bash
# Packages a release tarball (008 data model ReleaseArtifact):
#   mrun-X.Y.Z/{bin/mrun, bin/mlx.metallib, LICENSE-MIT, LICENSE-APACHE, NOTICE,
#               completions/{mrun.bash, _mrun, mrun.fish}}
# plus SHA256SUMS. Usage: scripts/release/package.sh <version> <target-dir> <out-dir>
set -euo pipefail

version="$1"
target_dir="$2"   # e.g. target/aarch64-apple-darwin/release
out_dir="$3"

name="mrun-${version}"
stage="$(mktemp -d)/${name}"
mkdir -p "${stage}/bin" "${stage}/completions" "${out_dir}"

cp "${target_dir}/mrun" "${stage}/bin/mrun"
if [[ -f "${target_dir}/mlx.metallib" ]]; then
  cp "${target_dir}/mlx.metallib" "${stage}/bin/mlx.metallib"
else
  echo "error: ${target_dir}/mlx.metallib is missing (build with --features mlx on macOS)" >&2
  exit 1
fi
cp LICENSE-MIT LICENSE-APACHE NOTICE "${stage}/"

"${stage}/bin/mrun" completions bash > "${stage}/completions/mrun.bash"
"${stage}/bin/mrun" completions zsh > "${stage}/completions/_mrun"
"${stage}/bin/mrun" completions fish > "${stage}/completions/mrun.fish"

tarball="${out_dir}/${name}-macos-arm64.tar.gz"
tar -C "$(dirname "${stage}")" -czf "${tarball}" "${name}"
(cd "${out_dir}" && shasum -a 256 "$(basename "${tarball}")" > SHA256SUMS)
echo "${tarball}"
