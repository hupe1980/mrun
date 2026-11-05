#!/usr/bin/env bash
# Packages a release tarball (008 data model ReleaseArtifact):
#   mrun-X.Y.Z/{bin/mrun, LICENSE-MIT, LICENSE-APACHE, NOTICE,
#               completions/{mrun.bash, _mrun, mrun.fish}}
# plus SHA256SUMS. Usage: scripts/release/package.sh <version> <target-dir> <out-dir>
#
# The binary contains the engine's Metal kernels, so nothing else ships next to it.
set -euo pipefail

version="$1"
target_dir="$2"   # e.g. target/aarch64-apple-darwin/release
out_dir="$3"

name="mrun-${version}"
stage="$(mktemp -d)/${name}"
mkdir -p "${stage}/bin" "${stage}/completions" "${out_dir}"

cp "${target_dir}/mrun" "${stage}/bin/mrun"
cp LICENSE-MIT LICENSE-APACHE NOTICE "${stage}/"

"${stage}/bin/mrun" completions bash > "${stage}/completions/mrun.bash"
"${stage}/bin/mrun" completions zsh > "${stage}/completions/_mrun"
"${stage}/bin/mrun" completions fish > "${stage}/completions/mrun.fish"

tarball="${out_dir}/${name}-macos-arm64.tar.gz"
# No extended attributes (macOS tags every file with com.apple.provenance, which bsdtar would
# archive and restore on the user's Mac), no AppleDouble `._` files, and root:wheel as the owner
# instead of the build user (008 local release dry run).
if tar --version 2>/dev/null | grep -q bsdtar; then
  owner=(--no-xattrs --no-mac-metadata --uid 0 --gid 0 --uname root --gname wheel)
else
  owner=(--no-xattrs --owner 0 --group 0)
fi
COPYFILE_DISABLE=1 tar "${owner[@]}" -C "$(dirname "${stage}")" -czf "${tarball}" "${name}"
(cd "${out_dir}" && shasum -a 256 "$(basename "${tarball}")" > SHA256SUMS)
echo "${tarball}"
