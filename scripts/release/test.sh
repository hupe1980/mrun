#!/usr/bin/env bash
# Tests of the release tooling (008 SC-007, T036): the NAX check on a metallib without and with
# `*_nax` kernels, and the Homebrew formula's tarball choice per macOS version.
# Usage: scripts/release/test.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

work="$(mktemp -d)"
trap 'rm -rf "${work}"' EXIT
printf 'MTLB\0gemm_bf16\0attention_f16\0' > "${work}/macos15.metallib"
printf 'MTLB\0gemm_bf16\0steel_gemm_fused_nax\0' > "${work}/macos26.metallib"
check=scripts/release/check_nax.sh

# A 15.0 build is never checked; a 26.2 one passes with NAX kernels.
"${check}" "${work}/macos15.metallib" 15.0
MLX_SYS_REQUIRE_NAX=1 "${check}" "${work}/macos26.metallib" 26.2
# A 26.2 target with a metallib that has no NAX kernels: a warning, and a failure when required.
"${check}" "${work}/macos15.metallib" 26.2 2> "${work}/warning.txt"
grep -q '^warning: .*holds no \*_nax kernels' "${work}/warning.txt"
if MLX_SYS_REQUIRE_NAX=1 "${check}" "${work}/macos15.metallib" 26.2 2> /dev/null; then
  echo "FAIL: a 15.0 metallib passed the NAX check for 26.2" >&2
  exit 1
fi
MLX_SYS_REQUIRE_NAX=1 "${check}" "${work}/macos15.metallib" 26.1
echo "check_nax.sh: ok"

# package.sh with a stand-in binary: the tarball layout of 008's ReleaseArtifact, its checksum,
# and no extended attributes or AppleDouble entries.
mkdir -p "${work}/target"
cat > "${work}/target/mrun" <<'STUB'
#!/bin/sh
echo "completions for $2"
STUB
chmod +x "${work}/target/mrun"
tarball="$(scripts/release/package.sh 9.9.9 "${work}/target" "${work}/out")"
[[ "${tarball}" == "${work}/out/mrun-9.9.9-macos-arm64.tar.gz" ]]
tar -tzf "${tarball}" | sort > "${work}/entries.txt"
diff -u - "${work}/entries.txt" <<'LIST'
mrun-9.9.9/
mrun-9.9.9/LICENSE-APACHE
mrun-9.9.9/LICENSE-MIT
mrun-9.9.9/NOTICE
mrun-9.9.9/bin/
mrun-9.9.9/bin/mrun
mrun-9.9.9/completions/
mrun-9.9.9/completions/_mrun
mrun-9.9.9/completions/mrun.bash
mrun-9.9.9/completions/mrun.fish
LIST
if gzip -dc "${tarball}" | LC_ALL=C grep -aq 'xattr\.'; then
  echo "FAIL: the tarball carries extended attributes" >&2
  exit 1
fi
(cd "${work}/out" && shasum -a 256 -c SHA256SUMS > /dev/null)
echo "package.sh: ok"

ruby packaging/homebrew/formula_test.rb
