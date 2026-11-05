#!/usr/bin/env bash
# Tests of the release tooling (008 SC-007, T036): the archive layout and checksum, and the
# Homebrew formula rendered for the one archive.
# Usage: scripts/release/test.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

work="$(mktemp -d)"
trap 'rm -rf "${work}"' EXIT
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
