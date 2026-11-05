#!/usr/bin/env bash
# Render the Homebrew formula for a release from its SHA256SUMS.
#
#   scripts/release/formula.sh 0.1.0 dist/SHA256SUMS > Formula/mrun.rb
set -euo pipefail
version="${1:?usage: formula.sh <version> <SHA256SUMS>}"
sums="${2:?usage: formula.sh <version> <SHA256SUMS>}"
template="$(cd "$(dirname "$0")/../.." && pwd)/packaging/homebrew/mrun.rb.in"
archive="mrun-$version-macos-arm64.tar.gz"
sum="$(awk -v a="$archive" '$2 == a || $2 == "*" a { print $1 }' "$sums")"
if ! [[ "$sum" =~ ^[0-9a-f]{64}$ ]]; then
  echo "formula: no checksum for $archive in $sums" >&2
  exit 1
fi
out="$(sed -e "s/@VERSION@/$version/g" -e "s/@SHA256_macos@/$sum/g" "$template")"
if grep -q '@[A-Z0-9_a-z-]*@' <<<"$out"; then
  echo "formula: a placeholder was left" >&2
  exit 1
fi
printf '%s\n' "$out"
