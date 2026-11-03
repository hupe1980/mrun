#!/usr/bin/env bash
# Ad-hoc signs the binary and verifies it (Developer ID and notarization are Later, D025).
# Usage: scripts/release/sign.sh <path-to-mrun>
set -euo pipefail

binary="$1"
codesign --force --sign - "${binary}"
codesign --verify --strict --verbose=2 "${binary}"
"${binary}" --version
