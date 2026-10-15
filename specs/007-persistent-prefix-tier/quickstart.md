# Quickstart: Persistent Prefix Tier

Validation scenarios for feature 007 (after implementation), on the target Mac.

## Prerequisites

`mrun` on `PATH`, `qwen3.6` and `qwen3-coder` pulled, bundled traces, a local `--record` directory
from real use for scenario 1.

## Scenarios

1. **Measure first** (FR-001):
   ```sh
   mrun bench --suite persist                       # read/write GB/s, 30K restore vs re-prefill
   mrun bench --suite persist --from ~/mrun-rec     # restart vs eviction attribution
   ```
2. **Restart restore** (SC-001):
   ```sh
   mrun serve --persist --detach
   curl -s localhost:8484/v1/chat/completions -d @req30k.json >/dev/null  # a 30K-token request
   sleep 3                                           # idle: live path written
   mrun stop && mrun serve --persist --detach
   curl -si localhost:8484/v1/chat/completions -d @req30k.json | grep x-mrun-cache
   # expect x-mrun-cache-cached ≈ 30K, x-mrun-cache-source: disk
   ```
   Compare TTFT with the same steps without `--persist` (ratio ≥ 20×); `req30k.json` is the last
   request of `agent/opencode-refactor` with `max_tokens: 1`. The automated cell is
   `persist/restore-30k`.
3. **Identity** (SC-002): restart with `--model qwen3-coder` → no disk hits (`source: none`, cause
   `new`); edit the template override → no hits; downgrade/upgrade mrun → no hits.
4. **Corruption**: flip a byte in a `.mrps` file → next match reports `disk-corrupt`, file gone,
   answer correct.
5. **Control**: `mrun cache` shows usage per model; `mrun cache clear --disk --model qwen3-coder`;
   `mrun stop; mrun cache clear --all` works without a server.
6. **Permissions**: `ls -la ~/Library/Caches/mrun/prefix` → `drwx------`, files `-rw-------`;
   `chmod 755` it and restart → persistence off with a `doctor` fix.

Automated equivalents: `cargo test -p mrun --test persist_format --test persist_crash --test
persist_identity --test persist_cap` (any OS).
