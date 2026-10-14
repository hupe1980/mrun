# Quickstart: Models and CLI Experience

Validation scenarios for feature 004 (after implementation).

## Prerequisites

Apple Silicon Mac, `mrun` on `PATH`, network for `pull`.

## Scenarios

1. **Fresh setup** (SC-001):
   ```sh
   mrun doctor              # expect: all ok or actionable warnings
   mrun pull qwen3-coder    # expect: size + license shown, progress, "verified"
   mrun serve &             # expect: URL http://127.0.0.1:8484 and connect hint
   mrun connect claude-code --write   # expect: diff, backup path, success
   claude                   # run a task; expect it to work against mrun
   ```
2. **Resume** (SC-003): interrupt `mrun pull` mid-download (Ctrl-C), rerun, expect no re-download
   of completed bytes (`--json` reports bytes fetched).
3. **Corruption**: flip a byte in a weight blob, run `mrun ls` → status `corrupt`; `mrun pull` →
   repaired.
4. **Doctor scenarios** (SC-002): enable Low Power Mode, remove `CLAUDE_CODE_ATTRIBUTION_HEADER`,
   stop the server; `mrun doctor` reports each with a fix.
5. **Offline**: disable network; `ls`, `show`, `config`, `doctor`, `connect` all succeed.

Automated equivalents run in `cargo test -p mrun --test hub_pull --test cli_snapshots --test
connect_write` with a mocked hub and temporary home directories.
