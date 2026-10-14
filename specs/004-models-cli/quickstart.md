# Quickstart: Models and CLI Experience

Validation scenarios for feature 004 (after implementation).

## Prerequisites

Apple Silicon Mac, `mrun` on `PATH`, network for `pull`.

## Scenarios

1. **Fresh setup** (SC-001):
   ```sh
   mrun doctor              # expect: all ok or actionable warnings
   mrun pull qwen3-coder    # expect: size + license shown, progress, "verified"
   mrun serve --detach      # expect: URL http://127.0.0.1:8484 and a launch hint
   mrun launch claude-code  # Claude Code starts against mrun; run a task
   mrun connect codex --write && mrun connect codex --undo   # diff, backup, restore
   ```
2. **Resume** (SC-003): interrupt `mrun pull` mid-download (Ctrl-C), rerun, expect no re-download
   of completed bytes (`--json` reports bytes fetched).
3. **Corruption**: truncate a weight blob → `mrun ls` shows `corrupt`; flip a byte keeping size and
   mtime → `mrun verify qwen3-coder` reports it; `mrun pull` repairs.
4. **Doctor scenarios** (SC-002): enable Low Power Mode, use a Codex config without
   `model_context_window`, stop the server; `mrun doctor` reports each with a fix (exit 8).
5. **Offline**: disable network; `ls`, `show`, `config`, `doctor`, `connect`, `launch`, `serve` all
   succeed.

Automated equivalents run in `cargo test -p mrun --test hub_pull --test cli_snapshots --test
connect_write --test offline` with a mocked hub and temporary home directories.
