# Quickstart: Models and CLI Experience

Validation scenarios for feature 004 (after implementation).

## Prerequisites

Apple Silicon Mac, `mrun` on `PATH`, network for `pull`.

## Scenarios

1. **Fresh setup** (SC-001): with `MRUN_HOME` pointing to an empty directory,
   ```sh
   mrun launch claude-code  # offers the default model (confirm), offers to start the server (confirm)
   ```
   → Claude Code opens against mrun after ≤ 2 confirmations; no other command was needed.
2. **Step by step**:
   ```sh
   mrun doctor              # expect: all ok or actionable warnings
   mrun pull qwen3.6        # expect: size + license shown, progress, "verified"
   mrun serve --detach      # expect: URL http://127.0.0.1:8484 and a launch hint
   mrun connect codex       # prints the profile and its path; no file changes
   ```
3. **Resume** (SC-003): interrupt `mrun pull` mid-download (Ctrl-C), rerun, expect no re-download
   of completed bytes (`--json` reports bytes fetched).
4. **Corruption**: truncate a weight blob → `mrun ls` shows `corrupt`; flip a byte keeping size and
   mtime → `mrun verify qwen3.6` reports it; `mrun pull` repairs.
5. **Doctor scenarios** (SC-002): enable Low Power Mode, occupy port 8484, stop the server, export
   `ANTHROPIC_API_KEY`; `mrun doctor` reports each with a fix (exit 8).
6. **Offline**: disable network; `ls`, `show`, `doctor`, `connect`, `launch`, `serve` all succeed.

Automated equivalents run in `cargo test -p mrun --test hub_pull --test cli_snapshots --test
offline` with a mocked hub and temporary home directories.
