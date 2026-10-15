# Quickstart: Distribution

Validation scenarios for feature 008 (after implementation), on the target Mac.

## Prerequisites

A clean macOS 15+ user account with Homebrew and Claude Code installed; a published release.

## Scenarios

1. **Install to agent** (SC-001):
   ```sh
   brew install hupe1980/tap/mrun
   mrun --version                  # version, commit, MLX version
   mrun launch claude-code         # accept the offered pull and server start; run the scripted task
   ```
   Expect no Gatekeeper prompt and ≤ 2 confirmations. Repeat 3× on fresh accounts.
2. **Signature** (SC-005): `codesign --verify --strict --verbose=2 $(brew --prefix)/opt/mrun/libexec/mrun`
   passes. Download the tarball in Safari, unpack, run `xattr -d com.apple.quarantine mrun` as the
   install page says, then `./mrun --version`.
3. **Sleep** (SC-004): System Settings → idle sleep 1 min; run a 3-minute request; `pmset -g
   assertions | grep mrun` shows the assertion during the run and ≤ 60 s after.
4. **Upgrade**: `brew upgrade mrun`, `mrun stop`, `mrun serve --detach`; `mrun --version` shows the
   new version; no network used by mrun itself (Little Snitch or `nettop` during `--version`, `ls`,
   `ps`).
5. **Platform floor**: on macOS 14, `brew install` refuses with the macOS 15 requirement.

Automated equivalents: `cargo test -p mrun --test power_state --test version_offline` (any OS);
the release workflow dry run on a fork.
