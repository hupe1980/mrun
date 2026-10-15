# Quickstart: Distribution

Validation scenarios for feature 008 (after implementation), on the target Mac.

## Prerequisites

A clean macOS 15+ user account with Homebrew and Claude Code installed; a published release.

## Scenarios

1. **Install to agent** (SC-001):
   ```sh
   brew install hupe1980/tap/mrun
   mrun --version                  # version, commit, MLX version
   mrun launch claude-code         # accept the offered pull; run the scripted task
   ```
   Expect no Gatekeeper prompt. Repeat 3× on fresh accounts.
2. **Signature** (SC-005): `codesign --verify --strict --verbose=2 $(brew --prefix)/opt/mrun/libexec/mrun`
   and `syspolicy_check distribution <same path>` pass; download the tarball in Safari, unpack, run
   `./mrun --version` without a prompt.
3. **Service** (SC-002):
   ```sh
   mrun service install            # label, plist, log path, model, expected memory
   sudo reboot                     # log in again
   curl -s localhost:8484/health   # ok within 10 s, loaded: false
   curl -s localhost:8484/metrics | grep mrun_model_loaded   # 0; no model/budget series
   mrun launch claude-code         # capacity shown as estimated (or from the cached probe)
   # first request loads the model; /health then reports loaded: true
   kill -9 $(cat ~/.mrun/server.pid); sleep 20; curl -s localhost:8484/health
   mrun stop                       # stays stopped; message about next login
   mrun service uninstall          # plist gone; ~/.mrun kept
   ```
   Repeat with `mrun service install --preload`: `/health` reports `loaded: true` after login.
   Also compare decode tok/s under the service with a foreground `mrun serve` (R5 measurement).
4. **Sleep** (SC-004): System Settings → idle sleep 1 min; run a 3-minute request; `pmset -g
   assertions | grep mrun` shows the assertion during the run and ≤ 60 s after.
5. **Metrics** (SC-003): `curl -s localhost:8484/metrics | promtool check metrics`; send one
   request per API and check that every contract metric is present.
6. **Upgrade**: `brew upgrade mrun && mrun service install`; `mrun --version` shows the new
   version; no network used by mrun itself (Little Snitch or `nettop` during `doctor`, `ls`, `ps`);
   `mrun doctor` names the install source (Homebrew).
7. **Platform floor**: on macOS 14, `brew install` refuses with the macOS 15 requirement.

Automated equivalents: `cargo test -p mrun --test metrics_exposition --test service_plist --test
power_state --test lazy_load --test doctor_offline` (any OS); the release workflow dry run on a
fork.
