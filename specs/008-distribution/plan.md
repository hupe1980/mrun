# Implementation Plan: Distribution

**Branch**: `008-distribution` | **Date**: 2026-10-15 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/008-distribution/spec.md`

## Summary

Ship mrun 0.1 to Macs: a GitHub Actions release pipeline that builds the arm64 binary with the
vendored MLX Metal library, signs and notarizes it, publishes a tarball with checksums, changelog
and SBOM, and updates a Homebrew tap formula. Add `mrun service install|uninstall` (launchd user
agent running `serve --service` with rotating logs), an idle-sleep power assertion while requests
run, and a Prometheus `/metrics` endpoint with a fixed, documented metric set. No updater.

## Technical Context

**Language/Version**: Rust 1.94.0 (edition 2024); Ruby (Homebrew formula); GitHub Actions YAML
**Primary Dependencies**: existing axum router (002) for `/metrics`; `objc2-io-kit` or a minimal
`extern "C"` IOKit binding; `plist` crate for writing the launchd plist; CI tools `codesign`,
`xcrun notarytool`, `syspolicy_check`, `cargo-cyclonedx`, `promtool` (tests)
**Storage**: `~/Library/LaunchAgents/io.github.hupe1980.mrun.plist`; `~/.mrun/{server.log*,
launchd.log, server.pid}`
**Testing**: cargo test (plist rendering snapshots, metric exposition snapshots and `promtool check
metrics` when available, power-assertion state machine with a fake clock, service command dry-run
on Linux); release workflow dry run on a fork with test credentials; manual SC-001/002/004 on the
target Mac
**Target Platform**: macOS 15+ arm64 (release); Linux for portable tests and CI
**Project Type**: Packaging, CI, and small runtime additions to the `mrun` binary
**Performance Goals**: `/metrics` render < 1 ms; metric updates lock-free on the hot path; no
measurable decode impact (bench delta < 0.5%)
**Constraints**: No network except `pull`; no root; no privileged OS tuning (constitution)
**Scale/Scope**: One formula, one plist, ~40 metric series

## Constitution Check

| Principle | Gate | Status |
| --- | --- | --- |
| I Agent latency | Power assertion keeps long runs alive; `/metrics` exposes TTFT and throughput as measured by the server | Pass |
| II Exactness | Metrics report real counts from the runtime (no estimates) | Pass |
| III Memory budget | Service install states resident memory; no new large allocations | Pass |
| IV Local, private | No update check; `/metrics` local with existing bind/auth rules, no prompt-derived labels; logs without content | Pass |
| V Own runtime | No new runtime dependencies beyond a plist writer and IOKit binding | Pass |
| VI Simple, DX | One install command, one service command pair, no `brew services` duplicate, hand-written encoder | Pass |

Exception (scoped): `crates/mrun/src/platform/macos/power.rs` may use `unsafe` for the IOKit FFI if
`objc2-io-kit` lacks the calls; module-level allow with a safety comment; expires when a safe
binding exists.

Post-design re-check: Pass.

## Project Structure

### Documentation (this feature)

```text
specs/008-distribution/
├── plan.md  research.md  data-model.md  quickstart.md  tasks.md
├── contracts/metrics.md
├── contracts/service-and-release.md
└── checklists/requirements.md
```

### Source Code (repository root)

```text
crates/mrun/src/
├── cli/service.rs            # service install|uninstall; launchctl bootstrap/bootout
├── cli/version.rs            # long --version (commit, MLX/mlx-c, build date) via build.rs env
├── platform/
│   ├── mod.rs                # install source detection, real binary path
│   ├── launchd.rs            # plist model + rendering (plist crate), label, paths
│   └── macos/power.rs        # IOPM assertion holder (no-op elsewhere)
├── runtime/power.rs          # queue-driven assertion state machine (acquire, 60 s release)
├── metrics.rs                # registry: counters, gauges, fixed-bucket histograms, text encoder
├── api/metrics.rs            # GET /metrics handler
└── logging.rs                # service-mode rotating file log (16 MiB × 3, 0600)
crates/mrun/build.rs          # embed git commit, build date, vendored MLX/mlx-c versions
crates/mrun/tests/
├── metrics_exposition.rs     # snapshot + promtool (if installed) + label-set checks
├── service_plist.rs          # plist snapshots for Homebrew and tarball paths
└── power_state.rs            # fake-clock state machine
.github/workflows/
├── ci.yml                    # PRs: Linux fmt/clippy/test; macOS arm64 release build; changelog check
└── release.yml               # tags: version check, build, sign, notarize, publish, tap PR
packaging/homebrew/mrun.rb.tmpl   # formula template rendered by release.yml
scripts/release/{sign.sh, notarize.sh, package.sh, sbom.sh}
CHANGELOG.md  RELEASING.md
```

**Structure Decision**: Runtime pieces live in the `mrun` crate (`platform/` isolates macOS-only
code with Linux no-ops so tests run everywhere); packaging lives in `packaging/`, `scripts/release/`
and workflows; the tap repository holds only the rendered formula.

## Cross-spec changes (to fold into the owning specs)

- 002 API contract: `GET /metrics` row; `/mrun/status` adds `power: {assertion: bool, since?}`.
- 004 CLI contract: `mrun service install|uninstall`, `serve --service`, long `--version`;
  `doctor` checks install source, service state, stale plist; `stop` message with a service.
- 006/007 own their metric families (`mrun_spec_*`, `mrun_cache_*{tier="disk"}`) using the names
  reserved in the metrics contract.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
| --- | --- | --- |
| Possible `unsafe` FFI in `platform/macos/power.rs` | IOKit power assertions have no std API | Spawning `caffeinate` adds a process and imprecise release |
