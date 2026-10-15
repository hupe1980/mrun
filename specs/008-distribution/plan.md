# Implementation Plan: Distribution

**Branch**: `008-distribution` | **Date**: 2026-10-15 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/008-distribution/spec.md`

## Summary

Ship mrun 0.1 to Macs (macOS 15+, arm64), lean (D025, 2026-10-15): a GitHub Actions release
pipeline that builds the arm64 binary with the vendored MLX Metal library, ad-hoc signs it,
publishes a tarball with checksums and changelog, and updates a Homebrew tap formula; a long
`--version`; an idle-sleep power assertion while requests run. No updater. Deferred (spec, Later):
Developer ID signing and notarization, SBOM, the launchd service with lazy model load, `/metrics`;
their design below and in research stays for that release.

## Technical Context

**Language/Version**: Rust 1.94.0 (edition 2024); Ruby (Homebrew formula); GitHub Actions YAML
**Primary Dependencies**: `objc2-io-kit` or a minimal `extern "C"` IOKit binding; CI tools
`codesign`; later: `plist`, `xcrun notarytool`, `syspolicy_check`, `cargo-cyclonedx`, `promtool`
**Storage**: none new (`serve --detach` logs per 004)
**Testing**: cargo test (power-assertion state machine with a fake clock, `--version` snapshot and
offline test); release workflow dry run on a fork; manual SC-001/004 on the target Mac
**Target Platform**: macOS 15+ arm64 (release); Linux for portable tests and CI
**Project Type**: Packaging, CI, and small runtime additions to the `mrun` binary
**Performance Goals**: none beyond not slowing the request path
**Constraints**: No network except `pull`; no root; no privileged OS tuning (constitution)
**Scale/Scope**: One formula, one release workflow, one CI job

## Constitution Check

| Principle | Gate | Status |
| --- | --- | --- |
| I Agent latency | Power assertion keeps long runs alive | Pass |
| II Exactness | Not affected | Pass |
| III Memory budget | No new large allocations | Pass |
| IV Local, private | No update check; nothing new listens | Pass |
| V Own runtime | No new runtime dependency beyond an IOKit binding | Pass |
| VI Simple, DX | One install command; ad-hoc signing needs no certificate | Pass |

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
├── cli/version.rs            # long --version (commit, MLX/mlx-c, build date) via build.rs env
├── platform/
│   ├── mod.rs                # real binary path, Metal library check, arm64/macOS 15 floor
│   └── macos/power.rs        # IOPM assertion holder (no-op elsewhere)
└── runtime/power.rs          # queue-driven assertion state machine (acquire, 60 s release)
crates/mrun/build.rs          # embed git commit, build date, vendored MLX/mlx-c versions
crates/mrun/tests/
├── version_offline.rs        # --version opens no socket
└── power_state.rs            # fake-clock state machine
.github/workflows/
├── ci.yml                    # owned by 001 T004a; T003 adds the macOS arm64 release build job
└── release.yml               # tags: version check, build, ad-hoc sign, verify, publish, tap PR
packaging/homebrew/mrun.rb.tmpl   # formula template rendered by release.yml
scripts/release/{package.sh, sign.sh}
CHANGELOG.md  RELEASING.md
```

Later (service, metrics, notarization): `cli/service.rs`, `runtime/lazy.rs`,
`platform/launchd.rs`, `metrics.rs`, `api/metrics.rs`, `logging.rs`, `scripts/release/{notarize.sh,
sbom.sh}` and their tests.

**Structure Decision**: Runtime pieces live in the `mrun` crate (`platform/` isolates macOS-only
code with Linux no-ops so tests run everywhere); packaging lives in `packaging/`, `scripts/release/`
and workflows; the tap repository holds only the rendered formula.

## Cross-spec changes (to fold into the owning specs)

- 002 API contract: `/mrun/status` adds `power: {assertion: bool, since?}`.
- 004 CLI contract: long `--version`.
- 001: macOS 15.0 deployment target; `ci.yml` stays 001's (T004a).
- Later: `GET /metrics`, `/health` `loaded` and estimated capacity before a lazy load,
  `mrun service install|uninstall [--preload]`, `serve --service`, `doctor` install source and
  service state; 006/007 metric families.
- Folded by T024.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
| --- | --- | --- |
| Possible `unsafe` FFI in `platform/macos/power.rs` | IOKit power assertions have no std API | Spawning `caffeinate` adds a process and imprecise release |
