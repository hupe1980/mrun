---
description: "Tasks for 008 Distribution"
---

# Tasks: Distribution

**Input**: [plan.md](plan.md), [spec.md](spec.md), [research.md](research.md), [data-model.md](data-model.md), [contracts/service-and-release.md](contracts/service-and-release.md)

**Scope**: Lean 0.1 (D025, 2026-10-15). Cut tasks keep their IDs and say so.

**Prerequisites**: 001 M0 spike (confirms macOS 15 floor and Metal library lookup) and 001 T004a
(`ci.yml`), 002 router and `/mrun/status`, 004 `launch`/`serve --detach`/`stop`. US1's SC-001
needs 001–004 complete.

## Phase 1: Setup

- [X] T001 `crates/mrun/build.rs`: embed git commit, build date, vendored MLX/mlx-c versions; `crates/mrun/src/cli/version.rs` long `--version`; snapshot test *Note*: `crates/mrun/build.rs` (commit from `MRUN_BUILD_COMMIT` or git, `SOURCE_DATE_EPOCH`-aware date, MLX pins checked against `mlx-sys` by a unit test) and `cli/version.rs`; unit tests instead of a snapshot.
- [X] T002 [P] `CHANGELOG.md` (Keep a Changelog, `Unreleased`) and `RELEASING.md` (steps per research R8) *Note*: `CHANGELOG.md`, `RELEASING.md` (with the release checklist).
- [X] T003 [P] Add one job to `.github/workflows/ci.yml` (owned by 001 T004a): macOS arm64 release-configuration build with `MACOSX_DEPLOYMENT_TARGET=15.0`, vendored MLX and Metal library (FR-015). Nothing else in `ci.yml` changes here *Note*: job `macos-release` (release build with `--features mlx`, deployment target 15.0, `--version` smoke test).

## Phase 2: Foundational

- [ ] T004 `crates/mrun/src/platform/mod.rs`: canonical binary path, Metal library presence check with a clear startup error; startup check that the machine is arm64 with macOS ≥ 15, exiting with the formula's requirement message otherwise (edge case). (Install-source detection: removed, 0.1 scope cut, 2026-10-15)
- [ ] T005 *Removed (0.1 scope cut, 2026-10-15)*: service-mode rotating log (`serve --detach` logs per 004 FR-009)

## Phase 3: US1 — install and start an agent (P1) 🎯 MVP

- [X] T006 [US1] `scripts/release/{package.sh, sign.sh}`: tarball layout per data model, ad-hoc signing (`codesign -s -`), verification (`codesign --verify --strict`), `SHA256SUMS`. (Developer ID, notarization, `syspolicy_check`, SBOM: removed, 0.1 scope cut, 2026-10-15) *Note*: `scripts/release/{package.sh,sign.sh}`; a `NOTICE` file was added for the tarball layout.
- [X] T007 [US1] `.github/workflows/release.yml`: version/changelog check, build (`MACOSX_DEPLOYMENT_TARGET=15.0`), T006 scripts, GitHub Release (pre-release for `-rc.N`), tap pull request for final releases (FR-001, FR-002, FR-012–FR-014) *Note*: `.github/workflows/release.yml`; the tap pull request needs a `TAP_TOKEN` secret. Unverified until the T009 dry run (for example where the cross-target build leaves `mlx.metallib`).
- [X] T008 [US1] `packaging/homebrew/mrun.rb.tmpl` + `hupe1980/homebrew-tap` repository: arm64/Sequoia requirements (`depends_on macos: :sequoia`), libexec install + `write_exec_script`, completions, caveats (`brew upgrade`, restart a running server), `test do`; `brew audit --strict` in the tap CI (FR-003) *Note*: `packaging/homebrew/mrun.rb.tmpl`; creating the `hupe1980/homebrew-tap` repository and its CI is outside this repository.
- [ ] T009 [US1] Dry run of the release workflow on a fork (SC-005); fix findings
- [ ] T010 [US1] SC-001 on clean accounts on the target Mac (3 runs), including the check that a Homebrew install shows no Gatekeeper prompt; document the `xattr` step for browser-downloaded tarballs; record results in the `RELEASING.md` checklist

## Phase 4: US2 — login service (P1) — Removed

- [ ] T011 *Removed (0.1 scope cut, 2026-10-15; Later)*: `platform/launchd.rs` plist model and rendering
- [ ] T012 *Removed (0.1 scope cut, 2026-10-15; Later)*: `cli/service.rs` install/uninstall
- [ ] T013 *Removed (0.1 scope cut, 2026-10-15; Later)*: `serve --service`, service-aware `stop` and `doctor`
- [ ] T014 *Removed (0.1 scope cut, 2026-10-15; Later)*: SC-002 reboots and crash recovery

## Phase 5: US3 — no idle sleep while working (P2)

- [X] T015 [P] [US3] `crates/mrun/src/platform/macos/power.rs`: IOPM assertion create/release (`objc2-io-kit` or documented `extern "C"`); no-op on Linux *Note*: `runtime/power.rs` `Caffeinate` holds a `caffeinate -i -w <pid>` child (the same IOPM idle-sleep assertion, no unsafe FFI; it ends with mrun); `NoAssertion` elsewhere.
- [X] T016 [US3] `crates/mrun/src/runtime/power.rs`: queue-driven state machine (acquire on queued/started, release 60 s after empty); `crates/mrun/tests/power_state.rs` with a fake clock; `/mrun/status` `power` field (FR-009) *Note*: `PowerState` (acquire on queued/running, release 60 s after empty) with a fake-clock unit test in `runtime/power.rs`; serve polls the queue each second; `/mrun/status` `power`.
- [ ] T017 [US3] SC-004 on the target Mac with a 1-minute idle-sleep setting

## Phase 6: US4 — metrics (P2) — Removed

- [ ] T018 *Removed (0.1 scope cut, 2026-10-15; Later)*: `metrics.rs` registry and encoder
- [ ] T019 *Removed (0.1 scope cut, 2026-10-15; Later)*: instrumentation and `/metrics` route
- [ ] T020 *Removed (0.1 scope cut, 2026-10-15; Later)*: exposition tests

## Phase 7: US5 and polish

- [ ] T021 [US5] First release candidate `v0.1.0-rc.1` through the full workflow; verify artifacts, signature, checksums
- [ ] T022 [P] Site pages: install (Homebrew, tarball with the `xattr` step), upgrading with `brew upgrade mrun`
- [ ] T023 Run quickstart scenarios; record results
- [ ] T024 Fold cross-spec changes into the 001–005 contracts and specs/README index (plan "Cross-spec changes")
- [ ] T025 *Removed (0.1 scope cut, 2026-10-15; Later)*: `runtime/lazy.rs` lazy engine creation (D022)
- [X] T026 [P] `crates/mrun/tests/version_offline.rs`: `--version` with networking blocked (no socket opened) (FR-004, FR-011) *Note*: `tests/version_offline.rs` (runs under `unshare -rn` where permitted).

## Dependencies

T001 → T006/T007; T003 after 001 T004a. T004 → T008/T026. T006 → T007 → T009 → T010 (also needs
001–004). T015 → T016 → T017. T021 after T007, T008, T016. T024 any time.
