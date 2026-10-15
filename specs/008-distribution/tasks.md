---
description: "Tasks for 008 Distribution"
---

# Tasks: Distribution

**Input**: [plan.md](plan.md), [spec.md](spec.md), [research.md](research.md), [data-model.md](data-model.md), [contracts/metrics.md](contracts/metrics.md), [contracts/service-and-release.md](contracts/service-and-release.md)

**Prerequisites**: 001 M0 spike (confirms macOS 14 floor and Metal library lookup), 002 router and
`/mrun/status`, 004 `launch`/`serve --detach`/`stop`/`doctor`. US1's SC-001 needs 001–004 complete.

## Phase 1: Setup

- [ ] T001 `crates/mrun/build.rs`: embed git commit, build date, vendored MLX/mlx-c versions; `crates/mrun/src/cli/version.rs` long `--version`; snapshot test
- [ ] T002 [P] `CHANGELOG.md` (Keep a Changelog, `Unreleased`) and `RELEASING.md` (steps per research R8)
- [ ] T003 [P] `.github/workflows/ci.yml`: Linux fmt/clippy/test; macOS arm64 release build with vendored MLX + Metal library; changelog check with `no-changelog` label (FR-015)

## Phase 2: Foundational

- [ ] T004 `crates/mrun/src/platform/mod.rs`: install-source detection (Homebrew Cellar/opt path, tarball, source build), canonical binary path, Metal library presence check with a clear startup error
- [ ] T005 [P] `crates/mrun/src/logging.rs`: service-mode rotating log (16 MiB × 3, 0600), no content (002 FR-014); tests

## Phase 3: US1 — install and start an agent (P1) 🎯 MVP

- [ ] T006 [US1] `scripts/release/{package.sh, sign.sh, notarize.sh, sbom.sh}`: tarball layout per data model, Developer ID signing with hardened runtime and timestamp, `notarytool submit --wait`, verification (`codesign --verify --strict`, `syspolicy_check distribution`), `SHA256SUMS`, CycloneDX SBOM + MLX/mlx-c components
- [ ] T007 [US1] `.github/workflows/release.yml`: version/changelog check, build (`MACOSX_DEPLOYMENT_TARGET=14.0`), temporary keychain from secrets, T006 scripts, GitHub Release (pre-release for `-rc.N`), tap pull request for final releases (FR-001, FR-002, FR-012–FR-014)
- [ ] T008 [US1] `packaging/homebrew/mrun.rb.tmpl` + `hupe1980/homebrew-tap` repository: arm64/Sonoma requirements, libexec install + `write_exec_script`, completions, caveats, `test do`; `brew audit --strict` in the tap CI (FR-003)
- [ ] T009 [US1] Dry run of the release workflow on a fork with test credentials (SC-005); fix findings
- [ ] T010 [US1] SC-001 on clean accounts on the target Mac (3 runs); record results in `RELEASING.md` checklist

## Phase 4: US2 — login service (P1)

- [ ] T011 [US2] `crates/mrun/src/platform/launchd.rs`: plist model and rendering (`plist` crate), Homebrew `opt` path vs canonical executable; `crates/mrun/tests/service_plist.rs` snapshots
- [ ] T012 [US2] `crates/mrun/src/cli/service.rs`: `install` (write 0644, bootout if loaded, bootstrap, wait for `/health`, summary output), `uninstall` (bootout, delete), refuse root, Linux message (FR-005, FR-006)
- [ ] T013 [US2] `serve --service` (no detach, rotating log, pid file, exit 0 on SIGTERM) in `cli/serve.rs`; `mrun stop` message with a service; `doctor` service state and stale plist checks (FR-007, FR-008)
- [ ] T014 [US2] SC-002 on the target Mac: 3 reboots, 5 `kill -9`; R5 foreground vs service decode comparison (set `ProcessType Interactive` only if > 2% slower)

## Phase 5: US3 — no idle sleep while working (P2)

- [ ] T015 [P] [US3] `crates/mrun/src/platform/macos/power.rs`: IOPM assertion create/release (`objc2-io-kit` or documented `extern "C"`); no-op on Linux
- [ ] T016 [US3] `crates/mrun/src/runtime/power.rs`: queue-driven state machine (acquire on queued/started, release 60 s after empty); `crates/mrun/tests/power_state.rs` with a fake clock; `/mrun/status` `power` field (FR-009)
- [ ] T017 [US3] SC-004 on the target Mac with a 1-minute idle-sleep setting

## Phase 6: US4 — metrics (P2)

- [ ] T018 [P] [US4] `crates/mrun/src/metrics.rs`: counters, gauges, fixed-bucket histograms over atomics; text encoder (format 0.0.4); unit tests
- [ ] T019 [US4] Instrument runtime (requests, TTFT, queue wait/depth, prefill/decode tokens and seconds), cache (lookups, hit tokens, misses by cause, evictions, bytes, branches, live tokens), memory (budget breakdown, state bytes, MLX counters, pressure events), power; `crates/mrun/src/api/metrics.rs` route with existing bind/auth rules (FR-010)
- [ ] T020 [US4] `crates/mrun/tests/metrics_exposition.rs`: every contract metric present with type and labels after one request per API on `ScriptedEngine`; label values from fixed sets; `promtool check metrics` when installed (SC-003); bench delta check (< 0.5%)

## Phase 7: US5 and polish

- [ ] T021 [US5] First release candidate `v0.1.0-rc.1` through the full workflow; verify artifacts, SBOM, checksums
- [ ] T022 [P] Site pages: install (Homebrew, tarball), service, metrics reference (generated from the contract table), upgrading with `brew upgrade mrun`
- [ ] T023 Run quickstart scenarios 1–6; record results

## Dependencies

T001 → T006/T007; T003 independent. T004 → T008/T011. T006 → T007 → T009 → T010 (also needs
001–004). T011 → T012 → T013 → T014. T015 → T016 → T017. T018 → T019 → T020. T021 after T007, T008,
T013, T016, T019.
