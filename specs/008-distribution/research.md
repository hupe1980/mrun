# Research: Distribution

## R1: Channel: tap with a prebuilt binary

- **Decision**: Tap `hupe1980/homebrew-tap`, formula `mrun.rb` downloading the notarized release
  tarball (`url` + `sha256`), `depends_on arch: :arm64`, `depends_on macos: :sonoma`. The real
  binary and `mlx.metallib` go to `libexec/`; `bin/mrun` is created with `write_exec_script`, so the
  process image path is `libexec/mrun` and MLX finds the colocated `mlx.metallib` (MLX looks next to
  the binary that contains it; a symlinked path in `bin/` would not resolve [U, verified in the
  M0 spike]). Completions via `generate_completions_from_executable(bin/"mrun", "completions")`.
- **Rationale**: homebrew-core requires a from-source build; MLX's Metal build needs full Xcode and
  takes long in Homebrew CI. oMLX and LM Studio ship through taps/casks [V landscape].
- **Alternatives**: Cask (for apps; mrun is a CLI); from-source formula (Xcode dependency, long
  builds on user machines); `curl | sh` installer (no upgrade path, weaker trust).

## R2: Platform floor

- **Decision**: macOS 15.0 deployment target, arm64 only, matching 001's build and test platform.
  MLX itself may run on macOS 14 [U]; the M0 binding spike decides whether to lower the floor.
- **Alternatives**: Universal binary (no Metal MLX on Intel; doubles size for nothing); macOS 13
  (older Metal and toolchain combinations, untested).

## R3: Signing and notarization

- **Decision**: `codesign --sign "Developer ID Application: …" --options runtime --timestamp`
  without extra entitlements (MLX compiles Metal source at runtime through the Metal API, which the
  hardened runtime allows; no CPU JIT). Notarize the zipped binary with `xcrun notarytool submit
  --wait` using an App Store Connect API key; bare Mach-O binaries cannot be stapled, so Gatekeeper
  checks the ticket online on first run of a quarantined download. Verify with `codesign --verify
  --strict --verbose=2`, `spctl`/`syspolicy_check distribution`. A notarized binary is smoke-tested
  by running `mrun --version` and `mrun run --check` with a tiny model on a macOS runner (CPU
  fallback if the runner exposes no GPU).
- **Alternatives**: `.pkg` installer (stapleable, but heavier and needs root for `/usr/local`);
  rejected for 0.1.

## R4: CI

- **Decision**: GitHub Actions. `ci.yml` (pull requests): Linux fmt/clippy/test; macOS arm64
  (`macos-15`) release build including the vendored MLX/mlx-c cmake build and Metal library
  (Xcode's Metal toolchain installed with `xcodebuild -downloadComponent MetalToolchain` when the
  selected Xcode needs it). `release.yml` (tags `v*`): version check (tag = workspace version =
  changelog section), build, sign, notarize, tarball + `SHA256SUMS` + SBOM (`cargo cyclonedx` plus
  a manual MLX/mlx-c component), GitHub Release with the changelog section, then a pull request to
  the tap with the new `url`/`sha256` (fine-grained token limited to the tap). Secrets: `.p12` +
  password into a temporary keychain, API key id/issuer/key.
- **Rationale**: Hosted macOS runners build Metal shaders but have no usable GPU for MLX
  performance; Metal correctness and speed stay on the target Mac (constitution workflow).

## R5: launchd user agent

- **Decision**: Label `io.github.hupe1980.mrun`; plist in `~/Library/LaunchAgents`, mode 0644
  (launchd requires it not be group/world writable). `ProgramArguments = [<real binary>, "serve",
  "--service"]` where `<real binary>` is `$(brew --prefix)/opt/mrun/libexec/mrun` for Homebrew
  installs (stable across upgrades) or the canonicalized current executable otherwise.
  `RunAtLoad true`, `KeepAlive {SuccessfulExit false}` (restart on crash or non-zero exit, not after
  a graceful `mrun stop`), `ThrottleInterval 10`, `ProcessType Standard` (default; `Background`
  would apply resource throttling), `StandardOutPath`/`StandardErrorPath` = `~/.mrun/launchd.log`,
  `EnvironmentVariables` only `MRUN_HOME` when set at install time. Load with `launchctl bootstrap
  gui/$UID <plist>`; unload with `bootout gui/$UID/<label>`; re-install = bootout + write +
  bootstrap. `--service` means: no detach, rotating log file, pid file, exit 0 on SIGTERM.
- **Measure**: quickstart compares decode tok/s under the service with a foreground server; if
  launchd's default QoS lowers it by > 2%, set `ProcessType Interactive`.
- **Alternatives**: `brew services` (Homebrew-only, plist owned by Homebrew, `doctor` cannot rely
  on it); `SMAppService` (needs an app bundle); socket activation (lazy model loading inside `serve --service` achieves the same without launchd
  sockets; decided 2026-10-15).

## R6: Sleep prevention

- **Decision**: `IOPMAssertionCreateWithName(kIOPMAssertionTypePreventUserIdleSystemSleep,
  kIOPMAssertionLevelOn, "mrun: inference in progress")` when the queue goes from empty to
  non-empty or a request starts; `IOPMAssertionRelease` 60 s after the queue empties (avoids
  flapping between agent turns, which are separated by tool runs). Bindings via `objc2-io-kit` if it
  exposes these functions, otherwise a 20-line `extern "C"` block in
  `crates/mrun/src/platform/macos/power.rs` with a module-level `#[allow(unsafe_code)]` and a
  documented reason (workspace lint is `deny`, not `forbid`). No-op on Linux.
- **Alternatives**: Spawning `caffeinate -i -w <pid>` (extra process, less precise release);
  preventing display sleep (not needed, wastes power).

## R7: `/metrics`

- **Decision**: Hand-written Prometheus text encoder over atomics and fixed-bucket histograms in
  `crates/mrun/src/metrics.rs` (~200 lines); served by the existing axum router. Names in the
  [contract](contracts/metrics.md), prefix `mrun_`, base units (seconds, bytes), `_total` for
  counters. Throughput is exported as token and second counters (rate = tokens/seconds) plus
  last-request gauges, not as averaged gauges.
- **Rationale**: Fixed, small metric set; no dependency or global registry; llama.cpp and oMLX
  expose comparable endpoints [V next-features §6].
- **Alternatives**: `prometheus-client` or `metrics` + exporter crates (more API than needed;
  exporter crates bring their own HTTP server).

## R8: Versioning and changelog

- **Decision**: SemVer with 0.x semantics (FR-012); one workspace version; tags `vX.Y.Z`; release
  candidates `vX.Y.Z-rc.N` publish to GitHub Releases as pre-releases and do not update the tap.
  `CHANGELOG.md` in Keep a Changelog format; every user-visible pull request adds an `Unreleased`
  entry (checked by a CI job that skips with a `no-changelog` label). Release steps documented in
  `RELEASING.md`: bump version, move `Unreleased` to the version, tag, watch the workflow, merge the
  tap pull request, verify SC-001 on a clean account.
- **Alternatives**: Conventional-commit generated changelogs (noisy, less useful for users).

## R9: Upgrades without an updater

- **Decision**: No update check (constitution IV, FR-011). Formula caveats and the site document
  `brew upgrade mrun` followed by `mrun service install` (restarts on the new binary). The prefix
  tier (007) is version-bound, so an upgrade invalidates persisted entries by design.
