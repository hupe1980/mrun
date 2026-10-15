# Feature Specification: Distribution

**Feature Branch**: `008-distribution`

**Created**: 2026-10-15

**Status**: Draft (release 0.1, milestone M6; lean 0.1 cut 2026-10-15, D025)

**Input**: User description: "Homebrew formula (tap first), signed + notarized arm64-only binary built in CI on macOS, `mrun service install|uninstall` (launchd user agent, KeepAlive, logs in ~/.mrun), prevent idle sleep while requests run (IOPMAssertion), Prometheus `/metrics` (TTFT, prefill/decode tok/s, cache hit tokens/queries, evictions, queue depth, memory budget), no `mrun update` (no network beyond pull) — document brew upgrade; release process (versioning, changelog), SBOM optional. SC: fresh Mac `brew install` → `mrun launch claude-code` works; service survives reboot; /metrics scrapeable with documented names."

## Clarifications

### Session 2026-10-15

- Q: Which install channels? → A: A Homebrew tap (`hupe1980/homebrew-tap`, `brew install hupe1980/tap/mrun`) installing the release tarball, and the same tarball on GitHub Releases. homebrew-core is later (it requires building MLX from source in Homebrew's CI).
- Q: Which platforms are released? → A: macOS 15 (Sequoia) or later on Apple Silicon (arm64) only (matching 001). Linux builds stay test-only.
- Q: Does mrun update itself? → A: No. There is no update check and no network use beyond `mrun pull`; `brew upgrade mrun` is documented.
- Q: Service, lazy load, `/metrics`, SBOM? → A: *Deferred by the lean cut (below).* The earlier decisions (launchd user agent with lazy model load and `--preload`, D022; `/metrics` on by default under the bind rules; a CycloneDX SBOM as SHOULD) stay recorded in Later.

### Session 2026-10-15 (lean 0.1 cut, D025)

- Q: What ships in 0.1? → A: The Homebrew tap and a CI-built tarball whose binary is ad-hoc signed (`codesign -s -`, which arm64 executables need), `mrun --version`, and the idle-sleep assertion while requests run. Release versioning, changelog and `SHA256SUMS` stay.
- Q: What is deferred? → A: Developer ID signing and notarization, the SBOM, the launchd service with lazy model load (D022), and `/metrics`.
- Q: Gatekeeper without notarization? → A: Homebrew does not quarantine formula downloads, so `brew install` runs without a prompt [U: confirmed by T010]. A tarball downloaded in a browser is quarantined; the install page documents `xattr -d com.apple.quarantine` for it.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Install and start an agent in one go (Priority: P1)

A developer on a fresh Apple Silicon Mac with Homebrew and Claude Code installed runs `brew install
hupe1980/tap/mrun` and then `mrun launch claude-code`. mrun offers the default model, downloads it,
starts the server, and Claude Code opens connected to it. No Gatekeeper warning appears.

**Why this priority**: This is the release's first-run promise (M6 gate).

**Independent Test**: On a clean macOS user account, run the two commands, accept the offered pull,
and complete a scripted read-edit-test task in Claude Code.

**Acceptance Scenarios**:

1. **Given** a Mac with Homebrew, **When** the developer runs `brew install hupe1980/tap/mrun`,
   **Then** `mrun --version` prints version, commit, MLX and mlx-c versions and build date, without
   a Gatekeeper prompt.
2. **Given** a fresh install, **When** the developer runs `mrun launch claude-code`, **Then** it
   offers the default model with size and fit, pulls it, starts the server and opens Claude Code
   configured for this process (004 FR-005).
3. **Given** a release tarball downloaded in a browser, **When** the developer follows the install
   page (`xattr -d com.apple.quarantine`), **Then** the binary runs.

---

### User Story 2 - Run mrun as a login service (Priority: P1) — Removed

*Removed (0.1 scope cut, 2026-10-15; see Later).* `mrun serve --detach` (004 FR-009) is the 0.1
background mode.

---

### User Story 3 - Long agent runs are not interrupted by idle sleep (Priority: P2)

While mrun is processing or has queued requests, the Mac does not go to idle sleep, so long
prefills and unattended agent loops finish.

**Why this priority**: Idle sleep during a multi-minute prefill aborts agent runs; the fix is
small.

**Independent Test**: Set a 1-minute idle-sleep timer, run a request that lasts 3 minutes, and
check `pmset -g assertions` and that the request completes.

**Acceptance Scenarios**:

1. **Given** a running or queued request, **When** the system idle timer expires, **Then** the Mac
   stays awake; display sleep is not prevented.
2. **Given** the queue becomes empty, **When** 60 s pass without a new request, **Then** the
   assertion is released.

---

### User Story 4 - Observe mrun with standard tools (Priority: P2) — Removed

*Removed (0.1 scope cut, 2026-10-15; see Later).* `mrun ps` and `/mrun/status` are the 0.1
observation surface.

---

### User Story 5 - Predictable releases (Priority: P3)

Maintainers cut a release by pushing a version tag; CI builds, ad-hoc signs, publishes artifacts
with checksums and changelog, and updates the tap. Users upgrade with `brew upgrade mrun`.

**Why this priority**: Repeatable, auditable releases; needed before 0.1 ships but invisible to
users when it works.

**Independent Test**: Push a release-candidate tag in a fork; verify artifacts, signature,
checksums and the tap pull request.

**Acceptance Scenarios**:

1. **Given** a `vX.Y.Z` tag on a commit with green CI and a changelog section for X.Y.Z, **When**
   the release workflow runs, **Then** it publishes the tarball and SHA-256 file and opens a tap
   update with the new URL and hash.
2. **Given** signing verification or the version check fails, **When** the workflow runs, **Then**
   nothing is published.

### Edge Cases

- Intel Mac or macOS older than 15: the formula refuses to install with the requirement stated; the
  tarball's binary exits with the same message.
- `brew upgrade` while a detached server runs: it keeps the old binary until `mrun stop` and a new
  `mrun serve --detach`; the formula caveats say so.
- Lid closed on battery: the assertion does not prevent clamshell sleep; documented.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Releases MUST be built in CI on macOS arm64 runners from a tagged commit, for
  `aarch64-apple-darwin` with a minimum macOS of 15.0 (`MACOSX_DEPLOYMENT_TARGET=15.0`), and ship as
  `mrun-X.Y.Z-macos-arm64.tar.gz` containing the binary, the MLX Metal library, licenses, notices
  and shell completions.
- **FR-002**: The binary MUST be ad-hoc signed (`codesign -s -`) and pass `codesign --verify
  --strict`; a failed verification blocks the release. *Removed (0.1 scope cut, 2026-10-15; see
  Later): Developer ID Application signature, hardened runtime, secure timestamp and notarization.*
- **FR-003**: A Homebrew tap formula MUST install the release tarball (verified by SHA-256),
  declare `depends_on arch: :arm64` and `depends_on macos: :sequoia`, install the binary so the Metal
  library resolves from the real binary location, install completions, and print caveats for
  `brew upgrade mrun` (restart a running server afterwards).
- **FR-004**: `mrun --version` MUST print version, git commit, vendored MLX and mlx-c versions and
  build date, with no network access (tested). *Removed (0.1 scope cut, 2026-10-15): the install
  source in `mrun doctor`.*
- **FR-005**: *Removed (0.1 scope cut, 2026-10-15; see Later)*: `mrun service install` (launchd user
  agent running `serve --service`, lazy model load, `--preload`; D022).
- **FR-006**: *Removed (0.1 scope cut, 2026-10-15; see Later)*: `mrun service uninstall`.
- **FR-007**: *Removed (0.1 scope cut, 2026-10-15; see Later)*: service-mode rotating log and pid
  file (`serve --detach`, 004 FR-009, logs to `~/.mrun/server.log`).
- **FR-008**: *Removed (0.1 scope cut, 2026-10-15; see Later)*: `mrun stop` and `doctor` service
  states.
- **FR-009**: While any request is running or queued, mrun MUST hold an IOKit power assertion that
  prevents idle system sleep (not display sleep), named "mrun: inference in progress", and release
  it 60 s after the queue empties; the assertion state appears in `/mrun/status`.
- **FR-010**: *Removed (0.1 scope cut, 2026-10-15; see Later)*: `GET /metrics` (Prometheus text
  format, [metrics contract](contracts/metrics.md)).
- **FR-011**: mrun MUST NOT check for updates or contact any host except during `mrun pull` (and a
  pull the user confirms inside `launch`); upgrade instructions are documentation only.
- **FR-012**: Versions MUST follow SemVer (`0.y.z` during 0.x: minor releases may break CLI or
  config, patch releases may not); the workspace version, tag `vX.Y.Z`, `--version` and the
  changelog MUST agree, and CI MUST fail a tag that disagrees.
- **FR-013**: `CHANGELOG.md` MUST follow Keep a Changelog sections (Added, Changed, Deprecated,
  Removed, Fixed, Security) with an `Unreleased` section; the release notes are that version's
  section.
- **FR-014**: Each release MUST publish `SHA256SUMS`. *Removed (0.1 scope cut, 2026-10-15; see
  Later): the CycloneDX SBOM.*
- **FR-015**: CI on every pull request (owned by 001 T004a: Linux fmt, clippy and tests, macOS
  build) MUST also build the release configuration on macOS arm64 (constitution workflow).
- **FR-016**: *Removed (0.1 scope cut, 2026-10-15; see Later)*: capacity and metrics before a lazy
  model load.

### Key Entities

- **Release Artifact**: Tarball, contents, checksum, ad-hoc signature.
- **Power Assertion**: Type, name, held since, release deadline.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: On a clean macOS 15+ user account on the target Mac with Homebrew and Claude Code
  installed, `brew install hupe1980/tap/mrun` followed by `mrun launch claude-code` meets 004 SC-001
  (≤ 2 confirmations, no other command) and the session completes the 002 SC-001 scripted task,
  with no Gatekeeper prompt, in 3 of 3 runs.
- **SC-002**: *Removed (0.1 scope cut, 2026-10-15)*: service across reboots and crashes.
- **SC-003**: *Removed (0.1 scope cut, 2026-10-15)*: `/metrics` exposition.
- **SC-004**: With a 1-minute idle-sleep setting, a 3-minute request completes in 3 of 3 runs and
  `pmset -g assertions` shows the assertion only while requests are running or queued (+ 60 s).
- **SC-005**: A release tag produces an ad-hoc-signed tarball whose `codesign --verify --strict`
  check passes, with checksums matching and a tap pull request opened, in one unattended workflow
  run.

## Assumptions

- No Apple Developer ID certificate is needed for 0.1.
- The GitHub repository `hupe1980/mrun` publishes releases; the tap repository
  `hupe1980/homebrew-tap` exists and accepts automated pull requests.
- `mrun launch`, `connect`, `serve --detach`, `stop` and `doctor` come from 004; this feature adds
  packaging and the power assertion.
- CI macOS runners can build Metal shaders but are not used for Metal performance tests (those run
  on the target Mac with `mrun bench`).

## Later

Moved out of 0.1 on 2026-10-15 (D025); IDs kept, design in [plan](plan.md), [research](research.md)
and the [contracts](contracts/):

- Developer ID signing with hardened runtime and timestamp, notarization, `syspolicy_check
  distribution` (FR-002, SC-005 extension); then the browser-downloaded tarball runs without a
  prompt.
- `mrun service install [--preload]` / `uninstall` as a launchd user agent (`RunAtLoad`,
  `KeepAlive {SuccessfulExit: false}`, `ThrottleInterval 10`), `serve --service` with rotating
  logs, lazy model load on the first request (D022), `/health` `loaded`, estimated capacity before
  load, `doctor` service states (US2, FR-005–FR-008, FR-016, SC-002).
- `GET /metrics` per the [metrics contract](contracts/metrics.md) (US4, FR-010, SC-003).
- CycloneDX SBOM (FR-014); install source in `doctor` (FR-004).
