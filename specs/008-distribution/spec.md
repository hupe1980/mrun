# Feature Specification: Distribution

**Feature Branch**: `008-distribution`

**Created**: 2026-10-15

**Status**: Draft (release 0.1, milestone M6)

**Input**: User description: "Homebrew formula (tap first), signed + notarized arm64-only binary built in CI on macOS, `mrun service install|uninstall` (launchd user agent, KeepAlive, logs in ~/.mrun), prevent idle sleep while requests run (IOPMAssertion), Prometheus `/metrics` (TTFT, prefill/decode tok/s, cache hit tokens/queries, evictions, queue depth, memory budget), no `mrun update` (no network beyond pull) — document brew upgrade; release process (versioning, changelog), SBOM optional. SC: fresh Mac `brew install` → `mrun launch claude-code` works; service survives reboot; /metrics scrapeable with documented names."

## Clarifications

### Session 2026-10-15

- Q: Which install channels? → A: A Homebrew tap (`hupe1980/homebrew-tap`, `brew install hupe1980/tap/mrun`) installing the release tarball, and the same tarball on GitHub Releases. homebrew-core is later (it requires building MLX from source in Homebrew's CI).
- Q: Which platforms are released? → A: macOS 14 or later on Apple Silicon (arm64) only. Linux builds stay test-only.
- Q: Does mrun update itself? → A: No. There is no update check and no network use beyond `mrun pull`; `brew upgrade mrun` is documented, and `mrun doctor` prints the installed version only.
- Q: How do service and Homebrew relate? → A: `mrun service install` is the only service mechanism; the formula defines no `brew services` block, and its caveats point to `mrun service install`.
- Q: Does the service load the model at login? → A: Yes; the model stays resident (D020) and the install output states the memory it will hold. A lazy mode that loads on the first request is not part of this feature. (Maintainer decision pending.)
- Q: Is `/metrics` on by default? → A: Yes. It is local, pull-only, contains no prompt data, and follows the same bind and authentication rules as every other endpoint.
- Q: SBOM? → A: Each release attaches a CycloneDX SBOM for the Rust dependency graph plus a hand-maintained component entry for vendored MLX and mlx-c (SHOULD; a missing SBOM does not block a release).

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
   **Then** `mrun --version` prints version, commit, MLX version and build date, and the binary
   passes Gatekeeper assessment.
2. **Given** a fresh install, **When** the developer runs `mrun launch claude-code`, **Then** it
   offers the default model with size and fit, pulls it, starts the server and opens Claude Code
   configured for this process (004 FR-005).
3. **Given** a downloaded release tarball (browser download with quarantine), **When** the binary
   is run, **Then** it starts without a Gatekeeper prompt.

---

### User Story 2 - Run mrun as a login service (Priority: P1)

The developer runs `mrun service install`; mrun starts at every login, restarts after a crash,
logs to `~/.mrun`, and `mrun launch` uses it.

**Why this priority**: Agents expect an always-available endpoint; starting a server by hand
every day is friction.

**Independent Test**: Install the service, reboot, log in, query `/health`; kill the process and
watch it come back; uninstall and confirm nothing remains.

**Acceptance Scenarios**:

1. **Given** no service, **When** the developer runs `mrun service install`, **Then** a launchd user
   agent is written and loaded, the server starts, and the output names the label, plist path, log
   path, model and resident memory.
2. **Given** an installed service, **When** the Mac reboots and the user logs in, **Then** the server
   is healthy without any command.
3. **Given** a running service, **When** the server process crashes, **Then** launchd restarts it
   (throttled to one restart per 10 s); **When** the user runs `mrun stop`, **Then** it stays stopped
   until the next login or `mrun service install`.
4. **Given** an installed service, **When** the developer runs `mrun service uninstall`, **Then** the
   agent is unloaded and its plist removed; logs and models are kept.

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

### User Story 4 - Observe mrun with standard tools (Priority: P2)

The developer scrapes `/metrics` with Prometheus (or reads it with `curl`) to watch TTFT,
throughput, cache efficiency, queue depth and memory in daily use.

**Why this priority**: Makes `mrun bench` claims checkable in real use; cheap and read-only.

**Independent Test**: Scrape `/metrics` with `promtool check metrics`, send requests, and verify
that every documented series exists and moves as expected.

**Acceptance Scenarios**:

1. **Given** a running server, **When** `/metrics` is fetched, **Then** it returns valid Prometheus
   text exposition with every metric in the contract, and no prompt-derived labels.
2. **Given** a completed request with a cache hit, **When** metrics are fetched, **Then** TTFT
   histogram, prompt/cached/generated token counters and the hit counters have advanced.

---

### User Story 5 - Predictable releases (Priority: P3)

Maintainers cut a release by pushing a version tag; CI builds, signs, notarizes, publishes
artifacts with checksums, changelog and SBOM, and updates the tap. Users upgrade with
`brew upgrade mrun`.

**Why this priority**: Repeatable, auditable releases; needed before 0.1 ships but invisible to
users when it works.

**Independent Test**: Push a release-candidate tag in a fork with test credentials; verify
artifacts, signatures, notarization status, checksums and the tap pull request.

**Acceptance Scenarios**:

1. **Given** a `vX.Y.Z` tag on a commit with green CI and a changelog section for X.Y.Z, **When**
   the release workflow runs, **Then** it publishes the tarball, SHA-256 file and SBOM, and opens a
   tap update with the new URL and hash.
2. **Given** notarization fails, **When** the workflow runs, **Then** nothing is published.

### Edge Cases

- Intel Mac or macOS older than 14: the formula refuses to install with the requirement stated; the
  tarball's binary exits with the same message.
- `brew upgrade` while the service runs: the running server keeps the old binary until restart;
  the formula caveats say `mrun service install` restarts it on the new version (the plist points to
  the stable `opt` path).
- Service installed from a tarball and later from Homebrew (different binary path): `service
  install` rewrites the plist to the current binary and reloads it.
- Port already in use when the service starts: the server exits non-zero, launchd retries after the
  throttle interval, `doctor` reports the conflict.
- `mrun stop` with the service installed: stops the server; message explains it restarts at next
  login and how to remove the service.
- Lid closed on battery: the assertion does not prevent clamshell sleep; documented.
- `/metrics` on a non-loopback bind: requires the API key like every endpoint.
- Metric cardinality: labels come from fixed sets (API, outcome, cause, component); no request,
  session or model-text labels.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Releases MUST be built in CI on macOS arm64 runners from a tagged commit, for
  `aarch64-apple-darwin` with a minimum macOS of 14.0, and ship as `mrun-X.Y.Z-macos-arm64.tar.gz`
  containing the binary, the MLX Metal library, licenses, notices and shell completions.
- **FR-002**: The binary MUST be signed with a Developer ID Application certificate, hardened
  runtime and secure timestamp, and notarized by Apple before publication; a failed signature or
  notarization blocks the release.
- **FR-003**: A Homebrew tap formula MUST install the release tarball (verified by SHA-256),
  declare `depends_on arch: :arm64` and `depends_on macos: :sonoma`, install the binary so the Metal
  library resolves from the real binary location, install completions, and print caveats for
  `mrun service install` and `brew upgrade mrun`.
- **FR-004**: `mrun --version` MUST print version, git commit, vendored MLX and mlx-c versions and
  build date; `mrun doctor` MUST report the install source (Homebrew, tarball, source build) and
  version, with no network access.
- **FR-005**: `mrun service install` MUST write and load (`launchctl bootstrap gui/<uid>`) a user
  agent `~/Library/LaunchAgents/io.github.hupe1980.mrun.plist` that runs `<binary> serve --service`
  with `RunAtLoad`, `KeepAlive {SuccessfulExit: false}`, `ThrottleInterval 10`, stdout/stderr to
  `~/.mrun/launchd.log`; it MUST be idempotent (re-running replaces and reloads it) and print label,
  paths, model and expected resident memory. Server settings come from `~/.mrun/config.toml`, not
  from the plist.
- **FR-006**: `mrun service uninstall` MUST unload (`launchctl bootout`) and delete the agent and
  keep logs, config and models; both commands MUST refuse to run as root.
- **FR-007**: In service mode the server MUST log to `~/.mrun/server.log` (rotated at 16 MiB, 3
  files kept, mode 0600), write `~/.mrun/server.pid`, and exit 0 on SIGTERM after graceful shutdown;
  logs never contain prompt or completion content (002 FR-014).
- **FR-008**: `mrun stop` with an installed service MUST stop the server and state that it restarts
  at the next login; `mrun doctor` MUST report service state (not installed, loaded, running,
  crashed with last exit status) and a stale plist pointing to a missing binary.
- **FR-009**: While any request is running or queued, mrun MUST hold an IOKit power assertion that
  prevents idle system sleep (not display sleep), named "mrun: inference in progress", and release
  it 60 s after the queue empties; the assertion state appears in `/mrun/status` and `/metrics`.
- **FR-010**: `GET /metrics` MUST serve Prometheus text exposition format 0.0.4 with the metric
  names, types, units and labels in the [metrics contract](contracts/metrics.md), under the same
  bind and authentication rules as other endpoints, with label values from fixed sets only.
- **FR-011**: mrun MUST NOT check for updates or contact any host except during `mrun pull` (and a
  pull the user confirms inside `launch`); upgrade instructions are documentation only.
- **FR-012**: Versions MUST follow SemVer (`0.y.z` during 0.x: minor releases may break CLI or
  config, patch releases may not); the workspace version, tag `vX.Y.Z`, `--version` and the
  changelog MUST agree, and CI MUST fail a tag that disagrees.
- **FR-013**: `CHANGELOG.md` MUST follow Keep a Changelog sections (Added, Changed, Deprecated,
  Removed, Fixed, Security) with an `Unreleased` section; the release notes are that version's
  section.
- **FR-014**: Each release MUST publish `SHA256SUMS` and SHOULD publish a CycloneDX SBOM.
- **FR-015**: CI on every pull request MUST run fmt, clippy and tests on Linux and build the
  release configuration on macOS arm64 (constitution workflow).

### Key Entities

- **Release Artifact**: Tarball, contents, checksum, signature identity, notarization ticket id,
  SBOM.
- **Service Definition**: Label, plist path, binary path, arguments, keep-alive policy, log paths.
- **Power Assertion**: Type, name, held since, release deadline.
- **Metric**: Name, type, unit, labels, source.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: On a clean macOS 14+ user account on the target Mac with Homebrew and Claude Code
  installed, `brew install hupe1980/tap/mrun` followed by `mrun launch claude-code` (accepting the
  offered pull) completes the scripted read-edit-test task with no other command and no Gatekeeper
  prompt, in 3 of 3 runs.
- **SC-002**: With the service installed, after 3 of 3 reboots and logins `/health` answers within
  the model load time plus 10 s, and after 5 of 5 `kill -9` the server is healthy again within
  load time plus 15 s.
- **SC-003**: `/metrics` passes `promtool check metrics` with zero errors, and 100% of the metrics
  listed in the contract are present with the documented type and labels after one request per API.
- **SC-004**: With a 1-minute idle-sleep setting, a 3-minute request completes in 3 of 3 runs and
  `pmset -g assertions` shows the assertion only while requests are running or queued (+ 60 s).
- **SC-005**: A release tag produces signed, notarized artifacts whose `codesign --verify --strict`
  and `syspolicy_check distribution` checks pass, with checksums matching, in one unattended
  workflow run.

## Assumptions

- The maintainer holds an Apple Developer ID certificate and an App Store Connect API key, stored
  as CI secrets.
- The GitHub repository `hupe1980/mrun` publishes releases; the tap repository
  `hupe1980/homebrew-tap` exists and accepts automated pull requests.
- `mrun launch`, `connect`, `serve --detach`, `stop` and `doctor` come from 004; this feature adds
  packaging, service, power and metrics.
- CI macOS runners can build Metal shaders but are not used for Metal performance tests (those run
  on the target Mac with `mrun bench`).
