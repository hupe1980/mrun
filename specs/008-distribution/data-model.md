# Data Model: Distribution

## ReleaseArtifact

| Field | Value |
| --- | --- |
| name | `mrun-X.Y.Z-macos-arm64.tar.gz` |
| contents | `mrun-X.Y.Z/{bin/mrun, bin/mlx.metallib, LICENSE-MIT, LICENSE-APACHE, NOTICE, completions/{mrun.bash, _mrun, mrun.fish}}` |
| signature | Developer ID Application, hardened runtime, secure timestamp (on `bin/mrun`) |
| notarization | submission id + status `Accepted` (recorded in release notes footer) |
| checksums | `SHA256SUMS` (tarball, SBOM) |
| sbom | `mrun-X.Y.Z.cdx.json` (CycloneDX; Rust graph + MLX + mlx-c components) |

## Formula (`hupe1980/homebrew-tap/Formula/mrun.rb`)

| Field | Value |
| --- | --- |
| url, sha256, version | release tarball |
| requirements | `depends_on arch: :arm64`, `depends_on macos: :sonoma` |
| install | `libexec.install "bin/mrun", "bin/mlx.metallib"`; `bin.write_exec_script libexec/"mrun"`; completions |
| caveats | service install/restart after upgrade; `brew upgrade mrun`; models live in the HF cache |
| test | `mrun --version`; `mrun config` |

## ServiceDefinition (plist)

| Key | Value |
| --- | --- |
| Label | `io.github.hupe1980.mrun` |
| ProgramArguments | `[<binary>, "serve", "--service"]`; `<binary>` = `<brew prefix>/opt/mrun/libexec/mrun` (Homebrew) or canonical current executable |
| RunAtLoad | true |
| KeepAlive | `{SuccessfulExit: false}` |
| ThrottleInterval | 10 |
| StandardOutPath, StandardErrorPath | `~/.mrun/launchd.log` |
| EnvironmentVariables | `{MRUN_HOME}` only if set at install |
| file | `~/Library/LaunchAgents/io.github.hupe1980.mrun.plist`, mode 0644 |

States (from `launchctl print gui/<uid>/<label>` and the plist): `not-installed`, `installed-not-loaded`,
`running (pid)`, `waiting (last exit status)`, `stale (binary missing)`.

## PowerAssertion

| Field | Type |
| --- | --- |
| held | bool |
| id | IOPMAssertionID (macOS) |
| since | Instant |
| release_at | Option<Instant> (queue empty + 60 s) |

Transitions: `idle --request queued/started--> held`; `held --queue empty--> releasing(60 s)`;
`releasing --new request--> held`; `releasing --timeout--> idle`. Linux: always `idle`.

## Metric

`{name, type: counter | gauge | histogram, unit, help, labels: fixed enum sets, buckets?}`; the
full list is the [metrics contract](contracts/metrics.md).

## VersionInfo (`mrun --version`)

`mrun X.Y.Z (<commit7>, <YYYY-MM-DD>) mlx <v> mlx-c <v> macos-arm64` — embedded at build time.
