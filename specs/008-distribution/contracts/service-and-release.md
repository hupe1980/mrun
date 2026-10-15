# Contract: Service, Install and Release

## CLI (extends the [004 CLI contract](../../004-models-cli/contracts/cli.md))

| Command | Behavior |
| --- | --- |
| `mrun --version` | `mrun X.Y.Z (<commit7>, <date>) mlx <v> mlx-c <v> macos-arm64` |
| `mrun service install [--preload]` | Write `~/Library/LaunchAgents/io.github.hupe1980.mrun.plist` (`ProgramArguments` includes `--preload` when given), `launchctl bootout` (if loaded) then `bootstrap gui/<uid>`; wait for `/health` (and `loaded: true` only with `--preload`); print label, plist, binary, log path, model, resident memory |
| `mrun service uninstall` | `launchctl bootout gui/<uid>/io.github.hupe1980.mrun`, delete the plist; keep `~/.mrun` and models |
| `mrun serve --service [--preload]` | Used by launchd only: lazy model load (engine created on the first dequeued request; keepalive/progress while loading; `/health` `{"loaded": bool}`) unless `--preload`, no detach, rotating `~/.mrun/server.log`, pid file, exit 0 on SIGTERM |
| `mrun stop` | Unchanged (004); with a service installed, also prints "the service restarts at next login; `mrun service uninstall` removes it" |
| `mrun doctor` | Adds: install source and version, service state, stale plist, power-assertion support; no network access |
| `mrun launch` (before load) | Shows capacity from the cached probe or the 004 fit estimate marked `estimated` |

Both `service` commands refuse root (exit 2) and are unavailable on Linux (exit 2 with a message).
Errors use the 004 error shape and exit codes (5 when the server does not become healthy).

## Install layout

Homebrew: `<prefix>/Cellar/mrun/X.Y.Z/libexec/{mrun, mlx.metallib}`, `<prefix>/bin/mrun` (exec
script), completions in Homebrew's standard locations. Tarball: unpack anywhere; keep `mrun` and
`mlx.metallib` in the same directory. A missing Metal library is a startup error naming the
expected path.

## Release workflow (`.github/workflows/release.yml`, trigger `v*` tags)

1. Check: tag `vX.Y.Z` = workspace version = `CHANGELOG.md` section `## [X.Y.Z]`; CI green on the
   commit.
2. Build `--release --target aarch64-apple-darwin` with `MACOSX_DEPLOYMENT_TARGET=15.0`.
3. Sign (hardened runtime, timestamp), notarize (`notarytool submit --wait`), verify (`codesign
   --verify --strict`, `syspolicy_check distribution`), smoke-test `mrun --version`.
4. Package tarball, `SHA256SUMS`, SBOM; create the GitHub Release (pre-release for `-rc.N`) with
   the changelog section as notes.
5. Final releases only: open a pull request on `hupe1980/homebrew-tap` updating `url`, `sha256`,
   `version`; `brew audit --strict` and `brew test` run in the tap's CI.

Any failure before step 4 publishes nothing.

## Versioning

SemVer, 0.x rules (FR-012). Changelog: Keep a Changelog; `Unreleased` entries required for
user-visible changes (review convention; no CI label job). The release workflow checks that the
version's section exists (step 1).
