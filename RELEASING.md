# Releasing mrun

Releases are cut from `main` by pushing a tag; `.github/workflows/release.yml` does the rest.

1. Make sure `main` is green (the release workflow refuses a tag whose commit has no successful
   CI run) and `CHANGELOG.md` has every user-visible change under `## [Unreleased]`.
2. Bump `version` in the workspace `Cargo.toml` (one version for all crates), update the exact
   `=X.Y.Z` versions of `mrun-engine` and `mrun-runpod` under `[workspace.dependencies]`, move the
   `Unreleased` entries to `## [X.Y.Z] - YYYY-MM-DD`, open a new empty `Unreleased` section, and
   merge.
3. Tag and push: `git tag vX.Y.Z && git push origin vX.Y.Z`. Release candidates use
   `vX.Y.Z-rc.N`: they become GitHub pre-releases and never update the Homebrew tap or crates.io.
4. Watch the workflow: it checks that the tag, the crate versions and the changelog section
   agree, waits for a green CI run on the tagged commit, builds for `aarch64-apple-darwin` with
   `MACOSX_DEPLOYMENT_TARGET=15.0` on an SDK ≥ 26.2 (nuvec's kernels precompiled, NAX ones
   included), signs ad hoc, smoke-tests `mrun --version` and `mrun doctor` with the binary alone,
   packages the one archive (`mrun-X.Y.Z-macos-arm64.tar.gz`) with its checksum, attests the build provenance of every
   artifact, and publishes the release: created as a draft, uploaded, then published, so a re-run
   finishes a draft but never overwrites a published release. Nothing is published if a step
   before publishing fails.
5. Final releases: the workflow publishes to crates.io in dependency order, `mrun-engine` →
   `mrun-runpod` → `mrun`, using `CARGO_REGISTRY_TOKEN` from the GitHub
   environment `release` (see the owner setup at the top of `release.yml`). The job fails when
   the token is missing; versions already on crates.io are skipped, so a re-run continues after a
   partial publish. To publish by hand instead:

   ```sh
   cargo publish -p mrun-engine
   cargo publish -p mrun-runpod
   cargo publish -p mrun
   ```

   Publishing verifies each crate by building it, so a Mac with Xcode's Metal toolchain is
   needed. Afterwards check that `cargo install mrun` works on a clean machine (about two
   minutes).
6. Final releases: the workflow renders `Formula/mrun.rb` with `scripts/release/formula.sh` from
   the release's `SHA256SUMS` and commits it to `hupe1980/homebrew-tap`, with a short-lived token
   of the GitHub App installed on the tap (`HOMEBREW_APP_ID`, `HOMEBREW_APP_PRIVATE_KEY` in the
   environment `release`). It then installs the release with `brew install hupe1980/tap/mrun` and
   runs `brew test mrun` on clean `macos-15` and `macos-26` runners.
7. Verify on a clean macOS account (008 SC-001): `brew install hupe1980/tap/mrun`, then
   `mrun --version` and `mrun launch claude-code`; no Gatekeeper prompt appears for the Homebrew
   install. A tarball downloaded with a browser needs `xattr -d com.apple.quarantine` on the
   unpacked directory (ad-hoc signed, not notarized).

## Checklist (copy into the release pull request)

- [ ] Changelog section for the version exists and reads well for users
- [ ] Release workflow green; artifacts and `SHA256SUMS` attached
- [ ] `codesign --verify --strict` passed in the workflow
- [ ] crates.io: `mrun-engine`, `mrun-runpod`, `mrun` at the new version (final releases)
- [ ] Tap formula updated and the Homebrew smoke job green (final releases)
- [ ] Clean-account install verified (3 runs)
