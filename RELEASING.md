# Releasing mrun

Releases are cut from `main` by pushing a tag; `.github/workflows/release.yml` does the rest.

1. Make sure `main` is green (the release workflow refuses a tag whose commit has no successful
   CI run) and `CHANGELOG.md` has every user-visible change under `## [Unreleased]`.
2. Bump `version` in the workspace `Cargo.toml` (one version for all crates) and in
   `crates/mrun-mlx-sys/Cargo.toml` (not a workspace member, so it carries its own copy; the
   workflow checks that both agree), update the exact `=X.Y.Z` versions of `mrun-engine` and
   `mlx-sys` under `[workspace.dependencies]`, move the `Unreleased` entries to
   `## [X.Y.Z] - YYYY-MM-DD`, open a new empty `Unreleased` section, and merge.
3. Tag and push: `git tag vX.Y.Z && git push origin vX.Y.Z`. Release candidates use
   `vX.Y.Z-rc.N`: they become GitHub pre-releases and never update the Homebrew tap or crates.io.
4. Watch the workflow: it checks that the tag, the crate versions and the changelog section
   agree, waits for a green CI run on the tagged commit, restores the MLX build from the CI cache,
   builds for `aarch64-apple-darwin` with `MACOSX_DEPLOYMENT_TARGET=15.0`, signs ad hoc,
   smoke-tests `mrun --version` with the binary alone (MLX's Metal kernels are embedded, D034),
   packages the single-binary tarball with `SHA256SUMS`, and publishes the release. Nothing is
   published if a step before publishing fails.
5. Final releases: the workflow publishes to crates.io in dependency order, `mrun-mlx-sys` →
   `mrun-engine` → `mrun`, using the `CARGO_REGISTRY_TOKEN` repository secret (a crates.io token
   with the `publish-update` scope for the three crates; `publish-new` for the first release).
   Without the secret the step is skipped; versions already on crates.io are skipped too, so a
   re-run continues after a partial publish. To publish by hand instead:

   ```sh
   cargo publish --manifest-path crates/mrun-mlx-sys/Cargo.toml
   cargo publish -p mrun-engine
   cargo publish -p mrun
   ```

   Publishing verifies each crate by building it, so a Mac with Xcode's Metal toolchain is
   needed; the MLX build comes from `~/Library/Caches/mrun-mlx-sys` when present. Afterwards
   check that `cargo install mrun` works on a clean machine (it builds MLX once, 15-40 min).
6. Final releases: merge the pull request the workflow opens on `hupe1980/homebrew-tap` once its
   `brew audit --strict` and `brew test` pass.
7. Verify on a clean macOS account (008 SC-001): `brew install hupe1980/tap/mrun`, then
   `mrun --version` and `mrun launch claude-code`; no Gatekeeper prompt appears for the Homebrew
   install. A tarball downloaded with a browser needs `xattr -d com.apple.quarantine` on the
   unpacked directory (ad-hoc signed, not notarized).

## Checklist (copy into the release pull request)

- [ ] Changelog section for the version exists and reads well for users
- [ ] Release workflow green; artifacts and `SHA256SUMS` attached
- [ ] `codesign --verify --strict` passed in the workflow
- [ ] crates.io: `mrun-mlx-sys`, `mrun-engine`, `mrun` at the new version (final releases)
- [ ] Tap pull request merged (final releases)
- [ ] Clean-account install verified (3 runs)
