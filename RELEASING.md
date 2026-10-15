# Releasing mrun

Releases are cut from `main` by pushing a tag; `.github/workflows/release.yml` does the rest.

1. Make sure `main` is green and `CHANGELOG.md` has every user-visible change under
   `## [Unreleased]`.
2. Bump `version` in the workspace `Cargo.toml` (one version for all crates), move the
   `Unreleased` entries to `## [X.Y.Z] - YYYY-MM-DD`, open a new empty `Unreleased` section, and
   merge.
3. Tag and push: `git tag vX.Y.Z && git push origin vX.Y.Z`. Release candidates use
   `vX.Y.Z-rc.N`: they become GitHub pre-releases and never update the Homebrew tap.
4. Watch the workflow: it checks that the tag, the workspace version and the changelog section
   agree, builds for `aarch64-apple-darwin` with `MACOSX_DEPLOYMENT_TARGET=15.0`, signs ad hoc,
   smoke-tests `mrun --version`, packages the tarball with `SHA256SUMS`, and publishes the release.
   Nothing is published if a step before publishing fails.
5. Final releases: merge the pull request the workflow opens on `hupe1980/homebrew-tap` once its
   `brew audit --strict` and `brew test` pass.
6. Verify on a clean macOS account (008 SC-001): `brew install hupe1980/tap/mrun`, then
   `mrun --version` and `mrun launch claude-code`; no Gatekeeper prompt appears for the Homebrew
   install. A tarball downloaded with a browser needs `xattr -d com.apple.quarantine` on the
   unpacked directory (ad-hoc signed, not notarized).

## Checklist (copy into the release pull request)

- [ ] Changelog section for the version exists and reads well for users
- [ ] Release workflow green; artifacts and `SHA256SUMS` attached
- [ ] `codesign --verify --strict` passed in the workflow
- [ ] Tap pull request merged (final releases)
- [ ] Clean-account install verified (3 runs)
