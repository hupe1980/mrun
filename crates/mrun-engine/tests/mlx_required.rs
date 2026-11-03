//! Zero-test guard (T072): the MLX suites are `#![cfg(feature = "engine")]`, so without the feature
//! they compile to nothing and "pass". A run that means to test MLX (`MRUN_TEST_MODELS` set, or
//! `MRUN_REQUIRE_MLX=1` as the macOS CI job and `scripts/test_mlx.sh` set it) fails here
//! instead. `scripts/test_mlx.sh` also fails when a suite reports 0 tests.

#[test]
fn mlx_suites_are_compiled_when_required() {
    if cfg!(feature = "engine") {
        return;
    }
    let required = std::env::var_os("MRUN_TEST_MODELS").is_some()
        || std::env::var("MRUN_REQUIRE_MLX").is_ok_and(|v| v == "1");
    assert!(
        !required,
        "mrun-engine was built without the `mlx` feature: its MLX test suites would run 0 tests \
         (add --features mlx)"
    );
}
