# mrun-engine

The engine side of [mrun](https://github.com/hupe1980/mrun): the primitive `Engine` contract,
model identity, the memory budget, the family registry and, with a backend feature, the engine
itself (Qwen3, Qwen3-MoE, Qwen3.5 and Qwen3.5-MoE families on Apple Silicon).

## Features

| Feature | What it adds |
|---|---|
| (none, the default here) | The contract, identity, budget and registry only. Builds anywhere in seconds; `mrun --no-default-features` and the Linux CI job use it. |
| `mlx` | The MLX engine through [`mrun-mlx-sys`](../mrun-mlx-sys/README.md), which builds MLX v0.32.4 and mlx-c v0.7.0 from source. `mrun` turns it on by default. |
| `nuvec` | The same engine on [nuvec](https://crates.io/crates/nuvec) 0.6.1, a pure-Rust port of MLX, instead of MLX (`mrun --no-default-features --features nuvec`). Under qualification; MLX stays the default until nuvec passes its gates. |
| `test-support` | Tiny random-weight checkpoints for tests (with a backend). |

## Building with `mlx`

- **macOS needs Xcode with the Metal toolchain.** The MLX build compiles a one-line kernel first
  and stops with instructions when that fails:

  ```sh
  sudo xcode-select -s /Applications/Xcode.app
  xcodebuild -downloadComponent MetalToolchain
  ```

  `MLX_SYS_METAL=0` builds the CPU backend only, for tests and machines without Xcode; it is far
  too slow for real models.
- **Linux** builds the CPU backend against OpenBLAS (`cmake libopenblas-dev liblapacke-dev`).
- **Build time**: the first MLX build takes 3 to 40 minutes depending on the machine. It is kept
  per configuration in the user cache, so later builds, profiles, worktrees and
  `cargo install` reuse it (seconds). OS file locks serialize concurrent first builds.

The main environment variables of the MLX build (the full list is in the
[`mrun-mlx-sys` README](../mrun-mlx-sys/README.md#environment-variables)):

| Variable | Meaning |
|---|---|
| `MLX_SYS_METAL` | `1` requires Metal, `0` builds CPU only; unset on macOS means Metal. |
| `MLX_SYS_CACHE_DIR` | Where builds and downloaded sources are kept. Default `~/Library/Caches/mrun-mlx-sys` (macOS), `$XDG_CACHE_HOME/mrun-mlx-sys` or `~/.cache/mrun-mlx-sys` (Linux). A CI cache restores this directory. |
| `MLX_SYS_SOURCE_DIR` | Offline or Nix builds: a directory with `mlx-<commit>.tar[.gz]` and `mlx-c-<commit>.tar[.gz]` (the pinned commits are in the [`mrun-mlx-sys` README](../mrun-mlx-sys/README.md#offline-builds)), checked against the same sha256 as downloads. |
| `MLX_SYS_PREBUILT_DIR` | An install prefix from an earlier build (`lib/libmlxc.a`, `lib/libmlx.a`, and `lib/mlx.metallib` on macOS); skips fetching and CMake. |
| `MACOSX_DEPLOYMENT_TARGET` | Part of the cache key. 26.2 or later with the macOS 26.2 SDK compiles MLX's NAX (M5 Neural Accelerator) kernels. |
| `MLX_SYS_REQUIRE_NAX` | `1` fails the build when the deployment target is 26.2 or later but no NAX kernel was compiled; otherwise a warning. |

## GPU kernels

With Metal, MLX's `mlx.metallib` (about 215 MB) is embedded zstd-compressed (about 15 MB), so
the binary needs no file next to it. On first use it is extracted once to
`~/Library/Caches/mrun/mlx-<MLX version>-<hash>.metallib` (temporary file plus rename, safe for
concurrent first runs). A `mlx.metallib` beside the executable's real location (symlinks
resolved) wins; development builds put one there.

## License

MIT OR Apache-2.0, like mrun. MLX and mlx-c are MIT-licensed by Apple.
