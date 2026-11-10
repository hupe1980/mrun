# mrun-engine

The engine side of [mrun](https://github.com/hupe1980/mrun): the primitive `Engine` contract,
model identity, the memory budget, the family registry and, with the `engine` feature, the engine
itself (Qwen3, Qwen3-MoE, Qwen3.5 and Qwen3.5-MoE families on Apple Silicon).

## Features

| Feature | What it adds |
|---|---|
| (none, the default here) | The contract, identity, budget and registry only. Builds anywhere in seconds; `mrun --no-default-features` and the Linux CI job use it. |
| `engine` | The engine on [nuvec](https://crates.io/crates/nuvec) 0.8.0, a pure-Rust port of MLX. Off by default here; `mrun` turns it on. |
| `test-support` | Tiny random-weight checkpoints for tests (with a backend). |

## Building with `engine`

On macOS, nuvec compiles its Metal kernels into the binary when Xcode's Metal toolchain is present
(`xcodebuild -downloadComponent MetalToolchain`). Without it they are compiled at first start, and
`mrun doctor` warns.

## License

MIT OR Apache-2.0, like mrun.
