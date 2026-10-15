# mrun

The local inference server built for coding agents on Apple Silicon.

Coding agents resend tens of thousands of tokens of instructions, tool schemas and history on
every turn. On a laptop GPU, re-reading that context costs tens of seconds per turn. mrun computes
it once, reuses it exactly across turns, branches and tool loops, explains every cache miss, and
never pushes your Mac into swap.

**[Documentation](https://hupe1980.github.io/mrun/)** ·
[Getting started](https://hupe1980.github.io/mrun/docs/getting-started/) ·
[How it works](https://hupe1980.github.io/mrun/docs/how-it-works/) ·
[Roadmap](https://hupe1980.github.io/mrun/docs/roadmap/)

> **Status: in development.** The commands below describe the first release; none of them works
> yet. No performance results are claimed until `mrun bench` publishes dated comparisons.

## Planned experience

```sh
mrun pull qwen3-coder      # Qwen3-Coder-30B-A3B, 4-bit MLX, verified download
mrun serve --detach        # http://127.0.0.1:8484
mrun launch claude-code    # or: codex, opencode
mrun ps                    # model, memory budget, cache hit rate, last cache miss
mrun doctor                # diagnose the machine and client setup
```

- **Three agent APIs.** Anthropic Messages, OpenAI Chat Completions and OpenAI Responses, with
  streaming, tool calls and structured output.
- **Exact prefix cache.** A radix tree of attention state reuses the longest identical token
  prefix and reports cached tokens and the reason for each miss.
- **Predictable memory.** Weights, attention state and scratch are budgeted at startup; requests
  that cannot fit are refused up front.
- **Instant interruption.** Cancellation is checked after every model layer.
- **Apple MLX from Rust.** One native binary; no Python at runtime.
- **Private by default.** Loopback only, no telemetry, no prompt logging, offline after `pull`.

Target hardware is an Apple Silicon Mac; the default model needs 48 GB of memory. Linux builds
compile and pass the portable tests in CI.

## Development

Requires the Rust toolchain pinned in `rust-toolchain.toml`.

```sh
cargo test --workspace     # portable tests, Linux or macOS
```

Building the MLX engine needs macOS 15+ on Apple Silicon with Xcode or the Command Line Tools.
Development follows the [Spec Kit](https://github.com/github/spec-kit) workflow.
The website is a [Zola](https://www.getzola.org/) site in `site/`
(`zola --root site serve`).

## License

MIT OR Apache-2.0; see [LICENSE-MIT](LICENSE-MIT) and [LICENSE-APACHE](LICENSE-APACHE). Model
weights keep their own licenses, shown by `mrun pull`.
