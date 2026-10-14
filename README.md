# mrun

The local inference server built for coding agents on Apple Silicon.

Coding agents resend tens of thousands of tokens of instructions, tool schemas and history on
every turn. mrun computes that context once, reuses it exactly across turns, branches and tool
loops, tells you why when it can't, and never pushes your Mac into swap.

> **Status: design complete, implementation starting.** Nothing below works yet. The specs and
> the roadmap describe what is being built; no performance results are claimed.

## Planned experience

```sh
mrun pull qwen3-coder            # Qwen3-Coder-30B-A3B, 4-bit MLX, verified download
mrun serve                       # http://127.0.0.1:8484
mrun connect claude-code         # or: codex, opencode
mrun ps                          # model, memory budget, cache hit rate, last cache miss
mrun doctor                      # diagnose the machine and client setup
```

- **APIs**: OpenAI Chat Completions, Anthropic Messages and OpenAI Responses (stateless), with
  streaming, tool calls and structured output, so Claude Code, Codex, OpenCode and other agents
  work out of the box.
- **Exact prefix cache**: radix-tree reuse of attention state with per-request cached-token counts
  and miss explanations.
- **Predictable memory**: a startup budget for weights, attention state and scratch; oversized
  requests are rejected up front.
- **Instant interruption**: cancellation takes effect within one processing chunk or one token.
- **Engine**: Apple [MLX](https://github.com/ml-explore/mlx), called from Rust.
- **Private by default**: loopback only, no telemetry, no prompt logging, offline after `pull`.

## Roadmap

| Milestone | Feature |
| --- | --- |
| M1 | Engine core: run Qwen3 and Qwen3-Coder (MoE) locally with `mrun run` |
| M2 | Agent APIs: Claude Code, Codex and OpenCode tool loops |
| M3 | Prefix cache with miss diagnostics |
| M4 | `pull`, `connect`, `doctor`, `ps` |
| M5 | `mrun bench` with a dated comparison against llama.cpp, Ollama, oMLX and mlx-lm |

Apple Silicon Macs are the target (first qualified on an M4 Pro with 48 GB). Linux builds compile
and run the portable tests; running models on Linux is a later evaluation.

## Development

Specs, plans and tasks live in [`specs/`](specs/README.md); design notes in
[`concepts/`](concepts/README.md). Work follows the Spec Kit workflow (`/speckit-plan`,
`/speckit-tasks`, `/speckit-analyze`, `/speckit-implement`) with skills for Claude Code and Copilot.

Use Rust 1.94.0 as pinned in `rust-toolchain.toml`:

```sh
cargo test --workspace
```

## License

MIT OR Apache-2.0; see [LICENSE-MIT](LICENSE-MIT) and [LICENSE-APACHE](LICENSE-APACHE). Model
weights retain their own licenses, shown by `mrun pull`.
