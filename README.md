# mrun

A local inference server for coding agents on Apple Silicon (M1 and later), written in Rust on
nuvec, a pure-Rust port of Apple's MLX.

Coding agents resend tens of thousands of tokens of instructions, tool schemas and history on
every turn. mrun computes that context once, reuses it exactly across turns, subagents and tool
loops, explains every cache miss, and stays inside a memory budget fixed at startup. It serves
the Anthropic Messages, OpenAI Chat Completions and OpenAI Responses APIs on `127.0.0.1`, and
`mrun ui` shows all of it in a window.

[![The mrun window: memory budget, live requests and how much of each prompt was reused](site/static/screenshots/now.png)](site/static/screenshots/now.png)

**[Documentation](https://hupe1980.github.io/mrun/)** ·
[Getting started](https://hupe1980.github.io/mrun/docs/getting-started/) ·
[How it works](https://hupe1980.github.io/mrun/docs/how-it-works/) ·
[Roadmap](https://hupe1980.github.io/mrun/docs/roadmap/)

## Status

**0.4**: `brew install hupe1980/tap/mrun`. mrun is qualified on one Mac
(M4 Pro, 48 GB); smaller Macs are simulated there with `--budget-gib`.

- **Models**: `qwen3.6` (Qwen3.6-35B-A3B 4-bit), the default wherever it leaves at least 64K
  tokens of context, and `qwen3.5` (Qwen3.5-9B 4-bit, a preview) on smaller Macs; also Qwen3-8B
  and the Qwen3-Coder family by repo name. See [Models](https://hupe1980.github.io/mrun/docs/models/).
- **Measured**: token-for-token parity with mlx-lm, speed against mlx-lm and Ollama,
  cancellation latency and speculative decoding (1.43× on copy-heavy replies with `qwen3.6`) are
  on [Benchmarks](https://hupe1980.github.io/mrun/docs/benchmarks/).
  The client gate (Claude Code, Codex and OpenCode on scripted tasks) is on
  [Clients](https://hupe1980.github.io/mrun/docs/clients/#client-gate).
- **Not yet measured**: a full comparison against llama.cpp, Ollama, oMLX and mlx-lm,
  cancellation on M1, and a pool on two real Macs (tested across processes on one Mac). Results from other Macs are welcome:
  [Hardware reports](CONTRIBUTING.md#hardware-reports).

## What mrun claims

Each is measured on the target Mac ([Benchmarks](https://hupe1980.github.io/mrun/docs/benchmarks/)):

- **Exact reuse, proven**: a request served from the cache gets bit-identical output to
  recomputing its prompt with the same chunk plan.
- **Every miss explained**: each response reports cached and processed tokens, and a miss names
  its cause (`new`, `diverged`, `evicted`, `gap`, `disabled`) and the message where the prompt
  changed.
- **Tool loops always hit**: the model's own generated tokens are spliced back into the next
  prompt, also on hybrid models.
- **Speculative decoding that changes nothing**: tokens drafted from the prompt are kept only
  where plain decoding would produce them, bit for bit, greedy or seeded.
- **A memory budget that holds**: weights, state, cache and scratch stay inside one budget
  measured at startup, including what other apps hold. Requests that cannot fit are refused up
  front with the numbers.
- **An honest benchmark**: `mrun bench` replays whole agent sessions against other servers;
  published results include every cell mrun loses.

mrun does not claim the fastest decode.

## Quick start

An Apple Silicon Mac with macOS 15 or later:

```sh
brew install hupe1980/tap/mrun

mrun ls                      # the aliases, the context each leaves here and this Mac's default
mrun pull qwen3.6            # or qwen3.5, where qwen3.6 leaves less than 64K tokens
mrun run                     # chat in the terminal
mrun serve --detach          # serve on http://127.0.0.1:8484
mrun launch claude-code      # or: codex, opencode
mrun ui                      # the window: status, chat, sessions, pool, models, agents
```

To build from source instead (Rust and Xcode's Metal toolchain): `cargo install mrun`, or see
[Getting started](https://hupe1980.github.io/mrun/docs/getting-started/).

`mrun launch` also offers to pull this Mac's default model and start the server. The binary
embeds its Metal kernels, so no files are needed next to it. `mrun doctor` checks the
machine, the server and the clients.

## The window

`mrun ui` opens a native window and a menu bar item (`--browser`: in a browser): what the server
is doing, each agent session with every miss and the text that changed, the Mac pool, a chat
that takes files, models with a fit verdict (any repo by name), one-click agent setup and code
for your own apps. Every action is an `mrun` command; Copy as Command shows it. See
[The mrun window](https://hupe1980.github.io/mrun/docs/ui/).

<table>
  <tr>
    <td><a href="site/static/screenshots/chat.png"><img src="site/static/screenshots/chat.png" alt="Chat: a reply with its reuse, time to first token and speed"></a></td>
    <td><a href="site/static/screenshots/models.png"><img src="site/static/screenshots/models.png" alt="Models: a fit verdict for this Mac, the model in use and the best one"></a></td>
  </tr>
  <tr>
    <td><a href="site/static/screenshots/agents.png"><img src="site/static/screenshots/agents.png" alt="Agents: connect Claude Code, Codex or OpenCode in one click"></a></td>
    <td><a href="site/static/screenshots/chat-thinking-dark.png"><img src="site/static/screenshots/chat-thinking-dark.png" alt="Dark mode: a reply while the model thinks, with seconds and tokens"></a></td>
  </tr>
</table>

## Mac pool (opt-in)

Macs on one network, colleagues' included, run a model none of them fits alone: `mrun pool start`
on one, `mrun pool join` on the others, compare a six-digit code, then `mrun pool serve <model>`.
Each Mac runs a range of layers; replies and reuse are as on one Mac, and agents keep using their
own `127.0.0.1:8484`. The host reads the prompts; traffic between the Macs is encrypted. See
[Mac pool](https://hupe1980.github.io/mrun/docs/pool/).

## Remote GPU (RunPod, opt-in)

For models that don't fit a Mac, `mrun cloud up --model <repo>` rents a GPU on RunPod, serves the
model there with vLLM and relays it on `127.0.0.1:8484`, so `launch` and `connect` work as usual.
**This sends your prompts and code to RunPod and costs money by the hour.** mrun shows the price
and asks first, the pod stops itself when idle, and `mrun cloud down` terminates it. In the
window it is Models → *On a rented GPU*. See
[Remote GPU](https://hupe1980.github.io/mrun/docs/cloud/).

## Configuration

mrun needs no configuration. Defaults can be changed with flags, `MRUN_*` variables or an optional
config file, `~/.mrun/config.toml`, whose tables mirror the commands (`[serve]`, `[cloud]`, …) and
whose keys are the flag names. `mrun config show` prints every setting and where it comes from;
`mrun config schema` gives editors a JSON Schema. See
[Configuration](https://hupe1980.github.io/mrun/docs/configuration/).

## Development

```sh
cargo test --workspace                         # on a Mac: every suite, CPU and GPU
cargo test --workspace --no-default-features   # without the engine (Linux, CI): the rest
```

The window's page lives in `crates/mrun/ui` (Svelte, built to `ui/dist`, which is committed and
embedded at build time, so Rust builds need no Node). After changing it:
`cd crates/mrun/ui && npm ci && npm test && npm run build`. `npm run shots` renders every place
in light and dark from fixed data into `ui/shots/` without a model; the screenshots in
`site/static/screenshots` come from there.

The website is a [Zola](https://www.getzola.org/) site in `site/` (`zola --root site serve`).

## License

MIT OR Apache-2.0; see [LICENSE-MIT](LICENSE-MIT) and [LICENSE-APACHE](LICENSE-APACHE). Model
weights keep their own licenses, shown by `mrun pull`.
