# Changelog

All notable changes to mrun are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/) with 0.x rules: minor versions may break.

## [Unreleased]

### Added

- **The mrun window** (018, 019): `mrun ui` opens a native window and a menu bar item
  (`--browser`: the default browser). Every action runs the `mrun` command it shows (Copy as
  Command). Loopback only, with a per-start key; no network requests besides your downloads.
  - **Now**: the memory budget as one bar, live requests with progress, and a reuse ribbon per
    request. Only unintended recomputes are amber (eviction, a disabled cache, a client re-sending
    an earlier message differently).
  - **Chat** through the Messages API: each reply shows its reuse, time to first token, speed and
    stop reason; thinking shows its seconds and tokens; Copy, Edit (also ↑) and Regenerate; an
    inspector with the system prompt and every sampling setting. Chats are kept only if you say so.
  - **Models** with the CLI's fit verdicts, Best for this Mac, In Use, download, Use and Remove; a
    first-run sheet. **Agents**: connect and restore in one click. Settings, ⌘K, context menus.
  - Native title bar, remembered window frame, switches, Reduce Motion and Increase Contrast; the
    menu bar bolt fills while a prompt is read.
  - `npm run shots` renders every place in light and dark from fixed data.
  - Files in the chat (020): drop, paste or attach text and code; the server's token count per
    file; refused with numbers when they don't fit the context. Add a model by repo or Hugging
    Face link, with its plan first. Copy-ready code for all three APIs and Zed and Crush
    settings. Chats export to Markdown, and search finds text inside messages.
- `repetition_penalty` on all three APIs (mlx-lm's rule over the last 20 tokens; unset or 1
  leaves output unchanged).
- `/mrun/status` `finished` (the last 20 requests with reuse, miss cause, time to first token,
  decode speed, stop); `GET /v1/models/{id}` carries the model's sampling `defaults` and
  `thinking`.
- `mrun pull --dry-run` shows the plan (size, family, fit) without downloading.
- Machine output for scripts: `mrun pull --json` (JSON lines), `mrun connect <client> --json`,
  `fit`/`verdict`/`recommended` in `mrun ls --json`, `mrun ls --budget-gib` and `--port`. The
  FITS column says `tight` below 64K tokens; verdicts group thousands (`221,819`).

- Config file (017, D066): optional `~/.mrun/config.toml` (or `MRUN_CONFIG`; empty disables it)
  with a table per command (`[serve]`, `[launch]`, `[run]`, `[cloud]`, `[bench]`) and top-level
  `port`, `model`, `budget-gib`; keys are the flags' long names. Precedence: defaults < file <
  `MRUN_*` < flags. Unknown keys warn with the closest name; invalid values stop every command but
  `mrun config`. Confirmations, remote mode, secrets and addresses cannot be set; mrun never
  writes the file.
- `mrun config path`, `mrun config show [--json]` (every setting with its value and source:
  `default`, `file:<line>`, `env MRUN_*`; valid TOML) and `mrun config schema` (JSON Schema for
  Taplo and other TOML editors).
- `mrun doctor` checks the config file; `mrun cloud up`'s confirmation marks values from the file
  with `(config)`.
- `cloud up --retry <DUR>` keeps trying sold-out GPUs, with `--max-price` and `--any-cloud` (both
  clouds, cheapest first).

- Remote GPU mode (opt-in, 013, D064): `mrun cloud up|down|status|gpus|models|logs` rents a
  RunPod GPU, runs vLLM there under `mrun agent`, and relays it on 127.0.0.1 (`mrun serve
  --relay`), so `launch`, `connect` and `ps` work unchanged. **Prompts and code are sent to
  RunPod.** The pod's endpoint needs a per-pod token; the pod stops itself after 30 idle minutes
  (counted from model ready) or 8 hours. A sold-out GPU lists the alternatives in stock;
  `--retry`, `--max-price` and `--any-cloud` keep trying. A failed start saves the pod's log to
  `~/.mrun/logs/` and deletes a volume it created; `mrun cloud logs [-f]` reads a running pod's
  log. vLLM runs with `--max-num-seqs 16` (`cloud up --max-num-seqs`). The relay drops Messages
  fields vLLM does not take and reports `backend: vllm <version>`; `mrun launch` waits for a pod
  that is still starting. New crate `mrun-runpod`. Measured: Qwen3-Coder-Next FP8 on a 96 GB
  RTX PRO 6000, about 190 tok/s, passes the client gate with all three clients.
- `mrun launch opencode` turns off OpenCode 2.x's Code Mode tool (`execute`): the served models
  route shell commands through it and fail. The relay logs every upstream error (URL, status,
  message) to the server log and never passes on an error with an empty message.
- A pod that stopped itself is reported by the relay with the limit and the command to start a
  new one, and its `/health` fails, so `mrun launch` does not start a client against it.
- `mrun serve`: Anthropic Messages (with `count_tokens`), OpenAI Chat Completions and OpenAI
  Responses (stateless) on `127.0.0.1:8484`, with streaming, tool calls, reasoning, keepalives
  sent 10 s after the last byte, a bounded queue (240 s deadline) and cancellation on disconnect.
- The MLX engine with three families: `qwen3` (alias `qwen3`, Qwen3-8B), `qwen3_5_moe` (alias
  `qwen3.6`, Qwen3.6-35B-A3B with Gated DeltaNet layers) and `qwen3_moe` (Qwen3-Coder, no alias).
  Greedy output matches mlx-lm 0.32.0 token for token; pipelined decode; prefill with at most two
  layers in flight and cancellation after every layer.
- Token splicing: the model's own reply ids replace the re-rendered previous turn, so a follow-up
  reuses the whole conversation (also in `mrun run`). Replies are matched by their exact canonical
  form (no hash), and each record carries the earlier replies spliced into its prompt, so a tool
  loop of any length needs only its newest record; `/mrun/status` reports
  `cache.spliced_tokens_total`.
- Prefix cache: exact reuse across turns, subagents and title requests; flat branches reachable
  through holder chains; multi-turn conversations protected from eviction; hybrid checkpoints by
  byte budget; a miss report (`new`, `diverged`, `evicted`, `disabled`, with message and role) on
  every response; macOS memory-pressure handling.
- Tool calls in each family's syntax (Hermes JSON, Qwen XML with schema coercion, Harmony);
  constrained decoding for JSON response formats and `required`/named tool choice.
- A loop guard that ends repeating output (in-band warning `repetition-stopped`).
- `mrun launch` (pulls and starts the per-machine default model), `connect`, `ps`, `doctor`,
  `stop`, `logs`, `cache clear`, `completions`; bare `mrun` prints status; `pull` with progress,
  retries and resume; aliases pinned to revisions and file hashes.
- `mrun bench` with whole-session validity, a truncation canary, paired per-turn comparisons with
  bootstrap intervals and a separate natural-stop quality pass.
- Idle-sleep prevention while requests run.
- `mrun doctor` checks that MLX's GPU kernels are extracted (`gpu-kernels`).
- The client gate: 10 scripted read-edit-test tasks (`crates/mrun/tests/clients/tasks/`) and a
  runner, `scripts/client_gate.py` (Python standard library), that drives Claude Code, Codex and
  OpenCode through `mrun launch` and reports pass rates with Wilson 95 % intervals. First run on
  an M4 Pro: `qwen3.6` meets the gate with all three clients.

### Changed

- **Breaking:** boolean preference flags take an optional value, `--flag[=<BOOL>]`:
  `--any-cloud=false` turns off what the config file or a variable turned on; plain `--flag` still
  means `true`. Boolean variables accept `true`/`false`, `1`/`0`, `yes`/`no`.
- nuvec backend (016), now on nuvec 0.6.1: memory counts memory-mapped weights as MLX does;
  vision, MTP and rotary-table tensors are not loaded (0.83 GiB of Qwen3.6); Qwen3.6 prefill at
  262K tokens is 1.17× MLX; `doctor` says whether NAX kernels run and why not. On MLX, a
  quantized checkpoint without biases loads. CPU bfloat16 casts now round as MLX's (nuvec 0.6.0).

- `mrun stop` stops any mrun server on the port, also one started in a terminal without `--detach`.
- `qwen3.5` is marked **preview**: it measured below the client gate (Claude Code 27/30, Codex
  26/30, OpenCode 28/30). `mrun ls` shows `preview`, `mrun show`, `mrun launch` and `mrun doctor`
  say so with the figures. It stays the default on Macs where `qwen3.6` does not fit.
- `launch codex` passes `model_context_window` instead of writing a model catalog file (its
  schema changes per Codex release).
- Claude Code side requests (titles, recaps) no longer evict the main conversation's cache on the
  hybrid models, and Claude Code's replies splice again: tool-call arguments equal to the schema
  default are ignored when matching the reply.
- `mrun doctor --report` prints a paste-ready hardware report (chip, working set, numerics class,
  calibration, findings; no user paths) for the hardware-report issue template.
- `MRUN_BUDGET_GIB` / `--budget-gib` now also drive `ls`, `doctor` and the default-model pick, so a
  smaller Mac can be simulated exactly on a larger one.
- Long prompts on small Macs: attention runs in query blocks when its score buffer would exceed
  512 MiB (prompts up to 8K unchanged), so `qwen3.5` gets about 84K tokens of context on a 16 GB
  Mac instead of 29K. Chunks stay at 2,048 tokens up to 16K unless a layer takes longer than
  250 ms on this machine; the timing is stored in `~/.mrun/calibration.toml`.
- Qwen XML tool parameters without a declared type become JSON when the value is an object or
  array (as mlx-lm 0.32); prompt tokenization is memoized per segment (`count_tokens` on a
  78K-token follow-up in about 2 ms); two log kinds (`ignored-field`, `unenforced`) replace eleven
  warning codes.
- `mrun --version`, `--info` and `/mrun/status` show the numerics class (NAX compiled and
  available, GPU family).

- Memory budget is the MLX recommended working set, lowered with `--budget-gib` and never above
  physical − 2 GiB; scratch is sized at the depth the capacity can reach. `--memory-fraction`,
  `--reserve-gib` and `--allow-over` are removed.
- Default model per machine: the largest alias that leaves at least 64K tokens of context —
  `qwen3.6` (Qwen3.6-35B-A3B) where it fits, else `qwen3.5` (Qwen3.5-9B, new `qwen3_5` family).
  Qwen3-8B (`qwen3`) stays pullable by repo id without an alias; `qwen3-coder` and `gpt-oss` are
  gone. `launch` refuses a model below the client's minimum context and names the budget needed.
- Miss reports add the cause `gap` (a hybrid resumed far behind the match), `unreused` (avoidable
  recompute) and `ring_wrapped`; the dropped-path ring holds 1024 entries.
- Splicing follows the template: replies are recorded with the cache path that holds them, the
  current turn is spliced with its reasoning (restored if a client dropped it), earlier turns are
  rendered as the template renders them.
- Streaming requests no longer time out in the queue; non-streaming ones are refused up front when
  the estimated wait exceeds 240 s. One default output limit, `min(remaining, 32,768)`, for all
  three APIs. Chat Completions keepalives are data chunks. On SIGTERM the server cancels queued
  work and gives running requests 5 s.
- `launch claude-code` disables Claude Code's non-streaming fallback, its cloud-provider switches,
  error reporting and feedback surveys, disallows WebSearch, forces the local model onto
  subagents and turns idle compaction off through `--settings`. `connect --write` writes client
  configuration with a diff and a backup; `connect --restore` undoes it. `connect` prints Crush
  and Zed settings. `mrun stop` exits 0 when nothing runs.
- New `/metrics` (Prometheus text) and per-request live status in `/mrun/status`; `mrun ps --watch`.
- Release builds: a macOS 26.2 artifact with MLX's M5 Neural Accelerator kernels (the build fails
  without them) and a macOS 15 artifact; the MLX build cache keys on the SDK version and
  deployment target.
- `mrun` builds the MLX engine by default and embeds the Metal library (no sidecar file); MLX and
  mlx-c are fetched as sha256-pinned source tars and built once per configuration into
  `~/Library/Caches/mrun-mlx-sys`. Building Metal needs Xcode with the Metal toolchain;
  `MLX_SYS_METAL=0` builds the CPU backend.
- The FFI crate is now `mrun-mlx-sys` (the name `mlx-sys` belongs to mlx-rs on crates.io); all
  three crates are publishable.
- Exit codes: `doctor` failures exit 7; the reserved code and the unused E011/E021 are gone.
- `POST /mrun/cache/clear` always answers 200 `{branches_dropped, bytes_freed}` once the worker has
  run it between requests (no 202 variant); the clear also forgets earlier evictions, so the next
  request reports `new`. `mrun cache clear` prints `cleared N branches (M MiB) and the live
  conversation`.
- Scratch is the warm-up peak plus an analytic bound for the largest chunk, × 1.25; MLX's buffer
  cache is cleared after state growth, each chunk and every 256 decode positions. Measured on
  48 GB (weights + scratch + state): Qwen3.6 18.2 + 5.4 + 13.8 GiB, Qwen3-8B 4.3 + 1.3 + 31.8 GiB.
- Pre-load fits (`pull`, `ls`, the per-machine default) use each family's measured scratch (1.4 GiB
  for Qwen3, 5.5 GiB for Qwen3.6, else 1 GiB): a 32 GB Mac now defaults to `qwen3`, 36 GB and up
  to `qwen3.6`.
- Without `--model`, `serve`, `run` and `launch` stop with E040 when no model leaves 32K tokens of
  context on this Mac, instead of picking one that cannot work; `launch --model` reads
  `MRUN_MODEL`.
- Memory pressure is also handled while the server is idle; critical pressure then frees the idle
  conversation too.
- Benchmark profiles not pinned to a release are compared but never headline.
- `/mrun/status` drops `model.architecture` and `model.quantization` (a copy of `family` and
  always `null`).

### Removed

- Harmony (gpt-oss) chat syntax and its parser; it returns with gpt-oss.
- Quantization modes other than affine 4- and 8-bit (mxfp4, nvfp4, mxfp8).

### Fixed

- On hybrid models (`qwen3.5`, `qwen3.6`) a plain chat follow-up re-read the whole conversation:
  the template re-renders the earlier reply, and the nearest checkpoint was the user message's
  start (an 8K-token file: 0 reused). A checkpoint at the reply's header now lets it resume there
  (021, D068). The window words this case "reply re-rendered" in grey instead of "client changed"
  in amber. A request near the memory limit drops its own new checkpoints instead of failing.
- `MRUN_AGENT_LOG`, `MRUN_CALIBRATION_DEBUG`, `MRUN_FAILED_STOP`, `MRUN_MAX_NUM_SEQS` and
  `MRUN_VLLM_ARGS` drew a false "unknown variable" warning; a test now keeps the known list equal
  to the variables the commands read.
- `mrun ps` and `mrun cloud status` kept adding cost after a pod had stopped (an exited pod showed
  "up 9h33m, about $16"): the estimate now ends when the agent stopped answering. `cloud status`
  showed "on CPU" for GPU pods once RunPod omitted the GPU; it now uses the GPU `cloud up` recorded.
- A keepalive test of the API handler depended on wall-clock timing and failed under load; it
  runs on paused time.

- OpenCode and Codex ran Qwen 3.5/3.6/3.8 with thinking off: Chat Completions and Responses requests
  without a reasoning effort now get the model's default (thinking on); `reasoning_effort: "none"`
  or `"minimal"` still turns it off.

- `mrun launch opencode` showed OpenCode's own default model: OpenCode 2.x's UI talks to a shared
  background service that never sees mrun's configuration. `launch` now starts OpenCode with
  `--standalone` (a private server that reads it), unless you pass `--standalone` or `--server`.

- The budget ignored memory other processes and VMs hold. It is now
  `min(working set, physical − 2 GiB, available − margin)`; `serve`, `run` and `launch` refuse a
  model that does not fit with E041, naming the largest processes, and `doctor`, `ls`, `bench`
  and the client gate account for them (D062).

- Constrained decoding (`tool_choice` required or named, JSON formats) panicked on real Qwen models
  because the mask had the tokenizer's vocabulary size, not the model's.
- Engine error paths left partial state (prefill, sampling readback, moves); a failed move now
  poisons the engine until reset, and `attach` no longer commits a pending token or over-allocates.
- A conversation lost its protection after its first subagent round trip and was evicted first.
- Hub metadata from a mirror (`HF_ENDPOINT`) could name paths outside the model cache; commit and
  blob ids are now validated, also before `mrun rm` deletes anything.
- Any request error cleared the whole prefix cache; capacity refusals were reported as 400 instead
  of `context_length_exceeded`; a cancelled queued request kept its queue slot.
- The loop guard could cut repetitive tool-call arguments.
- Template-rewritten content was tokenized with special-token parsing.
- `--queue 0` and `--queue-wait-s 0` made the server refuse everything.

- Anthropic Messages: a queue deadline is 529 `overloaded_error` (was 504 `timeout_error`), and a
  failed model load is 500 `api_error` instead of an overload that clients retry forever.
- A cache directory that cannot hold the GPU kernels stops engine load with an error naming the
  path, instead of an abort inside MLX.

- Multi-argument Hermes tool calls were cut at the first comma inside the arguments.
- Special tokens inside message content (for example `<|im_end|>` in a file) are no longer read as
  control tokens.
