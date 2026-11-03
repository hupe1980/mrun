+++
title = "Roadmap"
description = "What mrun 0.1 ships and the gates before its release, the 0.2 release with parallel lanes, exact speculative decoding and a restart-surviving SSD tier, then bigger Macs, images and more clients in 0.3."
weight = 110
+++

mrun is in development; nothing is released yet. Each release has measurable gates on the target
Mac (M4 Pro, 48 GB), the project's only machine, before it counts as done.
Smaller Macs are simulated there with `--budget-gib`; results from other Macs are welcome as
[hardware reports](https://github.com/hupe1980/mrun/blob/main/CONTRIBUTING.md#hardware-reports) and never gate a release.

The route: **0.1 ships the five claims** (exact reuse proven, every miss explained, tool loops
always hit, a budget that holds, an independent benchmark). **0.2 closes the gap** to what other
local servers already have: parallel lanes, speculative decoding and a restart tier, and
adds a sixth claim: speculation that never changes output. **0.3 widens the audience** to
64–128 GB Macs, images and more clients.

## Release 0.1: ship the claims

The mechanisms are built. 0.1 fixes the defects a review found in them, then closes its gates. The
steps, in order:

| Step | Work | Closed by |
| --- | --- | --- |
| **Correctness** | Fixes for the defects found: constrained decoding on real Qwen models, engine error paths, the protection of the main conversation, hub metadata validation, typed request errors, `mrun stop` while generating, the loop guard inside tool calls, Chat keepalives | A regression test for each |
| **Memory and the small Mac** | The budget becomes MLX's recommended working set, scratch is sized at the depth the state can reach, a 64K-token floor for the default alias; the `qwen3_5` family and the `qwen3.5` alias | Fit and parity gates; the default alias on simulated 16, 24 and 32 GB budgets (done: `qwen3.5` fits 84,333 tokens at 16 GB; parity 22/22 for Qwen3-8B and Qwen3.5-9B) |
| **Cache and splicing** | Five miss causes plus `unreused`, checkpoints at marks that mean the same in every API, reply records that live with the conversation they belong to | Prefix-cache gates on `qwen3.5` and `qwen3.6` |
| **Clients** | Verified Claude Code settings, client configuration passed through the environment, `connect --write` with `--restore`, refusal below the client's minimum context | Client gate: at least 10 scripted tasks × 3 runs per client, with a lower confidence bound of at least 0.8 (done for `qwen3.6`: 30/30, 29/30, 30/30; `qwen3.5` measured 27/30, 26/30, 28/30 and ships as a preview; [method](https://github.com/hupe1980/mrun/blob/main/scripts/client_gate.py)). Open: a long compacting session and a queued run |
| **Engine gates** | Parity sets, speed against mlx-lm, calibrated cancellation, admission, soak, cold start | On the target Mac (measured so far: prefill 92–98 %, decode 94–108 % of mlx-lm; cancel p95 ≤ 215 ms); M1 cancellation unmeasured until contributed |
| **Bench** | Committed traces; verified profiles for llama.cpp, Ollama, oMLX and mlx-lm | The first dated comparison, published whatever it shows |
| **Distribution** | Two macOS artifacts (macOS 26.2 and later with the M5 Neural Accelerator kernels, checked at build time; macOS 15 without them), dry run, release candidate | A clean user account on the M4 Pro goes from `brew install` to a working `mrun launch claude-code` |

Correctness comes first, because it changes behaviour the gates measure. A claim that fails its
gate is removed from these docs before the release, not reworded.

**Not in 0.1, and common elsewhere**: speculative decoding, parallel lanes and an SSD tier that
survives restarts (all 0.2); in 0.1 one request runs at a time. Also not in 0.1: non-loopback
addresses with API keys, a background login service, notarization and Developer ID signing.

## 0.2: close the gap

- **Parallel lanes**: 2–4 live sequences, so subagents, titles and compaction no longer wait
  behind the main turn. While one lane reads a prompt, the others keep decoding between its
  layers; how a prompt is split into chunks never depends on other traffic. A new request goes to
  the lane that already holds the longest part of its prompt.
- **Speculative decoding, exact by construction**: every verified token is bit-identical to what
  plain decoding would produce, also on hybrid models. Each alias ships it only if it reaches at
  least 1.15× median decode speed-up on recorded agent traces with no trace class slower;
  otherwise that alias decodes plainly, and the docs say which.
- **Restart tier**: exact prefix entries on SSD for one purpose, surviving a server restart.
- **Observability for lanes**: per-lane fields in `/mrun/status`, `/metrics` and
  `mrun ps --watch` (the endpoints themselves ship in 0.1).
- **`qwen3.8`** (Qwen3.8-27B) as the quality option on Macs with 32 GB or more.
- **[The mrun window](@/docs/ui.md)** (`mrun ui`): the budget, live lanes and every request's
  reuse at a glance, a chat, models that fit, and one-click agent setup.
- **A better small model**: Granite-4.2-8B, LFM2.5-24B-A2B and gpt-oss-20b run the client-gate
  suite against the `qwen3.5` preview; and a loop guard for near-repeating reasoning, the failure
  seen in the `qwen3.5` gate runs.

## 0.3: bigger Macs, more clients

- **64 GB**: Qwen3-Coder-Next (a new `qwen3_next` family).
- **96–128 GB**: Qwen3.5-122B-A10B on the existing `qwen3_5_moe` family.
- **Image input** for screenshots and image tool results, if at least 5 % of recorded sessions
  carry images, with images part of the exact cache key.
- **More clients**: Kilo Code or Cline, Qwen Code, Goose, Copilot BYOK.
- A background login service, and non-loopback addresses with API keys.

## Later

More model families, ordered by agentic score per GB. An opt-in 8-bit KV cache only after a
long-context agentic study shows it does not cost quality. Sharing one computation between lanes
that receive the same prompt at the same time. Notarization and an SBOM.

## Not planned

A model marketplace, running tools on the server, multi-model routing, multi-GPU or distributed serving,
training or fine-tuning, a custom quantization format, Ollama's native API, server-stored
Responses state, approximate or lossy context reuse by default, and performance work on Linux or
CUDA.
