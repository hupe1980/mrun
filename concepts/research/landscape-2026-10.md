# Local Runtime and Coding-Agent Landscape, October 2026

**Date**: 2026-10-14. **Scope**: This document verifies, corrects and extends
the former MARKET_LANDSCAPE.md (removed in the 2026-10-14 redesign) and the
[mistral.rs study](mistralrs-study.md). It does not repeat them.
Candle model coverage is in [candle-capabilities.md](candle-capabilities.md).

**Method**: Each project was shallow-cloned at its default branch on 2026-10-14, and its README, docs
and selected sources were read. Release tags and dates come from `git ls-remote` and the tag commit
date. Web search filled the remaining gaps. github.com was reachable. huggingface.co, lmstudio.ai
and omlx.ai were **not** reachable from this container, so model-card facts are second-hand unless
noted. Nothing was built or run, and no number in this file was measured by mrun.

**Markers**:
- **[V]** means verified in the primary source: the repository source or docs at the stated date.
- **[U]** means unverified: secondary sources, vendor claims, or values computed from configs.

## 1. Corrections to existing docs

| Topic | Earlier assumption | Current state (2026-10) |
|---|---|---|
| Ollama engine | llama.cpp comparator | Ollama now has its own **MLX runner** for MLX/safetensors models (`mlxrunner/`) and keeps llama.cpp for GGUF [V]. The MLX path shipped as a preview in 0.19, around 2026-03-30 [U: date] ([blog](https://ollama.com/blog/mlx)). Its `mlxrunner/prefix_cache.go` is a compressed prefix **trie** with per-layer snapshots, including whole-state snapshots for recurrent and rotating layers, an 8 GiB paged-out byte cap, MTP drafting (`mtp.go`) and DFlash (`dflash.go`) [V]. This is the closest existing equivalent of mrun's reuse design. |
| llama.cpp versioning | `bNNNN` builds only | Semver tags now exist: **v0.6.0 (2026-10-05)** [V]. Rolling builds continue (b11622, 2026-10-14) [V]. |
| "Exact prefix reuse is easy" | KV truncation | Most new 2026 coding models (Qwen3.5/3.6/3.8, Qwen3-Coder-Next, Nemotron 3) are **hybrid linear-attention** models. A recurrent state cannot be truncated, so reuse needs state checkpoints at chosen offsets (§3.2). |
| Comparator set | No MLX agent server listed | **oMLX** (jundot/omlx, v0.7.0, 2026-10-01 [V]) targets mrun's niche directly: an MLX server for Claude Code with a block KV cache, a RAM tier plus an SSD tier that survives restarts, and Homebrew service DX [V README]. |
| Rust newcomers | uzu, Imparo | Add **Crane** (Candle-based). Its README says it runs Qwen3.5/3.6/3.8 hybrid GDN on Metal and serves OpenAI chat [V README, unbenchmarked]. **Goose** embeds llama.cpp through `llama-cpp-2 =0.1.146` [V]. |

## 2. Inference runtimes (state on 2026-10-14)

### 2.1 Versions and API surfaces

| Runtime | Latest tag (date) | Chat Completions | Anthropic Messages | OpenAI Responses | Other |
|---|---|---|---|---|---|
| llama.cpp `llama-server` | v0.6.0 (2026-10-05); b11622 (10-14) | yes | `/v1/messages` + `count_tokens` (since PR #17570, blog 2026-01-19); tool use requires `--jinja` | `/v1/responses` (converted to chat internally) + `input_tokens` | `/completion`, `/infill`, `/slots`, `/metrics`, router `/models/*` [V README] |
| Ollama | v0.40.4 (2026-10-13) | yes | yes, subset; tool-choice not fully supported | **stateless only**: no `previous_response_id` or `conversation` | native `/api/*` [V docs/api] |
| LM Studio (closed core; `lms`, `llmster` daemon) | 0.4.x (0.4.0 2026-01-28; 0.4.1 2026-01-30 added `/v1/messages`) [U] | yes | yes (0.4.1) [U] | yes: Codex's built-in `lmstudio` provider uses Responses (`model-provider-info/src/lib.rs:686`) [V indirect] | native `/api/v1/*` stateful chat + MCP [U] |
| mlx-lm `mlx_lm.server` | v0.32.0 (2026-09-30) | yes; `/v1/completions` | **no** | **no** | reference server only [V `server.py`] |
| vllm-metal | v0.31.0 (2026-10-12) | via upstream vLLM API server [U] | upstream vLLM [U] | upstream vLLM [U] | Homebrew tap; paged varlen Metal kernel; MLX compute [V README] |
| mistral.rs | v0.9.4 (2026-09-24) | yes | yes | yes, plus `/v1/files` and Skills | web UI, MCP client, server-side tools [V README] |
| oMLX | v0.7.0 (2026-10-01) | yes | yes, incl. adaptive thinking | not listed | admin UI, `/v1/rerank` [V README] |
| LocalAI | v4.11.0 (2026-10-02) | yes | yes | yes | many backends incl. MLX and llama.cpp [V README grep] |
| llamafile | 0.10.6 (2026-09-15) | yes | yes | yes | single-file executable, now built on llama-server [V README grep] |
| Lemonade | v2026.42.1 (2026-10-12) | yes | yes | yes | also Ollama-compatible API; macOS supported [V README] |
| exo | v1.0.71 (2026-04-23); main active 10-13 | yes | yes | yes | also Ollama API; distributed MLX [V README grep] |
| shimmy (Rust, llama.cpp/MLX) | v2.6.4 (2026-08-30) | yes | yes | yes | `cargo install shimmy` [V README] |
| uzu (Rust, Apple-first) | 0.6.7 (2026-10-14) | SDK-first, no server documented | – | – | Rust/Swift/TS/Python bindings [V README] |
| Crane (Rust, Candle) | no tags; main 2026-10-14 | `crane-serve` | – | – | GGUF Qwen3.5 GDN [V README] |
| Candle | 0.11.0 (2026-06-26); main 12bb5d1 (10-14) | library only | – | – | see [candle-capabilities](candle-capabilities.md) |

**Finding**: In late 2026, OpenAI Chat Completions, Anthropic Messages and stateless Responses
together are **table stakes**. Every serious server except mlx-lm ships all three [V]. Nearly all
of them implement Messages and Responses by converting to an internal chat request.

### 2.2 Agent-relevant engine features

| Runtime | Prefix reuse | Persistence | Hybrid/SWA models | Speculative / MTP | Multi-model |
|---|---|---|---|---|---|
| llama.cpp | `--cache-prompt` (on by default); `--cache-reuse N` (KV-shift chunk reuse, default 0); `--cache-ram` host prompt cache, default 8192 MiB; `--cache-idle-slots` | `/slots/{id}?action=save\|restore` to `--slot-save-path` | `--ctx-checkpoints` (default 32 per slot) for SWA/hybrid; `--swa-full` | `--spec-type` draft-simple, draft-eagle3, **draft-mtp**, draft-dflash, n-gram variants | router mode: `--models-dir`, `--models-max 4`, autoload, `--sleep-idle-seconds` [V README] |
| Ollama (MLX runner) | trie with snapshots; re-evaluates ≥1 token | in-memory, paged-out snapshots ≤ 8 GiB | whole-state snapshots at node ends | MTP, DFlash | scheduler loads and unloads models [V source] |
| mlx-lm server | `LRUPromptCache`, `--prompt-cache-size`, `--prompt-cache-bytes` | none in server (`cache_prompt.py` CLI writes files) | `ArraysCache` (GDN), `RotatingKVCache` (SWA) | `--draft-model` | single model [V] |
| oMLX | vLLM-style blocks, prefix sharing, copy-on-write | **SSD tier** (safetensors) survives restart | via mlx-lm caches [U] | [U] | multiple, LRU [V README] |
| mistral.rs | see study §3a | none found | qwen3_next, qwen3_5 models present | yes | load/unload [V README] |

KV quantization: llama.cpp `-ctk/-ctv` (q8_0/q4_0 …), mlx-lm `--kv-bits`, `--kv-group-size` and
`--quantized-kv-start` [V]. Reasoning: llama-server `--reasoning-format none|deepseek|deepseek-legacy`
(default auto) [V]. Tool parsing: llama.cpp has a Qwen3-Coder XML parser ("also used by Nemotron
Nano 3, Qwen3.5", `common/chat.cpp:1232`) [V], and mlx-lm auto-selects `qwen3_coder` from the
template [V].

### 2.3 CLI and DX

| Tool | Install | Pull / run / serve | Distinctive DX |
|---|---|---|---|
| Ollama | app, brew, script | `pull`, `run`, `serve`, `ps`, `launch <client>` | `ollama launch claude\|codex\|opencode` writes client config. Its Claude launcher sets `CLAUDE_CODE_ATTRIBUTION_HEADER=0` to protect prefix caching (`cmd/launch/claude.go:73`) [V]. Context defaults by VRAM: <24 GiB 4k, 24–48 GiB 32k, ≥48 GiB 256k [V docs]. |
| llama.cpp | brew, binaries | `-hf user/repo:quant` downloads; router `/models` POST/DELETE | One binary. Every flag also has an `LLAMA_ARG_*` env var. INI presets. Web UI. |
| LM Studio | app; `lms`, `llmster` | `lms get/load/server start`, `lms daemon up` [U] | GUI-first; headless daemon since 0.4.0 [U] |
| mistral.rs | `curl … install.sh` | `run`, `serve`, `bench`, `tune`, `doctor`, `from-config` | `tune` emits a hardware-aware config; `doctor` diagnoses Metal and HF access [V README] |
| oMLX | brew tap, mac app | `omlx serve/start/stop`; `brew services` | Zero-config defaults (`~/.omlx/models`, port 8000); settings persisted on first run; admin UI with a built-in PP/TG benchmark [V] |
| vllm-metal | brew tap | `vllm serve` | Python under the hood, hidden by Homebrew [V] |

## 3. Coding-agent clients: what they require

| Client (latest tag) | Wire protocol to a local server | Notes |
|---|---|---|
| OpenCode v2.0.29 (2026-10-14; repo `anomalyco/opencode`) | **Chat Completions** via `@ai-sdk/openai-compatible`; Anthropic via `@ai-sdk/anthropic` + `baseURL` | Docs show llama.cpp and LM Studio as `openai-compatible` [V docs]. mrun's pinned client. |
| Claude Code | **Anthropic Messages** (`ANTHROPIC_BASE_URL`, `ANTHROPIC_AUTH_TOKEN`), streaming, `tool_use`, thinking blocks; `count_tokens` used if present | Before v2.1.181 a per-request attribution header at the top of the system prompt defeated exact prefix reuse unless `CLAUDE_CODE_ATTRIBUTION_HEADER=0` was set; since then it is stable per conversation for custom base URLs (fact check 2026-10-15). `CLAUDE_CODE_TOTAL_TOKENS_REMINDER=off` removes a volatile token-count reminder in `~/.claude/settings.json` [V via Ollama launcher; magnitude U]. Clients recommend 64k+ context [V Ollama docs]. |
| OpenAI Codex CLI rust-v0.162.1 (2026-10-09) | **Responses only**: `wire_api = "chat"` now errors ("no longer supported") [V `model-provider-info/src/lib.rs:100,130`] | Built-in OSS providers `ollama` and `lmstudio` (port 1234), both Responses [V]. |
| Crush v0.98.1 (2026-10-09) | `openai-compat` or Anthropic-compatible provider types | [V README] |
| Goose v1.54.0 (2026-10-08) | OpenAI, Anthropic and Ollama providers, plus **embedded llama.cpp** | [V source] |
| Aider v0.86.2 (2026-02-11) | Chat Completions via LiteLLM | Release activity has slowed [V tag] |
| Cline v4.1.24, Roo Code v3.54.0, Continue v2.1.0 | OpenAI-compatible, Ollama and LM Studio providers; Anthropic with custom base URL | [U: provider lists from docs memory/search] |
| Zed | Ollama, LM Studio, "OpenAI-compatible" (`api_url`) | [U: zed.dev docs via search] |

**Reach ranking**: (1) Chat Completions covers OpenCode, Aider, Continue, Cline/Roo, Zed, Crush and
Goose. (2) Anthropic Messages covers Claude Code, the most demanding client, plus Crush, Goose and
OpenCode. (3) Responses is needed **only** for Codex, which accepts nothing else.

## 4. Models for M4 Pro 48 GB (late 2026)

Budget: Metal's recommended working set is about 75% of 48 GB (≈36 GB), and mrun's own rule caps
it further. Weights plus KV plus about 2 GB of activations must fit, so practical weights are
≤ ~24 GB. Sizes below are computed at ≈4.85 bits per weight for Q4_K_M [U: computed], unless stated.

### 4.1 Candidates

| Model (release) | Type | Q4 size | Attention for prefix reuse | Tool format | Notes |
|---|---|---|---|---|---|
| **Qwen3-Coder-30B-A3B-Instruct** (2025-07-31 [U]) | MoE 30.5B / 3.3B active | ≈18.5 GB | **full GQA** (48 layers, 4 KV heads × 128 [U config]) → ≈96 KiB/token BF16, 1.5 GiB at 16k | Qwen3-Coder XML (`<function=…>`) [V parser] | Fast. Truncatable KV. Candle has `quantized_qwen3_moe`, but its MoE GEMM is CUDA-only, so Metal needs work ([candle-capabilities](candle-capabilities.md)). |
| **Qwen3.6-27B** (2026-04-22 [V Qwen README]) | dense 27B, hybrid | ≈16.4 GB (Q8 ≈29 GB) | **3:1 Gated DeltaNet : gated attention** (64 layers, 16 full; 4 KV heads × 256) [U card mirror; V `qwen3_5.py` `full_attention_interval=4`] → ≈64 KiB/token, plus fixed recurrent state per snapshot | Qwen3-Coder XML [V] | SWE-bench Verified 77.2 (vendor) [U]. Dense, so decode ≈ 11–14 tok/s on M4 Pro [U: bandwidth estimate]. Checkpoints carry MTP weights (`mtp.*`) [V]. |
| **Qwen3.6-35B-A3B** (2026-04-16 [V]) | MoE hybrid (qwen3_5_moe) | ≈21 GB | hybrid GDN, same as above | Qwen3-Coder XML | SWE-bench Verified 73.4 (vendor) [U]. M4 Pro MLX ≈85 tok/s TG at 1k (oMLX community runs) [U]. |
| **Qwen3.8-27B** (2026-08-14 [V Qwen README]) | dense hybrid, VL | ≈16.4 GB | same 3:1 hybrid (vLLM docs: 16 of 64 layers full) [U] | Qwen3-Coder XML [U] | Newest Qwen dense model; `reasoning_effort`, `preserve_thinking` [V README]. |
| gpt-oss-20b (2025-08 [U]) | MoE 21B / 3.6B active, MXFP4 | ≈13 GB [U] | **alternating SWA-128 / full**, with sinks [V `gpt_oss.py`] → rotating layers cannot be rewound past 128 tokens without `--swa-full`-style storage | Harmony | Absent in Candle; MXFP4 not a GgmlDType [V candle audit] |
| GLM-4.7-Flash (2026-01-20 [U]) | MoE ≈30B / 3B active | ≈18 GB | **MLA** (`kv_lora_rank 512`, `q_lora_rank 768`) [V `glm4_moe_lite.py`]: full attention with a small compressed KV | GLM XML [V `glm47.py` parser] | Truncatable; low KV bytes |
| Gemma 4 26B-A4B / 31B (2026-04 [U]) | MoE / dense | ≈15.5 / 19 GB | SWA-1024 + global layers [U] | Gemma 4 native [V parser] | Candle `gemma4/text.rs` uses `RotatingKvCache` [V audit] |
| Devstral Small 2 24B (2025-12-09 [U Mistral news]) | dense | ≈14.5 GB | full attention [U] | Mistral | SWE-bench Verified 68.0 (vendor) [U] |
| Muse Glimmer 30B (Meta, 2026-08-10 [U]) | ≈30B, Apache 2.0 | ≈18 GB | [U] | ATEM tool calling (mistral.rs) [V README] | AA Index 35 vs Qwen3.6-27B 38 [U] |
| Qwen3-Coder-Next 80B-A3B (2026-02 [U]) | MoE hybrid | ≈45–48 GB | hybrid GDN | Qwen3-Coder XML | **Does not fit** 48 GB with headroom |
| Qwen3-8B (mrun spec reference) | dense 8B | ≈5 GB | full GQA, 8 KV heads | Hermes JSON in `<tool_call>` | Engineering baseline, not a product model |

### 4.2 Architecture implications for exact prefix reuse

- **Full attention (Qwen3, Qwen3-Coder-30B-A3B, GLM-4.7-Flash MLA, Devstral)**: Reuse is a longest
  token-prefix match plus truncation. Any offset is reusable, and branches can share frozen KV
  blocks. This is what mrun's FR-006..FR-010 currently assume.
- **Sliding window (gpt-oss, Gemma 4)**: Rotating layers keep only the last W tokens, so a shorter
  shared prefix cannot be restored. The options are to keep full SWA KV (memory cost) or to
  checkpoint at offsets. llama.cpp exposes both (`--swa-full`, `--ctx-checkpoints`) [V].
- **Hybrid linear attention (Qwen3.5/3.6/3.8, Coder-Next, Nemotron-H)**: GDN layers hold a
  fixed-size state, about 75–150 MB per snapshot for 27B by config arithmetic [U]. It is valid only
  at the exact offset where it was captured. Reuse is limited to "deepest captured checkpoint ≤ match".
  Snapshot policy (where to capture: end of system prompt, end of each turn, before generated
  tokens) becomes the core algorithm. This is what Ollama's trie (`prefix_cache.go` invariants) and
  llama.cpp checkpoints do [V]. The upside is that the attention KV is only ¼ of the layers, so
  bytes per token are low.

## 5. Throughput evidence on M-series

| Source | HW | Model / quant | Prefill tok/s | Decode tok/s | Marker |
|---|---|---|---|---|---|
| llama.cpp discussion #4167 (build 8e672ef, 2023-11) | M4 Pro 20c, 273 GB/s | Llama-2 7B Q4_0 | 439.8 (pp512) | 50.7 (tg128) | [V page] |
| same | M4 Pro 20c | Llama-2 7B Q8_0 / F16 | 449.6 / 464.5 | 30.7 / 17.2 | [V] |
| same | M4 Max 40c, 546 GB/s | Llama-2 7B Q4_0 | 885.7 | 83.1 | [V] |
| mlx-lm `BENCHMARKS.md` (mlx 0.29.2, mlx-lm 0.28.2) | M4 Max 64 GB | Qwen3-30B-A3B-2507 4-bit | 1753.9 (2048) | 113.3 | [V repo] |
| same | M4 Max 64 GB | Qwen3-4B-2507 4-bit | 1622.3 | 134.5 | [V] |
| Ollama MLX blog (0.19 vs 0.18) | M-series (M5 emphasized) | Qwen3.5-35B-A3B | 1154 → 1810 | 58 → 112 | [U vendor] |
| Blog comparison | unspecified | Qwen3.6-35B-A3B | MLX 277 vs llama.cpp 460 | MLX 61.7 vs llama.cpp 39.1 | [U] |
| NixOS forum | M4 MBP | Qwen3-Coder-Next Q4_K_M, llama.cpp | – | 45.9 | [U] |
| arXiv 2511.05502 | M2 Ultra 192 GB | Qwen2.5 | – | MLX ≈230 > MLC ≈190 > llama.cpp ≈150 > Ollama 20–40 | [U abstract] |
| mistral.rs README | CUDA only (GB10/B200/H100) | Gemma 4 | beats llama.cpp Q8 | ≈ parity | [V]; **no Metal numbers published** |
| Candle / Crane | – | – | – | – | **No credible published Metal comparison found** |

**Reading**: Decode is bandwidth-bound. An M4 Pro has about 0.5× the decode throughput of an M4
Max 40c, so expect roughly 55–70 tok/s for a 4-bit A3B MoE and roughly 11–14 tok/s for a dense 27B
at Q4 [U: scaled]. MLX tends to lead on decode, and llama.cpp often leads on prefill for MoE and
hybrid models [U]. Nobody has published a Candle-on-Metal comparison. The mistral.rs study lists the
fused kernels (RMSNorm+residual, GLU, rotary, GPU top-k) that Candle lacks. A Candle runtime
without new kernels should be **expected to trail** llama.cpp and MLX on raw tok/s. Agent wall-clock
is dominated by prefill on 20–60k-token prompts, and reuse removes that cost. Competitiveness must
therefore be argued on **time-to-first-token after reuse**, not on decode rate.

## 6. DX patterns

**Beloved**, with evidence:
1. **One verb per intent**: `pull`, `run`, `serve`, `ps`, `stop` (Ollama), and `run`, `serve`,
   `bench`, `doctor` (mistral.rs).
2. **Zero-config start**: oMLX's `omlx serve` defaults, persisted on first run. Ollama sizes
   context from VRAM.
3. **Client launchers**: `ollama launch claude|codex|opencode` writes the client config and fixes
   cache-hostile settings (attribution header) [V].
4. **Every flag also an env var** (llama.cpp `LLAMA_ARG_*`), plus a preset file.
5. **Homebrew formula and `brew services`** (oMLX, vllm-metal, llama.cpp).
6. **Visible cache and throughput state**: llama.cpp `/slots`, `/metrics` and prefill-progress SSE.
   oMLX admin benchmark with "partial prefix cache hit testing".
7. **`doctor` / `tune`** commands that explain the hardware fit (mistral.rs).
8. **Model aliases** (`qwen3.6:27b`) that resolve to a pinned artifact digest.

**Anti-patterns**, with evidence:
- Silent context truncation from small default `num_ctx`. Ollama's docs now tell agent users to
  set 64k or more.
- Template or tool-parser regressions per build. Example: LM Studio 0.4.19 broke Qwen thinking
  templates on `/v1/messages` [U issue #2164].
- Binding to `0.0.0.0` by default (mistral.rs).
- Python environments surfacing to users.
- Feature sprawl: web search, code execution, image generation and cloud proxies in the inference
  server.
- Translating Messages or Responses to chat in ways that drop fields silently.

## 7. Implications for mrun (opinionated)

### 7.1 API surfaces, in priority order

1. **OpenAI Chat Completions (streaming + tools)**. This covers the pinned OpenCode client and the
   broadest reach. It is the minimum for the MVP.
2. **Anthropic Messages + `count_tokens`**. Claude Code is the most cache-sensitive client: it sends
   a large, stable system prompt and a growing history. That is exactly where verified exact reuse
   pays off. Build Messages **natively** on the same internal request IR, not as a lossy chat
   translation. Ship docs or a launcher that sets `CLAUDE_CODE_ATTRIBUTION_HEADER=0`. Report the
   first divergent token and the reason when reuse misses, rather than rewriting prompts.
3. **Stateless OpenAI Responses (subset)**, only for Codex, after 1 and 2 are qualified. Reject
   `previous_response_id`, `conversation` and hosted tools explicitly, as Ollama does.
4. Never build the Ollama-native `/api/*` API or stateful Responses.

### 7.2 Models

- **Keep Qwen3-8B** as the engineering and qualification reference. It uses full attention, Candle
  supports it, and it is small.
- **Primary product candidate: Qwen3-Coder-30B-A3B-Instruct (Q4_K_M ≈18.5 GB)**. Its full-attention
  KV makes FR-006..FR-010 directly applicable, MoE gives usable decode on 273 GB/s, and its XML tool
  format is well understood. Cost: Candle's quantized MoE path is CUDA-only, so mrun must provide a
  Metal MoE dispatch. That is a scoped exception to the no-new-kernels rule, or it composes
  existing `QMatMul` per expert.
- **Secondary / strategic: Qwen3.6-27B (or 3.8-27B) hybrid GDN**. It is the strongest coder that
  fits, and it is the family direction. It requires (a) a GDN implementation (absent in Candle; Crane
  shows it is feasible in Candle), (b) a checkpointed-state reuse design, and (c) acceptance of
  ≈11–14 tok/s dense decode. Design the prefix-entry data model **now** to support whole-state
  snapshots at offsets, so the hybrid models are not locked out later.
- **Avoid as targets**: gpt-oss-20b (MXFP4 + SWA-128 + Harmony are three new subsystems),
  Qwen3-Coder-Next (does not fit), and Gemma 4 (SWA plus weaker agentic coding [U]).

### 7.3 Realistic differentiation

The competition is stronger than MARKET_LANDSCAPE assumed. Ollama's MLX runner already has trie
reuse with recurrent snapshots, MTP and DFlash. llama.cpp has checkpoints, a host prompt cache, slot
persistence, MTP and all three APIs. oMLX has a persistent SSD tier aimed at Claude Code. mrun will
not win on tok/s or breadth. A defensible position is narrow:

1. **Provable reuse**: exact-token plus identity-tuple validation, with an explainable miss report
   (first divergent token, which identity field changed). No competitor exposes why a cache missed.
2. **Hard memory admission in bytes** on unified memory: never swap, never OOM mid-turn. The
   competitors do sequence-count or post-hoc checks (study §3c).
3. **Branch-safe snapshots** for agent sub-tasks and retries.
4. **Strict tool-call correctness**: grammar-constrained, no JSON repair, and explicit errors for
   unsupported fields.
5. **One signed Rust binary** with a `doctor` command that reports working set, KV budget and cache
   state.

Claim the advantage only on "TTFT and wall-clock on a recorded agent session" against llama.cpp
(with `--cache-ram` and checkpoints), Ollama MLX and oMLX, all with their reuse enabled.

### 7.4 Do NOT build

- A multi-model router, hot swapping or a model hub mirror. Use one resident model, aliases and
  `pull` from a pinned digest only.
- A web UI, admin panel or chat REPL beyond a minimal `mrun run`.
- Server-side agent tools: MCP client, web search, shell or code execution, file uploads, Skills.
- Continuous batching or paged attention for many users. Use one active request plus a bounded
  FIFO, per spec.
- Multimodal input, embeddings and rerank (beyond a possible later need), a custom quant format,
  fine-tuning or LoRA serving, distributed or Thunderbolt serving, or cloud proxying.
- Speculative decoding in v1. Revisit MTP only for a model that ships MTP heads (Qwen3.6+), and only
  after reuse is qualified.
- An SSD cache tier in v1. Persist prefixes to disk only after the in-memory identity and eviction
  model is proven. oMLX and llama.cpp slot files show the demand.

## Sources (accessed 2026-10-14 unless noted)

- llama.cpp server README (HEAD 23deec0, 2026-10-14): https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
- llama.cpp Anthropic API blog (2026-01-19): https://huggingface.co/blog/ggml-org/anthropic-messages-api-in-llamacpp
- llama.cpp Apple Silicon benchmarks: https://github.com/ggml-org/llama.cpp/discussions/4167
- Ollama repo (HEAD 292d7ce, 2026-10-14): `docs/api/*.mdx`, `docs/context-length.mdx`, `mlxrunner/prefix_cache.go`, `cmd/launch/claude.go`: https://github.com/ollama/ollama
- Ollama MLX announcement (around 2026-03-30): https://ollama.com/blog/mlx ; coverage: https://appleinsider.com/articles/26/03/31/ollama-is-supercharged-by-mlxs-unified-memory-use-on-apple-silicon
- LM Studio 0.4.0 (2026-01-28): https://lmstudio.ai/blog/0.4.0 ; 0.4.1 changelog (2026-01-30): https://lmstudio.ai/changelog/lmstudio-v0.4.1 ; Claude Code docs: https://lmstudio.ai/docs/integrations/claude-code ; issue: https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/2164
- mlx-lm (v0.32.0): https://github.com/ml-explore/mlx-lm (`SERVER.md`, `server.py`, `BENCHMARKS.md`, `models/`)
- vllm-metal (v0.31.0): https://github.com/vllm-project/vllm-metal
- mistral.rs (28ed6da, 2026-10-12): https://github.com/EricLBuehler/mistral.rs
- oMLX (v0.7.0, 2026-10-01): https://github.com/jundot/omlx ; https://github.com/ml-explore/mlx/discussions/3203
- LocalAI https://github.com/mudler/LocalAI ; llamafile https://github.com/mozilla-ai/llamafile ; Lemonade https://github.com/lemonade-sdk/lemonade ; exo https://github.com/exo-explore/exo ; shimmy https://github.com/Michael-A-Kuykendall/shimmy ; uzu https://github.com/trymirai/uzu ; Crane https://github.com/lucasjinreal/Crane ; Candle https://github.com/huggingface/candle
- Codex `codex-rs/model-provider-info/src/lib.rs` (rust-v0.162.1, 2026-10-09): https://github.com/openai/codex
- OpenCode providers docs (v2.0.29): https://github.com/anomalyco/opencode/tree/dev/packages/web/src/content/docs
- Crush README: https://github.com/charmbracelet/crush ; Goose: https://github.com/block/goose ; Zed: https://zed.dev/docs/ai/llm-providers
- Qwen news/models (README, 2026-10): https://github.com/QwenLM/Qwen3.8 ; Qwen3-Coder: https://github.com/QwenLM/Qwen3-Coder
- Qwen3.6-27B card mirror: https://huggingface.co/coder3101/Qwen3.6-27B-int4-AutoRound ; Qwen3.8-27B: https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Qwen3.8-27B.html
- Devstral 2 (2025-12-09): https://mistral.ai/news/devstral-2-vibe-cli/ ; Muse Glimmer (2026-08-10): https://artificialanalysis.ai/articles/muse-glimmer
- Claude Code cache/attribution header: https://mykolaaleksandrov.dev/posts/2026/06/claude-code-llamacpp-prompt-cache-fix/
- arXiv 2511.05502: https://arxiv.org/abs/2511.05502 ; MLX vs llama.cpp blog: https://yage.ai/share/mlx-apple-silicon-en-20260331.html

## Addendum 2026-10-15 (fact check)

- **Models**: add Laguna XS 2.1 (33B total / 3B active MoE, sliding-window 512 + global layers,
  fits 48 GB, mlx-lm `laguna.py`); Qwen3.8-Flash-Next (~125B/6B) does not fit; Qwen4-27B was
  announced 2026-09-22 without weights. Qwen3-Coder-Next released 2026-02-03.
- **llama.cpp** is at b11633; `llama-server`'s default port changed to 9931 (2026-10-09); it gained
  checkpoint save/restore and a MoE host-memory cache.
- **Ollama** v0.40.5-rc0; its Claude Code launcher also sets `CLAUDE_CODE_TOTAL_TOKENS_REMINDER=off`,
  `AUTO_MODE_SERVER=0` and `MAX_CONTEXT_TOKENS`.
- **oMLX main** added MTP snapshot restore on cache hits, requeueing under memory pressure, caching
  of cancelled prompts and batched prefill.
- **Claude Code ≥ 2.1.273** sends gateway hint headers with `CLAUDE_CODE_GATEWAY_HINT_HEADERS=1`
  (e.g. `x-claude-code-context-compacted`).
- **Codex** added `model_auto_compact_token_limit_scope`; **OpenCode** accepts `limit.input`.
Source: [fact-check-2026-10-15](fact-check-2026-10-15.md).
