# Fact Check, 2026-10-15

**Scope**: This file re-checks the version, model, client and competitor claims in `concepts/*.md`,
`concepts/research/landscape-2026-10.md`, `concepts/research/mlx-rust.md`, `specs/README.md`, and
the `plan.md`/`research.md` files of specs 001, 002 and 004, all written on 2026-10-14.

**Method**: Tags and HEADs come from `git ls-remote` and blobless clones in the session scratchpad.
Crate and npm versions come from the crates.io and npm registry APIs. Claude Code facts come from
code.claude.com docs fetched today. huggingface.co, opencode.ai and poolside.ai were **not reachable**
from this container, and the GitHub REST/GraphQL APIs were blocked, so model-card facts rely on web
search. Markers: **[V]** means checked against a primary source today; **[U]** means second-hand or
not checkable here.

**Status key**: correct · outdated (it was true on 10-14, or newer facts change it) · wrong ·
unverifiable.

## 1. MLX stack

| Claim | Location | Status | Evidence | Recommended change |
|---|---|---|---|---|
| Pin MLX **v0.32.3** (2026-09-28) | ARCHITECTURE.md:11; mlx-rust.md:12, 256-257; 001 plan.md:22, 73; 001 research.md:8 | **outdated** [V] | **v0.32.4** was tagged 2026-10-14 23:07 +0300 at `36969c0`, which is also `main` HEAD. It adds 97 commits over v0.32.3, and the relevant ones are listed in the next row. Public headers only add ops (`minmax`, `FloorDivide`, `stft`), and the memory API is unchanged (`git diff v0.32.3 v0.32.4 -- mlx/memory.h` is empty). | Bump the pin to v0.32.4 (`36969c0`) and re-hash the tarball. Ollama already runs MLX `bebd642` (one commit before v0.32.4) with mlx-c `a341b49` (ollama `292d7ce`, 2026-10-14), which shows that mlx-c v0.7.0 builds against it. |
| Fixes relevant to mrun in v0.32.4 | (new) | info [V] | SDPA: #4643 (max threadgroup on M1/M2; Ollama carried it as a patch), #4663 (missing stream in layer_norm/SDPA fallbacks), #4650 (negative-stride SDPA mask), #4596 (`sdpa_vector_2pass` unroll). gather_qmm: #4687 (missing stream in the gather-matmul indices cast; this matters for an engine thread with its own stream), #4684 (non-transposed NAX `gather_qmm_rhs`). Quantized matmul: #4641 (split-K partials kept in fp32), #4673 (batched `qvm_split_k` row mismatch), #4516 (fast qmv when N % 8 ≠ 0). Deadlocks: #4556 (eval cleanup), #4689 (Metal fence). Residency: #4623, #4633. | Cite these in mlx-rust.md §5 as the reason for the bump. #4687 and #4556/#4689 directly affect the one-engine-thread, per-layer-cancel design. |
| mlx `main` HEAD = `bebd642` (2026-10-14) | mlx-rust.md:12 | outdated [V] | HEAD is now `36969c0` = v0.32.4 | Update the row. |
| gather_qmm fix #3922 is in v0.32.3 ("should contain" [U]) | mlx-rust.md:223-224, 257 | correct, now **[V]** | Commit `d73eb75` (2026-08-26) "Fix sorted gather_qmm NAX row overflow above 32K (#3922)"; `git tag --contains` lists v0.32.3 and v0.32.4 | Change the [U] to [V]. |
| mlx-c pinned **v0.7.0 = `a341b49`** (2026-09-28) | mlx-rust.md:13, 256; 001 plan.md:22, 73 | correct [V] | `refs/tags/v0.7.0` is an annotated tag object `fa74b25`; it peels to `a341b49`, which is also `main` HEAD. No newer tag exists. Its `CMakeLists.txt:37-38` still fetches MLX `v0.32.2`. | Keep it. Pin the submodule by the peeled commit, not the tag object. Keep overriding `FETCHCONTENT_SOURCE_DIR_MLX`. |
| mlx-lm at `ba8a08f` (2026-10-10); latest tag v0.32.0 (2026-09-30) | mlx-rust.md:14, 84; 001 research.md:20; 002 research.md:8; landscape:37 | correct [V] | `ls-remote` HEAD is `ba8a08f…` ("Support unconverted models #1962"). v0.32.0 is still the latest tag. `pyproject.toml` requires `mlx>=0.32.2`. | None. |
| mlx-rs HEAD `9051034`; crates mlx-rs 0.32.0 / mlx-sys 0.6.0 (2026-09-12) | mlx-rust.md:11, 31 | correct [V] | Repo HEAD is `9051034`, the latest tag is v0.32.0, and crates.io's max_stable is 0.32.0 / 0.6.0, updated 2026-09-12 | None. The rejection rationale (D010) stands. |
| safemlx-sys 0.2.3 (2026-09-07) | mlx-rust.md:32 | correct [V] | crates.io | None. |

## 2. Models

| Claim | Location | Status | Evidence | Recommended change |
|---|---|---|---|---|
| `mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit` and `mlx-community/Qwen3-8B-4bit` exist | mlx-rust.md:172-173; 004 research.md:34 | correct [U] | A search result shows the HF page for the Coder repo (https://huggingface.co/mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit). huggingface.co was unreachable, so the sha and files were not checked. | Before release, pin the repo **revision sha** in `mrun pull` aliases and check it on a host that can reach HF. |
| Qwen3-Coder-30B-A3B-Instruct released 2025-07-31 [U] | landscape:108 | correct [U] | Matches the widely reported Qwen3-Coder Flash release date; the model card was not reachable | None. |
| Qwen3-Coder-Next 80B-A3B, "2026-02 [U]", hybrid, does not fit 48 GB | landscape:24, 117, 218 | correct, now dated | Launched 2026-02-03/04 on the Qwen3-Next-80B-A3B base (hybrid attention + MoE), per https://technews.tw/2026/02/04/qwen3-coder-next-significantly-cuts-inference-costs/ (secondary). One guide (blog.kilo.ai) says it suits "48 GB VRAM"; at ≈45–48 GB Q4 it still exceeds the ≈36 GB Metal working set. | Write "2026-02-03". Keep "does not fit". |
| Qwen3.8-27B released 2026-08-14, dense hybrid 3:1 GDN, 64 layers | landscape:111; DECISIONS.md:65; ROADMAP.md:22 | correct [U→stronger] | NVIDIA NIM catalog: HF release 08/14/2026, 64 layers in 16 × (3 GDN + 1 gated attention), Apache-2.0 (https://catalog.ngc.nvidia.com/orgs/nim/qwen/models/qwen3.8-27b) | Cite the NVIDIA catalog in place of the vLLM-Ascend tutorial. |
| Qwen3.6-27B released 2026-04-22 | landscape:109 | correct [U] | Official blog exists (https://qwen.ai/blog?id=qwen3.6-27b); the date was not re-read | None. |
| Candidate table is complete for 48 GB coding models | landscape §4.1 | **outdated** [V/U] | (a) **Laguna XS 2.1** (poolside) is missing: MoE 33B / 3B active, 40 layers = **30 sliding-window (512) + 10 global**, 262k context, open weights `poolside/Laguna-XS-2.1`. mlx-lm supports it (`mlx_lm/models/laguna.py` at `ba8a08f`, using `RotatingKVCache` + `KVCache`) [V code; card U via https://huggingface.co/poolside/Laguna-XS-2.1]. ≈19 GB at Q4, so it fits. (b) **Qwen3.8-Flash-Next** (`qwen4_exp` preview arch, ≈125B / 6B active [U]) is supported by llama.cpp (#30397), mistral.rs (`3f2515e`, 2026-10-01) and oMLX. Its 4-bit MLX build is ≈112 GB, so it does **not fit**. (c) **Qwen4-27B** was announced 2026-09-22 (Apsara) with **no weights, card or architecture** yet (https://llmlearner.com/models/qwen4-27b, https://www.yottalabs.ai/post/qwen-4-27b-release-date-specs-hardware-what-is-known-2026) [U]. | Add a Laguna XS 2.1 row ("SWA-512 + global, the same reuse class as gpt-oss/Gemma 4"). Add a Flash-Next row marked "does not fit". Add Qwen4-27B to the watch list: when its weights land, re-check whether it is GDN-hybrid before M4 picks the default model. The Qwen3-Coder-30B-A3B primary choice is unaffected. |
| Muse Glimmer 30B, GLM-4.7-Flash, Devstral Small 2, Gemma 4 rows | landscape:113-116 | correct (arch) [V partial] | mlx-lm `ba8a08f` has `muse_glimmer.py`, `glm4_moe_lite.py`, `mistral3.py`/`mistral4.py`, `gemma4*.py` | None. |

## 3. Clients

| Claim | Location | Status | Evidence | Recommended change |
|---|---|---|---|---|
| Claude Code **2.1.300** (npm, 2026-10-14) | 002 plan.md:14; 002 research.md:7, 117 | correct [V] | npm `@anthropic-ai/claude-code` dist-tag latest is 2.1.300, published 2026-10-14T14:27Z | None. |
| `ANTHROPIC_BASE_URL` without `/v1`; `ANTHROPIC_AUTH_TOKEN` (Bearer) / `ANTHROPIC_API_KEY` (x-api-key) | 002 research.md:120-121; 004 research.md:44 | correct [V] | https://code.claude.com/docs/en/llm-gateway-protocol (API formats; Request headers) | None. |
| `ANTHROPIC_DEFAULT_{OPUS,SONNET,HAIKU}_MODEL`, `CLAUDE_CODE_SUBAGENT_MODEL`; "Haiku is used for background tasks" | 002 research.md:122 | correct, needs nuance [V] | /model-config defines the three alias vars and `CLAUDE_CODE_SUBAGENT_MODEL`. The gateway doc says that on an `ANTHROPIC_BASE_URL` gateway, background tasks use **the main model unless `ANTHROPIC_DEFAULT_HAIKU_MODEL` pins one**. | Reword: "background tasks use the main model unless DEFAULT_HAIKU is set; `mrun launch` sets it". |
| `CLAUDE_CODE_MAX_CONTEXT_TOKENS` applies directly to unrecognized IDs; `CLAUDE_CODE_AUTO_COMPACT_WINDOW` sets the compaction window | 002 research.md:123; mlx-rust.md:196 | correct [V] | /model-config: it applies directly when the ID is unrecognized and has no `[1m]`. AUTO_COMPACT_WINDOW takes plain token counts, is capped at the model window, and overrides `/autocompact`. Unknown IDs otherwise assume 200K. | Optionally add `CLAUDE_CODE_DISABLE_UNKNOWN_MODEL_WINDOW_ENFORCEMENT` as a documented non-default. |
| Attribution block stable per conversation since v2.1.181; `CLAUDE_CODE_ATTRIBUTION_HEADER=0` removes it | 002 research.md:144; 004 research.md:64 | correct [V] | Gateway doc §System prompt attribution block | None. |
| "A **per-request** attribution header … defeats exact prefix reuse unless `CLAUDE_CODE_ATTRIBUTION_HEADER=0`" | landscape:86 (also the 197 recommendation) | **outdated** [V] | This holds only before v2.1.181. On ≥ 2.1.181 the block is stable per conversation, so it costs a few tokens of prefix but does not defeat reuse. | Align with 002 research: "per-request only before 2.1.181; mrun's launcher still sets `=0` (Ollama does too)". |
| Gateway hint headers | (missing) | **new** [V] | Since v2.1.273 Claude Code can send `x-claude-code-request-class` (main/subagent/compaction/auxiliary), `x-claude-code-compaction`, **`x-claude-code-context-compacted`** ("the conversation prefix before this request is no longer used, so a cache keyed on it can be dropped"), `x-claude-code-prompt-id` (≥ 2.1.283) and `x-claude-code-agent-id`. They are **off by default for custom base URLs** and enabled with `CLAUDE_CODE_GATEWAY_HINT_HEADERS=1`. | Add to 002 research and to 003 prefix-cache eviction hints. Have `mrun launch claude` set `CLAUDE_CODE_GATEWAY_HINT_HEADERS=1`, and make mrun accept unknown headers. |
| Ollama launcher env (claude.go:69-73, 172-178) | 002 research.md:120-122; landscape:74 | correct, extended [V] | ollama `main` `cmd/launch/claude.go`: lines 69-73 as cited, plus `CLAUDE_CODE_TOTAL_TOKENS_REMINDER=off` (l.74), `CLAUDE_CODE_DISABLE_FEEDBACK_SURVEY=1`, `CLAUDE_CODE_AUTO_MODE_SERVER=0` (l.82-83), and `MAX_CONTEXT_TOKENS` + `AUTO_COMPACT_WINDOW` from the model window (l.183-185). The tier vars are now at l.174-177. | Consider copying `CLAUDE_CODE_TOTAL_TOKENS_REMINDER=off` [U: the variable is not in public docs]. Update the line refs. |
| Codex rust-v0.162.1 (2026-10-09); `wire_api="chat"` is a hard error; `model_providers`, `env_key`, `model_context_window`, `model_auto_compact_token_limit` | landscape:87; 002 research.md:147-156 | correct [V] | Latest `rust-v0.162.1`, npm `@openai/codex` 0.162.1 (2026-10-09). `model-provider-info/src/lib.rs:102` (ollama-chat removed), `:130` (`"chat"` → error), `:149` `env_key`. The app-server config schema has `model_context_window`, `model_auto_compact_token_limit` and a **new `model_auto_compact_token_limit_scope`**. OSS providers ollama and lmstudio:1234 use `WireApi::Responses` (l.653-686). | Note the new `_scope` key. The cited `main ae11049` is now `160b2b1` (2026-10-15), which changes nothing material. |
| OpenCode v2.0.29 (2026-10-14), `@ai-sdk/openai-compatible`, `limit.context/output` | landscape:85; 002 research.md:174-183 | correct [V] | Tag `v2.0.29` is the newest `v*` tag of `anomalyco/opencode`. `packages/core/src/aisdk-native.ts:73` maps the npm id to `@opencode/ai/providers/openai-compatible`. The `limit` schema (`v1/config/provider.ts:42`) also accepts **`limit.input`**. Caveat: npm `opencode-ai` latest is **1.19.0** (2026-10-11), so npm installs may lag v2. | Mention `limit.input`, and in `mrun doctor`/docs which channel (brew/GitHub vs npm) gives v2. |

## 4. Competitors since early October 2026

| Claim | Location | Status | Evidence | Recommended change |
|---|---|---|---|---|
| llama.cpp v0.6.0 (2026-10-05), b11622 (10-14) | landscape:34 | outdated (minor) [V] | Latest build tag is b11633 (`5cdadfc`, 2026-10-14). Agent-relevant commits since 10-01: **server default port changed to 9931** (#30159, 10-09); context checkpoints kept across slot save/restore (#26004); a GPU cache for MoE experts held in host memory (#29887); Metal 128/96 flash-attention kernels (#30209); `--spec-draft-n-ubatch` (#30459); MTP for Qwen3.8-Flash-Next and GLM5Next. | Update the tag. In the 005 bench harness, never assume port 8080 for llama-server. |
| Ollama v0.40.4 (2026-10-13); MLX runner with snapshots, MTP, DFlash | landscape:22, 35, 59 | correct; newer RC [V] | `v0.40.5-rc0` = `292d7ce` (2026-10-14) bumps MLX to `bebd642`. Since 10-01: compacting restored recurrent state after speculative rollback (`a5b659e`), fp32 candidate logits for quantized heads, populating the Codex model picker from installed models (#18887), accepting Responses custom tool calls in history (#18911), "decision models". | Note the rc. No change to the architecture conclusions. |
| oMLX v0.7.0 (2026-10-01), HEAD `6feea9f` | landscape:25, 40, 266; mlx-rust.md:15, 180 | correct tag; HEAD outdated [V] | v0.7.0 = `4d4f5a2` is still the latest stable (v0.7.1.dev1 pre-release). Main on 10-14/15 added: **MTP prefix snapshots restored on prompt-cache hits** (#4540), a **scheduler that requeues under memory pressure and caches cancelled prompts** (#4509), batched-prefill of concurrent prompts (#4361), DFlash in the batched engines (#4508), GDN state kept through padded prefill (#4526), reasoning tokens in usage (#4546), and a keyless-inference toggle. | Update the HEAD. Note "caches cancelled prompts" as a parity target for 003 (cancelled turns still seed the prefix cache). |
| LM Studio 0.4.x: `/v1/messages` (0.4.1), Responses; mlx-engine | landscape:36 | unverifiable for October [U] | No October changelog was found; lmstudio.ai was not reachable. Known: 0.4.2 brought continuous batching in mlx-engine 1.0.0, and 0.4.9 added `output_config.effort` support on `/v1/messages`. The `lmstudio-ai/mlx-engine` commit "Update MLX dependencies" is from 2026-10-05. | Add the 0.4.2 batching and 0.4.9 effort facts. Re-check the changelog from a networked host. |
| mlx-lm `mlx_lm.server`: no Messages, no Responses | landscape:37 | correct [V] | No commits after `ba8a08f` (2026-10-10) | None. |
| mistral.rs v0.9.4 (2026-09-24): chat + Messages + Responses | landscape:38 | correct [V] | v0.9.4 = `4400935` is still the latest. `main` adds Qwen3.8-Flash-Next (`3f2515e`, 10-01) and CUDA 13.4 (`28ed6da`, 10-12); nothing new on Metal or the agent APIs. | None. |

## 5. Crate pins

| Claim | Location | Status | Evidence | Recommended change |
|---|---|---|---|---|
| tokenizers 0.23.2 | Cargo.toml:23; 001 plan.md:23; 001 research.md:103 | correct [V] | crates.io max_stable 0.23.2 (2026-09-21). Its optional `http` feature depends on `hf-hub ^0.4.1`. | Keep `default-features = false`, which avoids a second, 0.4 hf-hub next to 1.0. |
| minijinja 3.0.0 (+ contrib 3.0.0) | Cargo.toml:16-17; 001 plan.md:24 | correct [V] | Both 3.0.0, published 2026-10-08 | None. |
| llguidance 1.9.1 + toktrie 1.9.1 | 002 plan.md:21; 002 research.md:9, 262 | correct [V] | Both 1.9.1, 2026-09-30 | None. |
| hf-hub 1.0 with a "tokio API" | 004 plan.md:18-19 | correct [V] | 1.0.0 (2026-07-10) is the latest. tokio, reqwest ^0.13 and sha2 ^0.11 are non-optional deps; the features are `blocking`, `rustls-tls`, `socks` (the async API is the default). | Write "async API (default)". Enable `rustls-tls` explicitly. |
| axum 0.8.9 / "axum 0.8" | 002 plan.md:20 | correct [V] | 0.8.9 is max_stable (2026-04-14) | None. |
| clap 4.6 (+ clap_complete 4.6) | crates/mrun/Cargo.toml:14 (4.6.7); 001 plan.md:26; 004 plan.md:18 | correct [V] | clap 4.6.7 (2026-09-14), clap_complete 4.6.11 | None. |
| tokio 1.53, reqwest 0.13, sha2 0.11, tower-http (no version) | 001 plan.md:26; 002 plan.md:20; 004 plan.md:19 | correct [V] | tokio 1.53.3 (10-13), reqwest 0.13.5, sha2 0.11.1 (10-12), tower-http **0.7.1** (2026-08-31) | Pin tower-http 0.7 in the 002 plan and T001. |

## 6. Sources (fetched 2026-10-15)

- MLX tags/log: https://github.com/ml-explore/mlx (v0.32.4 `36969c0`; compare v0.32.3...v0.32.4)
- mlx-c: https://github.com/ml-explore/mlx-c ; mlx-lm: https://github.com/ml-explore/mlx-lm ; mlx-rs: https://github.com/oxiglade/mlx-rs
- Ollama MLX pins: https://github.com/ollama/ollama/commit/292d7ce58c2fce64100e4ec7ae9977ef4fdd481a
- Claude Code: https://code.claude.com/docs/en/llm-gateway-protocol ; https://code.claude.com/docs/en/model-config ; https://www.npmjs.com/package/@anthropic-ai/claude-code
- Codex: https://github.com/openai/codex (`160b2b1`, `codex-rs/model-provider-info/src/lib.rs`)
- OpenCode: https://github.com/anomalyco/opencode (`37df62f`)
- Models: https://catalog.ngc.nvidia.com/orgs/nim/qwen/models/qwen3.8-27b ; https://huggingface.co/poolside/Laguna-XS-2.1 ; https://technews.tw/2026/02/04/qwen3-coder-next-significantly-cuts-inference-costs/ ; https://llmlearner.com/models/qwen4-27b
- LM Studio: https://lmstudio.ai/changelog/lmstudio-v0.4.2 ; https://lmstudio.ai/changelog/lmstudio-v0.4.9
- crates.io API (`/api/v1/crates/<name>`), npm registry (`registry.npmjs.org/<pkg>`)
