# Research: Speculative Decoding

Evidence marks as in [next-features](../../concepts/research/next-features-2026-10.md): **[V]**
verified in source at the stated revision, **[U]** unverified, **[I]** own inference.

## R1: Source of MTP weights

- **Finding [V]**: mlx-lm `ba8a08f` drops every key containing `mtp.` in `qwen3_5.py:354`
  (`sanitize`), which `qwen3_5_moe.py` calls after remapping; `convert` loads through `load` →
  `load_model` → `sanitize` (`utils.py:493`, `convert.py:85`). mlx-community conversions made with
  mlx-lm therefore lack the head. Ollama loads the head "only when its tensors are present"
  (`mlxrunner/model/qwen3_5/qwen3_5.go:956`) and its importer copies `mtp.*` from the source
  (`compatmigrate/qwen35.go`).
- **Decision**: Keep the pinned mlx-community base repository. Add an MTP source to the alias
  (`qwen3.6` → `Qwen/Qwen3.6-35B-A3B` at a pinned revision, with the SHA-256 of every `mtp.*`
  tensor's bytes, computed once by the maintainer alias-refresh script from a full download).
  `mrun pull` reads `model.safetensors.index.json`, then each shard's safetensors header (first
  8 + N bytes by HTTP `Range`), then fetches only the `mtp.*` tensor byte ranges. Each tensor is
  verified, quantized with the base scheme (affine, 4-bit, group 64; `mlp.gate` and
  `shared_expert_gate` 8-bit, matching mlx-lm's `quant_predicate`), and written to
  `~/.mrun/sidecars/<base-identity>/mtp.safetensors` (outside the shared HF cache) atomically. The
  manifest records source repo, revision, tensor hashes and the sidecar SHA-256; the model identity
  (001 R10) gains `mtp = <sidecar sha256>`, so 003 keys never mix states from with- and
  without-head loads (the head does not change target state, but the identity stays honest).
  When the mirror refuses range requests, pull falls back to downloading the shards that contain
  `mtp.*` (listed in the index), verifying them by their LFS hash, then extracting.
- **Size [I from U config]**: hidden 2,048, 256 experts × 3 × 2,048 × 512 ≈ 805M expert params plus
  ≈ 35M attention, shared expert and `fc` params: ≈ 1.7 GB to fetch in bf16, ≈ 0.47 GB after 4-bit
  quantization. Measured at implementation.
- **Alternatives**: (a) A community repository that kept `mtp.*` (unpinned provenance, often
  missing `mtp.fc.weight` [U]); rejected. (b) mrun converts the whole model itself from bf16 (70 GB
  download); rejected. (c) Ship our own converted repo; rejected (hosting models is a non-goal).
  (d) Keep the head in bf16 (1.7 GB resident, slower draft steps); rejected for memory.

## R2: N-gram/suffix drafter

- **Decision**: Per request, a suffix automaton over token ids of the prompt, the copy-aware
  renderings (R3) and the output so far, extended incrementally per emitted token. Lookup: longest
  suffix of the committed context with length ≥ `min_match`; proposal = tokens following the most
  recent occurrence (most recent wins on ties: agents re-read the latest file version). Defaults
  `min_match` 4 (dense) / 8 (MoE), tuned in FR-001. Built on a helper thread during prefill
  (O(n), ≈ 10 ms per 100K tokens [I]); the engine thread only queries it.
- **Rationale**: SuffixDecoding-style model-free drafting was best on SE tasks (2604.26469 [V
  listing]: 2.27× SWE-bench repair, 1.51× Aider edits). Agent replies copy from context. llama.cpp
  notes "MoEs require long drafts" for `ngram-mod` (n_match 24, drafts 48–64) [V `docs/speculative.md`].
- **Alternatives**: Cross-request global suffix trees (AgSpec workspace corpora): the prompt already
  holds the history, so per-request is enough; revisit if traces show cross-session copies.
  Frequency-weighted continuation trees: more state, little gain for verbatim copies.

## R3: Copy-aware renderings

- **Finding [V]**: Claude Code's `Read` output prefixes lines with a right-aligned number and `→`
  (older versions: a tab); `Edit.old_string` must not include those prefixes. In Qwen3 (Hermes JSON)
  tool calls, argument strings are JSON-escaped; Qwen3-Coder and Qwen3.6 XML parameters are raw
  (002 research §3). AgSpec (2610.01108) indexes files "in the agent's own emission format" [V listing].
- **Decision**: For every tool-result and message text block, add renderings: (1) line-number
  prefixes stripped; (2) JSON-escaped (models with JSON tool syntax only); (3) both. Each rendering is
  tokenized separately and appended to the index as an unrendered source with a separator token.
  Cost is bounded by input caps (002 body ≤ 8 MiB).
- **Alternatives**: Character-level matching and re-tokenizing drafts: token boundaries differ by
  context, but the token-level index already absorbs that within a few tokens; rejected for
  complexity.

## R4: MTP drafter and its attention state

- **Finding [V]**: The head fuses `embed(token[s+1])` with the target hidden at `s` through
  `pre_fc_norm_embedding`, `pre_fc_norm_hidden` and `fc`, runs one full-attention decoder layer
  (MoE in the MoE model) and the shared output head (Ollama `qwen3_5.go:1375-1396`). Its KV lags the
  target by one token; Ollama keys its prefix trie with a one-token look-ahead to keep it consistent
  (`mtp.go:24-30`).
- **Decision**: The head's attention covers only the last 4,096 positions (a ring kept with the
  live path, ≈ 8 MiB at ≈ 2 KiB/token [I]). It is filled from target hidden states of the last
  1,024 prompt positions during prefill (only when the controller's MTP depth is > 0) and from every
  decode/verify step. After a branch switch the ring restarts from the new prefill. The ring is
  never stored in the prefix store, so 003 needs no change and no look-ahead keys.
- **Rationale**: Drafts are verified, so a shorter window can only lower acceptance, never change
  output; storing head KV for whole branches would add a one-token-lagged state to every segment and
  checkpoint (Ollama's approach) for an unmeasured gain. FR-001 measures acceptance; if the window
  costs too much acceptance, a later change can widen it.
- **Alternatives**: Full-context head KV stored with segments (Ollama); rejected for now.

## R5: Rollback on hybrid models

- **Finding [V]**: Recurrent state is not trimmable. Ollama schedules recurrent snapshots at every
  draft offset by splitting the recurrent kernel there (`speculate.go:337`,
  `cache/recurrent.go:42-65`); vLLM keeps an SSM replay buffer.
- **Decision**: During a verify forward of N+1 positions, capture recurrent snapshots (conv and
  delta state of all linear-attention layers) at most at 8 offsets: every position when N ≤ 8,
  otherwise every ⌈(N+1)/8⌉ positions (the kernel is segmented at those offsets). On acceptance of a
  tokens, restore the deepest snapshot ≤ a+1 and re-forward the remaining < stride tokens (all
  layers; attention KV for those positions is rewritten identically within tolerance). Snapshots
  (≈ 60 MiB each on Qwen3.6-35B-A3B [U]; ≤ 480 MiB) are part of the scratch envelope; FR-013
  lowers N_max when the probe cannot fit them. The controller's cost includes the replay.
- **Alternatives**: Snapshot every position for long drafts (64 × 60 MiB); rejected for memory.
  Replay from the round start always; rejected (doubles the cost of long accepted drafts).

## R6: Controller

- **Finding [V]**: Ollama's depth controller maximizes committed(N)/cost(N) over depths 0…frontier+1,
  probes one deeper every 4 rounds with back-off to 512, and persists state across requests
  (`speculate_depth.go:1-70`). Cascade (2506.20675) and "The Limits of Speculation" (2609.22156)
  show MoE verify cost grows with the union of experts and the optimal stop is a linear boundary in
  (marginal cost, marginal accepted progress) [V listings]. Adversarial prompts can collapse
  acceptance (2607.21804).
- **Decision**: Per model, per drafter: acceptance as Beta counts per n-gram match-length bucket
  ({4–7, 8–15, 16–31, 32+}) and per MTP position; cost(N, depth bucket) from the probe (widths 1, 2,
  4, 8, 16, 32, 64 at a short and the deepest admissible position, cached with 001's probe in
  `~/.mrun/probe.toml`) refined by an EWMA of measured rounds per context bucket ({<8K, 8–32K,
  32–128K, ≥128K}). Expected committed(N) = 1 + Σᵢ₌₁ᴺ Πⱼ≤ᵢ αⱼ. Choose argmax committed(N)/cost(N)
  over N ∈ {0} ∪ [N_min, N_max]; depth 0 has committed 1 at pipelined decode cost. N_min is the
  smallest N whose measured cost/committed beats depth 0 at the bucket's observed acceptance (this
  is "long drafts only" on MoE, derived rather than hard-coded); N_max = 64 (n-gram), 4 (MTP).
  Probe one step deeper every 4 rounds, doubling the interval up to 512 while probes change nothing.
- **Alternatives**: Fixed draft lengths (llama.cpp defaults); rejected: cannot guarantee SC-001.

## R7: Verification and sampling on device

- **Decision**: The verify forward returns logits for N+1 positions; greedy acceptance compares
  `argmax` on device; sampling applies the request's transforms once per position and runs the
  acceptance test and residual sampling on device with the request's RNG stream. One host sync per
  round (accepted length + bonus token). Plain decode keeps its one-token-ahead pipelining at
  depth 0; switching depth 0 → N drains the pipeline first.
- **Rationale**: Self-Speculation (2608.20359) calls suffix drafts "approximate"; a deterministic
  draft is a point-mass proposal and the rule in FR-005 keeps the target distribution exactly [I].

## R8: Measure-first protocol

- **Decision**: `mrun bench --suite spec` (mrun only). Offline cell: for each trace turn, tokenize
  the recorded reply (taken from the next request's history) and simulate the n-gram drafter against
  it: acceptance-length histogram, share of reply tokens in matches ≥ 16 (defines "edit-heavy",
  SC-002). Online cells: decode tok/s per turn with `off|ngram|mtp|auto`, drafted/accepted, MTP
  acceptance per position. Gate: ship `auto` as default only when SC-001 and SC-002 pass; publish
  the cells with the bench report.
- **Alternatives**: Synthetic copy prompts only; rejected (overstates acceptance, 2601.11580).
