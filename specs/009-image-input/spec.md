# Feature Specification: Image Input

**Feature Branch**: `009-image-input`

**Created**: 2026-10-15

**Status**: Draft (after 0.1; gated by the measure-first check in FR-001)

**Input**: User description: "Image input for vision-capable models (Qwen3.6-35B-A3B ships a vision
encoder): accept base64 and data-URL images in user messages and tool results on all three APIs, never
fetch remote URLs; size, count and memory limits with the vision encoder inside the 001 budget;
images are part of the exact prefix identity (hash of decoded pixels plus preprocessing parameters);
a vision-embedding cache keyed by image content; M-RoPE positions survive prefix reuse; models
without a vision tower keep 002 FR-020 behaviour; `mrun pull` provides a vision-capable variant.
Measure first how often agent traces contain images. Success: screenshot round trip in Claude Code,
prefix reuse across turns with an image in history, bounded TTFT overhead."

## Clarifications

### Session 2026-10-15

- Q: Which image sources are accepted? → A: Inline only: Anthropic `source.type: "base64"`, OpenAI
  `image_url.url` and Responses `input_image.image_url` as `data:image/…;base64,` URLs. Remote
  `http(s)` URLs, Anthropic `url`/`file` sources and Responses `file_id` are never fetched
  (Principle IV): rejected in user messages, replaced with a placeholder in tool results.
- Q: What identifies an image for reuse? → A: SHA-256 over the preprocessing version, the resize
  parameters and resulting grid, and the resized RGB pixels. Pixel-identical images re-encoded by the
  client still hit; a changed resize setting misses. The full digest is compared; no folded hashes.
- Q: Is vision on by default? → A: Yes when the loaded model has a vision tower; its weights and
  scratch are measured at startup and shown in the budget. `--no-vision` skips loading it and frees
  the memory; behaviour is then 002 FR-020.
- Q: Video, audio, PDFs? → A: Out of scope; still rejected per 002 FR-020.
- Q: Are vision embeddings persisted? → A: No. Memory only, like 003 (007 may extend later).

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Show the agent a screenshot (Priority: P1)

A Claude Code user pastes a screenshot of a failing UI or a stack trace and asks the agent to fix
it. The local model sees the image and answers about its content.

**Why this priority**: Screenshots are the most common image an agent user sends; today mrun
rejects them in user messages.

**Independent Test**: Run Claude Code (pinned version) via `mrun launch claude-code` against
Qwen3.6 and paste a screenshot containing a known error string and colour; the answer names both.

**Acceptance Scenarios**:

1. **Given** a vision-capable model, **When** a user message contains a base64 PNG, JPEG, GIF or
   WebP image, **Then** it is decoded, resized within the limits, encoded and answered about.
2. **Given** a Read tool result (Claude Code) or `view_image` output (Codex) carrying an image,
   **When** the next turn runs, **Then** the model sees the image, not a placeholder.
3. **Given** an image given as a remote URL, **When** it is in a user message, **Then** the request
   is rejected naming the JSON path and saying mrun does not fetch remote content.
4. **Given** an image larger than the token cap, **When** it arrives, **Then** it is downscaled by
   the model's reference resize and usage counts its actual image tokens.

---

### User Story 2 - Images in history cost nothing on later turns (Priority: P1)

After a screenshot, the conversation continues for many turns. Each turn reuses the cached state
including the image; the vision encoder does not run again.

**Why this priority**: Agents resend the whole history every turn (Principle I); an image that
breaks reuse or is re-encoded every turn would make vision unusable.

**Independent Test**: A 30K-token conversation with a screenshot in turn 1 continues for 10 turns;
every turn reports full reuse, no `image` miss and zero vision encodes.

**Acceptance Scenarios**:

1. **Given** an image in the live path, **When** the next turn resends it unchanged, **Then** the
   prefix through and past the image is reused and the encoder is not run.
2. **Given** the same pixels re-encoded by the client (different PNG bytes), **When** resent,
   **Then** reuse still holds.
3. **Given** a subagent whose prompt carries the same screenshot after a different prefix, **When**
   it runs, **Then** the embedding cache supplies the encoder output and the miss report shows the
   text divergence, not the image.
4. **Given** a different image at the same place, **When** the next turn arrives, **Then** reuse
   stops at the image span start with cause `image`.

---

### User Story 3 - Text-only models and settings stay predictable (Priority: P1)

Users of Qwen3-Coder, Qwen3-8B, a text-only conversion or `--no-vision` see exactly today's
behaviour, and clients are told the truth about image support.

**Why this priority**: Not every model has a tower; clients must not offer image tools a model
cannot use.

**Independent Test**: Run 002's image tests (`*_image_placeholder`, `*_image_part_rejected*`,
`--reject-tool-images`) on Qwen3-Coder and on Qwen3.6 with `--no-vision`; all pass unchanged.
`/v1/models` and `mrun launch codex --dry-run` show image input only when vision is active.

**Acceptance Scenarios**:

1. **Given** a model without a vision tower, **When** images arrive, **Then** 002 FR-020 applies.
2. **Given** vision is active, **When** `mrun launch codex` or `opencode` configures the client,
   **Then** the model is declared image-capable; otherwise text-only.

---

### User Story 4 - Vision fits the memory budget (Priority: P2)

Enabling vision never makes the Mac swap or a request fail mid-way: the encoder's weights and
scratch are budgeted at startup, and oversized inputs are refused up front.

**Why this priority**: Principle III; the vision path must not be the exception.

**Independent Test**: Start with a lowered budget; read `mrun run --info`; submit the maximum
number of maximum-size images under memory pressure; peak memory stays within the budget.

**Acceptance Scenarios**:

1. **Given** startup with vision enabled, **When** the model loads, **Then** `mrun run --info`,
   `mrun ps` and `/mrun/status` show vision weights, vision scratch and the embedding-cache cap as
   budget terms, and the state capacity they report is what remains after those terms.
2. **Given** a request exceeding image count, size or byte limits, **When** it is validated,
   **Then** it is rejected before queuing, naming the limit and either the setting that raises it
   or, for the fixed safety caps, that the cap is fixed.

### Edge Cases

- Image placeholder text typed literally by a user (`<|image_pad|>`): never becomes an image span;
  spans come only from parsed image blocks and are marked by position by the renderer (FR-005).
- Animated GIF: first frame only, with an `image-first-frame` warning.
- Corrupt or undecodable data, or a declared media type that differs from the content: rejected
  naming the path (user messages); placeholder plus `image-omitted` in tool results.
- Image smaller than the minimum: upscaled by the reference resize to the minimum grid.
- Tool result with several images: each becomes its own span in order.
- Cancel during vision encoding: takes effect after the current encoder layer; nothing is cached
  for a partly encoded image.
- Prompt fits only without its images: admission rejects with token numbers including image tokens.
- Hybrid resume from a checkpoint before an image: the gap is re-prefilled from cached embeddings.
- OpenCode replaces old images with a placeholder above 25 MiB: miss cause `image` at that span.
- `count_tokens`: counts image tokens from decoded dimensions without running the encoder.
- Encoder or embedding output contains non-finite values: request fails with an engine error;
  nothing is cached.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001** (gate): Before implementation beyond the measurement tasks, mrun MUST count requests
  containing image blocks (by API and location: user message or tool result) live, in
  `/mrun/status` under the key `images` (and in `/metrics` once 008 FR-010 returns; the gate reads
  `/mrun/status`); sessions are counted offline:
  `mrun bench --trace-stats` MUST report the share of recorded sessions (002 FR-021) containing
  images per client. The feature proceeds when, in each of at least two clients, ≥ 5% of recorded
  sessions contain an image, or when ≥ 3 independent users request image input; the decision and
  data are recorded in research.
- **FR-002**: When the loaded model's family declares a ported vision tower (in this feature:
  `qwen3_5_moe` only), the checkpoint holds it and vision is enabled, mrun MUST accept
  images in user messages and tool results on all three APIs: Anthropic `image` blocks with
  `source.type: "base64"` (also inside `tool_result.content`), Chat Completions `image_url` parts
  with `data:` URLs (user messages and, where the client sends arrays, tool messages), Responses
  `input_image` with `data:` URLs (user content and `function_call_output.output` arrays). Formats:
  PNG, JPEG, GIF (first frame), WebP, detected from content.
- **FR-003**: mrun MUST NOT fetch remote content. Remote URLs, file ids and file references MUST be
  rejected in user messages naming the JSON path, and replaced with
  `[image omitted: remote images are not fetched]` plus an `image-omitted` warning in tool results
  (`--reject-tool-images` rejects them).
- **FR-004**: Limits, validated before queuing and each named in errors: ≤ 64 images per request
  (`max_images`); decoded image bytes ≤ 48 MiB per request (`max_image_bytes_mib`; the base64 form
  must also fit 002's 64 MiB body cap); image tokens per image ≤ `max_image_tokens` (default
  2,560; images above are downscaled, not rejected); `detail: "low"` caps an image at 256 tokens.
  Two fixed safety caps have no setting, and their errors say the cap is fixed: encoded size
  ≤ 20 MiB per image, measured after base64 decoding, and decoded size ≤ 50 megapixels. Decoding
  runs one image at a time; resized pixels are dropped once the digest is computed, and the image
  is decoded again from the request bytes when the encoder needs it.
- **FR-005**: Preprocessing MUST follow the family's reference processor (for `qwen3_5_moe`: smart
  resize to multiples of 32 px within `[min_pixels, max_pixels]`, RGB, rescale and normalize,
  temporal patch 2) and be versioned; images are rendered as the model's template expands them
  (for `qwen3_5_moe`: start, `t·h·w/4` placeholder tokens, end), and usage counts placeholder tokens as prompt tokens. Spans MUST come only from
  parsed image blocks; placeholder text inside message text never creates a span.
- **FR-006**: Every image MUST have a digest: SHA-256 over the preprocessing version, resize
  parameters, resulting grid and resized RGB pixels. A bounded memo from the SHA-256 of the raw
  bytes to the digest and grid MAY skip decoding repeated images.
- **FR-007**: The vision tower MUST be loaded at startup (never lazily), its weights measured and
  its peak scratch probed at the largest admissible image as part of 001 FR-008; state capacity is
  sized from what remains. `--no-vision` / `MRUN_NO_VISION` (config key once 004 FR-010
  returns) skips loading it. `mrun run --info`,
  `mrun ps` and `/mrun/status` show vision weights, vision scratch and the embedding-cache cap.
- **FR-008**: Prefix identity (003 FR-001) MUST include images: each placeholder position is keyed
  by (image digest, index within the image), so two prompts match through an image only if the full
  digests are equal. With vision enabled, model identity (001 FR-003) MUST additionally cover
  `preprocessor_config.json` and, when used, the vision sidecar's hash: the tower changes prompt
  state, so it is part of the cache key.
- **FR-009**: The engine MUST compute multimodal positions as the family's reference does (for
  `qwen3_5_moe` M-RoPE: text equal on all axes; image offset by the merged grid; text after an
  image resumes at the maximum plus one). The
  position offset from images MUST travel with the live state, segments and checkpoints, so any
  reuse, move or resume restores exactly the positions a cache-off run computes.
- **FR-010**: Vision encoder output MUST be cached in memory keyed by (model identity, image
  digest), LRU, default cap 256 MiB, inside the state budget and shed before branches under memory
  pressure or eviction. Images MUST be encoded one at a time so cached and recomputed embeddings are
  identical. Encoding runs on demand when prefill reaches an image not covered by reused state; only
  the image being prefilled holds uncached embeddings.
- **FR-011**: For families whose attention over image tokens is causal (`qwen3_5_moe`), prefill MAY
  end chunks inside an image span; other families MUST keep each image span within one chunk.
  Checkpoint placement follows 003 FR-004 unchanged. Cancellation during
  encoding MUST be checked after every encoder layer.
- **FR-012**: The miss report (003 FR-010) MUST gain cause `image`: divergence at an image span
  whose surrounding text matches but whose digest differs or that the client replaced with text.
  mrun metadata (002 FR-022) MUST report image count, image tokens, encoder runs and embedding-cache
  hits per request.
- **FR-013**: Without an active vision tower (the family has no ported tower, e.g. Devstral Small 2,
  whose `mistral3` tower is dropped by `sanitize`; the model has none; a text-only conversion; or
  `--no-vision`), behaviour MUST be exactly 002 FR-020.
- **FR-014**: `/v1/models` MUST report image input capability. `mrun launch` and `mrun connect`
  (004 FR-005/006) MUST declare it truthfully: Codex model catalog `input_modalities`, OpenCode
  `modalities.input`. Claude Code has no such setting; `launch` warns when vision is off.
- **FR-015**: `mrun pull qwen3.6` MUST provide the vision tower: if the pinned repository holds
  vision tensors they are verified with the rest; otherwise a vision sidecar is fetched from the
  pinned original repository by range requests (mechanism owned by 009, T027), verified against pinned hashes and stored
  with the manifest. `mrun show` and `mrun ls` show whether a model is vision-capable and the vision
  memory cost. A text-only model is never silently replaced.
- **FR-016**: Images and their digests MUST NOT be logged or written to disk, except by
  `--record` (002 FR-021), which stores request bodies as sent.

### Key Entities

- **Image**: source location (API path), raw bytes, format, decoded size, resized grid, digest.
- **Image span**: start position, token count, grid (t, h, w), digest; part of the prompt.
- **Position offset**: difference between the next position and the token count, carried by state.
- **Embedding entry**: (model identity, digest) → encoder output, bytes, last use.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: With Claude Code (pinned version) via `mrun launch claude-code` on Qwen3.6 4-bit, a
  pasted screenshot round trip (known error string and colour) is answered correctly in 5 of 5
  runs; the same holds for a Codex `view_image` round trip.
- **SC-002**: On the target Mac (M4 Pro, 48 GB) with Qwen3.6-35B-A3B 4-bit, a 30K-token
  conversation with a ≈1,500-token screenshot in turn 1, continued for 10 turns of 200–1,000
  appended tokens with replies that contain no tool calls, shows on every turn: cached ≥ the previous turn's committed
  length, zero encoder runs, no miss cause; median TTFT within 10% of the same conversation with
  the image replaced by an equal number of text tokens.
- **SC-003**: On the target Mac (M4 Pro, 48 GB) with Qwen3.6-35B-A3B 4-bit, a turn appending one 2,560-token image to a cached 30K-token
  conversation has TTFT ≤ TTFT of the same turn with 2,560 text tokens + 1.5 s; with the embedding
  cache warm, ≤ that text-only TTFT + 0.1 s.
- **SC-004**: Cache-on output equals cache-off output bit for bit (token ids and logits, same chunk
  split) for ≥ 20 image cases (image in history, edit before and after an image, subagent with the
  same image, checkpoint resume across an image, embedding-cache hit vs miss).
- **SC-005**: Greedy outputs on ≥ 10 image prompts, on identical pixel tensors, match mlx-vlm
  `4623cf9` (MLX v0.32.4) token for token for the first 32 tokens (001 conformance tolerance).
- **SC-006**: With a request of 64 images of 2,560 tokens each whose total decoded bytes reach the
  per-request cap, under injected memory pressure, peak memory stays ≤ budget + 1 GiB (001 SC-006)
  and no admitted request fails.
- **SC-007**: All 002 image tests pass unchanged on a text-only model and with `--no-vision`.

## Assumptions

- Vision per supported family (0.1, and Devstral from 0.2): `qwen3_5_moe` (Qwen3.6-35B-A3B): Qwen3-VL tower, ported here;
  `mistral3` (Devstral Small 2): the original has a vision tower, but `sanitize` drops it and no
  port is planned, so it is text-only (FR-013); `qwen3_moe` (Qwen3-Coder-30B-A3B), `gpt_oss`
  (gpt-oss-20b) and `qwen3` (Qwen3-8B): no tower. Other vision families are later work.
- 001 provides the engine, budget and scratch probe; 002 the request parsing, renderer and metadata;
  003 the prefix store; 004 `pull`, manifests and `launch`; 005 trace recording and replay; 008
  `/metrics` (optional, once 008 FR-010 returns). The range-request sidecar mechanism for FR-015, if needed, is owned by 009 (T027).
- Clients send images inline (verified for Codex and OpenCode; Claude Code per its API usage).

## Open Questions

Decisions above are defaults; these need data or a maintainer call before `/speckit-plan` is
final (research references in brackets).

- **OQ-1**: Does the pinned `mlx-community/Qwen3.6-35B-A3B-4bit` revision contain the vision tower,
  and in which precision? Decides whether FR-015 needs the sidecar path [R2]. Checked by plan T001.
- **OQ-2**: Does the pinned Claude Code version send Read-tool images to a custom
  `ANTHROPIC_BASE_URL`? Issue reports say some versions do not. If not, SC-001 covers pasted
  screenshots only [R4].
- **OQ-3**: Is a 5% session share the right gate threshold [R10]?
- **OQ-4**: Can mrun's resize reproduce the reference (PIL bicubic) bit for bit, or does SC-005
  compare on identical pixel tensors only, as written [R3]?
- **OQ-5** (resolved 2026-10-15): 002 FR-009's body cap was raised to 64 MiB for 0.1, so
  image-carrying sessions reach the placeholder path; this feature keeps that cap.
- **OQ-6**: Is 2,560 tokens per image enough for screenshots with small text, and is 1.5 s
  (SC-003) achievable for the encoder on an M4 Pro? Both are measured in the gate phase.
