# Research: Image Input

Evidence marks as in [next-features](../../concepts/research/next-features-2026-10.md): **[V]**
verified in source at the stated revision, **[U]** unverified (secondary source or search
summary), **[I]** own inference. Hugging Face was not reachable from the research environment, so
model-card and config facts are **[U]** until the plan's T001 reads the pinned files.

Revisions: mlx-lm `ba8a08f` (2026-10-10), mlx-vlm `4623cf9` (2026-10-14), Codex `160b2b1`
(2026-10-15), OpenCode `37df62f` (2026-10-15), Ollama `8e4dad4` (2026-10-14), oMLX `958974c`
(2026-10-15), LM Studio mlx-engine `ab52366` (2026-10-05), vllm-mlx `9a939f8` (2026-10-11).

## R1: Is Qwen3.6-35B-A3B natively multimodal?

- **Finding [U]**: Yes. The official card describes a "Causal Language Model with Vision Encoder",
  `pipeline_tag: image-text-to-text`, Apache-2.0, released 2026-04
  ([model card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B),
  [MarkTechPost, 2026-04-16](https://www.marktechpost.com/2026/04/16/qwen-team-open-sources-qwen3-6-35b-a3b-a-sparse-moe-vision-language-model-with-3b-active-parameters-and-agentic-coding-capabilities/)).
  Image, document and video input are reported.
- **Finding [V]**: mlx-vlm implements it as `qwen3_5_moe`: the Qwen3.5 hybrid text backbone plus a
  Qwen3-VL vision tower and spatial-merge connector; its README lists `Qwen/Qwen3.6-35B-A3B` under
  this architecture (`mlx_vlm/models/qwen3_5/README.md`). Deepstack is forced off for Qwen3.5/3.6
  (`qwen3_5/config.py:47-58`).
- **Consequence [I]**: Image input needs no second model; it is a capability of the default model
  that mrun's loader currently discards (001 FR-002).

## R2: Do MLX 4-bit conversions keep the vision tower?

- **Finding [V]**: mlx-lm drops it at load and at convert: `qwen3_5.py:429-433` and
  `qwen3_5_moe.py:23-27` skip every key starting with `vision_tower` or `model.visual`; `convert`
  loads through `sanitize` (which also drops `mtp.*`). A repository converted with mlx-lm is
  therefore text-only.
- **Finding [U]**: The `mlx-community/Qwen3.6-35B-A3B-*` repositories (4-bit, mxfp4, bf16) were
  converted with mlx-vlm 0.4.4 and their cards show image prompts, so they carry the vision tower;
  some third-party 4-bit builds (for example an OptiQ build) show text-only usage
  ([search summary](https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-mxfp4),
  [vision quant](https://huggingface.co/spicyneuron/Qwen3.6-35B-A3B-MLX-4.8bit-vision)).
- **Finding [V]**: mlx-vlm quantizes vision modules unless `vision_config.skip_vision` is set or the
  checkpoint lacks quantized tensors for them (`mlx_vlm/utils.py:165-181, 605-620`), so the tower
  may be stored quantized or in bf16 depending on the conversion.
- **Decision**: 004's pinned `qwen3.6` repository stays the source. `mrun pull` inspects the
  safetensors index: vision tensors present → the manifest records `vision: present` with their
  precision and bytes. Absent → a vision sidecar is fetched from the pinned original repository by
  range requests (mechanism owned by 009, T027; cast to bf16, no quantization, hashes pinned).
  Which case applies to the pinned revision is open question OQ-1.

## R3: How the vision path works (mlx-vlm reference)

- **Preprocessing [V]** (`qwen3_vl/processing_qwen3_vl.py`): RGB, smart resize so both sides are
  multiples of `patch_size × merge_size` = 16 × 2 = 32 px and the pixel count lies within
  `[min_pixels, max_pixels]`, rescale 1/255, normalize mean = std = 0.5, temporal patch 2 (a still
  image is repeated to fill the temporal pair).
- **Encoder [V]** (`qwen3_vl/vision.py`): `PatchEmbed` is a Conv3d with kernel = stride =
  (2, 16, 16) (`:63-98`); transformer blocks with 2D rotary embeddings and attention split per image
  by `cu_seqlens` (`:129-157`); learned position embeddings interpolated to the grid; `PatchMerger`
  merges 2 × 2 patches and projects to the text hidden size. One image token = one 32 × 32 px
  block. Size of the tower for 35B-A3B (depth, width) is **[U]** (Qwen3-VL family: ≈0.4–0.6 B
  parameters, ≈0.8–1.2 GB in bf16); measured at load (FR-007).
- **Prompt [V]**: the template renders `<|vision_start|><|image_pad|><|vision_end|>`; the processor
  expands `<|image_pad|>` (id 248056; start 248045, end 248046, `qwen3_5/config.py:116-121`) to
  `t·(h/2)·(w/2)` copies, and `merge_input_ids_with_image_features` replaces those embeddings with
  the encoder output (`qwen3_5/qwen3_5.py:121-135`).
- **M-RoPE [V]** (`qwen3_5/language.py:1456-1621`): attention layers use interleaved multimodal
  RoPE (`mrope_section` [11, 11, 10]). Text tokens get the same position on all three axes. An
  image span starting after text index `s` gets positions `s + (t, h, w)` over its merged grid, and
  the text after it resumes at `max(position) + 1`. So after an image of `H × W` merged tokens the
  position runs `H·W − max(H, W)` behind the token index (`rope_deltas`). Without images positions
  equal token indices (`language.py:993-1007` tiles the 1D offset ×3), which is why the text-only
  path (mlx-lm, mrun 001) is exact with plain RoPE **[I]**.
- **Gated DeltaNet layers [I]**: position-free; they consume image embeddings like any other
  input. In `qwen3_5_moe` LLM attention over image tokens is causal, so a prefill chunk may end
  inside an image span and recomputation is exact; families with bidirectional image attention keep
  each span in one chunk (FR-011) (unlike bidirectional-image models, where Ollama forbids resume points
  inside a media item, `mlxrunner/prefix_cache.go:22`).

## R4: How agent clients send images

- **Claude Code [U]**: pasted screenshots become Anthropic `image` blocks (base64) in user messages;
  the Read tool on PNG/JPEG returns an `image` block inside `tool_result`. Several issue reports say
  some versions send no image (or fail) for Read-tool images against a custom `ANTHROPIC_BASE_URL`
  while pasted images work ([#67511](https://claudeissues.com/issue/67511-read-tool-sends-no-image-content-to-custom-anthropic-base-url-endpoints-clipboar),
  [#60601](https://claudeissues.com/issue/60601-read-tool-fails-on-every-local-image-with-400-could-not-process-image)).
  Claude Code has no capability switch: it always sends images. Open question OQ-2.
- **Codex [V]**: `view_image` returns `function_call_output.output = [input_image{image_url:
  "data:<mime>;base64,…", detail}]` (`protocol/src/models.rs:2232-2250`). Images are resized client
  side to ≤ 2,048 px and ≤ 2,500 patches of 32 px in high detail (`utils/image/src/lib.rs`), so at
  most ≈2,500 Qwen image tokens. The tool is offered only when the model's `input_modalities`
  contains `image` (`core/src/tools/handlers/view_image.rs:98-108`); the default for unknown models
  is `[text, image]` (`protocol/src/openai_models.rs:193`). Consequence: 004's Codex model catalog
  should state `input_modalities` truthfully, which also stops Codex from offering `view_image` to
  text-only models.
- **OpenCode [V]**: model `modalities.input` gates media (`core/src/v1/config/provider.ts:49-55`);
  unsupported media are replaced client-side by `ERROR: Cannot read … (this model does not support
  image input)` (`core/src/session/model-request.ts:146`). When inline images exceed 25 MiB the
  oldest are replaced by a placeholder until 15 MiB remain (`model-request.ts:43-44, 189-216`):
  this rewrites history and shows up as an image-position cache miss.

## R5: Competitors

| Server | Image sources | Prefix identity with images | Vision embedding cache |
| --- | --- | --- | --- |
| Ollama MLX runner [V] | Anthropic: base64 only (`anthropic/anthropic.go:1031`) | Trie key substitutes each media item by a 32-bit FNV fold of bytes + preprocessing dims (`mlxrunner/media.go:30-43`) | Encoded on first use, released after the span is evaluated |
| Ollama, no tower [V] | — | — | 400 "model has no vision tower" (`mlxrunner/embed.go:257`) |
| LM Studio mlx-engine [V] | — | Token prefix cut back to the first image span whose (start, end, SHA-256 of resized image) differs (`batched_vision/prompt_cache/image_spans.py`) | Prepared-image cache keyed by SHA-256 of mode + size + bytes (`prompt_inputs.py:398`) |
| oMLX [V] | data URLs | SHA-256 over size + RGB bytes per image (`utils/image.py:475`) | Projected features keyed (model, image hash), memory LRU + SSD (`cache/vision_feature_cache.py`) |
| vllm-mlx [V] | Fetches http(s) URLs with redirect checks (`models/mllm.py:743-788`) | — | Two levels: pixel values and vision encodings; claims ≈2 s → 0.01 s per repeated image (`vision_embedding_cache.py`) |

- **Lessons [I]**: (1) everyone caches encoder output by image content hash; (2) token ids alone
  are not an identity (all placeholder tokens are equal); (3) a 32-bit fold (Ollama) is a collision
  risk that mrun's exactness principle rules out; (4) remote fetching is an SSRF and privacy surface
  mrun avoids (Principle IV).

## R6: Prefix identity with images

- **Decision**: Each placeholder position is keyed in the token walk by the pair
  `(image digest, index within the span)` instead of the bare `<|image_pad|>` id. The digest is
  SHA-256 over: preprocessing version, resize parameters (`min_pixels`, `max_pixels`, resulting
  grid), and the resized RGB8 pixels. Matching compares the full 32-byte digest, never a fold.
- **Rationale**: Equal digest ⇒ identical encoder input ⇒ identical embeddings (R7), so reuse is
  exact. Hashing decoded, resized pixels (not file bytes) makes re-encoded but pixel-identical
  images hit; including the parameters makes a changed `--max-image-tokens` miss instead of reusing
  a different grid.
- **Constitution note**: Principle II requires hash lookups to be confirmed by token comparison.
  Images have no tokens; constitution 3.2.0 states that media reuse is confirmed by the full digest
  of the preprocessed input, which is what comparing the full SHA-256 at every placeholder position
  does. Model identity additionally covers `preprocessor_config.json` and the vision sidecar
  (FR-008).
- **Alternatives**: raw-byte hash (misses on re-encoding, and does not cover preprocessing);
  keeping pixels for byte comparison (≈8 MB per image held for the cache's lifetime).

## R7: Vision embedding cache

- **Decision**: Encoder output (merged embeddings, `tokens × hidden` bf16, ≈4 KiB per image token
  on Qwen3.6 [U hidden 2,048]) is cached in memory keyed by `(model identity, image digest)`, LRU,
  default cap 256 MiB, counted inside the state budget and shed before branches. Every image is
  encoded alone, so its embeddings never depend on which other images share the request; cached and
  recomputed embeddings are bit-identical.
- **Raw-bytes memo**: each turn resends every image in history; decoding a PNG screenshot costs
  milliseconds to tens of milliseconds. A memo `SHA-256(raw bytes) → (digest, grid)` (bounded,
  host memory) skips decode for repeated images; it is exact because decoding is deterministic.
- **When it matters [I]**: on plain appended turns the image's attention and recurrent state is
  already in the live path and the encoder is not run at all. The embedding cache pays when the
  prefix before the image changed (edit, subagent carrying the same screenshot, eviction, hybrid
  resume from a checkpoint before the image).

## R8: M-RoPE and prefix reuse

- **Decision**: The sequence state carries `position = active_len + rope_delta`. `rope_delta` is a
  pure function of the image grids in the prefix, so it is recorded with every tree node,
  segment and checkpoint and restored with them; equal token-walk keys imply equal deltas.
  Text-only sequences have delta 0 and run the existing 1D path unchanged.
- **Risk**: a resume or trim that restores tokens without the matching delta would silently shift
  every later position. Mitigation: the delta travels inside the engine's state objects, debug
  builds recompute it from spans on every restore, and SC-004 compares cache-on and cache-off runs
  with images.

## R9: Limits and memory

- **Token cost [I]**: Claude-style screenshots (long edge ≈1,568 px, e.g. 1,568 × 980; smart
  resize rounds to 1,568 × 992) → 49 × 31 = 1,519 image tokens; Codex high detail ≤ 2,500. Default `max_image_tokens` = 2,560 (max_pixels
  = 2,560 × 32²) keeps both at native resolution; larger images are downscaled by the reference
  smart resize. `detail: "low"` caps at 256 tokens.
- **Memory [I]**: weights of the tower (measured), encoder scratch for the largest admissible image
  (patches = 4 × 2,560 = 10,240 per image; attention is per image via fused SDPA, probed), one
  image's embeddings in flight (≈10 MiB), the embedding cache (≤ 256 MiB), and the image tokens'
  attention state (≈20 KiB per token on Qwen3.6: ≈50 MiB per 2,560-token image) inside the normal
  admission.
- **Host side**: decode bounded by fixed per-image safety caps (encoded ≤ 20 MiB after base64
  decoding, decoded ≤ 50 MP; errors say they are fixed); one image decoded at a time; resized
  pixels dropped after the digest and re-decoded when the encoder needs them, so a request never
  holds more than one image's pixels.
- **002 interaction**: 002's body cap (FR-009) is 64 MiB (OQ-5). Image bytes count inside it: the
  per-request cap is 48 MiB of decoded image bytes (`max_image_bytes_mib`), whose base64 form
  (×4/3 = 64 MiB) is the most the body can carry.

## R10: Measure-first

- **Signal**: 002 already emits `image-omitted` warnings; this feature adds live counters of
  requests with image blocks (by API location) to `/mrun/status` (key `images`) and `/metrics`
  (008). Sessions are counted offline: `mrun bench --trace-stats` reports the share of recorded
  sessions (002 `--record`) containing images per client.
- **Gate**: proceed past the gate when, in each of at least two clients, ≥ 5% of recorded sessions
  contain an image, or when ≥ 3 independent users request image input. Otherwise the
  feature stays specified and parked.
