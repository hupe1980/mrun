# Data Model: Image Input

## ImageInput (API layer, per image block)

| Field | Type | Notes |
| --- | --- | --- |
| path | String | JSON path, for errors and warnings |
| location | UserMessage \| ToolResult | decides reject vs placeholder (FR-003, FR-013) |
| source | Inline{bytes} \| Remote{kind} | `Remote` never fetched |
| detail | Auto \| Low \| High \| Original | `Low` caps at 256 tokens; others use `max_image_tokens` |
| declared_mime | Option<String> | compared with the sniffed format |

## PreparedImage

| Field | Type | Notes |
| --- | --- | --- |
| format | Png \| Jpeg \| Gif \| Webp | sniffed |
| decoded | (w, h) | ≤ 50 MP |
| resized | (w, h) | multiples of 32 px |
| grid | (t = 1, h/16, w/16) | patch grid; merged grid = (1, h/32, w/32) |
| tokens | u32 | `(h/32)·(w/32)`, ≤ `max_image_tokens` |
| pixels | Vec<u8> (transient) | resized RGB8; held only while the digest is computed, then dropped; re-decoded from the request bytes when the encoder needs the image; never stored |
| digest | [u8; 32] | SHA-256(preproc_version, min_pixels, max_pixels, grid, pixels) |

`RawMemo`: `SHA-256(raw bytes)` → `(digest, grid, resized dims)`, bounded (default 1,024 entries),
host memory only.

## ImageSpan (renderer → store and engine)

`{start: u32, len: u32, merged_grid: (t, h, w), digest: [u8; 32]}`; `start` is the first
placeholder position (after `<|vision_start|>`). Added to 003's `Prompt` as `images:
Vec<ImageSpan>` (ascending, non-overlapping).

## Key symbol (003 tree)

Positions outside spans: token id. Positions inside a span: `Image{digest_id, index}` where
`digest_id` interns the 32-byte digest in a per-store table (refcounted; equality of ids ⇔
equality of digests). Edge labels compare symbols, so a match through an image requires equal
digests at every position.

## Position offset

`offset: i32` with `next_position = active_len + offset`. For each image span with merged grid
`(1, H, W)`: offset changes by `max(H, W) − H·W`. Stored on: the live state, each `Segment`
(offset at its start and end), each `Checkpoint`, each tree node (offset at node end). Text-only
sequences: always 0.

## EmbeddingEntry

| Field | Type | Notes |
| --- | --- | --- |
| key | (model identity, digest) | |
| id | EmbeddingId (engine) | `H·W × hidden` bf16 |
| bytes | u64 | ≈ 4 KiB per image token on Qwen3.6 [U] |
| last_used | Instant | LRU; cap 256 MiB; evicted before branches |

## VisionInfo (engine, startup)

`{present: bool, enabled: bool, weights_bytes, scratch_bytes (probed at max_image_tokens),
max_image_tokens, min_pixels, max_pixels, preproc_version, image_token_id, vision_start_id,
vision_end_id}`; part of `ModelInfo` and the budget breakdown.

## MissReport additions

Cause `image` when the divergence position is an image span start (or inside a span) and the text
before it matches. Full precedence: `disabled > model > evicted/capacity > image > template >
system > edit > new > checkpoint`. 003's `MessageMark` content hash includes the digests of the
message's images, so a changed image is not mistaken for an unchanged message (002 T004 and 003
data model updated when this feature lands, T022). Request metadata adds `images`,
`image_tokens`, `encoder_runs`, `embedding_hits`.

## Counters (FR-001)

Live request counts only; session shares come offline from `mrun bench --trace-stats`.
`/mrun/status` key `images` (and `/metrics` once 008 FR-010 returns): `requests_with_images{api, location}`, `images_total`,
`images_omitted_total{reason}`, `encoder_runs_total`, `embedding_cache_hits_total`.
