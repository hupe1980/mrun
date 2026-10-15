# Contract: Image Input

## Wire behaviour (002 APIs)

| API | Accepted (vision active) | Never fetched |
| --- | --- | --- |
| Anthropic Messages | `{"type":"image","source":{"type":"base64","media_type":M,"data":B}}` in user `content` and in `tool_result.content` | `source.type` `url`, `file` |
| Chat Completions | `{"type":"image_url","image_url":{"url":"data:image/…;base64,…","detail"?}}` in user content and tool message content arrays | `http(s)://` URLs |
| Responses | `{"type":"input_image","image_url":"data:…","detail"?}` in user content and `function_call_output.output` arrays | `http(s)://` URLs, `file_id` |

| Situation | User message | Tool result |
| --- | --- | --- |
| Vision active, inline, valid | accepted | accepted |
| Remote URL / file reference | 400, path named, "mrun does not fetch remote content; send base64" | placeholder `[image omitted: remote images are not fetched]` + `image-omitted` (`--reject-tool-images`: 400) |
| Undecodable, unsupported format | 400, path named | placeholder + `image-omitted` (`--reject-tool-images`: 400) |
| No active vision tower | 002 FR-020: 400 | 002 FR-020: placeholder `[image omitted: model has no vision input]` |
| Limit exceeded (count, bytes, tokens) | 400 before queuing, limit and setting named | same |
| Fixed safety cap exceeded (20 MiB encoded after base64 decoding, 50 MP decoded) | 400 before queuing, cap named and stated to be fixed | same |

Status codes and error shapes follow [002 contracts/api.md](../../002-agent-apis/contracts/api.md).
Warnings: `image-omitted`, `image-first-frame`, `image-downscaled` (via 002 FR-022 metadata).
Metadata fields: `images`, `image_tokens`, `encoder_runs`, `embedding_hits`; miss cause `image`.

`GET /v1/models` adds per model: `"capabilities": {"input": ["text", "image"]}` (or `["text"]`).

## CLI and config

| Item | Default | Notes |
| --- | --- | --- |
| `mrun serve --no-vision` / `vision = false` | vision on when the tower is present | frees tower memory |
| `--max-image-tokens N` / `max_image_tokens` | 2,560 | changes digests (images miss once) |
| `--max-images N` / `max_images` | 64 per request | |
| `--max-image-bytes MiB` / `max_image_bytes_mib` | 48 per request | decoded bytes; the base64 form must also fit 002's 64 MiB body cap |
| `--embedding-cache-mib N` / `embedding_cache_mib` | 256 | 0 disables |
| `mrun show`, `mrun ls` | — | `vision: yes (≈X GB) \| no \| sidecar missing` |
| `mrun launch codex\|opencode`, `mrun connect` | — | Codex catalog `input_modalities`; OpenCode `modalities.input` |

These rows extend the [004 CLI contract](../../004-models-cli/contracts/cli.md), which stays the
single source once this feature lands.

## Engine additions (001 `Engine` trait)

```rust
pub struct MediaSpan { pub start: u32, pub len: u32, pub patch_grid: (u32, u32, u32), pub embedding: EmbeddingId }

pub trait Engine {
    // … existing methods …
    fn vision(&self) -> Option<&VisionInfo>;
    /// Encode one image (resized RGB8, patch grid). Memory checked first (NeedsMemory); the check
    /// excludes embeddings already owned by the caller. Cancel checked after every encoder layer.
    /// The result is owned by the caller's cache.
    fn encode_image(&mut self, pixels: &[u8], patch_grid: (u32, u32, u32), cancel: &CancelToken)
        -> Result<Embedding, EngineError>;
    fn drop_embedding(&mut self, id: EmbeddingId);
    /// Position of the next token (= active_len() + offset).
    fn next_position(&self) -> u32;
    // allocated() (001) includes the bytes of every live embedding, so 003's invariant
    // (store accounting = engine allocated()) holds with the embedding cache counted.
}

// PrefillPlan gains `media: &[MediaSpan]` (absolute, ascending). Positions inside a span take the
// embedding rows in order; chunk ends may fall inside a span only for causal-image-attention
// families (`qwen3_5_moe`); others keep each span in one chunk (FR-011). Every embedding is resolved by the
// caller before the call; the engine never encodes during prefill. Segments and checkpoints record
// the position offset at their boundaries; split_off/append/restore_checkpoint restore it.
pub struct Embedding { pub id: EmbeddingId, pub tokens: u32, pub bytes: u64 }
```

`ScriptedEngine` models encoder cost, embedding bytes, offsets and per-layer cancel.

## Prefix store additions (003 contract)

- `Prompt.images: Vec<ImageSpan>`; lookup compares key symbols (data model).
- `Plan.media_needed`: spans at or after the resume position. The worker splits prefill at the
  start of each uncached span, resolves that span's embedding (embedding cache or `encode_image`,
  pixels re-decoded) before the prefill call that reaches it, and releases uncached embeddings
  once the span is committed.
- `initial_need` adds the largest single uncached image's embedding bytes; embedding-cache entries
  are evicted before branch units.
