# Implementation Plan: Image Input

**Branch**: `009-image-input` | **Date**: 2026-10-15 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/009-image-input/spec.md`

## Summary

Turn on the vision tower Qwen3.6 already ships. Phase 0 measures how often agent traffic carries
images (FR-001) and settles OQ-1 (does the pinned repository hold the tower). If the gate passes:
the API layer decodes inline images into resized pixels with a content digest; the renderer expands
them into placeholder spans; the prefix store keys placeholder positions by (digest, index); the
engine gains a vision encoder (port of mlx-vlm's Qwen3-VL tower), multimodal RoPE with a position
offset carried by every state object, and on-demand encoding with an in-memory embedding cache.
Text-only models and `--no-vision` keep 002 FR-020 unchanged.

## Technical Context

**Language/Version**: Rust 1.94.0
**Primary Dependencies**: `image` 0.25 (PNG, JPEG, GIF, WebP decoding; `default-features = false`
with only those codecs), `base64`, `sha2` 0.11 (already used by 004); resize is mrun's own port of
the reference smart resize and PIL bicubic filter (R3, OQ-4); MLX ops through `mlx-sys`: `conv`
(Conv3d as reshape + matmul if mlx-c lacks it; checked in T002), fused SDPA, `rope` variants
**Storage**: Memory only (embedding cache, raw-bytes memo); vision sidecar (if needed) under
`~/.mrun/sidecars/` (mechanism owned by 009, T027)
**Testing**: `ScriptedEngine` gains media spans, position offsets and an encoder cost model; golden
preprocessing fixtures (pixels and grids) generated from mlx-vlm `4623cf9` (MLX v0.32.4); macOS
equality and conformance tests on Qwen3.6
**Target Platform**: macOS (Metal); parsing, preprocessing, digests and store logic run on Linux
**Project Type**: Changes across `mrun-engine` and `mrun`
**Performance Goals**: encoder ≤ 1.5 s for a 2,560-token image on M4 Pro (SC-003, OQ-6); digest
and memo lookup ≤ 5 ms per repeated image; zero encoder runs on appended turns (SC-002)
**Constraints**: no network fetches; full-digest identity; one image encoded at a time; encoder
cancel per layer; everything inside the 001 budget
**Scale/Scope**: ≤ 64 images and ≤ 48 MiB of decoded image bytes per request (base64 within 002's
64 MiB body cap); ≤ 2,560 tokens per image

## Constitution Check

| Principle | Gate | Status |
| --- | --- | --- |
| I Agent latency | Gate on measured image share (FR-001); SC-002 zero re-encodes on appended turns; SC-003 TTFT bound | Pass |
| II Exactness | Placeholder positions keyed by the full SHA-256 digest of the preprocessed input, compared at every position (R6); model identity covers `preprocessor_config.json` and the vision sidecar (FR-008); positions carried with state (R8); SC-004 bit equality | Pass (3.2.0: media reuse confirmed by full digest of preprocessed input) |
| III Memory budget | Tower weights measured, encoder scratch probed at the largest admissible image, embedding cache inside the budget and shed first; input caps before queuing | Pass |
| IV Local, private | No remote fetching (FR-003); nothing on disk; no image logging | Pass |
| V Own runtime | Encoder ported from mlx-vlm and tested against it (SC-005); no new kernels planned | Pass (3.2.0 names mlx-vlm as reference; pinned 4623cf9 on MLX v0.32.4) |
| VI Simple | Vision on when present, one off switch, one token cap; no video, audio, PDF | Pass |

Post-design re-check: Pass.

## Project Structure

### Documentation (this feature)

```text
specs/009-image-input/
├── spec.md  plan.md  research.md  data-model.md  quickstart.md  tasks.md
├── contracts/vision.md
└── checklists/requirements.md
```

### Source Code (repository root)

```text
crates/mrun-engine/src/
├── models/blocks/vision/qwen3_vl.rs  # [mlx] patch embed, blocks with 2D RoPE, pos-embed interpolation, merger
├── models/qwen3_5_moe.rs       # [mlx] M-RoPE (interleaved, [11,11,10]), embedding injection
├── vision.rs                   # VisionInfo, MediaSpan, position offset helpers (no MLX)
└── load.rs                     # [mlx] keep vision tensors when enabled; sidecar merge
crates/mrun/src/
├── media/
│   ├── mod.rs                  # Image parsing per API location, limits (FR-002–FR-004)
│   ├── decode.rs               # sniff, decode (first GIF frame), RGB8
│   ├── resize.rs               # reference smart resize + bicubic, preprocessing version
│   ├── digest.rs               # SHA-256 digest, raw-bytes memo
│   └── embed_cache.rs          # LRU (model identity, digest) → EmbeddingId
├── chat/render.rs              # image spans in the rendered prompt
├── cache/                      # 003: (digest, index) key symbols, `image` cause, offsets in nodes
├── api/                        # 002: accept/reject/placeholder per FR-002/003/013; metadata
├── cli/launch.rs, connect.rs   # 004: truthful image capability
└── hub/                        # 004: vision presence check, sidecar fetch (FR-015)
crates/mrun/tests/
├── media_limits.rs  media_digest.rs  media_api.rs  prefix_images.rs
crates/mrun-engine/tests/
├── vision_conformance.rs  vision_equality.rs   # macOS, ignored by default
```

**Cross-spec (010, not edited here)**: 010's family contract needs
`fn vision(&self, cfg) -> Option<VisionSpec>` (default `None`; `qwen3_5_moe` returns the Qwen3-VL
spec), and its locality rule must cover `models/blocks/vision/**` as shared blocks. Recorded as
task T034.

**Structure Decision**: Image parsing, preprocessing and digests live in `mrun` (portable, tested
on Linux); the engine sees only pixel tensors, grids, spans and embedding ids, keeping engine types
out of the API (Principle V).

## Phases

0. **Measure (gate)**: counters and bench report (FR-001); OQ-1 check of the pinned repository;
   encoder timing spike in Python mlx-vlm on the target Mac for OQ-6. Stop here unless the gate
   passes.
1. **Inputs**: parsing, limits, decode, resize, digest, memo; renderer spans; `count_tokens`.
2. **Engine**: tower port, M-RoPE, offset in state objects, encoder probe, embedding injection.
3. **Store**: key symbols, offsets in nodes and checkpoints, `image` cause, embedding cache.
4. **Clients and models**: `/v1/models`, `launch`/`connect`, `pull`/`show`/`ls`.
5. **Validation**: SC-001…SC-007.

## Complexity Tracking

| Item | Why needed | Simpler alternative rejected because |
| --- | --- | --- |
| Own resize port | Digest and conformance need deterministic, reference-like pixels | Generic resize crates use different bicubic kernels; output differs from the reference |
| Position offset in every state object | M-RoPE positions differ from token indices after an image | Recomputing from spans on every restore is possible but leaves a silent-shift failure mode; carrying it plus a debug recheck is safer |
