---
description: "Tasks for 009 Image Input"
---

# Tasks: Image Input

**Input**: [plan.md](plan.md), [spec.md](spec.md), [research.md](research.md),
[data-model.md](data-model.md), [contracts/vision.md](contracts/vision.md)

## Format: `[ID] [P?] [Story] Description`

## Phase 0: Measure first (gate, FR-001)

- [ ] T001 [P] Read the pinned `qwen3.6` revision's `model.safetensors.index.json`, `config.json` and `preprocessor_config.json`: vision tensors present, precision, bytes, `vision_config`, `min_pixels`/`max_pixels`; record in research R2/R3 (OQ-1)
- [ ] T002 [P] Check mlx-c v0.7.0 for every op the tower needs (Conv3d or its reshape-matmul equivalent, fused SDPA with per-image splits, interleaved M-RoPE inputs); record in research
- [ ] T003 [P] Live request counters `requests_with_images{api,location}`, `images_total`, `images_omitted_total{reason}` in `/mrun/status` (key `images`) and `/metrics` (needs 008's exporter; 002 parsing already sees the blocks); no session counting live; tests
- [ ] T004 [P] `mrun bench --trace-stats`: share of recorded sessions with images per client (needs 005 trace format); add the flag to the 005 contract and 004 CLI contract when it lands
- [ ] T005 Encoder timing spike with Python mlx-vlm on the target Mac at 256, 1,500 and 2,560 image tokens; record (OQ-6)
- [ ] T006 Gate decision recorded in research R10 with the data (in each of at least two clients ≥ 5% of recorded sessions contain an image, or ≥ 3 independent users request image input); stop here if it fails

## Phase 1: Setup

- [ ] T007 Add `image` (png, jpeg, gif, webp only), `base64` to `crates/mrun/Cargo.toml`; skeletons `crates/mrun/src/media/{mod,decode,resize,digest,embed_cache}.rs`, `crates/mrun-engine/src/vision.rs`
- [ ] T008 [P] Golden fixtures from mlx-vlm `4623cf9` on MLX v0.32.4 (`scripts/vision_fixtures.py`): resized pixels, grids, placeholder counts, positions, first-32-token greedy outputs for ≥ 10 images

## Phase 2: Foundational (blocks all stories)

- [ ] T009 `media/decode.rs` + `media/resize.rs`: sniff, decode, first GIF frame, RGB8, smart resize and bicubic port, preprocessing version; fixture tests (OQ-4 tolerance recorded)
- [ ] T010 `media/digest.rs`: digest per FR-006, raw-bytes memo; pixels dropped after the digest and re-decoded for encoding (FR-004); tests: re-encoded identical pixels share a digest, changed `max_image_tokens` changes it
- [ ] T011 `media/mod.rs`: parse image blocks per API and location, limits FR-004 before queuing (48 MiB decoded per request; fixed 20 MiB / 50 MP caps reported as fixed), remote and file sources per FR-003, `detail`; tests in `crates/mrun/tests/media_limits.rs` and `media_api.rs`
- [ ] T012 Renderer: image spans (start, placeholders, end) from parsed blocks only; literal placeholder text never forms a span; `count_tokens` with image tokens; golden renders vs HF template
- [ ] T013 `ScriptedEngine`: media spans, embeddings with bytes, encoder cost and per-layer cancel, position offsets on live state, segments and checkpoints (needs 001 scripted engine)

## Phase 3: US1 — Show the agent a screenshot (P1) 🎯 MVP

- [ ] T014 [US1] `models/qwen3_vl_vision.rs` [mlx]: patch embed, blocks with 2D RoPE, position-embedding interpolation, merger; loader keeps vision tensors when enabled (`--no-vision` skips them)
- [ ] T015 [US1] `models/qwen3_5_moe.rs` [mlx]: interleaved M-RoPE `[11,11,10]` with text positions equal on all axes (text-only path bit-identical to before); embedding injection at span positions; `next_position`
- [ ] T016 [US1] Budget: tower weights measured, encoder scratch probed at `max_image_tokens` (extends 001 FR-008 cache key with vision settings); `--info`, `mrun ps`, `/mrun/status` breakdown (FR-007)
- [ ] T017 [US1] Worker: split prefill at each uncached span start, resolve its embedding (cache or encode from re-decoded pixels) before that prefill call, one image at a time; release uncached embeddings after commit; engine `allocated()` includes embedding bytes (003 invariant); cancel during encoding
- [ ] T018 [US1] API acceptance per FR-002 on all three APIs; metadata `images`, `image_tokens`, `encoder_runs`; warnings `image-first-frame`, `image-downscaled`
- [ ] T019 [US1] macOS `crates/mrun-engine/tests/vision_conformance.rs`: ≥ 10 prompts vs mlx-vlm `4623cf9` (MLX v0.32.4) on identical pixel tensors, token for token for 32 tokens (SC-005)

## Phase 4: US2 — Images in history cost nothing (P1)

- [ ] T020 [US2] 003 tree: `Image{digest_id, index}` key symbols with interned digests; offsets on nodes, segments and checkpoints; debug recheck of offsets from spans on every restore
- [ ] T021 [US2] `media/embed_cache.rs`: LRU keyed (model identity, digest), cap, evicted before branches and under pressure; `embedding_hits` metadata
- [ ] T022 [US2] Miss cause `image` with precedence per data model; `MessageMark` content hash includes image digests (update 002 T004 and the 003 data model when this lands); tests including OpenCode-style replacement by text
- [ ] T023 [US2] `crates/mrun/tests/prefix_images.rs` on `ScriptedEngine`: appended turns with an image in history (zero encodes), subagent with the same image (embedding hit), edit before and after an image, hybrid resume across an image, cancel mid-encode
- [ ] T024 [P] [US2] macOS `crates/mrun-engine/tests/vision_equality.rs`: ≥ 20 cache-on vs cache-off cases bit-identical at the same split (SC-004)

## Phase 5: US3 — Text-only models stay predictable (P1)

- [ ] T025 [US3] No active tower → 002 FR-020 path untouched; run 002's image tests on a text-only model and with `--no-vision` (SC-007)
- [ ] T026 [US3] `/v1/models` capabilities; 004 `launch`/`connect`: Codex catalog `input_modalities`, OpenCode `modalities.input`, Claude Code warning when vision is off; snapshot tests; update the 004 CLI contract when it lands
- [ ] T027 [US3] 004 `pull`/`show`/`ls`: vision presence from the index; sidecar fetch via 006's range-request path if T001 found no tower (FR-015), implemented here if 006 has not shipped it; identity includes `preprocessor_config.json` and the sidecar hash (FR-008); test that changing either changes the identity

## Phase 6: US4 — Vision fits the budget (P2)

- [ ] T028 [US4] Admission counts image tokens; `initial_need` includes the largest uncached embedding; tests at maximum count and size under injected pressure (SC-006 structural)

## Phase 7: Polish

- [ ] T029 SC-001 (Claude Code paste and Read tool, Codex `view_image`, pinned versions, 5 runs each), SC-002, SC-003, SC-006 on the target Mac; record versions and numbers in research (needs 005 harness)
- [ ] T030 [P] `mrun bench --suite vision` cells: encoder time per size, TTFT overhead cold and warm; add the suite to the 005 contract and 004 CLI contract when it lands
- [ ] T031 [P] Site page `site/content/docs/images.md` (sources, limits, no remote fetching, memory cost, `--no-vision`); update 002/003/004 public docs where behaviour changes
- [ ] T032 [P] Audit test: with images in requests, server logs, `/mrun/status`, `/metrics` and error messages contain no pixels, raw bytes or digests (FR-016)
- [ ] T033 When this feature lands, reconcile upstream specs: 002 FR-020 (images accepted with vision), 002 contract metadata, 003 contract and data model (key symbols, `image` cause, offsets), 004 CLI contract, 005 contract, 008 metrics contract

## Dependencies

- Gate: T001–T006 before everything else; T006 decides whether Phase 1 starts.
- 001: engine, budget, probe (T014–T016). 002: parsing, renderer, metadata (T011, T012, T018).
  003: store (T020–T023). 004: launch, pull (T026, T027). 005: trace stats and harness (T004, T029).
  006: sidecar mechanism (T027, only if needed; built in 009 if 006 has not shipped it).
  008: `/metrics` exporter (T003).
- T007 → T008 → T009/T010 → T011 → T012 → T013 → T014 → T015 → T016 → T017 → T018 (MVP) → T019;
  T020 → T021 → T022 → T023 → T024; T025–T028 after T018; T027 → T014 when T001 found no tower; T029–T033 last.

## Parallel opportunities

T001–T004; T009 and T010; T024, T025 and T026; T030, T031 and T032.
