# MLX from Rust: Bindings, mlx-lm Internals, and Design Inputs for mrun

**Date**: 2026-10-14. **Context**: mrun is being rebuilt as a Rust server that calls Apple MLX for inference.
mrun owns the scheduler, the KV cache with exact prefix reuse through snapshots, the memory budget, the APIs and
the CLI. Read [landscape-2026-10.md](landscape-2026-10.md) first.

**Method**: The repos were shallow-cloned on 2026-10-14 and read. The commits were:

| Repo | Commit (date) | Latest tag (date) |
|---|---|---|
| oxideai/mlx-rs (now `oxiglade/mlx-rs`) | `9051034` (2026-09-15) | crates `mlx-rs 0.32.0`, `mlx-sys 0.6.0` (2026-09-12) |
| ml-explore/mlx | `36969c0` (2026-10-14) | v0.32.4 (2026-10-14; was v0.32.3 when this study was written) |
| ml-explore/mlx-c | `a341b49` = v0.7.0 (2026-09-28) | v0.7.0 |
| ml-explore/mlx-lm | `ba8a08f` (2026-10-10) | v0.32.0 (2026-09-30) |
| jundot/omlx | `6feea9f` (2026-10-14) | v0.7.0 (2026-10-01); v0.7.1.dev1 |
| jbg/eredu (`safemlx*` crates) | `2f1d623` (2026-10-14) | safemlx-v0.3.1 |

Crate versions come from the crates.io API. huggingface.co and omlx.ai were not reachable, either from the
container or through WebFetch, so model sizes are computed and benchmark numbers are second-hand. The only build
attempted was one Linux CPU probe of mlx-rs (§1.4). Nothing was run on Apple hardware.

**Markers**: **[V]** means verified in source or docs at the commit above. **[U]** means unverified: secondary
sources, estimates, or inference.

## 1. Rust bindings for MLX

### 1.1 Options

| Option | Version / date | Wraps | Maintenance | License |
|---|---|---|---|---|
| **mlx-rs** (`mlx-rs`, `mlx-sys`, `mlx-macros`, `mlx-internal-macros`) | 0.32.0 / mlx-sys 0.6.0, 2026-09-12 [V crates.io] | mlx-c `c74db530` (v0.6.0-7), MLX **0.32.2**, as a git submodule. CMake FetchContent pulls MLX from GitHub [V `mlx-sys/Cargo.toml`, `.gitmodules`, mlx-c `CMakeLists.txt:37-38`] | Dormant from March to August 2026, then a burst: 21 commits since 2026-06-01, 18 of them by one maintainer (dcvz) [V git log]. The workspace has a `mlx-lm` crate (Llama + Qwen3, KV snapshots, prefix reuse) marked `publish = false` [V `mlx-lm/README.md`] | MIT OR Apache-2.0 [V] |
| **safemlx** (`safemlx-sys` 0.2.3, `safemlx` 0.3.1; part of the *Eredu* runtime) | 2026-09-07 [V crates.io] | Vendored MLX **0.32.3** tarball with a SHA-256 check and local patches, including an added `mlx_get_cache_limit`/cache-policy getter, plus checked-in bindings [V `safemlx-sys/README.md`, `vendor/`] | Very active: 1,370 commits since 2026-07-03, essentially one author [V git log]. Has Linux x86-64 CPU (+CUDA) and Windows CI [V `.github/workflows/linux.yml`] | safemlx-sys MIT; workspace MIT OR Apache-2.0 [V] |
| `mlx-nn 0.1.0` (2026-02-23), `mlx-lm 0.0.1` (2025-10-04) on crates.io | stale placeholders, no repository link [V crates.io] | – | – | – |
| `rlx-mlx-sys` (MIT-RLX) | 0.2.17, 2026-10-04 | hand-written C++ shim over vendored MLX [V crates.io description] | part of another framework | – |
| **Own `-sys` over mlx-c** | – | mlx-c v0.7.0 → MLX 0.32.2 [V `CMakeLists.txt`] | mrun-owned | MIT (mlx-c), MIT (MLX) |

### 1.2 mlx-rs API coverage (at `9051034`)

| Need | Status | Where |
|---|---|---|
| Arrays, lazy graph, `eval`, `async_eval` | yes | `mlx-rs/src/transforms/mod.rs:87` [V] |
| Streams | `with_stream`, `with_device` scopes and a `thread_local!` default stream. **No** mlx-c thread-local streams (`mlx_stream_thread_local` exists only in mlx-c v0.7.0, not in the pinned `c74db53`) | `stream.rs:9-84` [V] |
| `fast::scaled_dot_product_attention`, `rope`, `rope_dynamic`, `rms_norm`, `layer_norm` | yes | `fast.rs:13,84,184,247,294` [V] |
| `quantized_matmul`, `gather_qmm`, `gather_mm`, `quantize`/`dequantize`, `qqmm` | yes, but `quantized_matmul` and `gather_qmm` **hard-code `DEFAULT_MODE` (affine)**, so mxfp4/nvfp4 weights need a raw `mlx_sys` call | `ops/quantization.rs:104-135, 232-282` [V] |
| `slice_update` (in-place KV write) | yes, via `TryIndexMutOp`/`UpdateMode` | `ops/indexing/indexmut_impl.rs:27-65` [V] |
| safetensors load/save, with metadata | yes (`Array::load_safetensors*`); GGUF too | `ops/io.rs:57-133` [V] |
| Memory: active, peak, reset peak, cache, limit, `set_memory_limit`, `set_cache_limit`, `set_wired_limit`, `clear_cache` | **all present** (added 2026-09-01, `04a722b`) | `memory.rs:13-61` [V] |
| `metal::is_available`, `device_info` (`max_recommended_working_set_size`) | **not wrapped**. Present in mlx-sys (`mlx_metal_is_available`, `mlx_device_info_get`) | mlx-c `metal.h:32`, `device.h:92-130` [V] |
| `compile` | present, but ROADMAP says `compile_with_state` **re-traces every call** (measured at mlx-c v0.5.0) | `ROADMAP.md` "Known-weak coverage" [V] |
| Thread safety | `unsafe impl Send for Array`, no `Sync`. The project's own note: graph construction across threads is safe, but "concurrent *eval* on the shared default stream" is not | `array/mod.rs:75`, `ROADMAP.md` [V] |

The project is candid about its quality ([V] `ROADMAP.md`): `assert_array_eq!` tolerances are too loose, 47
doctests never run, and **"`mlx-lm` has zero executing tests"**. Eval-stage C++ exceptions used to abort the
process (measured at MLX 0.30.6) and are said to be catchable at 0.32.2. `CHARTER.md` defers 75 newly added
mlx-c ABI entries.

### 1.3 Build requirements (macOS)

- `mlx-sys/build.rs` runs CMake on mlx-c, which fetches and builds **all of MLX from source** (C++ and Metal
  shaders) and links it statically (`libmlx.a`, `libmlxc.a`, `gguflib`) [V `build.rs:110-175`]. It calls
  `xcrun`/`xcode-select` to find `libclang_rt.osx.a`, and needs bindgen (libclang) [V]. **Compiling the Metal
  shaders needs the full Xcode Metal toolchain.** The Command Line Tools alone do not have `metal` [U: MLX build
  docs; ROADMAP: "Metal needs the system Xcode"].
- `mlx.metallib` is written to `~/.mlx/lib/<mlx-c key>/`, which `MLX_RS_METAL_PATH` overrides. The path is baked
  in at build time, and `cargo install` binaries look it up at runtime [V `build.rs:85-100`, CHANGELOG 0.6.0].
  **For distribution, mrun must ship the metallib next to the binary** and call `set_metallib_path`
  (`metal.rs:64`) [V API; U packaging].
- Build time: one full MLX C++/Metal build. Expect minutes per clean build. It is cached by Cargo after that [U:
  not measured on a Mac].
- CI: mlx-rs runs only on macOS (Blaze `macos-latest` runners) [V `.github/workflows/validate.yml`].

### 1.4 Linux (CPU backend) for CI

- MLX itself supports a Linux CPU backend, and Eredu/safemlx proves the Rust path works: its CI runs
  `ubuntu-22.04` CPU jobs plus a self-hosted CUDA job [V workflows].
- **mlx-rs does not build on Linux as shipped.** `build.rs` links `framework=Foundation`, `dylib=objc` and `c++`
  unconditionally, and forces `CMAKE_C(XX)_COMPILER=/usr/bin/cc|c++` [V `build.rs:116-160`]. The probe (`mlx-rs`
  with `default-features = false` on x86-64 Ubuntu 24.04, cmake 3.28) failed first in MLX's CMake
  `FindBLAS/FindLAPACK`, which needs `libopenblas-dev liblapacke-dev`. With those installed, MLX + mlx-c CPU
  **compiled in 3 min 19 s wall time (10.5 CPU-min, 4 cores)**. The build then stopped with `error: library kind
  'framework' is only supported on Apple targets` [V probe]. A Linux fix needs a `target_os` gate in `build.rs`,
  which mlx-rs does not have; safemlx-sys does.

## 2. mlx-lm internals to learn from (`ba8a08f`)

### 2.1 Models

- **`models/qwen3.py`** (234 lines): Attention is q/k/v/o `Linear`, plus **per-head `q_norm`/`k_norm` RMSNorm
  before RoPE** (`:60-61, 80-84`). RoPE uses `offset=cache.offset` (`:89-90`), then
  `base.scaled_dot_product_attention` (`:96`), which dispatches to `mx.fast.scaled_dot_product_attention` or to
  the quantized-KV SDPA when the cache has `bits` (`base.py:110-139`). The MLP is SwiGLU (`:103-112`).
  Embeddings are tied when `tie_word_embeddings` is set (`:180-199`). `rope_parameters` is normalised for
  transformers ≥ 5 (`:32-40`) [V].
- **`models/qwen3_moe.py`** (336 lines): `Qwen3MoeSparseMoeBlock` (`:112-150`) does `gate` Linear →
  `softmax(precise=True)` → `argpartition` top-k → `take_along_axis` → optional `norm_topk_prob` renormalisation
  → `SwitchGLU` → weighted sum. `decoder_sparse_step` and `mlp_only_layers` choose which layers are dense
  (`:25-26`). `quant_predicate` keeps the **router `mlp.gate` at 8 bits, group 64** (`:326-331`) [V].
- **`models/switch_layers.py`**: `SwitchGLU` calls `mx.gather_qmm` per projection (`:93-105`). When
  `indices.size >= 64` it **sorts tokens by expert** (`_gather_sort`/`_scatter_unsort`) and passes
  `sorted_indices=True` (`:201-218`). This matters for prefill speed [V].

### 2.2 KV cache (`models/cache.py`)

- **`KVCache`** (`:339-421`): `step = 256`. On overflow it allocates `ceil(L/256)*256` more positions with
  `mx.zeros` and `concatenate`s them. It first slices off unused tail capacity when `offset % step != 0`
  (`:347-365`). It writes with `keys[..., prev:offset, :] = keys` (slice_update). `keys_and_values()` returns a
  `[:offset]` view. **`trim(n)` only decrements `offset`** (`:391-394`). `state = (keys, values, offset)`.
  `to_quantized()` exists [V].
- **`QuantizedKVCache`** (`:251`, step 256, default 8-bit g64), **`RotatingKVCache`** (`:423`, `max_size`,
  `keep`; trimmable only before wrap), **`ArraysCache`** (`:596`, the recurrent/conv state of GDN/Mamba layers;
  **not trimmable**; the old `MambaCache` name is gone), `ChunkedKVCache`, `CacheList` (a composite per layer),
  `BatchKVCache`/`BatchRotatingKVCache` (left-padded batching) [V].
- `_BaseCache` has `state`/`meta_state`-style properties, `is_trimmable`, `nbytes` and `from_state`
  (`:154-192`). `can_trim_prompt_cache` returns true only if **all** layers can be trimmed (`:115-138`).
  `save_prompt_cache`/`load_prompt_cache` write one safetensors file holding the flattened states, with class
  names and scalars in its metadata (`:43-112`) [V].
- `make_prompt_cache` defers to `model.make_cache()` when present, so a hybrid model (`qwen3_5.py:347,575`,
  `qwen3_next.py:477`) gets a mixed list (`:15-40`) [V].

### 2.3 Generation loop (`generate.py`)

- A module-level `generation_stream = mx.new_thread_local_stream(default_device)` (`:225`) [V].
- `wired_limit()` context (`:229-263`): sets the wired limit to
  `device_info()["max_recommended_working_set_size"]` (`utils.py:74-83`) and warns when the model exceeds 90% of
  it. On exit it **synchronizes the streams before restoring the limit**. Never change the wired limit during an
  async eval [V].
- `generate_step` (`:304-473`) has default `prefill_step_size=2048`. It prefills every token except the last in
  chunks: model call, then `mx.eval([c.state for c in cache])`, then `mx.clear_cache()` (`:427-454`). The **last
  prompt token goes through `_step` alone** (L = 1). Decode is **pipelined**: it builds step n+1 and
  `async_eval`s it before `y.item()` blocks on step n (`:456-471`), and calls `clear_cache()` every 256 tokens
  [V].
- Samplers (`sample_utils.py`): `make_sampler` (temp, top-p, min-p, top-k, XTC), `make_logits_processors`
  (repetition, presence and frequency penalties over a `context_size=20` window). Sampling happens on logprobs:
  `logits - logsumexp` [V `:13-361`, `generate.py:411-413`].

### 2.4 Server prompt cache (`server.py` + `cache.py:1462-1741`)

- **`PromptTrie`** (`:1508`) is a token-level dict trie per model. `search` returns `exact`, `shorter` (the
  longest stored prefix of the request), `longer` (the shortest stored sequence that extends the common prefix)
  and `common_prefix` [V].
- **`LRUPromptCache.fetch_nearest_cache`** (`:1652-1672`) checks three cases in order:
  1. Exact match: return a `deepcopy` of the entry.
  2. A longer entry whose common prefix beats the shorter one, and every layer can be trimmed: deepcopy it, then
     **trim to `min(len(tokens)-1, common_prefix)`**. At least one token is always left to compute logits.
  3. Otherwise, a shorter prefix entry. **Every hit deep-copies the whole cache** [V].
- **`insert_cache`** (`:1674-1715`): when the cache can be trimmed, it **drops every stored prefix of the new
  entry**, because the longer entry subsumes them. Eviction is by count (`--prompt-cache-size`, default 10) and
  by bytes (`--prompt-cache-bytes`). The LRU is split by `cache_type` into `assistant`, `user` and `system`.
  `pop()` evicts from the earliest type in the order `assistant, user, system` that has at least as many entries
  as the next type. The effect is that system-prompt checkpoints live longest [V `:1599-1650`].
- **Segmenting** (`server.py:583-631`): when the last message comes from the user, the prompt is split into
  `system` (found by re-templating only the system messages), `user` and a short `assistant` tail (the
  generation prompt, or a think tag within the last 11 tokens). The batch generator **snapshots the cache at
  each segment end** (`:830-846`) and after generation (`:876-885`). After every insert, `trim_to(n_bytes =
  total - active)` (`:762-765`) [V].
- Tool parsing: `tokenizer_utils.py:619-660` infers the parser from chat-template markers.
  `<tool_call>\n<function=` selects `qwen3_coder`, and `<tool_call>` with `tool_call.name` selects `json_tools`
  (Hermes, which covers Qwen3-8B). There are 14 parser modules in `tool_parsers/` [V].

## 3. MLX quantized checkpoints on Hugging Face

- **Format** [V `utils.py:382-406, 496-516, 1022-1027`]: config.json has `"quantization": {"group_size": G,
  "bits": B, "mode": ...}`, with an optional per-path override dict under the same key (for example
  `"model.layers.0.mlp.gate": {"group_size":64,"bits":8}`). Each quantized Linear stores `weight` (uint32,
  packed `32/B` values per word), `scales` and, for affine, `biases`. Affine scales are in the weight dtype
  (bf16 or f16). The mxfp4/nvfp4/mxfp8 modes use **uint8 scales and no biases**. A loader can infer the mode
  from the arrays: `group = in_dims / scales.shape[-1]`, `bits = weight.shape[-1]*32/in_dims`.
- **Defaults** per mode: affine (64, 4), mxfp4 (32, 4), nvfp4 (16, 4), mxfp8 (32, 8). Legacy
  `quantization_config.quant_method` values (mxfp4, mxfp8, fp8 ue8m0, AWQ/GPTQ repack) are mapped at load time
  [V].
- **Repos** [U: HF unreachable; names are confirmed by search hits and the mlx-lm benchmark table]:
  `mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit` (it ships a `qwen3coder_tool_parser.py` file), `-8bit`,
  `mlx-community/Qwen3-8B-4bit`/`-8bit`, `lmstudio-community/Qwen3-8B-MLX-4bit` (mlx-lm 0.24.0),
  `mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit`. A third-party `…Coder-30B-A3B-Instruct-MLX-4bit-mxfp4`
  exists.
- **Sizes** [U: computed at 4.5 bits per weight for affine 4-bit g64]: Qwen3-8B 4-bit ≈ 4.6 GB; Qwen3-30B-A3B
  4-bit ≈ 17.2 GB. mlx-lm reports **18.20 GB peak memory** for Qwen3-30B-A3B-2507-4bit and 33.46 GB for the
  8-bit version (M4 Max) [V `mlx_lm/BENCHMARKS.md`].

## 4. oMLX (v0.7.0, Apache-2.0)

- Python on `mlx==0.32.3` and a **git-pinned mlx-lm** commit (`pyproject.toml:12,48`) [V]. Started from vllm-mlx
  v0.1.0 [V README:462].
- **Cache design** [V `omlx/cache/*`, about 19.7k lines]: a vLLM-style `PagedCacheManager` (`paged_cache.py`)
  with **256-token blocks** (`prefix_cache.py:257`), ref counts, copy-on-write and an O(1) LRU free list. Each
  block hash is a **SHA-256 chain**: `model_name ‖ parent_hash ‖ str(tuple(token_ids)) ‖ extra_keys`
  (`paged_cache.py:84-125`). Blocks are logical. KV lives in mlx-lm's contiguous caches, and on a hit
  `reconstruct_cache` **loads per-block tensors and concatenates them per layer** (`prefix_cache.py:3176-3210,
  3866`). The SSD tier (`paged_ssd_cache.py`, 5.5k lines) writes **one safetensors file per block**. A
  background writer drains a RAM-bounded queue (10–30% of RAM), and a startup scan reuses the files after a
  restart. The default SSD budget is 50% of free disk (README:366). Hybrid models store non-sliceable
  `ArraysCache` state as **boundary snapshots per block** (`boundary_snapshot_store.py`). The text describes the
  hot RAM tier as "write-back".
- **Claude Code integration** [V]: `omlx/integrations/claude.py:86-150` launches `claude` with
  `ANTHROPIC_BASE_URL`, an empty `ANTHROPIC_API_KEY`, `API_TIMEOUT_MS=3000000`,
  `CLAUDE_CODE_AUTO_COMPACT_WINDOW=<ctx>`, model-tier env vars and a 48k minimum context window. The Anthropic
  adapter sends SSE `ping` keep-alives during long prefills (`anthropic_models.py:354`). It **strips the
  `x-anthropic-billing-header:` system block and the per-request `<total_tokens>N tokens left</total_tokens>`
  marker**. These "break prefix cache… measured as full ~60k-token re-prefills per turn"
  (`api/anthropic_utils.py:688-707`).
- **Learn**: canonicalise the prompt head; segment boundaries matter more than block granularity; use a disk
  tier for restarts; show cache-hit stats; keep the SSE alive; report the real context window.
- **Avoid**: block-granular KV that has to be concatenated back into a contiguous cache (an O(prefix) copy on
  every hit); hash chains over `str(tuple)`; 20k lines of cache code with a history of TOCTOU fixes
  (`prefix_cache.py:796-806`); write-back complexity before measuring the need.

## 5. Determinism and batch invariance in MLX

- **Kernel choice depends on shape** [V MLX source]. SDPA uses `sdpa_vector` (and a 2-pass variant) for query
  length ≤ 8 and the full kernel above that (`backend/metal/scaled_dot_product_attention.cpp:956-1030`).
  Quantized matmul uses `qmv`/`qmv_fast`/`qmv_wide` up to a K/N-dependent `vector_limit` and `qmm_t` or
  `qmm_t_splitk` beyond it. On M5 it uses `*_nax` (`backend/metal/quantized.cpp:465-545, 852-1250, 1840-1906`).
  The summation order differs between these kernels. **So K/V for the same token computed in an L=1 step, a
  512-token chunk or a 2048-token chunk are not guaranteed bit-identical** [U: not measured; follows from the
  dispatch].
- mlx-lm itself mixes schedules. The last prompt token runs at L=1, and every generated token's K/V comes from
  L=1 decode steps. A "cold" re-prefill of the same conversation therefore computes those positions in large
  chunks and can differ in low bits [V code path; U magnitude].
- **Real bug, now fixed**: mlx#3856, opened 2026-07-17 and closed 2026-08-26 by PR #3922. A one-shot forward on
  affine-quantized MoE (Qwen3-Coder-30B-A3B-8bit) diverged from 2048-token chunked prefill when the gathered
  rows were > 32768 and not a multiple of 64. The cause was a `short` overflow in `affine_gather_qmm_rhs_nax`,
  on **M5/NAX only, not M1–M4**. Logits differed by about 17, and needle recall fell from 100% to 0%. A sibling
  issue, #3887 (K % 64 ≠ 0, also affects mxfp4), is mentioned [V issue page via WebFetch]. MLX ≥ 0.32.3 should
  contain the fix [U].
- No MLX document promises batch invariance [U: search found none]. Run-to-run determinism for identical shapes
  on one machine is expected but not documented [U].

## 6. Performance reference numbers

| Source | HW | Model / quant | Prefill tok/s | Decode tok/s | Marker |
|---|---|---|---|---|---|
| mlx-lm BENCHMARKS.md (mlx 0.29.2, mlx-lm 0.28.2, `-p 2048 -g 128`) | M4 Max 64 GB | Qwen3-30B-A3B-2507 4-bit | 1753.9 | 113.3 | [V repo] |
| same | M4 Max 64 GB | Qwen3-30B-A3B-2507 8-bit | 1719.5 | 83.2 | [V] |
| same | M4 Max 64 GB | Qwen3-4B-2507 4-bit | 1622.3 | 134.5 | [V] |
| oMLX community bench, v0.4.1 | M4 Pro 20c **48 GB** | Qwen3-Coder-30B-A3B 4-bit, 1k / 4k | – / 768 | 72.6 / 61.1 | [U search snippet] |
| oMLX community bench, v0.2.7 | M4 Pro 20c 64 GB | same, 1k / 4k | 734 / 717 | 76.5 / 62.3 | [U] |
| oMLX community bench, v0.2.6 | M4 Pro 16c 24 GB | Qwen3-8B (quant unstated), 1k / 4k | 350 / 338 | 49.5 / 42.4 | [U] |
| Bandwidth scaling (M4 Pro 273 GB/s vs M4 Max 546 GB/s) | M4 Pro | 30B-A3B 4-bit; 8B 4-bit | ≈ 0.5× Max | ≈ 55–75; ≈ 45–55 | [U estimate] |

The figures mrun should qualify against on the target box are **Qwen3-8B-4bit: about 45–55 tok/s decode, a few
hundred to about 1k tok/s prefill**, and **Qwen3-Coder-30B-A3B-4bit: about 60–75 tok/s decode, about 700–900
tok/s prefill**. Measure them with `mlx_lm.benchmark -p 2048 -g 128` on the same machine as the oracle.

## 7. Implications for mrun

### 7.1 Binding strategy: own `mrun-mlx-sys` over mlx-c, with a thin safe layer of our own

- **Do not depend on `mlx-rs` as-is.** It cannot build on Linux, its safe wrappers hard-code the affine quant
  mode, it does not wrap `device_info`, `metal_is_available` or thread-local streams, and its single maintainer
  has a bursty history while its own ROADMAP admits weak tests. Use it as a **reference** for the FFI patterns
  that matter: error-handler setup, `Guarded` result conversion, `mlx_*_free` ownership, the metallib path, and
  the snapshot/ledger design in `mlx-lm/src/cache`.
- **Do not adopt safemlx/Eredu as a dependency.** It is a full competing runtime, has one author, and patches
  MLX. Copy its *build approach* instead: a vendored, SHA-pinned MLX tarball; `target_os` gating; Linux CPU and
  BLAS detection; checked-in bindgen output.
- **Plan**: a crate `mrun-mlx-sys` with mlx-c as a git submodule, **pinned to v0.7.0 (`a341b49`), with MLX
  pinned to v0.32.4** (updated 2026-10-15 per the fact check; includes the #3922 gather_qmm fix and later SDPA/gather_qmm fixes) by overriding `FETCHCONTENT_SOURCE_DIR_MLX` to a
  vendored, hash-checked source. Check in the generated bindings so normal builds need no libclang. `build.rs`
  links Foundation, Metal and Accelerate only on macOS, turns Metal off and requires OpenBLAS/LAPACKE on Linux,
  and puts the metallib next to the binary.
- Above that, `mrun-mlx` exposes only what mrun uses, about 60 functions: an `Array` RAII handle (`!Sync`,
  owned by one engine thread); eval and async_eval; stream creation; the Qwen3/MoE ops (matmul,
  quantized_matmul and gather_qmm **with mode**, the fast ops, argpartition, take_along_axis, softmax(precise),
  slice and slice_update, concatenate, astype); safetensors load; memory and device_info.
- **Threading**: one engine thread owns every MLX object and one dedicated GPU stream, and the async (tokio)
  side talks to it over channels. This matches the ROADMAP's warning about concurrent eval and mlx-lm's
  thread-local generation stream.

### 7.2 Model port plan

1. **Qwen3 dense (Qwen3-8B-4bit)**: port `qwen3.py` 1:1. Watch for per-head q/k RMSNorm before RoPE, a RoPE
   offset taken from the cache, GQA through `fast::sdpa`, and a lm_head that is tied or quantized. The loader
   reads `quantization` with per-path overrides and infers the mode from the scales dtype and shapes (§3). Gate
   it on a logits oracle: mlx-lm at the pinned version on the same Mac, with a tolerance-based comparison on
   fixed prompts, plus greedy token-match over 256 tokens.
2. **Qwen3-MoE (Qwen3-Coder-30B-A3B-4bit)**: add the router (8-bit `mlp.gate`, `softmax(precise)`,
   `argpartition` top-8, norm_topk_prob) and SwitchGLU via `gather_qmm`, with the **sort-by-expert path once
   `indices.size >= 64`**. Use the Qwen3-Coder XML tool parser. This is the product model.
3. Later: hybrid GDN (Qwen3.6/3.8) needs `ArraysCache`-style whole-state snapshots (see the landscape §3.2).

### 7.3 KV cache design

- Per sequence, keep per-layer K and V buffers `[1, n_kv, cap, d]`, **pre-allocated in steps of 256 positions**
  (the same as mlx-lm), with an `offset`. **Trim = decrement `offset`**. Writes use `slice_update`.
- **Copy semantics, which are critical**: MLX `SliceUpdate` writes in place only when the input is *donatable*:
  use_count 1 and sole data owner (`array.h:284-287`; `metal/indexing.cpp:709-727` copies `in`→`out` and donates
  when possible). Any extra live handle (a snapshot sharing the buffer, or a lingering view) turns **every
  subsequent append into a full-buffer copy**. Rules:
  - The live cache must hold the only reference.
  - A **snapshot is a materialised slice copy** `keys[..., :n, :]` → `eval` into a new, compact, owned buffer.
  - **Restore copies the snapshot into a fresh step-aligned buffer.** Never alias. (mlx-rs's `Cache::snapshot`
    shares handles: cheap to take, but the next decode step then copies the whole buffer. mlx-lm's server
    `deepcopy`s on every hit.)
- **Size accounting by construction**: bytes = layers × 2 × n_kv × d × cap × dtype. For Qwen3-Coder-30B-A3B that
  is 48 × 2 × 4 × 128 × 2 B = 96 KiB per token, about 1.5 GiB at 16k. Charge `cap`, not `offset`.

### 7.4 Prompt-cache design (from mlx-lm server and oMLX)

- Use a **token trie with whole-sequence entries** (mlx-lm), not paged blocks (oMLX). A lookup returns the
  longest usable entry. For full-attention models, a *longer* entry can be trimmed down to the common prefix
  (mlx-lm's `longer` case). Always leave at least 1 token to prefill.
- **Checkpoint at segment boundaries**: the end of the system prompt and tools, the end of the previous turn,
  and the end of generation. Evict assistant entries before user entries before system entries. Drop stored
  prefixes that a newer trimmable entry subsumes.
- **Canonicalise the request before tokenising**: strip Claude Code's billing header and `<total_tokens>` marker
  (oMLX §4). Otherwise there are zero hits past the system prompt.
- Disk tier (v1.x): one safetensors file per *snapshot* (not per block), keyed by SHA-256 over (model digest,
  tokenizer digest, template digest, token IDs), using mlx-lm's `save_prompt_cache` layout as a reference. Write
  it asynchronously off the engine thread after `eval`.
- **Define "exact reuse" honestly** (§5): a restored snapshot reproduces the *bytes* that were computed earlier.
  It is not bit-identical to a cold prefill with different chunking. To make hit and miss paths comparable in
  tests, fix the chunk boundaries at absolute positions that are multiples of the prefill step. Assert equality
  only for hit-versus-hit replays. Test hit-versus-cold with a logits tolerance and greedy-token agreement.

### 7.5 Memory APIs for budget-by-construction

At startup:

1. Read `device_info().max_recommended_working_set_size` (about 75% of 48 GB, or about 36 GB [U]).
2. Compute the budget: weights (from safetensors headers, before loading) + KV arena (sum of `cap`) +
   prompt-cache pool + activation headroom (measured via `peak_memory` on a calibration prefill at
   `prefill_step_size`).
3. Refuse to load a configuration that does not fit.
4. Call `set_wired_limit(min(budget, recommended))`, `set_memory_limit(budget)` and `set_cache_limit(small, for
   example 1–2 GiB)`.

Call `clear_cache()` after each prefill chunk and every 256 decode tokens (as mlx-lm does). Change the wired
limit only after `synchronize` (§2.3). Export `active_memory`, `peak_memory` and `cache_memory` as metrics.

### 7.6 CI strategy

- **Linux (every PR)**: build `mrun-mlx-sys` against the MLX CPU backend (`MLX_BUILD_METAL=OFF`, OpenBLAS +
  LAPACKE) and cache the CMake build directory. Run unit tests for the cache, trie, scheduler, API and template
  code, plus a **tiny random-weight Qwen3 and Qwen3-MoE** forward pass checked against committed
  NumPy/MLX-Python golden values (the mlx-rs conformance idea).
- **macOS arm64 runner (merge queue / nightly)**: build with Metal, then run the mlx-lm oracle comparison on
  Qwen3-0.6B/8B-4bit and a perf smoke test. GitHub-hosted `macos-15`/`26` arm64 runners have no GPU passthrough
  guarantees, and Blaze or self-hosted runners cost money [U].

### 7.7 Top risks

1. **FFI ownership and C++ exceptions**: leaks, use-after-free, or aborts from eval-stage errors. Mitigate with
   one engine thread, RAII, an mlx-c error handler, and a leak test (mlx-rs `xtask verify-ffi` idea).
2. **Hidden full-buffer copies** from non-donatable `slice_update`, which turn decode into O(context) work.
   Mitigate with assertions on `use_count` in debug builds and by benchmarking decode tok/s against context
   length.
3. **Quantized-kernel correctness bugs per chip and shape** (#3856 class). Pin MLX, add a
   chunked-versus-one-shot regression test, and track MLX releases.
4. **Build and distribution**: the Xcode Metal toolchain is required, the metallib location matters, and each
   clean build is long. Prebuild MLX in CI and ship the metallib with the binary.
5. **Binding upkeep**: mlx-c moves with MLX, about 75 ABI additions per bump according to the mlx-rs CHARTER. A
   small owned surface keeps bumps cheap.
6. **Performance gap**: compiled graphs (`mx.compile`) and the async pipelining in mlx-lm. Port async_eval
   pipelining on day one, and treat compile as optional.

## Sources

- mlx-rs: https://github.com/oxideai/mlx-rs (→ oxiglade/mlx-rs); crates.io `mlx-rs`, `mlx-sys`
- Eredu/safemlx: https://github.com/jbg/eredu ; crates.io `safemlx-sys`
- MLX: https://github.com/ml-explore/mlx ; mlx-c: https://github.com/ml-explore/mlx-c
- mlx-lm: https://github.com/ml-explore/mlx-lm (`mlx_lm/models/{cache,qwen3,qwen3_moe,switch_layers,base}.py`,
  `generate.py`, `server.py`, `utils.py`, `tokenizer_utils.py`, `BENCHMARKS.md`)
- oMLX: https://github.com/jundot/omlx (`omlx/cache/*`, `omlx/integrations/claude.py`,
  `omlx/api/anthropic_utils.py`)
- MLX issue #3856 / PR #3922: https://github.com/ml-explore/mlx/issues/3856
- oMLX community benchmarks (via search snippets): https://omlx.ai/benchmarks/fh0qhmnx ,
  https://omlx.ai/benchmarks/kp4yo25l , https://omlx.ai/benchmarks/r7hwunza
- Batch-invariance background:
  https://mohakchugh.is-a.dev/blog/batch-invariant-kernels-deterministic-llm-inference

