# Research: Persistent Prefix Tier

Evidence marks as in [next-features](../../concepts/research/next-features-2026-10.md).

## R1: Value and bandwidth

- **Finding**: Restore vs recompute [I]: Qwen3-Coder 30K tokens = 2.75 GiB of attention state,
  ≈ 0.6 s at ≈ 5 GB/s (M4 Pro 2 TB internal SSD [U]) against ≈ 40–60 s prefill. Qwen3.6-35B-A3B:
  30K × ≈ 20 KiB ≈ 0.6 GiB plus one checkpoint of ≈ 61 MiB [I] ≈ 0.2 s against ≈ 30–40 s.
  2603.04428 measured ≈ 500 ms reload vs 15.7 s re-prefill on an M4 Pro (Q4 storage) [V listing].
  Reuse is bimodal with a long tail after tens of minutes (2608.11668) [V listing].
- **Decision**: Measure first (FR-001), with the model loaded (realistic page-cache and memory
  pressure): `mrun bench --suite persist` reports sequential read/write GB/s in the tier directory
  (4 MiB blocks, 8 GiB, `F_NOCACHE`) and a 30K restore; `--from
  <record-dir>` replays the arrival order of a local recording through a simulated memory store
  (capacity from `/mrun/status`, LRU) with server-start markers, attributing re-prefilled tokens to
  `restart`, `evicted` or other causes, and simulating a disk tier at the cap. Ship only if restart
  plus eviction re-prefill is ≥ 10% of prefill time in the maintainers' recordings and SC-001
  holds.
- **Alternatives**: Trust literature numbers; rejected (constitution I).

## R2: File format

- **Decision**: Own container `*.mrps`, one file per entry:
  `magic "MRPS" | u32 format_version | u32 header_len | header JSON | token ids (u32 LE) |
  tensor blocks (16 KiB aligned) | footer {xxh3-64 per 4 MiB block, header xxh3}`. Header: cache
  key fields, MLX version, parent entry id and its `end`, `start`, `end`, `checkpoint: bool`, tensor
  table (layer, kind `k|v|conv|delta`, dtype, shape, offset, length; attention tensors
  `[heads, tokens, dim]`). Publish: write temp, `fcntl(F_FULLFSYNC)` on the file, rename,
  `fcntl(F_FULLFSYNC)` on the directory (plain `fsync` does not reach stable storage on macOS;
  Linux uses `fsync`). Load allocates destination
  arrays through the engine (inside the budget), then `pread`s directly into their host pointers
  (unified memory, no staging copy) and verifies each block's checksum as it lands.
- **Rationale**: Raw reads into budgeted buffers keep the 20× target and add no transient memory;
  xxh3 runs far above SSD speed; a self-describing header lets startup rebuild the index without a
  separate index file that could disagree with the entries.
- **Alternatives**: safetensors via `mlx_load_safetensors` (no per-block checksums, metadata only
  as strings, loader copies through its own path); one big append-only log (compaction and crash
  recovery complexity); SQLite blobs (extra dependency, double buffering).

## R3: What to persist and when

- **Decision**: Two sources, both immutable while read: (1) a branch segment as soon as it is moved
  out of the live path (003 moves create exactly these objects); (2) the live path's committed
  tokens not yet on disk, after 2 s of server idleness, as an entry whose parent is the previous
  live-path entry; live-path entries are split at persisted checkpoints. Before a branch is
  written, the writer persists the live path up to the branch point `at`, so the branch's parent
  exists on disk. Hybrids persist per turn only the end-of-prompt checkpoint (plus the system+tools
  checkpoint once); restore imports only the deepest checkpoint at or below the match. Writes
  happen once per range; LRU touches update mtime only. Tiny tails (< 256 tokens, which 003
  discards) are not persisted. Requests with `x-mrun-cache: off` are never persisted.
- **Rationale**: Write-on-evict cannot work under pressure (writing 2 GiB takes ≈ 0.5 s while the
  memory is needed now); writing early from immutable sources makes eviction free and keeps the
  write volume to new tokens only: per 2K-token turn ≈ 40 MiB attention + ≈ 61 MiB checkpoint
  ≈ 100 MiB on Qwen3.6, ≈ 190 MB on Qwen3-Coder [I].
- **Alternatives**: Write every turn synchronously (adds latency, constitution I); write only at
  shutdown (crashes and `kill` lose everything).

## R4: Concurrency with the engine thread

- **Decision**: The writer thread never calls MLX. The engine thread hands it `(host pointer,
  length)` views of evaluated arrays plus a generation counter. Before any operation that could
  free or overwrite those bytes (trim below the written range, per-layer growth, move-out of a
  layer, eviction of the source branch, shutdown), the engine thread calls `quiesce(range)`; it
  waits at most for one block, eviction included: the writer finishes or abandons its current
  4 MiB block (≤ ≈ 2 ms at ≥ 2 GB/s) and drops the views;
  the job is discarded and its temp file deleted. Appends need no quiesce.
- **Rationale**: Keeps ARCHITECTURE's single-MLX-thread rule (competitors that drove MLX from
  several threads hit stream errors and kernel panics); bounded critical-path cost (SC-003).

## R5: Lookup and restore decision

- **Decision**: The index maps a 64-bit hash of tokens `[0, end)` at each entry end, plus every
  boundary recorded in the header, to entries. Lookup walks the request's boundary hashes deepest
  first, then confirms with a full token comparison (FR-005). Restore decision: use disk when the
  disk match exceeds the memory match by ≥ 512 tokens (at ≥ 20× restore vs prefill, smaller gains
  are not worth a disk read); ties go to memory.
- **Alternatives**: Read-ahead at queue admission (`F_RDADVISE`) and an EWMA read-speed model with
  a `disk-slower` cause (previous design): dropped after analyze as unmeasured complexity.
  Prefetch into mrun buffers before admission (spends budget on requests that may be cancelled);
  rejected.

## R6: Identity binding

- **Decision**: Eligibility requires equality of: weights identity (001 R10; the 006 MTP sidecar is
  never part of it, D024), tokenizer hash, template hash, state format (attention dtype, layer
  layout, checkpoint format version), on-disk format version, and the vendored MLX/mlx-c version.
  mrun patch releases that keep these keep their entries. Directory layout `prefix/<sha256(key)[0..16]>/` groups files
  per key; the key itself is in every header and compared in full.
- **Rationale**: Stored bytes are exact for their tokens, but a version change alters recompute
  bits and possibly layouts, so the MLX version is part of the key.
- **Alternatives**: Bind to the full mrun version (simplest, but every patch release discards the
  whole tier).

## R7: Location, permissions, privacy

- **Decision**: `~/Library/Caches/mrun/prefix` (macOS excludes `~/Library/Caches` from Time
  Machine by default [U, re-checked at implementation]); additionally set the
  `com.apple.metadata:com_apple_backup_excludeItem` attribute and a `.metadata_never_index` file.
  Directories 0700, files 0600 created with `O_CREAT|O_EXCL` and `fchmod`; refuse a directory not
  owned by the user or with group/other bits. No app-level encryption; `doctor` warns when FileVault
  is off and persistence is on. Constitution IV opt-in is satisfied by FR-002.
- **Alternatives**: Keychain-held per-file keys (complexity, Keychain prompts for a background
  service); rejected for a single-user cache.

## R8: Eviction and cap

- **Decision**: LRU over leaf entries (entries with no children on disk) by mtime, touched on use;
  deleting a parent requires deleting its descendants first (they are unusable without it). Cap
  default 32 GiB; free-space floor 10% of the volume; max age 30 days. A sweep runs before each
  write that would exceed the cap and hourly.
- **Rationale**: Plain LRU is near-optimal for single long sessions (2609.28870 [V listing]);
  matches 003 D017.

## R9: Locking

- **Decision**: `flock(LOCK_EX|LOCK_NB)` on `prefix/lock` at startup. `mrun cache clear --disk`
  without a server takes the same lock; with a server it calls `POST /mrun/cache/clear?tier=disk`.
