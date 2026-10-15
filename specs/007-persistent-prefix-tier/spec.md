# Feature Specification: Persistent Prefix Tier

**Feature Branch**: `007-persistent-prefix-tier`

**Created**: 2026-10-15

**Status**: Draft (after 0.1; gated by the measure-first check in FR-001)

**Input**: User description: "Opt-in SSD tier for branches and checkpoints (`--persist` / config), exact bytes (no quantization), bound to full model identity + state format + MLX version, token-verified on load, files in ~/Library/Caches/mrun mode 0600, size cap with LRU, survives restarts, write-behind off the engine thread's critical path, read-ahead on lookup, crash safety (atomic files, checksums); requires constitution IV opt-in (already amended); `mrun cache` commands for size/clear; measure-first: share of misses due to restart/eviction in traces, SSD read GB/s. SC: 30K-token prefix restored from SSD ≥ 20× faster than re-prefill on the target Mac; zero stale reuse after model change."

## Clarifications

### Session 2026-10-15

- Q: Is the tier on by default? → A: No. Constitution IV (v3.1.0) requires an explicit opt-in: `mrun serve --persist`, `[cache] persist = true` or `MRUN_CACHE_PERSIST=1`. `mrun launch` does not enable it.
- Q: What is written, and when? → A: Exact sequence-state bytes of branches (attention state for a token range) and hybrid checkpoints, written by a background writer: when a branch is moved out of the live path, and for the live path's committed tokens once the server is idle for 2 s after a turn. Eviction from memory never waits for a write.
- Q: What binds an entry to a model? → A: The full cache key of 003 (weights, tokenizer, template, state format) plus the on-disk format version and the MLX version (decided 2026-10-15: mrun patch releases keep entries). Any difference means the entry is invisible; it is never converted.
- Q: Default size cap? → A: 32 GiB, capped further so the volume keeps at least 10% free; LRU by last use; entries unused for 30 days are deleted. All three are configurable.
- Q: Encryption at rest? → A: No application-level encryption; files are user-only (0700 directory, 0600 files) under `~/Library/Caches`, and `mrun doctor` warns when FileVault is off and persistence is on.
- Q: Two servers on one machine (different ports)? → A: The first server to start holds an exclusive lock on the tier directory; a second server runs with persistence off and says so.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Resume fast after a restart (Priority: P1)

A developer with persistence enabled stops mrun overnight, reboots, or upgrades it. The next
morning `claude --resume` sends yesterday's 30K-token conversation; mrun restores the state from
disk instead of recomputing it, and the first turn answers in about a second.

**Why this priority**: Human pauses and restarts are where in-memory caches lose most; a cold 30K
re-prefill takes tens of seconds to a minute on the target Mac.

**Independent Test**: With `--persist`, run a 30K-token conversation, restart the server, resend
the conversation plus one message; compare time to first token with a cold restart without
persistence.

**Acceptance Scenarios**:

1. **Given** persistence on and a finished conversation, **When** the server restarts and the
   conversation is resent, **Then** the prefix is restored from disk, usage reports it as cached,
   and mrun metadata names the source `disk`.
2. **Given** a hybrid model, **When** a stored conversation is resumed, **Then** the deepest stored
   checkpoint at or below the match is restored with its attention state and only the gap is
   recomputed.
3. **Given** a memory-evicted branch that was written to disk, **When** a later request matches
   it, **Then** it is restored from disk if that is faster than recomputing the gap.

---

### User Story 2 - Never reuse the wrong state (Priority: P1)

Persistence never feeds a request state computed for other tokens, another model, tokenizer,
template, state format or MLX version, and a damaged file is never used.

**Why this priority**: Stale or corrupted reuse silently corrupts answers; exactness is a
constitutional rule.

**Independent Test**: Run the mutation suite across restarts with model, template and version
changes, truncated, bit-flipped and half-written files; assert zero stale or corrupt loads.

**Acceptance Scenarios**:

1. **Given** entries written for model A, **When** the server restarts with model B (or A with a
   changed template or after an mrun upgrade), **Then** no entry of A is used and the miss cause
   is `new` (not a disk hit).
2. **Given** a file whose checksum does not match, **When** it is matched, **Then** it is deleted,
   the request falls back to recompute, and the miss cause is `disk-corrupt`.
3. **Given** a crash during a write, **When** the server restarts, **Then** the partial file is
   removed and never considered.

---

### User Story 3 - See and control what is stored (Priority: P2)

The developer sees how much disk the tier uses, per model, and can clear it, with or without a
running server.

**Why this priority**: Stored state is prompt-equivalent data; users must control its footprint
and lifetime.

**Independent Test**: Fill the tier, run `mrun cache`, clear one model and everything; verify
sizes and files.

**Acceptance Scenarios**:

1. **Given** persistence on, **When** the developer runs `mrun cache`, **Then** it shows location,
   cap, used bytes, entry count and per-model usage (alias or identity prefix), plus memory-tier
   stats when a server runs.
2. **Given** stored entries, **When** the developer runs `mrun cache clear --disk`, **Then** every
   file is removed (through the server when it runs, directly otherwise) and `mrun cache` shows 0.
3. **Given** the cap is reached, **When** new entries are written, **Then** least recently used
   entries are deleted first and usage never exceeds the cap.

### Edge Cases

- Disk full or below the free-space floor: writes stop, a warning appears in `/mrun/status` and the
  server log, no request fails.
- Slow or external volume (`MRUN_CACHE_DIR` on a USB disk): measured read speed feeds the
  restore-or-recompute decision; a slow disk simply loses to recompute.
- A trim or growth on the live path while it is being written: the writer stops at its next block
  (≤ 4 MiB); the partial file is discarded.
- An entry whose parent chain is incomplete (parent evicted): its tokens are still verified; it is
  usable only with an available parent chain, otherwise it is deleted by the next sweep.
- The tier directory is deleted while the server runs: the server recreates it, drops its index
  entries and continues.
- Wrong permissions on an existing directory (group/world readable): the server refuses to use it,
  persistence is off, and `doctor` names the fix.
- Clock changes: LRU uses the server's monotonic order plus file mtimes; a backward clock jump can
  only make an entry look older.
- `mrun cache clear` (no flag) clears memory only, as in 003; `--disk` or `--all` are explicit.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001 (measure first)**: Before this feature ships, mrun MUST be able to report, from a local
  `--record` log of real use, the share of prompt tokens re-prefilled because of server restarts
  and because of memory eviction, and the share a disk tier at the default cap would have served
  (`mrun bench --suite persist --from <record-dir>`), and the target Mac's sequential read and
  write speed for the tier directory and a 30K-token restore time (`mrun bench --suite persist`).
- **FR-002**: Persistence MUST be off unless enabled by `--persist`, `[cache] persist = true` or
  `MRUN_CACHE_PERSIST=1`; `--no-persist` overrides config.
- **FR-003**: Entries MUST hold the exact bytes of the in-memory sequence state (attention state
  ranges and hybrid checkpoints) with their token ids; no conversion, quantization or compression
  of state bytes.
- **FR-004**: Every entry MUST record the cache key (weights, tokenizer, template, state format),
  the on-disk format version and the MLX version; an entry is eligible only when all match the
  running server exactly.
- **FR-005**: Before any state byte of an entry is used, the stored token ids for the whole prefix
  `[0, end)` MUST be compared with the request's tokens; hashes are used only to find candidates.
- **FR-006**: Writes MUST be atomic (temporary file, flush to stable storage, rename, directory
  sync) and every block of state MUST carry a checksum verified on load; a mismatch deletes the
  entry and reports `disk-corrupt`. Temporary files are deleted at startup.
- **FR-007**: Writes MUST happen on a background writer: branches when they are moved out of the
  live path, and the live path's committed tokens after 2 s of idleness. The engine thread waits at
  most for one in-flight block (≤ 4 MiB) before mutating state the writer reads; eviction never
  waits for a write.
- **FR-008**: Lookup MUST consider memory and disk; when the disk match is longer, mrun restores
  from disk only if the estimated read time (bytes / measured read speed) is lower than the
  estimated time to recompute the extra tokens; restored state is admitted against the budget like
  any restore (003).
- **FR-009**: When a request is accepted into the queue, mrun MUST issue read-ahead for the disk
  entries that best match it, so reads at the head of the queue come from the OS file cache.
- **FR-010**: The tier MUST live in `~/Library/Caches/mrun/prefix` (`$XDG_CACHE_HOME/mrun/prefix`
  on Linux; `MRUN_CACHE_DIR` overrides) with directories 0700 and files 0600, marked to be skipped by
  backup and Spotlight indexing; an existing directory with broader permissions MUST be refused.
- **FR-011**: Total size MUST stay at or below the cap (default 32 GiB, `[cache] persist_gib`) and
  the volume's free space at or above 10%; least recently used entries are deleted first; entries
  unused for `persist_max_age_days` (default 30) are deleted at startup and hourly.
- **FR-012**: `mrun cache` MUST show location, cap, used bytes, entries, and per-model usage;
  `mrun cache clear --disk [--model M]` and `mrun cache clear --all` MUST remove entries through the
  running server or, without one, directly (holding the tier lock).
- **FR-013**: Miss reports (003) MUST distinguish `source: memory | disk` for reused tokens and add
  causes `disk-corrupt` and `disk-slower` (a disk match was skipped because recompute was faster).
- **FR-014**: Only one server process MAY use a tier directory (exclusive lock); others run without
  persistence and report it.
- **FR-015**: Persisted entries MUST NOT include request text, headers, session ids or any data
  beyond token ids, state bytes and the fields in the data model; nothing is written when
  persistence is off.

### Key Entities

- **Tier Entry**: One file: cache key, format and MLX versions, parent entry, token ids `[0, end)`,
  state range `[start, end)`, optional checkpoint at `end`, per-block checksums, size, last use.
- **Tier Index**: In-memory view built at startup from entry headers: candidate lookup by token
  hash, parent chains, LRU order, sizes.
- **Writer Job**: Entry being written from an immutable source (branch segment or live-path range),
  cancellable per block.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: On the target Mac (M4 Pro, 48 GB, internal SSD), after a server restart, restoring a
  30K-token conversation prefix from the tier is at least 20× faster than re-prefilling it (time
  from request arrival to first token minus the same request's decode time), for Qwen3.6-35B-A3B
  and Qwen3-Coder-30B-A3B 4-bit.
- **SC-002**: In a mutation suite of at least 100 cases spanning restarts (model, tokenizer,
  template, state-format and mrun-version changes; edited, truncated, bit-flipped, half-written and
  foreign files), zero stale or corrupt state is ever loaded.
- **SC-003**: With persistence on, median warm-turn TTFT (memory hits) on the bench agent suite is
  within 1% of persistence off, and no request fails because of the tier.
- **SC-004**: Under a write load exceeding the cap by 3×, tier size never exceeds the cap, measured
  every second.
- **SC-005**: After `kill -9` at 50 random points during writes, every restart yields only complete,
  verified entries (zero partial files used).

## Assumptions

- Builds on 003 (live path, branches, checkpoints, cache key, miss reports) and 004 (identity,
  config, CLI); FileVault is the at-rest protection.
- The tier is a cache, not a conversation store: entries may vanish at any time; correctness never
  depends on them.
- Sharing entries across machines or users is out of scope.
