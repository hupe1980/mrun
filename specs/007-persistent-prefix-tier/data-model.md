# Data Model: Persistent Prefix Tier

## TierConfig

| Key | Default | Source |
| --- | --- | --- |
| `[cache] persist` | false | `--persist` / `--no-persist`, `MRUN_CACHE_PERSIST` |
| `[cache] persist_gib` | 32 | `MRUN_CACHE_PERSIST_GIB` |
| `[cache] persist_max_age_days` | 30 | `MRUN_CACHE_PERSIST_MAX_AGE_DAYS` |
| directory | `~/Library/Caches/mrun/prefix` | `MRUN_CACHE_DIR` (reserved name, not a config key) |
| free-space floor | 10% of the volume | fixed |

## EntryKey (binding)

`{weights_identity (001 R10, incl. MTP sidecar component), tokenizer_hash, template_hash,
state_format {attention dtype, layer layout hash, checkpoint format version}, format_version: 1,
mlx_version}`. Directory: `prefix/<hex(sha256(EntryKey))[0..16]>/`; the full key is in every
header and compared field by field.

## TierEntry (file `<entry_id>.mrps`, entry_id = 128-bit random hex)

| Field | Type | Rules |
| --- | --- | --- |
| magic, format_version | "MRPS", u32 = 1 | unknown version → ignored and deleted by the sweep |
| key | EntryKey | must equal the server's exactly |
| parent | Option<{entry_id, end}> | parent's tokens `[0, end)` equal this entry's `[0, start)` (checked at write and load) |
| start, end | u32 | state range `[start, end)`; `start == parent.end` or 0 |
| tokens | u32 × end | full prefix token ids |
| boundaries | list of {pos, kind} | message boundaries inside `(start, end]` (lookup hashes) |
| checkpoint | bool | recurrent state at `end` included (hybrids) |
| tensors | list of {layer, kind `k`\|`v`\|`conv`\|`delta`, dtype, shape, offset, len} | 16 KiB aligned |
| checksums | xxh3-64 per 4 MiB block + header hash | verified on load |
| size | u64 | file length |
| last_use | file mtime | touched on reuse |

Invariants:
- A published file (no `.tmp` suffix) is complete and its checksums matched at write time.
- Tokens are never trusted without comparison against the request (FR-005).
- Deleting an entry deletes its descendants first (LRU removes leaves).

States: `writing (.tmp) → published → (touched)* → deleted`; `corrupt` (checksum or parse failure) →
deleted.

## TierIndex (in memory)

`{by_hash: HashMap<u64 (hash of tokens [0,pos)), Vec<(entry_id, pos)>>, entries: {id → {parent,
start, end, checkpoint, size, mtime}}, children: id → set, lru: ordered by mtime, used_bytes,
read_gbps (EWMA, initialized by a 64 MiB probe read at open)}`.

## WriterJob

`{entry_id, source: Branch(segment ids) | LivePath(range), views: [(ptr, len)] valid for
generation g, state: queued | writing(block n) | published | abandoned}`. `quiesce(range)` from the
engine thread abandons any job whose views overlap `range` after its current block.

## Lookup result (extends 003 Lookup)

`{memory: Lookup (003), disk: Option<{entry chain, matched, checkpoint_at, bytes,
est_read_s, est_recompute_s}>, choice: memory | disk}`.

## MissReport (extends 003)

`source: memory | disk | none` for the reused tokens; causes add `disk-corrupt`, `disk-slower`.

## TierStats (`/mrun/status`, `mrun cache`, metrics in 008)

`{enabled, dir, cap_bytes, used_bytes, entries, per_key: [{key16, model alias?, bytes, entries}],
writes_total, write_bytes_total, abandoned_total, disk_hits_total, disk_hit_tokens_total,
corrupt_total, read_gbps, last_error?}`.
