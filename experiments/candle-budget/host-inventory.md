# Host Allocation/Work Inventory (T115)

Status: **source review only**. No tests, builds of mrun code, or allocation measurements were run.
Every byte figure below is derived from reading source, not measured; T118 must reproduce or refute it.
The two ceilings remain **targets, not established capabilities**:

- Render/tokenize: **"64 MiB per active request; reject overflow"**.
- Grammar/parser: **"16 MiB per active request; reject unsupported/excess complexity"**.

Limit kinds used below: **fuel** (work/steps), **state** (count of items/states/nodes), **output**
(bytes emitted to a sink), **bytes** (a cap checked against intended allocation size *before* the
allocation). Only "bytes" (or a pre-entry admission whose worst-case footprint is proven) can
satisfy "reject before exceeding" on its own.

## Pinned source identities

Fetched with `cargo fetch` (cargo 1.97.0) into a scratch probe package; checksums are crates.io
`.crate` SHA-256 values from the probe `Cargo.lock`. VCS sha1 is from each tarball's
`.cargo_vcs_info.json` and matches the research C-05 revisions where C-05 names one.

| Crate | Version | Cargo.lock checksum (sha256) | VCS sha1 (path) | C-05 match |
|---|---|---|---|---|
| tokenizers | 0.23.1 | `44e5bea67576e04b6ff8564c5d9e09c2ef0cf476502245f2f120e497769d3112` | `7f1623b90b5adfb9bc327d4c3468d2f70bbce262` (tokenizers) | yes |
| minijinja | 2.24.0 | `86886cf6dbf4e614b19c9a1eec9775f021869d7eadde0fc73921a81b90c9b4c9` | `0ca749f7ba507514fa6b052c74130ae6ae472e03` (minijinja) | yes |
| minijinja-contrib | 2.24.0 | `bd3e5f077bc2379f0f7d911e7cfdd921114ed99fc884533dca502944cb355b11` | `0ca749f7…` (minijinja-contrib) | yes |
| llguidance | 1.9.1 | `04ede1356c8ee73a0d29ef573dbeb2123a1667cfc3ebfc7e2c41346a6e92c3c8` | `f0971424ec072d3e4d4196bcc7f31a2f60527df9` (parser) | yes |
| toktrie | 1.9.1 | `1bd80f04d68b86f1a4bf03bd4bed87b31341e74c48035697cded09d797cb1c05` | `f0971424…` (toktrie) | yes |
| derivre (llguidance dep, `=0.3.13`) | 0.3.13 | `2a4fbc4bbc70b71fbf8f4d9270006e8b238b91c6179f8aad06b6d25589494347` | not checked | n/a, **unaudited** |
| jsonschema | 0.33.0 | `d46662859bc5f60a145b75f4632fbadc84e829e45df6c5de74cfc8e05acb96b5` | `c6ee21efc29083422e466aceb2a15bf1e50c83b3` | n/a (same checksum as root `Cargo.lock`) |
| serde_json | 1.0.154 | `e7e9cc8b1b85264074fbcc02a88680c4096b1e47df8f739dceb03bf482f04bd6` | `20572a6d066958542d79cdd6cf1a7cd9f0c8e1f8` | n/a (same checksum as root `Cargo.lock`) |
| onig / onig_sys (tokenizers `onig` feature) | 6.5.3 / 69.9.3 | see probe lock | not checked | **unaudited C code** |
| Candle `gguf_file.rs` | git `c68b24997319b8d76e5a4e775dab197c7f26322a` | n/a (git source) | same | T112 candidate |

The probe lock resolved on cargo 1.97.0, not the Rust 1.94.0 pin. It is not the frozen
experiment lock (T113). Paths below are relative to each crate root (`<crate>-<ver>/…`).

## 1. GGUF metadata/loading (Candle `candle-core/src/quantized/gguf_file.rs`)

Allocation sites:
- `read_string` `vec![0u8; len]` (:143). It runs after `len <= 1<<30` (:136) and `len <= remaining file bytes` (:140).
- Array `Value::Array` built with `Vec::new()` + `push` per element (:360-370). It runs after `len <= 1<<30` (:350) and `len*min_disk_size <= remaining` (:353-359). Depth is capped at 64 (:19, :327).
- Header counts are checked against the file size (:497-506) before the `HashMap` metadata/tensor-info loops (:507-547). Dims are capped at 4 (:518).
- `TensorInfo::read` `vec![0u8; size_in_bytes]` (:110), checked against the remaining file size (:105).
- `quantized/tokenizer.rs` rebuilds vocab/merges (`value_to_string_array` :45-55, `merges_from_value` :58-66) by cloning metadata into new `Vec<String>`s.

Available limits: **state** (count caps, depth 64) and **file-size-relative counts**. There is no
in-memory byte cap.

Analysis: every count is checked against the *file* size, so allocation is bounded by
`file_size × amplification`, not by a byte budget. `Value` is an enum containing `String`/`Vec`
(:186-200), presumably 32 bytes on 64-bit (**not measured**). A `U8`/`Bool` array element takes 1 disk
byte but becomes a 32-byte `Value`, and `Vec` doubling growth adds up to about 2x more, so the
amplification is about 32-64x.

**Decisive counterexample (for any "Candle checks bound bytes" claim):** a GGUF v3 header with one
metadata KV of type `Array<U8>`, `len = 1_000_000_000`, followed by 1 GB of payload, passes :350 and
:353-359. It then grows a `Vec<Value>` to about 32 GB (about 64 GB transient while reallocating)
before any check trips. That is far beyond 64 MiB.

Enforceable path: GGUF is a provisioned model artifact, not request input. It belongs to
model-load admission, not the per-request ceilings. mrun can (a) admit only the pinned model file
by SHA-256, and (b) pre-scan the GGUF header with its own non-allocating reader that computes the
worst-case in-memory metadata footprint from counts/lengths *before* calling `Content::read`. Both
are pre-entry checks. Verdict: **enforceable path (pre-entry, mrun-owned); none in Candle**. The
C-05 preference for a separately provisioned `tokenizer.json` avoids the per-request overlap. The
GGUF tokenizer reconstruction is load-time only.

## 2. tokenizers 0.23.1 (render/tokenize target: 64 MiB)

Allocation sites (per `encode` call):
- `NormalizedString` holds `original`, `normalized` and `alignments: Vec<(usize,usize)>`, one entry per normalized byte, so about 16 B/byte (`src/tokenizer/normalizer.rs:105-117`).
- `AddedVocabulary::extract_and_normalize` (`src/tokenizer/added_vocabulary.rs:523`), then pre-tokenize/`do_tokenize` (`src/tokenizer/mod.rs:734-748`, :1144).
- The `Encoding` holds 8 parallel `Vec`s (ids, type_ids, `tokens: Vec<String>`, words, offsets, masks, overflowing, `sequence_ranges` map) (`src/tokenizer/encoding.rs:11-30`).
- The BPE `thread_local!` cache `BPE_LOCAL_CACHE` (`src/models/bpe/model.rs:82-89`) inserts words `< MAX_LENGTH=256` bytes up to `DEFAULT_CACHE_CAPACITY=10_000` entries per thread (:582, `src/utils/cache.rs:7,10`). That residency is **cross-request, per thread**, not per request.
- Regex pre-tokenization with the `onig` feature (`src/utils/mod.rs:9-10`) calls Oniguruma, which is **C code allocating with C `malloc`**. `fancy-regex` is only selected under `unstable_wasm` (`Cargo.toml` features :54-67). Oniguruma memory is **unaudited**.
- Loading `tokenizer.json` (`Tokenizer::from_file`, `src/tokenizer/mod.rs:438`) materializes the vocab/merges maps. That is load-time and trusted-file only.

Available limits:
- `TruncationParams.max_length` (`src/utils/truncation.rs:23-29`) is a **state/output** limit applied in `post_process` (`src/tokenizer/mod.rs:1238-1251`) **after** normalization, pre-tokenization and model tokenization are complete. It bounds nothing that came before it.
- `MAX_LENGTH`/capacity on the BPE cache is a **state** limit.
- There is no fuel and no byte limit.

Analysis: work and allocation are roughly linear in input bytes for a fixed tokenizer, but the
linearity constant is **unmeasured**, and the normalizer and onig paths are **unaudited**.

Enforceable path: a **pre-entry input-byte admission** at the narrow bridge (C-05: "bounded byte
input"). mrun rejects a render output larger than `B_in` before calling `encode`, with `B_in`
chosen so that `B_in × measured_peak_amplification + per-thread cache residency ≤ 64 MiB`. This
needs T118 to measure the amplification on adversarial inputs: max-expansion NFC/byte-level
input, no-merge inputs, inputs that produce many added-vocabulary matches, and inputs that drive
onig backtracking.

Verdict: **partial**. Rejection before exceeding is achievable only through a pre-entry bound plus
a reviewed, measured amplification. The library's truncation is not such a bound. C-allocated onig
memory is invisible to a Rust global allocator.

## 3. MiniJinja / minijinja-contrib 2.24.0 (render target: 64 MiB)

Allocation sites:
- Template compilation (`Environment::add_template` `src/environment.rs:180`, `template_from_str` :439) runs lexer/parser/codegen. The only constants are codegen pool caps (`src/compiler/codegen.rs:65-66`) and `MAX_LOCALS=50` (`src/compiler/instructions.rs:22`). There is no source-size or AST byte cap.
- Context conversion: `render(ctx)` serializes the whole `ctx` into `Value`s (`src/template.rs:188`). The cost is proportional to request messages/tools.
- `string_concat` `format!("{left}{right}")` (`src/value/ops.rs:463-465`). String repeat is capped at `MAX_REPEATED_STRING_LEN = 100_000_000` bytes (:6, :334). Sequence repeat/merge goes through `repeat_iterable` (:360) and `MergeSeq` (:295-300).
- `range` is capped at 100,000 items (`src/functions.rs:330`).
- Filters `tojson(indent=N)` `" ".repeat(N)` (`src/filters.rs:1152`), `indent(width)` `" ".repeat(width)` (:1234), `replace` (:270), `join` (:465), `format` (:1841) and `pprint` (:1817) are uncapped.
- Contrib `truncate` uses `with_capacity` (`minijinja-contrib-2.24.0/src/filters/mod.rs:251`). `wordwrap` is at :300. The pycompat string/map methods (`src/pycompat.rs:74-333`, for example `replace` :136, `split` :151, `join` :267, `format` :201) are uncapped.
- Captures: `BeginCapture` (`src/vm/mod.rs:610`, `src/output.rs:51`) buffer into memory regardless of the final writer.

Available limits:
- `set_fuel` (`src/environment.rs:613`; the feature is in the default set, `Cargo.toml:44,69`) is **fuel**. `FuelTracker::track` (`src/vm/fuel.rs:29-38`) is checked once per instruction *before* execution (`src/vm/mod.rs:316-320`). Some instructions cost 0 (`src/vm/fuel.rs:52-60`, including `BeginCapture`).
- `set_recursion_limit` (`src/environment.rs:660`, max `MAX_RECURSION=500` :53) is a **state** limit.
- `render_to_write` (`src/template.rs:340`) with a bounded writer is an **output** limit on emitted bytes only.
- The repeat/range caps are per-operation **bytes/state** limits, but they are larger than the target (100 MB > 64 MiB).
- I found no output- or allocation-byte limit in the engine (`grep` for size/output limits finds none; not proven exhaustive).

**Decisive counterexample (fuel/recursion/output ≠ byte bound):** `{% set s = "x" * 100000000 %}`
is a handful of instructions (far below fuel 1,000,000) and allocates about 95 MiB in one
`s.repeat(n)` (`ops.rs:334-340`) with zero output. `{% set s = "x" * 50000000 %}{% set s = s ~ s %}`
reaches the same result through the uncapped `string_concat`. Fuel, recursion 32 and a bounded
writer all pass. C-05 already forbids request-provided templates, so this defeats the engine-level
claim, not the pinned-template policy.

Enforceable path for the pinned template only:
1. Compile the single provisioned template at startup, outside request accounting.
2. Review the template to show that every allocation is linear in context bytes with constant
   template arguments (no `*`, no data-derived `indent`/`tojson` widths, no unbounded concat loops).
3. Admit a request only when its serialized context bytes are at most `B_ctx`, sized from that
   review.
4. Render via `render_to_write` into a writer capped at the 64 MiB share, with fuel 1,000,000 and
   recursion 32 as work/state backstops.

Verdict: **partial**. A pre-entry path exists only with a reviewed template amplification bound,
and the engine itself gives no byte rejection.

## 4. LLGuidance / toktrie 1.9.1 (grammar/parser target: 16 MiB)

Trie residency (toktrie, per model, shared):
- `TokTrie::from` (`toktrie-1.9.1/src/toktree.rs:196-240`) allocates `token_offsets` with capacity `vocab_size` (:198), `token_data` with capacity Σ token bytes (:203), an index sort vector (:211), and `TrieBuilder` nodes with capacity `3 × vocab_size` (:1276-1277), then the serialized `nodes` (:233).
- The vocab is the pinned tokenizer's, so this is a load-time, model-scoped charge, sized once, not per-request.
- Per-request masks come from `alloc_token_set` (:362-363), about `vocab/8` bytes per `SimpleVob`. Token-to-bytes helpers use `with_capacity(tokens.len()*6+32)` (:524, :550).
- The `TokenizerEnv::tokenize_bytes` trait (`src/tokenv.rs:13-25`) is called by the grammar compiler. mrun's narrow bridge owns it and can bound its input.

Grammar construction (request-provided tool schemas):
- The JSON schema compiler counts compiled nodes, `n_compiled > max_size` (default 50,000), at `llguidance-1.9.1/src/json/shared_context.rs:141` (**state**). Stack level is limited to 128 (`src/json/schema.rs:272`, :546-547).
- `GrammarBuilder::check_limits` (`src/grammar_builder.rs:187-201`) compares regex spec cost to `initial_lexer_fuel` (**fuel**) and `num_symbols` to `max_grammar_size` (**state**). It is called *after* building steps (:224, :294, :376, :403, :432; `src/json/compiler.rs:259`, :709; `src/lark/compiler.rs:132`, :247, :427). Growth up to that point is therefore unchecked between calls. The overshoot per step is **unaudited**.
- Lexer setup: `initial_lexer_fuel` (`src/earley/parser.rs:634-651`, `src/earley/regexvec.rs:622-639`, `src/earley/grammar.rs:1533`) is **fuel**. `precompute_large_lexemes` should be off per C-05.
- The derivre `ExprSet` hash-cons growth is bounded only through fuel (derivre 0.3.13 **unaudited**).

Matcher growth (per token):
- Per mask step, `step_lexer_fuel` (**fuel**) and `max_lexer_states` (**state**; documented as about 1-2 kB/state, `src/api.rs:288-292`) are set at `src/earley/parser.rs:813-814`. `step_max_items` (**state**) applies at :816 and :1424, and `max_items_in_row` (**state**) at :1388.
- The history is never popped in definitive mode (comment at `src/earley/parser.rs:361-371`). `rows.push` (:2213) and `row_infos.push` (:2238) run per lexeme. `items: Vec<Item>` holds 8-byte `Item`s (:78-80, :245), and `ensure_items` reserves (:496-503). Captures accumulate (:1961, :1976).
- Grammar state therefore grows with the number of consumed lexemes. It is bounded by **state × output length**, not by bytes.
- With C-05 row items 512 × 8 B ≈ 4 KiB/row and the 256 KiB tool-call output bound with 1-byte lexemes, the item history alone could reach about 1 GiB in the worst case. That is a **computed upper bound, not a demonstrated input**.
- `Matcher::last_step_stats` (`src/matcher.rs:305`) and `ParserStats { rows, all_items, num_lexemes, … }` (`src/earley/parser.rs:88-101`) expose counters that mrun can poll between `consume_token` calls. Whether the counters are cumulative or per-step is **unaudited**.
- Errors from panics are caught (`src/matcher.rs:92`), but allocation failure is not a panic (see §6).

Verdict: **partial**. Construction is bounded by state/fuel limits that are checked *after*
building segments. Matcher growth is bounded by state × lexeme count. A conservative pre-entry
bound requires (a) a proven bytes-per-compiled-node/symbol/lexer-state factor, so that 50,000-node
/ 16,384-symbol / 1,024-state limits imply ≤ 16 MiB including the post-hoc overshoot, and (b) an
mrun-side cap on grammar-constrained lexemes, enforced between tokens, of about
`(16 MiB − construction) / (row_items × 8 B + row overhead)`. That cap is far below 256 KiB of
1-byte lexemes, so the 256 KiB output bound must be either constrained further or justified by
measured item counts. A decisive counterexample is **not** established. T118 must try the
optional/additional-property schema cases.

## 5. jsonschema 0.33 / serde_json 1.0.154 (grammar/parser target: 16 MiB)

- serde_json: the `Deserializer` has a depth limit of 128 (`serde_json-1.0.154/src/de.rs:63`, :1375-1376) (**state**; `disable_recursion_limit` :215 under `unbounded_depth`). String scratch buffering is proportional to input string length. `Value` maps and arrays grow with input.
- mrun's existing direction in `src/backend/qualification.rs:334-420` shows the required pattern. It rejects on `bytes.len() > MAX_BYTES` before parsing (**pre-entry bytes**), and a `DeserializeSeed` visitor enforces `MAX_DEPTH`, `MAX_NODES` and `MAX_STRING` before each node/string is materialized (**state**, checked before collection growth). This yields a provable bound of about `MAX_BYTES + MAX_NODES × node_size` plus scratch.
- jsonschema: no `jsonschema` call site exists in `src/` today. Only the root `Cargo.toml:20` dependency (default features off: no http/file resolvers, `Cargo.toml` features :34-46) exists.
- `pattern` compilation honors optional `size_limit`/`dfa_size_limit`. These are regex-crate compiled-program **byte** limits checked during compile (`src/regex.rs:49-80`, `src/keywords/pattern.rs:71-89`, `src/keywords/additional_properties.rs:1181-1183`). They are `None` by default (`src/options.rs:710-712`), and `backtrack_limit` is **fuel** (fancy-regex).
- The schema compile/`$ref` registry, keyword node allocation and validation error collection have no limits (**unaudited**).

Verdict: **enforceable path for argument JSON** (the existing bounded visitor plus pre-entry
byte cap). **Partial for schema compilation/validation**: bound the schema by the same visitor,
set regex `size_limit`/`dfa_size_limit`, and use `is_valid`/first-error rather than collecting all
errors. The compiled-validator footprint per schema node is unaudited.

## 6. Process-wide counting/limiting global allocator with per-request scopes

- **Denial cannot be a returned error.** A `GlobalAlloc` returning null makes the infallible
  `Vec`/`String`/`Box`/`HashMap` paths that these libraries use call `handle_alloc_error`. On
  stable Rust that aborts the process, and the `oom=panic` handling is unstable. The allocator also
  must not unwind (unwinding out of `GlobalAlloc` is undefined behavior). Only `try_reserve`-style
  call sites see a recoverable `Err`. None of the hot paths cited above use them (for example
  `vec![0u8; len]` gguf :143, `with_capacity` toktrie :198/:203, `ensure_items` `reserve`
  parser.rs:498).
- **What it can do:** precisely attribute Rust-heap bytes to a thread-local request scope, raise an
  "over budget" flag, and let mrun poll the flag at cooperative points: between `consume_token`
  calls, after `encode`, and inside a capped render writer. That is post-allocation detection with
  unbounded single-allocation overshoot. The 95 MiB MiniJinja repeat above happens in one call. It
  is accounting, not "reject before exceeding".
- **Blind spots:** onig_sys C `malloc` (tokenizers), any rayon or worker-thread hop that breaks
  thread-local scope attribution (`encode_batch`/parallelism, **unaudited**), cross-request
  thread-local caches (BPE cache) charged to whichever request happened to warm them, and
  shared/global LLGuidance factory state (**unaudited**).
- **Viable role:** a measurement instrument for T118 (peak bytes per adversarial case, to fit
  amplification constants), plus a defense-in-depth tripwire that fails the request after the
  fact. The hard "reject before" guarantee must come from **pre-entry admission** sized by a proven
  bound. A hard cap independent of library cooperation would require process isolation (a worker
  process with an RLIMIT/cgroup memory limit, where overflow kills the worker and the request is
  rejected). That is a different architecture, not evaluated here.

## Summary: targets vs verdict

| Target / path | Library limits present (kind) | Pre-allocation byte cap in library? | Verdict |
|---|---|---|---|
| GGUF metadata/tokenizer reconstruction (load-time) | count caps vs file size, depth 64 (state) | no; ~32-64x file-byte amplification | **Counterexample** to library bound (`Array<U8>` 1e9); **enforceable path** via pinned SHA-256 + mrun header pre-scan |
| Tokenize 64 MiB | truncation (post-tokenize state/output), BPE cache caps (state) | no; onig C memory uncounted | **Partial**: pre-entry input-byte cap × measured amplification (T118) |
| Template compile 64 MiB | none beyond MAX_LOCALS | no | **Enforceable path** only for the single provisioned template compiled at startup |
| Template render/filters 64 MiB | fuel, recursion, repeat 100 MB / range 100k caps, writer (output) | no (repeat cap > target) | **Counterexample** for engine-level bound (`"x"*100000000`); **partial** for reviewed pinned template + context-byte admission |
| Trie residency | none (sized by vocab) | n/a, load-time | **Enforceable path**: model-scoped one-time charge, excluded from per-request share |
| Grammar construction 16 MiB | max_size 50k, grammar symbols, initial fuel, stack 128 (state/fuel, post-hoc) | no | **Partial**: needs bytes-per-unit proof and overshoot bound |
| Matcher growth 16 MiB | step fuel, step items, row items, lexer states (fuel/state) | no; history grows per lexeme | **Partial**: mrun lexeme cap between tokens; 256 KiB × 512-item rows can exceed 16 MiB in the computed worst case |
| Tool-argument JSON parse 16 MiB | serde depth 128; mrun visitor depth/nodes/string + MAX_BYTES | yes (mrun pre-entry + visitor) | **Enforceable path** (existing pattern) |
| jsonschema compile/validate 16 MiB | regex size_limit/dfa_size_limit (bytes, off by default), backtrack (fuel) | partial (regex only) | **Partial**; validator footprint unaudited |
| Global counting allocator | n/a | cannot deny without abort | **Not sufficient alone**: accounting/tripwire only; hard cap needs pre-entry admission or process isolation |

No target is an established capability. T118 must measure the amplification constants and attempt
the counterexamples before any verdict is upgraded.
