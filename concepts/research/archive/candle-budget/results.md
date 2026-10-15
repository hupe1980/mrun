# Feasibility Results: Enforceable Allocation Bounds for Candle

> **Historical**: Candle was dropped on 2026-10-14 (D010). Kept as evidence for that decision and for D012.

**Date**: 2026-10-14 · **Tasks**: T112..T120 (results), T124 (handoff) · **Disposition**: **BLOCKED**
(no gate pass). The experiment produced enforcement mechanisms and decisive counterexamples, not a
complete bound. Planning gate *Principle III / FR-014 enforceable allocation bounds* stays closed.

## Identities

| Item | Value |
| --- | --- |
| Toolchain | Rust 1.94.0 (`rustc 1.94.0 (4a4ef493e 2026-03-02)`) |
| Candle | `c68b24997319b8d76e5a4e775dab197c7f26322a` (0.11.0): `candle-core`, `candle-metal-kernels` |
| Patch | `patches/0001-metal-allocation-gate.patch` (2 files, +38/-1; applies cleanly at the pin) |
| Host libraries | tokenizers 0.23.1 (`fancy-regex`), minijinja 2.24.0, llguidance/toktrie 1.9.1 (pins: [host-inventory](host-inventory.md)) |
| Lockfile | `Cargo.lock` in this directory (git sources pinned to the Candle revision) |
| Host | Linux x86_64 container; **no macOS, no Metal device, no M4 Pro** |

## Commands and outcomes

| Command | Outcome |
| --- | --- |
| `cargo build --locked` | Pass (CPU) |
| `cargo check --locked --tests --features metal --target aarch64-apple-darwin` | Pass (type-check only; not linked) |
| `./check-patched.sh` (patched Candle, `metal-gate`, macOS target) | Pass (type-check only) |
| `cargo test --locked` | 20/20 pass; 15 consecutive clean runs |
| `cargo clippy --locked --all-targets -- -D warnings` | Pass |
| `cargo test --features metal-gate` on macOS | **Blocked**: no hardware |

Tests-first evidence (T116): all seven admission probes failed behaviorally against the stub
ledger (`Unimplemented` denials, wrong totals), before `src/budget.rs` was implemented.

## Probe results

### Allocation contract (T116/T117, real Candle CPU constructors)

| Probe | Result |
| --- | --- |
| Lawful reservation covers `Tensor::zeros` (32×32 f32) | Pass: constructor allocations ≤ reserved (output + 1 KiB metadata) |
| Insufficient reservation | Denied before constructor entry; constructor never ran; denying thread made 0 allocations |
| Checked size/rounding | `tensor_bytes` and `pool_bucket` reject overflow that Candle leaves unchecked (`device.rs:203`, `:327`) |
| Aliased views vs independent staging | Views share one charge; staging is charged separately; release-once verified |
| Constructor failure | Charge rolled back exactly once |
| Simple CPU constructors (zeros, from_vec, to_dtype, contiguous, cat) | Peak within output + 1 KiB |
| **Counterexample**: CPU matmul 256×256 f32 | Peak ≈ 6.6 MB for a 256 KiB output (~25×): gemm packing scratch has no Candle hook |

### Metal path (T114, T117, T119)

The 22 inventoried paths are in [gpu-inventory](gpu-inventory.md). Patch 0001 places a fallible
gate inside the two raw constructors of `candle-metal-kernels` (`Device::new_buffer`,
`Device::new_buffer_with_data`), so **admission** sees every buffer Candle itself creates, after
power-of-two rounding: G-01..G-03, G-05..G-14 (including the seed G-07 and sort scratch G-13,
which bypass `MetalDevice`). **Release** is reported only where the pool frees after a completed
wait (`drop_unused_buffers`), which is the safe release event for pooled buffers.

| Finding | Consequence |
| --- | --- |
| Pools keep capacity after tensors drop; trimmed only after a completed wait | Charges must cover pool capacity, modeled by `Pool` and verified in portable probes |
| `Buffer` is `Clone` and Metal retains buffers referenced by command buffers | A Rust `Drop` is not a safe release point; release is tied to observed completion |
| Unpooled G-06, G-07, G-13 are admitted but never reported released | Fail-safe overcharge that leaks budget until teardown; mrun must forbid G-06/G-13 and bound G-07 (8 B, once) |
| G-20 foreign/no-copy imports, public `MetalStorage::new` | Invisible to any gate; must be forbidden by mrun code policy |
| G-15..G-19 command buffers, fences, residency set, pipeline cache, `setBytes`, device setup | **Unaudited**: no byte bound established |
| Panics instead of errors (lock unwraps, `Fence::new` expect, unchecked math) | Allocation failure is not uniformly fallible |

Portable lifetime controls (6/6 pass) cover pooled reuse, larger-bucket reuse, denied growth,
submission retention until observed completion, quarantine on uncertain completion blocking
readiness, refusal to release while leases live, and the size-only gate ledger. They model the
patched behavior; **they are not Metal evidence**. The real-Metal probe compiles for macOS but has
not run.

### Host libraries (T115, T118, measured on Linux debug build)

| Target | Measurement | Verdict |
| --- | --- | --- |
| Render ≤ 64 MiB | Lawful ~1 MiB ChatML render: peak 4.6 MB (4.3× output) | Enforceable only for a reviewed template, via context-byte cap before render |
| Render ≤ 64 MiB | `{{ 'x' * 100000000 }}` with fuel 100 and 64 KiB capped sink: **peak 200 MB** | **Counterexample**: fuel and output caps do not bound intermediates |
| Tokenize ≤ 64 MiB | Byte-level worst case: 260× amplification at 64 KiB; 256 KiB input → peak 42 MB | Enforceable by input-byte cap ≈ 256 KiB per encode call (or chunked encode); production Oniguruma allocations would be invisible to Rust accounting |
| Grammar ≤ 16 MiB | Lawful 16-property tool schema: peak 187 KB | Within target |
| Grammar ≤ 16 MiB | 20,000-property schema rejected by `max_grammar_size`, but only after **peak 26 MB** | **Counterexample**: symbol limits trip after bytes exceed target; mrun needs a pre-construction schema size bound |
| Parser ≤ 16 MiB | ~47 B per generated byte inside a JSON string; projected 12.3 MB at 256 KiB payload | Within target for this shape; not a proof for all schemas |
| GGUF load | Array length checked only against file size (source review) | Enforceable only by admitting the pinned SHA-256 artifact plus an mrun header pre-scan |
| Global counting allocator as a cap | Null return aborts; C allocations invisible | Accounting/verification only, never enforcement |

## Gate disposition

Feasibility requires every reachable allocation and host-library path to have a reviewed bound and
enforced denial. Not met: G-15..G-19 unaudited, G-20 unenforceable, CPU op scratch unbounded,
template and grammar counterexamples, and no Metal run. **Planning remains blocked; T122/T123 are
not started.** Nothing here measured model fit, 16K context, cancellation latency, numerical
quality or any SC gate.

## Defensible path to closing the gate

1. **Metal**: adopt patch 0001 (or upstream an equivalent hook), forbid G-06/G-13/G-20 by code
   policy, and bound G-15..G-19 by construction: precompile the finite kernel set at load and
   measure it, cap in-flight command buffers through mrun's synchronized chunking, and verify all of
   it on the M4 Pro with `metal-gate` probes.
2. **CPU ops**: production inference runs on Metal; CPU ops are limited to reviewed host paths with
   measured per-op scratch envelopes (or excluded).
3. **Templates**: render only the pinned, reviewed chat template; cap context bytes before render;
   reject templates that are not on the allow-list.
4. **Grammars**: mrun-owned schema admission (node/property/enum/byte limits from the HTTP matrix)
   before llguidance construction, sized from measured bytes per schema node.
5. **Tokenizer**: cap per-call input bytes from measured amplification of the real Qwen3 tokenizer.
6. **Re-run** T119 on the M4 Pro and T120 with the above; only then reconcile the gate (T121).

No production manifest, root source or CI was changed by this experiment, no weights were
downloaded, and no private data was persisted.

## Handoff (T124)

Verified: root `Cargo.toml`, `Cargo.lock`, `rust-toolchain.toml`, `src/` and CI are unchanged by
this experiment; no weights were downloaded; no private data was persisted; no historical report
eligibility was transferred. Downstream T121..T123 were not executed: the gate is blocked, and the
2026-10-14 redesign replaces the per-allocation enforcement approach (see `concepts/`).
