# Candle Allocation-Bound Feasibility Experiment

Feature `001-agent-runtime`, tasks T112..T120 ([tasks](../../specs/001-agent-runtime/tasks.md)).
This package answers one planning question: can mrun enforce finite, pre-allocation byte bounds
(FR-014) around Candle and its host libraries? It is not a second production backend, a public
model, a serving path or evidence for any SC gate.

## Baseline (recorded 2026-10-14)

| Item | Value |
| --- | --- |
| Root package | `mrun` 0.0.1, Rust 1.94.0, no Candle dependency; portable suite 39/39 passing |
| Historical tasks | T001..T111 text preserved; removed llama.cpp artifacts stay removed |
| Toolchain | Rust 1.94.0 (`rust-toolchain.toml` here, separate workspace) |
| Candle candidate | `c68b24997319b8d76e5a4e775dab197c7f26322a` (workspace 0.11.0), `candle-core` only |
| Model weights | None. No download, no inference, no tokenizer/model artifact |
| Serving | None. No HTTP, CLI or client integration |

## Boundaries

- The experiment is its own Cargo workspace. It never changes root `Cargo.toml`, `Cargo.lock`,
  `rust-toolchain.toml`, CI or source; the root package `include` list excludes it from packaging.
- No kernels are patched. Candle patches under `patches/` only add accounting or denial hooks
  around existing allocation entry points, and must be reviewed before any later use.
- Experimental GPU backing is capped at **64 MiB** per test process. Every test records a
  cleanup baseline (reserved, committed, live leases, quarantined bytes) and must return to it.
- Missing tools or hardware are recorded as **blocked**. Nothing is silently installed or
  substituted with a different backend.

## Host status

| Lane | Host | Status |
| --- | --- | --- |
| CPU build, locked | Linux x86_64, Rust 1.94.0 | Builds |
| Metal type-check, locked | `--target aarch64-apple-darwin --features metal` from Linux | Type-checks; not linked |
| Metal build/run | Requires macOS with Metal (target: M4 Pro 48 GB) | **Blocked**: not available on this host |

## Layout

- `src/budget.rs`: experimental reservation ledger, leases, submissions and quarantine.
- `tests/admission.rs`: allocation-contract probes against real Candle CPU constructors (T116).
- `tests/host_limits.rs`: host-library limit probes (T118).
- `tests/metal_lifetime.rs`: portable lifetime controls plus Metal probes gated on macOS (T119).
- `gpu-inventory.md`, `host-inventory.md`: source inventories (T114, T115).
- `patches/`: reviewed-candidate Candle hook patches (T117).
- `results.md`: consolidated disposition (T120) and handoff (T124).

## Commands

```sh
cd experiments/candle-budget
cargo test --locked                       # portable lanes
cargo check --locked --tests --features metal --target aarch64-apple-darwin
cargo test --locked --features metal      # macOS with Metal only
```
