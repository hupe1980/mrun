# Contract: `mrun bench` CLI and Result Schema

## CLI

```text
mrun bench [--suite agent|synthetic|cancel|all] [--trace NAME]...
           [--system NAME=URL[,apis=chat+messages+responses][,model=M][,pid=N][,profile=P]]...
           [--compare] [--reps N] [--out DIR] [--baseline FILE]
```

- Without `--system`, benchmarks the local mrun server (`http://127.0.0.1:8484`).
- `--compare` with several `--system` entries runs them one at a time (profile start → health →
  reset per repetition → stop; or a "Stop X, start Y, press Enter" pause without commands) and
  writes one combined report. The `cancel` suite runs for mrun only.
- Exit codes per the [CLI contract](../../004-models-cli/contracts/cli.md#exit-codes): 7 when
  `--baseline` thresholds fail; 5 only when every system (or the only system) is unreachable. An
  unreachable competitor under `--compare` yields "not run" cells, not an error exit.

## Profile (`bench/profiles/<name>.toml`)

`start`, `stop`, `health` (optional; all or none), `reset` (required), `pid` (optional command),
`context_length` flags, cache flags, model name, format and quantization.

## Result schema (v1)

Stable fields are listed in [data-model](../data-model.md#runresult-json-schema_version-1).
Additive changes keep `schema_version: 1`; removals or meaning changes bump it. The hand-written
`bench/result.schema.json` is the contract; tests validate emitted results against it.

## Report (markdown)

Sections: Method (in-order replay, reuse measured up to the previous reply, cache resets, equal
context, repetitions) · Machine · Systems (versions, flags, context length, model name, format,
bits per weight) · Agent suite (per-turn TTFT table, session totals, peak memory, malformed tool
calls and early EOS per system) · Synthetic cells · Cancellation (mrun) · Warnings (power,
throttling, memory not recovered) · Losing cells (always present, possibly "none").
