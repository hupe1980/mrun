# Contract: `mrun bench` CLI and Result Schema

## CLI

```text
mrun bench [--suite agent|synthetic|cancel|memory|quality|all] [--trace NAME]...
           [--system NAME=URL[,apis=chat+messages+responses][,model=M][,pid=N][,profile=P]]...
           [--compare] [--reps N] [--out DIR] [--baseline FILE]
```

- Without `--system`, benchmarks the local mrun server (`http://127.0.0.1:8484`).
- `--compare` with several `--system` entries runs them sequentially and writes one combined
  report.
- `--baseline FILE` runs regression mode; exit code 7 when thresholds fail; 5 when a system is
  unreachable (exit codes per the [CLI contract](../../004-models-cli/contracts/cli.md#exit-codes)).

## Result schema (v1)

Stable fields are listed in [data-model](../data-model.md#runresult-json-schema_version-1).
Additive changes keep `schema_version: 1`; removals or meaning changes bump it. The hand-written
`bench/result.schema.json` is the contract; tests validate emitted results against it.

## Report (markdown)

Sections: Method (teacher-forced replay, repetitions) · Machine · Systems (versions, flags, model
name, format, quantization) · Agent suite (per-turn TTFT table and session totals) · Synthetic
cells · Cancellation · Memory · Quality · Warnings (power, throttling, early EOS) · Losing cells
(always present, possibly "none").
