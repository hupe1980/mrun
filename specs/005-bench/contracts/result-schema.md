# Contract: `mrun bench` CLI and Result Schema

## CLI

```text
mrun bench [--suite agent|synthetic|cancel|memory|quality|all] [--system NAME=URL[,api=openai|anthropic][,pid=N]]...
           [--compare] [--reps N] [--out DIR] [--baseline FILE] [--model M]
```

- Without `--system`, benchmarks the local mrun server (`http://127.0.0.1:8484`).
- `--compare` with several `--system` entries runs them sequentially and writes one combined
  report.
- `--baseline FILE` runs regression mode; exit code 1 when thresholds fail.

## Result schema (v1)

Stable fields are listed in [data-model](../data-model.md#runresult-json-schema_version-1).
Additive changes keep `schema_version: 1`; removals or meaning changes bump it. A JSON Schema
file is generated from the Rust types (`schemars`) into `bench/result.schema.json` and
snapshot-tested.

## Report (markdown)

Sections: Machine · Systems (versions, flags, models) · Agent suite (per-turn TTFT table and
session totals) · Synthetic cells · Cancellation · Memory · Quality · Warnings · Losing cells
(always present, possibly "none").
