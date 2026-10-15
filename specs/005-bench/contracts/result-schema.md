# Contract: `mrun bench` CLI and Result Schema

## CLI

```text
mrun bench [--suite agent|synthetic|cancel|all] [--trace NAME]...
           [--system NAME=URL[,apis=chat+messages+responses][,model=M][,profile=P]]...
           [--compare] [--reps N] [--out DIR]
```

- `--reps` raises, never lowers, the FR-004 minimums.
- Without `--system`, benchmarks the local mrun server (`http://127.0.0.1:8484`).
- `--compare` with several `--system` entries runs them one at a time ("Stop X, start Y, press
  Enter" pause → health → reset per repetition) and writes one combined report. The `cancel` suite
  runs for mrun only.
- Exit codes per the [CLI contract](../../004-models-cli/contracts/cli.md#exit-codes): 5 only when
  every system (or the only system) is unreachable. An unreachable competitor under `--compare`
  yields "not run" cells, not an error exit. (`--baseline` and exit 7: Later.)

## Profile (`bench/profiles/<name>.toml`)

`health` (optional), `reset` (required: an HTTP call, or `manual` for a restart pause),
`context_length` flags, cache flags, model name, format, quantization and KV format. Bundled:
`mrun`, `llama.cpp`, `ollama`, `omlx`, `mlx-lm`, `splash` (cells `splash-bf16` and `splash`).
(`start`/`stop`/`pid` commands: Later.)

## Result schema (v1)

Stable fields are listed in [data-model](../data-model.md#runresult-json-schema_version-1).
Additive changes keep `schema_version: 1`; removals or meaning changes bump it. The hand-written
`bench/result.schema.json` is the contract; tests validate emitted results against it.

## Report (markdown)

Sections: Method (in-order replay, reuse measured up to the previous reply, cache resets, equal
context, repetitions) · Machine · Systems (versions, flags, context length, model name, format,
bits per weight, KV format) · Agent suite (per-turn TTFT table, session totals, malformed tool
calls and early EOS per system, mrun `template` re-renders) · Synthetic cells · Cancellation
(mrun) · Warnings (power, thermal state, free memory) · Losing cells (always present, possibly
"none").
