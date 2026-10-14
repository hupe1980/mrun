# Contract: Engine trait and `mrun run`

## Internal `Engine` trait (crates/mrun-engine/src/engine.rs, no MLX dependency)

```rust
pub trait Engine {
    fn info(&self) -> &ModelInfo;
    fn budget(&self) -> &MemoryBudget;
    fn active_len(&self) -> u32;
    /// Ensure capacity for the active cache to reach `total_tokens`; evicting is the caller's job
    /// (003) via `drop_segment`. Err(Capacity) when even an empty cache cannot fit it.
    fn reserve(&mut self, total_tokens: u32) -> Result<(), EngineError>;
    /// Append `tokens` to the active cache in chunks; cancel checked after every layer.
    /// Returns how many tokens were processed (less than `tokens.len()` only on cancel).
    fn prefill(&mut self, tokens: &[u32], cancel: &CancelToken, progress: &mut dyn FnMut(u32))
        -> Result<u32, EngineError>;
    /// Feed `last` (the previous token) and return the next sampled token. Pipelined with
    /// async evaluation when `mask` is None.
    fn decode(&mut self, last: u32, sampling: &Sampling, mask: Option<&TokenMask>)
        -> Result<u32, EngineError>;
    fn trim(&mut self, len: u32);
    /// Owned deep copy of active tokens [start, end).
    fn snapshot(&mut self, range: std::ops::Range<u32>) -> Result<Segment, EngineError>;
    /// Replace the active cache: drop it, copy `segments` (contiguous from position 0) into fresh
    /// buffers, trim to `keep` tokens.
    fn restore(&mut self, segments: &[KvSegmentId], keep: u32) -> Result<(), EngineError>;
    fn drop_segment(&mut self, id: KvSegmentId);
    fn reset(&mut self);
}
pub struct Segment { pub id: KvSegmentId, pub bytes: u64 }
```

Implementations: `MlxEngine` (feature `mlx`), `ScriptedEngine` (deterministic tokens, structural
tracking of which token ranges each segment holds; used by `mrun` tests on any OS).

Guarantees:
- No method blocks longer than one layer of one chunk without checking `cancel` (prefill) or one
  token (decode).
- `snapshot` never shares buffers with the active cache; `restore` never aliases segments.
- Engines are created on the worker thread by `EngineHandle::spawn(factory)` and never move.

## Generation loop (crates/mrun/src/runtime/generate.rs)

Generic over `Engine`: admission (`reserve`), prefill of all but the last prompt token, decode loop
with stop-sequence and UTF-8 hold-back, events on a bounded channel (cap 64, stall timeout 10 s),
exactly one `Finished`, trim to committed tokens.

## CLI: `mrun run`

```text
mrun run <model> [--prompt TEXT] [--system TEXT] [--temperature F] [--top-p F] [--top-k N]
                 [--min-p F] [--seed N] [--max-tokens N] [--stop S]... [--think]
                 [--chunk N] [--memory-fraction F] [--reserve-gib N] [--allow-over]
                 [--info] [--check]
```

- `<model>`: local path or HF repo id (`mlx-community/Qwen3-8B-4bit`); aliases arrive with 004.
- Interactive REPL when `--prompt` is absent; `/clear`, `/info`, `/exit` commands.
- After each answer: `42 tok · 1.2 s TTFT · 512 tok/s prefill · 61 tok/s decode · 3,104 cached`.
- `--info`: budget breakdown (incl. current free memory) and exit after load.
- `--check`: conformance set for this model; exit non-zero at the first mismatch, printing prompt
  id, token position and logit margin.
- Input caps: prompt ≤ 8 MiB, ≤ 4 stop sequences of ≤ 256 bytes.
- Ctrl-C cancels the current generation; a second Ctrl-C within 1 s exits.
- Exit codes follow [004 CLI contract](../../004-models-cli/contracts/cli.md).
