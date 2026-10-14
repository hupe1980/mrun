# Contract: Engine trait and `mrun run`

## Internal `Engine` trait (crates/mrun/src/runtime/engine.rs)

```rust
pub trait Engine: Send + 'static {
    fn info(&self) -> &ModelInfo;                 // identity, capacity, kv bytes/token, template
    fn budget(&self) -> &MemoryBudget;
    fn generate(&mut self, req: GenerateRequest, events: &EventSink) -> Result<(), EngineError>;
    fn snapshot(&mut self, range: std::ops::Range<u32>) -> Result<KvSegmentId, EngineError>; // owned copy of active KV tokens [start, end), used by 003
    fn restore(&mut self, segments: &[KvSegmentId], keep: u32) -> Result<(), EngineError>; // copy segments (contiguous from 0) into the active cache, then trim to `keep` tokens
    fn drop_segment(&mut self, id: KvSegmentId);
    fn reset(&mut self);
}
```

Guarantees:
- `generate` emits exactly one `Finished` event, also on cancel and error.
- Cancel observed between prefill chunks and between decode steps; no event after `Finished`.
- A request whose `tokens.len() + max_tokens` exceeds free KV capacity returns
  `EngineError::Capacity { requested, available }` before any computation.
- Implementations: `MlxEngine` (mrun-engine) and `ScriptedEngine` (deterministic, for tests).

## CLI: `mrun run`

```text
mrun run <model> [--prompt TEXT] [--system TEXT] [--temperature F] [--top-p F] [--top-k N]
                 [--min-p F] [--seed N] [--max-tokens N] [--stop S]... [--think]
                 [--chunk N] [--memory-fraction F] [--reserve-gib N] [--allow-over]
                 [--info] [--check]
```

- Interactive REPL when `--prompt` is absent; `/clear`, `/info`, `/exit` commands.
- After each answer: one dim line `42 tok · 1.2 s TTFT · 512 tok/s prefill · 61 tok/s decode`.
- `--info`: print the budget breakdown and exit after load.
- `--check`: run the conformance set for this model (stored references) and exit non-zero on the
  first mismatch, printing prompt id and token position.
- Ctrl-C cancels the current generation; a second Ctrl-C within 1 s exits.
- Exit codes follow [004 CLI contract](../../004-models-cli/contracts/cli.md).
