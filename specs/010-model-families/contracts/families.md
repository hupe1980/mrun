# Contract: Families, registry, syntaxes, locality

## Family module (`crates/mrun-engine/src/models/<family>.rs`, feature `mlx` except config/layout)

```rust
/// One per architecture family. Implemented by a unit struct; registered in `registry.rs`.
pub trait Family: Sync {
    /// Stable name shown by `mrun show` (e.g. "gpt_oss").
    fn name(&self) -> &'static str;
    /// Parse and validate (`config.json` root or `text_config`). No MLX; runs before weights load.
    fn parse_config(&self, raw: &serde_json::Value) -> Result<FamilyConfig, LoadError>;
    /// Per-layer state layout; no MLX. Drives budget, admission and `--info`.
    fn state_layout(&self, cfg: &FamilyConfig) -> Vec<StateLayout>;
    /// Accepted chat syntaxes, default first; the template probe picks within them (R8).
    fn syntaxes(&self, cfg: &FamilyConfig) -> &'static [SyntaxId];
    /// Expected tensor names and shapes after sanitize, for the pre-load check (FR-004).
    fn expected_tensors(&self, cfg: &FamilyConfig) -> TensorManifest;
    /// Port of mlx-lm `sanitize` (drop vision/mtp, split fused projections, MXFP4 `_blocks`).
    #[cfg(feature = "mlx")]
    fn sanitize(&self, cfg: &FamilyConfig, w: WeightMap) -> Result<WeightMap, LoadError>;
    /// Compose blocks into the layer stack.
    #[cfg(feature = "mlx")]
    fn build(&self, cfg: &FamilyConfig, w: WeightMap, q: &QuantPlan) -> Result<Graph, LoadError>;
    /// Tiny random-weight config covering every option this family enables (Tier 2 parity).
    fn tiny_configs(&self) -> &'static [&'static str];
}

/// The stack the engine runs. Blocks come only from `models::blocks`.
pub struct Graph { /* embed, layers: Vec<Layer>, norm, head, softcap */ }

impl Graph {
    /// One layer over `x` for positions [pos, pos + L) using that layer's state slot.
    fn layer(&self, i: usize, x: &Array, st: &mut LayerState, pos: u32) -> Result<Array, EngineError>;
    fn logits(&self, h: &Array) -> Array;
}
```

Rules:

- `Graph` evaluates one layer at a time so 001's per-layer cancel and yield checks keep working.
- Masks are built from absolute positions; windowed layers slice keys to `[q − W + 1, q]` (R3).
- Modules never touch `LayerState` allocation, growth, checkpoints or moves; `state.rs` does, from
  `StateLayout`.
- A module may contain family-only arithmetic; it may not contain logic for another family.

## Registry (`crates/mrun-engine/src/models/registry.rs`)

```rust
pub fn lookup(model_type: &str, text_model_type: Option<&str>) -> Result<&'static dyn Family, LoadError>;
pub fn supported() -> &'static [(&'static str, Option<&'static str>)];
```

## Load errors (extends 001 load errors; exit code 3 per 004)

```rust
pub enum LoadError {
    UnsupportedArchitecture { model_type: String, text_model_type: Option<String>, supported: Vec<String> },
    UnsupportedQuantization { path: String, mode: String, bits: u32, group_size: u32 },
    UnknownTensor { name: String },
    MissingTensor { name: String },
    ShapeMismatch { name: String, expected: Vec<u32>, found: Vec<u32> },
    InvalidConfig { key: String, reason: String },
    SyntaxMismatch { family: String, accepted: Vec<SyntaxId>, detected: Option<SyntaxId> },
}
```

`UnsupportedArchitecture` message: 004 E034 (supported families with their aliases).

## Chat syntax (`crates/mrun/src/chat/syntax/{mod.rs, hermes.rs, qwen_xml.rs, harmony.rs, mistral.rs, glm_xml.rs, gemma.rs}`)

```rust
pub trait ChatSyntax {
    fn id(&self) -> SyntaxId;
    fn probe_signature(&self) -> &'static str;
    fn markers(&self) -> &'static [&'static str];
    /// Template variables and message field mapping for history (FR-015).
    fn history(&self) -> HistoryOptions;
    /// New per-request parser; `prompt_tail` decides the initial state (open `<think>`, Harmony
    /// header already rendered), `tools` gives schemas for string/non-string parameter typing.
    fn parser(&self, tools: &[ToolSchema], prompt_tail: &str) -> Box<dyn SyntaxParser>;
}

pub trait SyntaxParser {
    /// Feed one sampled token (id and decoded text). Emits zero or more events; holds back text
    /// that may be a marker prefix.
    fn feed(&mut self, id: u32, text: &str, out: &mut Vec<ParseEvent>);
    /// End of generation with the 001 finish reason; flushes held text, closes or marks
    /// incomplete an open call.
    fn finish(&mut self, reason: FinishReason, out: &mut Vec<ParseEvent>);
}

pub enum ParseEvent {
    Content(String),
    Reasoning(String),
    ToolStart { index: u32, name: String },
    ToolArgsRaw { index: u32, delta: String },          // RawJson syntaxes only
    ToolEnd { index: u32, args: ToolArgs },              // ToolArgs::Json(Value) | ToolArgs::Raw(String)
    ToolIncomplete { index: u32, raw: String },          // cut by length
    Warning(Warning),                                    // tool-unknown, tool-args-invalid
}
```

002 maps events to the three APIs and applies FR-004 fallbacks; the parser never repairs or
coerces.

## Template probe (`crates/mrun/src/chat/syntax/probe.rs`)

```rust
pub fn probe(template: &ChatTemplate, vocab: &Vocab, accepted: &[SyntaxId]) -> Result<ProbeResult, LoadError>;
```

Detected syntax in `accepted` → used; otherwise `LoadError::SyntaxMismatch { family, accepted,
detected }` (002 FR-032). Runs once at load (and in `mrun pull` for the manifest). Deterministic: fixed messages, fixed
tool schema, no date dependence (a `strftime_now` call is stubbed to a fixed date in the probe;
at serve time it returns the server start date, OQ-4). The probe also records whether tools render
after the first user message; if so `ProbeResult.tools_late` is set and `mrun doctor`/`mrun ps`
warn `template-tools-late` (the prefix changes every turn, so caching is defeated).

## Locality rule

A family change (PR labelled `family:<name>`) may touch only:

- `crates/mrun-engine/src/models/<family>.rs` and its line in `models/mod.rs` and
  `registry.rs`;
- `crates/mrun/src/chat/syntax/<syntax>.rs` and its line in `syntax/mod.rs` (new syntax only);
- the alias entry in `crates/mrun/src/hub/aliases.rs`;
- tests and fixtures: `crates/mrun-engine/tests/families/<family>/**`,
  `crates/mrun/tests/templates/<family>/**`, `crates/mrun/tests/syntax/<syntax>/**`,
  reference outputs and tiny weights generated by `scripts/gen_reference.py`;
- the 005 cell list and `site/content/docs/models.md`.

`scripts/check_family_diff.sh <base>` lists every other path and fails if any exists. Block-library
changes (`models/blocks/**`, `state.rs`, loader quantization) are separate changes with block-level
tests; a family change that needs one depends on it.
