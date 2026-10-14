# Engine and Product Landscape

**Basis**: Source review recorded on 2026-10-13 in the
[feature research](../specs/001-agent-runtime/research.md). No local comparative benchmarks exist.
These are evaluation roles, not a ranking or qualified support matrix.

## Candidates and Comparators

| Project | Why evaluate it | What must be proved |
| --- | --- | --- |
| llama.cpp/Metal | GGUF reference and initial integration candidate | State ownership, allocation bounds, cancellation and template/tool controls |
| Ollama | Established local-product comparator | Actual runner, optimized cache configuration and comparable quality/resources |
| mistral.rs | Serious Rust-native candidate | Admitted model/precision, state controls, tools and physical M4 Pro behavior |
| MLX LM | Apple-focused engine/product comparator | Conversion provenance, template agreement and quality-matched comparisons |
| uzu | Apple-first Rust/native candidate | Format provenance, architecture coverage and controllable lifecycle |
| Imparo | Agent-oriented Rust/Metal research candidate | Selected-model admission and reproducible bounded-workload behavior |
| SGLang | Scheduling reference and conditional comparator | Relevant local Apple route and model/tool compatibility |

Original source URLs, observed revisions and capability caveats belong to feature research.
Do not turn authors' benchmark tables into mrun evidence or copy mutable dependency pins here.

## Defensible Differentiation

Exact prefix caching already exists elsewhere. The potential advantage is a better combination of
correct tool loops, branch-safe reuse, predictable shared-memory residency, interruption and useful
output latency on a developer's workload. That advantage remains hypothetical. Language preference
and a private shim are engineering choices, not product differentiation.

A small model may be fast but fail useful tasks. Active MoE parameter count does not describe all
resident weights. Separate engine quality from model quality and never choose a weak reference to
manufacture a speedup.

## Comparison Rules

Compare identical artifacts/settings where possible and explicitly quality-matched products where
conversion formats differ. Enable useful competitor optimizations, record memory/power conditions,
and retain unsupported, failed and competitor-winning cells. Self-improvement, serving overhead
and optimized products are different lanes.

The [claim policy](../specs/README.md#performance-and-quality-policy) and
[benchmark protocol](../specs/001-agent-runtime/benchmark-protocol.md) own the comparative standard.
A Rust-native backend may prove better; no backend earns that conclusion in advance.