# Product Overview

## Purpose

mrun targets interactive local coding-agent work on Apple Silicon. The primary user runs an
existing OpenCode client with repository instructions, tool schemas and growing conversation
history. Success means correct completed work with less waiting and predictable resource use,
not simply a larger tokens-per-second number or a Rust rewrite of a serving layer.

The initial qualification host is an M4 Pro with 48 GB unified memory. That target does not
establish a supported hardware matrix. Exact machine, software and workload identities belong
in evidence. The current package is a placeholder, not an inference runtime.

## Product Boundary

mrun serves provisioned models, emits text/tool-call streams, owns inference-state reuse,
controls admission and cancellation, and reports sanitized local measurements. The client owns
conversation history, tool authorization and execution, edits, and fixture-test execution.
The runtime must not become a shell executor or an integrated agent merely to simplify a demo.

One trusted OS user, one active model and one foreground generation define the first scope.
Loopback applications under that OS identity are not isolated tenants. Offline runtime inference
does not guarantee that the client or its tools are offline; both require separate traffic checks.

General Ollama parity, universal model support, distributed serving, training, remote execution,
durable state, approximate matching and speculative decoding are not first-feature promises.
Detailed scope and rejection behavior belong to the [feature specification](../specs/001-agent-runtime/spec.md).

## What Best-in-Class Means

It is a comparative result for named coding-agent workloads under fixed quality, memory and power
conditions. It is not an engine label, programming-language preference, or universal ranking.
The [product policy](../specs/README.md#performance-and-quality-policy) owns the claim boundary;
the [benchmark protocol](../specs/001-agent-runtime/benchmark-protocol.md) owns fair comparisons.

Improvement over cache-disabled mrun demonstrates a mechanism benefit, not leadership over
competitors whose caches and other useful optimizations are enabled. A fast invalid tool call,
failed coding task or cancelled response cannot count as successful useful output.
Whether competitor leadership must be achieved in the first feature remains an unanswered
clarification; the current acceptance gates have not been amended.

## Documentation Boundary

These notes explain why the product is shaped this way. Specs own requirements, contracts own
the planned external behavior, code establishes implementation facts, and reproducible checks
establish evidence. Public README/rustdoc/package material must not depend on this ignored folder.