# Roadmap Entry Point

The [spec of specs](../specs/README.md#feature-sequence) owns feature sequence, product IDs and
reserved slots. This explains the route through that work, not another backlog, new thresholds
or completion claims. Implementation items belong in their numbered feature's tasks.

## Immediate Direction

1. Finish the product clarification: qualified baseline versus mandatory optimized-competitor win
   for first-feature completion. Do not choose on the user's behalf.
2. Reconcile [001's spec](../specs/001-agent-runtime/spec.md),
   [plan](../specs/001-agent-runtime/plan.md), contracts and tasks with the accepted choice.
3. Prove required native capabilities before production runtime work. Stop or revise selection
   if sharing, bounded allocation, tools, cancellation or faults cannot meet the contract.
   Rust-native alternatives are eligible; a full rewrite is not the default remedy for a failed spike.
4. Qualify the complete client/runtime slice and evidence layers in [VERIFICATION.md](VERIFICATION.md).
   Publish scoped comparisons, retaining failures and competitor-winning cells.

## Later Work

Durable checkpoints, scheduling, quality-preserving acceleration, model/backend expansion and
distribution/hardening have reserved slots in the canonical sequence. They are not already
specified features or promised production adapters. Promote work when evidence identifies
its benefit and prerequisites, not because a mechanism is fashionable.

Custom kernels/engines need reviewed scope, a profiled bottleneck, reference correctness,
safety/fallback consequences and measurable user benefit. Broader contexts, speculation and
KV precision changes also require their own quality and memory evidence.

## Completion Rule

Implementation requires code and checks that distinguish correct behavior from adverse cases.
Performance qualification additionally requires the physical-host evidence in the owning spec.
Best-in-class needs a dated optimized-comparator result, not feature completion or self-cache gains.
Preserve original failed evidence when proposed targets are formally amended.