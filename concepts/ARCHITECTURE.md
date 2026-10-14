# Architecture and Ownership

**Status**: Intended architecture, not implemented or native-capability-qualified.
The [implementation plan](../specs/001-agent-runtime/plan.md) owns concrete technical commitments.

## Layers

| Layer | Owns | Must not own |
| --- | --- | --- |
| CLI/configuration | Explicit local model and policy inputs, controls, readiness | Hidden model downloads or privileged OS tuning |
| HTTP boundary | Bounded parsing, schema/history validation, truthful stream/error encoding | Native context mutation or tool execution |
| Runtime scheduler | FIFO admission, reservations, lifecycle, cancellation arbitration | Blocking native work on async executor threads |
| Prefix index | Exact identity/tokens, immutable entries, pin/eviction decisions | Sampler/RNG state or approximate text matches |
| Resource ledger | Unique backing ownership, finite growth, pressure disposition | Treating CPU/GPU views as separate memory pools |
| Backend adapter | Template/tokenizer, prefill/decode, grammar/parser, GPU-safe lifetime | Publicly exposed native handles or independent admission policy |
| Evidence boundary | Stage/token/resource metadata and reviewed public-fixture results | Private content logging or performance claims without comparisons |

## Control Flow

Validate bounded input and tool-result linkage; reserve waiting demand; enter the finite queue.
The native owner prepares a qualified model, rerenders/tokenizes the complete request, checks
context/output demand, and attaches only an exact compatible retained prefix. It evaluates the
unmatched suffix and counted boundary replay, then generates with fresh request-local state.
The owner publishes only synchronized, validated prompt checkpoints. Output crosses a bounded
byte-leased bridge and is encoded without inventing usage or a successful interrupted tool call.

Cancellation, disconnect, pressure and failure converge on one cleanup arbiter. Storage remains
alive through submitted GPU work; releasing an HTTP socket is not releasing GPU capacity.
Native owner serialization and immutable shared membership protect branches from mutation.
Details and state transitions belong to the [data model](../specs/001-agent-runtime/data-model.md).

## Backend Strategy

The current plan selects one pinned llama.cpp/Metal integration behind a private C ABI/C++ shim
for an initial capability spike. It is neither permanent nor proof of engine superiority.
If a required capability cannot be safely exposed, the choice must be revised before production
runtime work. Do not disguise a subprocess HTTP wrapper as ownership of fine-grained native state.

Evaluate alternatives without building multiple production adapters or a speculative plugin
framework. Public runtime types must not leak engine types, so a future evidence-backed replacement
does not require redefining client behavior. Rust-native engines remain candidates; a fully custom
engine is a separate investment, not implied by the project's language.

## Unified Memory and Failure

Count each unique backing allocation once, including physical pools, scratch and serving demand.
Reclaiming cache cells does not necessarily reclaim a preallocated device buffer. Observe process,
OS pressure and Metal allocations separately; accounted limits are not a hard RSS guarantee.
When pressure cannot be relieved safely, stop admission, cancel affected work, synchronize and
teardown rather than truncate context silently.

If native work leaves validity uncertain, invalidate affected state or the entire context.
An unobservable asynchronous driver failure is a qualification gap, not proof of safe recovery.
The [native contract](../specs/001-agent-runtime/contracts/backend.md) owns those stop conditions.