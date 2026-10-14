# Quickstart and Validation Guide

**Candle replan notice (2026-10-14)**: This is the superseded llama.cpp validation guide, not the
current Candle procedure. Its Rust 1.85/CMake/native commands and placeholder-only description below
are historical. The active plan selects Rust 1.94.0; replacement Phase 1 instructions remain blocked
on enforceable allocation contracts. Do not run the commands below as Candle qualification.
See [current planning status](plan.md#current-planning-status).

**Status**: Design-stage guide. Only a dependency-free placeholder Cargo package exists; there is
no inference executable, harness or qualified model today. Only the documentation check below runs
now. Remaining commands specify the future workflow
and must not be presented as already working or passing. Implementation belongs to later tasks.

## Check the Current Documents

From the repository root:

```sh
.specify/scripts/bash/check-prerequisites.sh --json --require-spec
```

Expected: feature `specs/001-agent-runtime`, plan/spec present, research/data-model/quickstart/contracts
discoverable. This says nothing about runtime readiness. See [plan](plan.md) and [contracts](contracts/http.md).

## Prerequisites After Implementation

- Native arm64 M4 Pro with 48 GB, macOS 15+, macOS 15+ SDK/Apple Clang, CMake 3.31.6 and Rust 1.85.0.
- Pinned native source and reviewed locked Rust dependencies, provisioned before offline use.
- Official hash-verified primary GGUF below; no inference-time downloads.
- Released OpenCode v1.19.0 (`add9b4ba5adfb895934cb1bf46f590342cec68d2`) and its dependencies
  provisioned before offline tests; verify release checksum and binary hash in evidence.
- Redistributable fixture manifest generated from [catalog](contracts/qualification.md), including
  schema/request captures, licenses, hashes, expected tests and objective quality outcomes.
- Prepare draft inputs first; derive/seal token expectations after the verified renderer exists.
  Dependent evidence runs reject draft manifests. Client capture revisions require affected reruns.
- Idle and editor-open host conditions recorded separately; no privileged wired-memory changes.

## Explicit Provisioning (Future Workflow)

Only perform downloads as an explicit provisioning step, never as a request fallback:

```sh
mkdir -p models
curl --fail --location --output models/Qwen3-8B-Q4_K_M.gguf \
  https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/7c41481f57cb95916b40956ab2f0b139b296d974/Qwen3-8B-Q4_K_M.gguf
shasum -a 256 models/Qwen3-8B-Q4_K_M.gguf
wc -c < models/Qwen3-8B-Q4_K_M.gguf
```

Expected SHA-256: `d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785`;
length: `5027783488`. Abort on mismatch. Never treat the download as a qualification result.
Native source provisioning must check out the exact research D-01 commit and verify a clean tree
at `native/llama.cpp`; implementation must provide a clean-package-compatible vendoring/provisioning
procedure and retain native MIT notices. Cargo/build scripts must not fetch during offline builds.

## Build and Capability Proof (Future Commands)

```sh
cargo +1.85.0 test --locked --no-default-features
cargo +1.85.0 build --locked --release --features metal
./target/release/mrun model verify \
  --path models/Qwen3-8B-Q4_K_M.gguf \
  --sha256 d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785
cargo +1.85.0 run --locked --release --features metal --example qualify -- \
  --suite seal --runtime-binary ./target/release/mrun \
  --model models/Qwen3-8B-Q4_K_M.gguf --manifest tests/fixtures/manifest.json
cargo +1.85.0 run --locked --release --features metal --example qualify -- \
  --suite adapter --model models/Qwen3-8B-Q4_K_M.gguf \
  --manifest tests/fixtures/manifest.json --output benchmark-results/adapter
cargo +1.85.0 run --locked --release --features metal --example qualify -- \
  --suite capability --model models/Qwen3-8B-Q4_K_M.gguf \
  --runtime-binary ./target/release/mrun \
  --manifest tests/fixtures/manifest.json --output benchmark-results/capability
```

Expected: deterministic portable positive/rejected paths pass; verified identities match; the native
capability runner reports real sharing/boundary/geometry/accounting/fault/unload/cancellation outcomes.
Primitive construction probes establish the renderer before fixture sealing; draft inputs do not
authorize dependent gates. The adapter suite authorizes continued implementation only. The capability
suite exercises the selected executable through the private fixture-scoped harness channel, not just
the example's library code, and records both executable identities. It requires production admission,
pressure/reservation and safe cleanup already implemented. The report authorizes cache-off serving
only for the exact candidate/model/profile. Rebuilds invalidate it: repeat candidate capability and
affected reuse/acceptance suites before another changed binary serves. Never edit an old subject hash.
A failure/blocked/inconclusive capability report cannot authorize `serve`; a 128-token chunk must
actually meet SC-004, not merely compile. These commands describe staged future artifacts, not code today.

## Local Service Checks (Future Commands)

After capability evidence authorizes this binary/model/profile:

```sh
./target/release/mrun serve --model models/Qwen3-8B-Q4_K_M.gguf \
  --qualification benchmark-results/capability/qualification.json --reuse off
```

In a second terminal:

```sh
curl --fail http://127.0.0.1:8012/v1/models
curl --fail --no-buffer http://127.0.0.1:8012/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-8b-q4-k-m","messages":[{"role":"user","content":"Reply with a short greeting."}],"stream":true,"stream_options":{"include_usage":true},"max_tokens":64}'
./target/release/mrun status --json
./target/release/mrun cache clear
./target/release/mrun model unload
```

Expected: configured model discovery does not load by itself; stream reconstructs valid UTF-8 and
one terminal result; usage reconciles. Clear invalidates all retained hits, unload waits for safe
release. Probe oversize context/schema/queue and unknown semantic fields using frozen fixtures;
expect published codes and no inference work. Do not infer correctness from curl output alone.

## Pinned OpenCode Configuration (Future Validation)

Use explicit local `opencode.json` in a public fixture directory, not a private repo:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "model": "mrun/qwen3-8b-q4-k-m",
  "provider": {
    "mrun": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "mrun",
      "options": {"baseURL": "http://127.0.0.1:8012/v1"},
      "models": {
        "qwen3-8b-q4-k-m": {
          "name": "Qwen3-8B Q4_K_M",
          "reasoning": false,
          "attachment": false,
          "tool_call": true,
          "limit": {"context": 16384, "input": 14336, "output": 2048},
          "modalities": {"input": ["text"], "output": ["text"]}
        }
      }
    }
  }
}
```

Preprovision client metadata/providers/tools and disable its external services through settings
supported by this pinned release; observe traffic independently. `$schema` is editor metadata,
not permission for runtime network access. Run the unmodified release with this provider and the
IC-020 read-edit-test fixture; verify authorized edits and supplied tests, matching tool-call/result
IDs, no partial-call execution and no external traffic. The model runtime never executes tools.
Capture only public-fixture requests; if release behavior differs from the reviewed source matrix,
stop and reconcile the contract, not patch the client or rewrite requests invisibly.

## Full Acceptance and Regression (Future Commands)

The example harness CLI is a planned validation target, not a product command. First obtain reuse
correctness through candidate-only public-fixture execution; no production `serve --reuse on` bypass
is available. Its run-scoped endpoint lets the unmodified client complete the paired coding suite:

```sh
cargo +1.85.0 run --locked --release --features metal --example qualify -- \
  --suite reuse --model models/Qwen3-8B-Q4_K_M.gguf \
  --runtime-binary ./target/release/mrun \
  --manifest tests/fixtures/manifest.json --client-binary /path/to/pinned/opencode \
  --output benchmark-results/reuse
cargo +1.85.0 run --locked --release --features metal --example qualify -- \
  --suite acceptance --model models/Qwen3-8B-Q4_K_M.gguf \
  --runtime-binary ./target/release/mrun \
  --manifest tests/fixtures/manifest.json --client-binary /path/to/pinned/opencode \
  --output benchmark-results/acceptance
cargo fmt --check
cargo clippy --locked --all-targets --features metal -- -D warnings
cargo +1.85.0 check --locked --all-targets --all-features
```

Expected evidence: IC=20 and state=100 distinct fixtures; 100 cancellation trials per phase;
30-minute churn with independent 100 ms memory samples; 60 paired task outcomes per path;
ten ten-turn traces; exact/tolerance lanes and protocol statistics. Fixture guard tests must reject
missing counts/licenses/hashes and run from a clean packaged tree. Production/default reuse requires
matching capabilities, state/numerical, IC-001..020 and SC-007 passes. Candidate experiments do not
confer eligibility until their records pass verification. Report all SC-001..009 as
pass/fail/blocked/inconclusive; all are still required for feature completion/performance qualification.

Provision direct llama.cpp and Ollama comparators plus compatible MLX/Rust-native contenders before
offline product comparison. Freeze their binary/model hashes and exact commands in the run manifest;
report unsupported cells with reasons. Follow the [benchmark protocol](benchmark-protocol.md),
not arbitrary engine defaults or advertised speedups. Keep raw failures and statistics under ignored
benchmark-results/ and publish only reviewed sanitized evidence. All gates are currently blocked.