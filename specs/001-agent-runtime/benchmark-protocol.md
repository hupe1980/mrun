# Benchmark and Qualification Protocol

**Created**: 2026-10-13
**Status**: Proposed procedure; zero experiments run
**Scope**: Acceptance evidence for [001-agent-runtime](spec.md)

## Freeze Before Measuring

Record a versioned manifest containing exact CPU/GPU model and core counts, physical bytes,
macOS/SDK/toolchain, engine/source/binary hashes, build flags, client/SDK version, weight artifact
hashes and license, tokenizer/template/render-control hashes, quantization, KV format, context,
prefill chunk, GPU offload, threading, speculation, sampling, output allowance, and cache policy.

Provision models explicitly. Keep small redistributable fixture inputs in the repository; larger
weights use documented revision/hash-verified downloads outside Git. Freeze 20 interoperability
cases, 100 cache-state cases, and 20 self-contained coding tasks before optimization. Publish their
identities and expected checks, not private repository content. Do not tune on held-out task cases.

Physical M4 Pro 48 GB is the required acceptance host. Record AC/battery and power mode, background
memory, GPU working-set recommendation, swap/pressure at start/end, and observable thermal state.
Do not change OS wired-memory policy. Normal background apps may run only if fixed and disclosed;
use the same conditions for all paired engines. Separate a realistic editor-open cohort from an
idle-host microbenchmark. Qualification must name which cohort supports each claim.

## Comparator Lanes

| Lane | Control | What it can establish |
| --- | --- | --- |
| A: mechanism | Same mrun binary/backend/artifact/settings, reuse off vs on | Benefit attributable to prefix reuse |
| B: serving overhead | mrun vs direct serving by same pinned backend, reuse/speculation off | Runtime overhead rather than kernel superiority |
| C: matched engine | Ollama, llama.cpp and compatible candidates with identical artifact/token IDs when possible | Engine/runtime differences under matched controls |
| D: optimized product | Each qualified engine with documented useful optimizations enabled and fixed quality/memory budget | Real user outcome against strong configurations |

Ollama and llama.cpp are mandatory comparisons. Include MLX LM and qualified Rust-native/other
Apple Silicon contenders where the chosen model and tool contract are supported. Unsupported cells
are reported as unsupported with reasons, not as losses. Lack of a fair comparison blocks a claim
about that competitor, not portable correctness development.

GGUF and MLX/uzu conversions may differ in numerics and size even with the same nominal bits.
Label those as quality-matched product lanes, retain conversion provenance, and run quality gates.
Never call them identical-weight kernel comparisons. Record Ollama's actual runner/backend.

## Workload Cells

- Cold model start: process launch through first generated output; record model load separately.
  Distinguish process-cold from OS-file-cache-cold. Do not claim disk-cold without proving it.
- Warm misses: 512, 8,192, and 14,336 rendered prompt tokens, up to 256 output tokens, reuse disabled.
  Longer cells still fit the 16,384 total-context qualification ceiling.
- Warm hits: 8,192-token retained prefix plus suffix lengths 32, 128, and 512; freeze actual rendered
  IDs and overlap after template application, not nominal message-length counts.
- Branch/edit: identical prefix, changed system/tool controls, early/middle edits, full identical prompt,
  model/template/adapter/precision changes, eviction and cache-clear recovery.
- Ten-turn tool trace: fixed system/tools, growing history, deterministic recorded tool results,
  5-10k prompts within the context ceiling; preserve every turn and replay the exact inputs per engine.
- Live coding tasks: real client read/edit/test loops, with sandboxed fixture repos and objective
  test outcomes; three paired seeds per task, bounded wall-time and tool budget.
- Cancellation: 100 trials each during queue, prefill, decode, and slow-reader states; use a defined
  trigger tied to state/event observation, not guessed delays.
- Memory/queue pressure: 30-minute warm mixed-context prefix churn, eight queued requests, oversize
  admission attempts, eviction, cancellation, failure, clear and unload. Report foreground latency.

Replay and live-agent lanes are different. Identical replay proves runtime effects; live runs can
change tool sequences and output lengths and therefore need task-success and total-time analysis.

## Metric Definitions

- Client request origin: monotonic timestamp immediately before issuing the request.
- First generated output latency: origin to first model-generated token payload, excluding empty
  role events and heartbeat comments; mark whether it was reasoning, text, or tool arguments.
- First useful output latency: origin to first visible non-reasoning text, or to a complete valid
  tool call ready for client execution, whichever occurs first. Report both clocks; no useful output
  before error/timeout is a failure, not a fast sample.
- Queue/load/render/cache/prefill/decode/cleanup durations: backend/runtime stage observations,
  reported separately from the client measurements. TTFT is not a pure prefill timer.
- Prompt throughput: actually evaluated prompt tokens divided by measured prefill time. Report
  retained reused, new, and replayed boundary tokens separately; do not count hits as evaluated work.
- Decode throughput: `(output_tokens - 1) / (last_token_time - first_token_time)` for at least two
  output tokens using actual token IDs, not SSE chunk counts. Record engine decode timers separately.
  Text-only and all-generated (including reasoning/tool payload) denominators are separate.
- Runtime-controlled loop time: sum of client request-to-terminal-response intervals over the frozen
  replay trace; tool intervals excluded. Live end-to-end task time includes tools and client work.
- Cancellation release: runtime-observed cancellation to safe release of the execution slot and
  request-owned capacity. Also record trigger-to-observation and observation-to-last-GPU-completion.
- Memory: owned live/peak budget, process footprint/RSS, GPU allocations, and system pressure/swap.
  Use documented macOS measurements with units and sampling cadence (target at most 100 ms);
  allocator peaks cover short transients missed by sampling. Do not sum overlapping unified-memory
  views. Coarse sampled RSS alone is insufficient to prove all resource bounds. For SC-005, sustained
  swap growth means system swap usage rises more than 256 MiB above the pre-run baseline and stays
  above it for at least 30 seconds. Run an otherwise identical idle control cohort and retain
  available process pageout evidence; ambiguous attribution is inconclusive, not a pass. Measure
  post-unload footprint 30 seconds after all in-flight work completes, using the same process
  footprint metric as the post-initialization unloaded baseline.
- Task quality: objective fixture-test success, tool correctness, structured-output validity,
  context-faithfulness probes, and numerical/state checks, each reported independently.

## Repetition and Analysis

For warm timing cells, use three unreported warmup requests per configuration and **30 paired
measured observations per engine/cell** distributed across at least three sessions. Seed identical
inputs; alternate order using recorded ABBA blocks or balanced randomization. For cache-hit samples,
reset or establish the exact intended prefix before each measured turn; never mix hit/miss regimes.
For cold-load cells, use ten fresh-process observations and label OS-cache state honestly.

Publish all observations, median, p95 (nearest-rank), paired differences/ratios, and session/block-aware
95% bootstrap intervals for median ratios. Ratios use candidate/reference times for latency and
candidate/reference throughput for decode. For gate claims, the confidence interval must remain
on the passing side of the stated median threshold; p95 gates use the reported p95 plus uncertainty.
Thirty observations are a minimum, not permission to call noisy differences decisive. Freeze a
maximum expanded sample count of 100 per timing cell before running; otherwise report inconclusive.

Retain failed, cancelled, timed-out, invalid-tool, and wrong-answer runs with their reasons. Never
drop slow observations merely as outliers. Timing gates require a valid-output cohort with no new
failure rate; publish the complete cohort and failure count alongside it. Live task timing must also
report the success count and timeout ceiling, so fewer successful tasks cannot look like speed.

Compute the loop ratio as the geometric mean of per-trace optimized/reference runtime-controlled
times over at least ten frozen distinct ten-turn traces, with 30 paired observations each. Do not
pool raw token rates across different lengths into one representative performance number.

## Correctness Gates

Freeze numerical tolerances per selected precision/model in the plan before comparing routes.
Use reference logits/continuation likelihood plus state-transition tests and code/task outcomes.
Greedy token equality is required only for declared exact-replay reference cases using identical
execution settings; different prefill batches can alter logits and must be judged by the frozen
numerical/quality contract. A fixed seed is not proof of bitwise cross-backend equality.

Include prefix off/on, boundary replay, branch isolation, eviction, cancellation, model replacement,
malformed schemas, stream fragmentation, UTF-8, tool-result linkage, pressure, and GPU-error recovery.
State tests require positive lawful reuse and deliberate invalidation cases. Supported schema validity
does not prove semantic tool choice; live tasks test that separately.

The 100 cache-state cases MUST include at least 20 positive exact-replay cases, 20 prefix edit/branch
cases, 20 compatibility-identity changes, 20 scope/isolation cases, and 20 eviction/cancellation/failure
cases. The 20 interoperability cases MUST cover both text modes, fragmented tool handoff and result
continuation, finish/usage reconstruction, malformed input, unsupported options/schemas, and overload.
The plan freezes the individual named fixtures; these category counts cannot be satisfied by rerunning
one happy-path fixture under different case names.

## Evidence and Release Disposition

Store run manifests, raw measurements, summary calculations, output/test identities, engine commands,
and failure ledger under a versioned ignored local results directory; publish only sanitized reviewed
artifacts for releases. Model weights, caches, credentials, and private source remain outside Git.
Document reproduction commands when the harness exists; do not present proposed CLI flags as working.

Ordinary CI runs portable contracts and documentation checks. Hardware qualification runs explicitly
on physical M4 Pro, not an arbitrary hosted macOS runner. Use the numerical gates in [spec.md](spec.md)
as the acceptance authority. Gate status is pass, fail, blocked, or inconclusive; no evidence means
blocked, not pass. A best-in-class claim additionally requires the matrix policy in the
[spec of specs](../README.md), not just the cache-off self-comparison.