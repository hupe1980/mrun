# arXiv SOTA Scan for mrun (2025–2026)

**Date**: 2026-10-15. **Status**: Research input, not a decision. **Scope**: This scan covers only
papers that [literature-study.md](literature-study.md) and
[next-features-2026-10.md](next-features-2026-10.md) do not already cover. Papers covered there
(RadixAttention, Marconi, Continuum, KVFlow, TraceLab, SuffixDecoding, EAGLE-3, DFlash, Cascade,
EcoSpec, MoE-Spec, KIVI, KVQuant, AttentionStore, LMCache, vllm-mlx, BaseRT, and others) are
referred to by name only.

**Method and evidence marks.** arxiv.org could not be reached from this environment: DNS failed
for WebFetch, and the proxy returned 403 for curl, export.arxiv.org, Semantic Scholar and alphaxiv.
So each ID was checked through web search. **[A]** means a search result pointed to an
`arxiv.org/{abs,pdf,html}/<id>` URL with the same title. The headline result comes from abstract or
snippet text. **[S]** means the claim appears only in a secondary summary. **[I]** marks this
document's own inference. No full PDF was read. All numbers are the authors' own figures, mostly
measured on datacenter GPUs, and none was measured on an M4 Pro.

## Ranked recommendations

| # | Action | Verdict | Why |
|---|---|---|---|
| 1 | **Make `mrun bench` replay token IDs, enforce the recorded outputs and tool gaps, and add a fidelity gate** (XPerf, AgentReplay, SiliconBench) | **Adopt now (M5)** | Replay that keeps only sequence lengths misstates prefix hits and MoE routing. SiliconBench found that fast Apple stacks can silently produce wrong output or take memory macOS needs. These are cheap to build in and make every later claim checkable. |
| 2 | **Keep eviction plain LRU plus a pinned boundary for the last tool call. Defer Marconi-style scoring for pure-attention models** (2609.28870, 2609.16215) | **Adopt now (M3), simplifies plan** | Two independent 2026 studies found LRU close to the offline optimum, and identical to the other policies for a single long session. Literature item A-3 is over-engineered for concurrency 1. |
| 3 | **Copy-aware speculation: a suffix-tree or n-gram drafter that also indexes tool-call renderings, with a B=1 MoE utility controller that can choose depth 0** (2604.26469, AgSpec 2610.01108, EVICT 2605.00342, 2609.22156, 2607.21804) | **Roadmap: first speculative feature** | This is the strongest evidence in the scan. On software-engineering tasks, Suffix Decoding was the best model-free method: 2.27× on SWE-bench repair and 1.51× on Aider edits. Those are realistic numbers, not the 5× headline. On Qwen3-30B-A3B, gains were 1.08–2.35×. The MoE cost of verification makes a controller mandatory. |
| 4 | **Hybrid-model checkpoint placement by exact dynamic programming over the expected overlap depth** (Sparse Prefix Caching 2605.05219) | **Roadmap (hybrid spec)** | This is exact, unlike Tail-Replay, DASC, LinearKV and HYPIC, which are all approximate. It gets the best Pareto frontier at low checkpoint budgets, which is the regime of a 48 GB Mac with checkpoints of about 61–147 MiB. |
| 5 | **Exact bf16 SSD tier** (2603.04428 on an M4 Pro; 2608.11668 on bimodal reuse) | **Roadmap (persistent tier)** | Measured on the same chip: about 500 ms reload against 15.7 s re-prefill per agent at 4K. Reuse has a long tail of returns after tens of minutes, which is exactly what RAM loses. Reject that paper's Q4 storage (+2.8–3.0% perplexity). Store exact bytes. |
| 6 | **Verify-and-rollback determinism for future batching** (LLM-42 2601.17768) | **Roadmap (parallel sessions)** | It gives determinism without batch-invariant kernels. It is not needed at batch 1. |
| 7 | **Q8 KV only, opt-in. No 4-bit** (2607.05399, SemKV 2608.28911, UltraQuant 2606.20474) | **Roadmap, unchanged** | Still no study measures long-context code under KV quantization. UltraQuant's 4-bit result relies on FP4/MFMA hardware (AMD CDNA4). |
| 8 | Position-independent or spliced KV reuse (EPIC, Irminsul, Leyline, LinearKV, HYPIC, Tail-Replay, AsymSpec) | **Reject** | None of these is exact: downstream KV was computed against different context. Irminsul and Leyline are also specific to MLA, while Qwen3 uses GQA. |
| 9 | MoE expert offload to SSD (FlashMoE, SpecMD, ZipMoE) | **Reject for 48 GB target** | Qwen3-Coder-30B-A3B at 4 bits (about 17 GB) fits entirely in memory. Offloading costs decode speed and energy (2508.06978). Revisit only for Macs with 16–24 GB or MoEs over 100B. |
| 10 | CPU+GPU split prefill (FusionML 2607.22785) | **Reject now; keep one insight** | +18–25% TTFT on a dense 7B. It has not been shown for 4-bit MoE, and decode does not gain. The useful insight: MLX lazy graphs serialize work across streams unless evaluation boundaries are explicit, which also matters for mrun's per-layer cancellation points. |

## 1. Prefix/KV reuse and agentic serving

| Paper | arXiv / date | Key result | Exact? | Verdict for mrun |
|---|---|---|---|---|
| Stateful Inference for Low-Latency Multi-Agent Tool Calling (Norgren, LayerScale) | 2605.26289 [A], May 2026 | Persistent per-session KV that advances by appending only, a radix cache and prompt-lookup speculation. Per turn it is 2.1× faster (6-turn workflow) and 4.2× faster (median turn of a 35-turn workflow) than vLLM and SGLang with caching disabled. | Yes | **Confirms the design.** This is essentially mrun's architecture. Hype flag: the baseline is "nothing served from cache", the author benchmarked their own system, and there is one author. |
| When Fancy Eviction Fails | 2609.28870 [A], Sep 2026 | Plain LRU "reliably converg[es] to the optimal offline ceiling on every trace"; no single best policy. | n/a | **Adopt now:** LRU as the baseline, with add-ons only where traces show a gap. |
| Where Should the KV Cache Live? (simulation) | 2609.16215 [A], Sep 2026 | For "one isolated long chat, every policy is identical and cheap LRU is right". Prefetch on a contended link costs more than it saves. | n/a | **Adopt** the conclusion. The tiering and PCIe results do not apply to unified memory. |
| Learning Agent Execution for KV-Cache Management (CacheScout?) | 2608.14624 [A title; content S] | Learns Markov agent transitions at runtime: +10–18 points hit rate and −29–38% per-turn latency on multi-agent workloads [S]. | Yes | **Roadmap, gated on traces.** It matters only if subagents alternate prefixes under memory pressure. |
| Irminsul: MLA-native position-independent caching | 2605.05696 [A], May 2026 | Content-defined chunk hashes plus closed-form re-rotation of the 64-dim RoPE part. Recovers about 83% of prompt tokens above the exact prefix. Cites operators reporting 10–16 s TTFT spikes on unchanged content. | **No** (chunk KV was computed under other context) | **Reject.** It is MLA-only and approximate. The motivation is still useful: agent harnesses that reorder content kill exact hits, so A-5 divergence diagnostics should be built. |
| Leyline: KV cache directives | 2606.01065 [A], Jun 2026 | Agent-issued span-replacement directives with δ-rotation of downstream KV. +11.2 points replay hit rate and up to 241 ms saved on SGLang. Argmax flips at low precision. | **No** (downstream KV attended the removed span) | **Reject.** Watch only as an opt-in idea if clients ever send edit directives. |
| EPIC (ICML 2025) | 2410.15332 [A], Oct 2024 | Position-independent caching with LegoLink recompute; up to 8× TTFT. | **No** (the authors claim accuracy, not exactness) | **Reject** (same class as CacheBlend). |
| KV Cache SSD tier: Agent Memory Below the Prompt | 2603.04428 [A], Feb 2026 | On an **M4 Pro**: disk Q4 KV reload takes about 500 ms against 15.7 s re-prefill (4K, per agent), up to 136× TTFT. Q4 changes perplexity by −0.7 to +3.0%. | No (Q4) | **Roadmap:** same mechanism, but with exact bf16 segments. The best on-target evidence for the SSD tier. |
| HBF Sucks? Full-stack characterization of high-bandwidth flash | 2608.11668 [A], Aug 2026 | Production reuse is bimodal: most within minutes, with a long tail after tens of minutes. | n/a | Supports the SSD tier for human pauses and next-day `--resume`. |

## 2. Speculative decoding

| Paper | arXiv / date | Key result | Verdict for mrun |
|---|---|---|---|
| An Empirical Study of Speculative Decoding on SE Tasks | 2604.26469 [A], Apr 2026 | Qwen3-32B and Llama-3.1-8B/70B. Model-free **Suffix Decoding is best**: mean 2.27× on SWE-bench repair and 1.51× on Aider Polyglot editing. Agent infinite loops add noise to the results. | **Roadmap, first.** Use these realistic numbers as the target, not SuffixDecoding's "5.3×". |
| AgSpec: retrieval-based SD in coding-agent pipelines | 2610.01108 [A], 1 Oct 2026 | Session, workspace and global corpora. **Opened files are indexed in the agent's own emission format.** A per-agent draft cap is tuned online. Up to 4.37× at batch 1, beating EAGLE-3 and 5 retrieval drafters. The abstract does not name the models or benchmarks [S]. | **Roadmap, adopt the idea.** Index both raw and JSON-escaped renderings of file content. This matters for Claude Code `Edit.old_string` inside JSON tool arguments, and less for Qwen3-Coder's raw XML parameters [I]. Hype flag: models and benchmarks are not named. |
| EVICT: adaptive verification for MoE SD | 2605.00342 [A], May 2026 | Lossless and training-free; trims the draft tree before verification. On **Qwen3-30B-A3B**: 1.08–2.35× over autoregressive decoding and about 1.21× over EAGLE-3. Cutting tokens 74.7% removes only 32.5% of the experts touched [S, via AcceptMoE]. | **Roadmap:** the cost-aware truncation rule goes into the utility controller. |
| The Limits of Speculation: bounding SD in MoE | 2609.22156 [A], Sep 2026 | Batch-1 cost model with **Qwen3-Coder plus EAGLE-3**. The optimal stopping rule is a linear boundary in (marginal cost, marginal accepted progress) space. | **Roadmap:** direct theory for the controller on the default model. |
| Adversarial Prompts for Acceptance Collapse | 2607.21804 [A], Jul 2026 | Inputs can drive acceptance toward 0. | **Adopt as a requirement:** the controller must be able to fall back to depth 0, so speculation is never slower than plain decode. |
| Self-Speculation for Faster Reasoning Models | 2608.20359 [A], 2026 | Calls suffix decoding "approximate" because drafts carry no probabilities. | **Correction [I]:** a deterministic draft is a point-mass proposal. Accept with probability p(x), otherwise resample from p with x removed and renormalized. This keeps the target distribution exactly. Greedy decoding is trivially exact up to floating-point differences from wider verification. Do not import this caveat. |
| ESP: training-free MTP via embedding probing (ICML 2026) | 2603.17942 [A], Mar 2026 | +7–8% acceptance length over lookahead decoding on Qwen3, and up to 15–19% throughput. | **Reject:** gains are marginal compared with n-gram drafting on code. |
| AsymSpec: context-asymmetric SD | 2608.26004 [A], Aug 2026 | The verifier sees compressed context: about 90% of full accuracy at 1.3–1.7× throughput. | **Reject:** lossy. |
| AgentSpec: SD for batch inference of agents | 2608.24004 [A], Aug 2026 | Up to 2.02× on code generation. | Later (batching). |
| DFlash for Qwen3-Coder-30B-A3B | (2602.06036, known) | A z-lab drafter exists. The MLX backend advises block_size ≤ 5 for quantized targets because quantized matmul slows at wider verify widths [S]. | Later. The note on block width carries over to any MLX verification path. |

## 3. KV cache compression and quantization

| Paper | arXiv / date | Key result | Verdict |
|---|---|---|---|
| Benchmarking KV-cache optimizations across quality and system performance | 2607.05399 [A], May 2026 | **KIVI4 has the most stable quality.** TurboQuant-3 and KIVI-2 degrade, especially on generation-heavy tasks. Compression ratio predicts end-to-end performance poorly. | Supports "Q8 first; Q4 experimental". |
| SemKV: the quality cliff | 2608.28911 [A], Aug 2026 | Llama-3.1-8B: uniform quantization looks like FP16 down to about 2.3 bits, then collapses at 2.0. | Avoid anything below 4 bits. |
| KVarN | 2606.03458 [A], Jun 2026 | Near-lossless at 2.3 bits; includes HumanEval. | Reject (needs new kernels; short-context code only). |
| Kitty | 2511.18643 [A], Nov 2025 | KIVI-K2V2 loses about 13 points at 32K-token AIME generation, and Kitty narrows that to 3–4. | Evidence that **long generations amplify KV-quantization error**, which is relevant to long agent sessions. |
| UltraQuant: 4-bit KV for context-heavy agents (AMD) | 2606.20474 [A], Jun 2026 | On replayed **Claude Code traces**: 2.71× (MiniMax-M2.5) and 4.38× (Qwen3-235B) qualified throughput against BF16. Uses FP4 with MFMA on CDNA4. | **Reject for now.** It depends on the hardware and the benefit is throughput under batching. The quality tables were not seen. Hype flag: the speedup comes from cache residency at high concurrency, not single-stream speed. |

Token eviction methods (H2O, SnapKV, StreamingLLM and their successors) remain **rejected** because they are not exact (literature section 2.3). No 2026 paper changes that.

## 4. Hybrid and linear attention, and prefix caching

| Paper | arXiv / date | Key result | Exact? | Verdict |
|---|---|---|---|---|
| Sparse Prefix Caching for Hybrid and Recurrent LLM Serving | 2605.05219 [A], May 2026 | Exact recurrent-state checkpoints at a few positions. Placement is an O(NM) DP over the overlap-depth distribution. Matches or beats block caching with far fewer checkpoints. | **Yes** | **Roadmap (hybrid spec):** feed the DP with overlap depths from mrun's miss diagnostics. Combine with the boundary-aligned chunk planner (next-features section 3). |
| Tail-Replay | 2608.30310 [A], Aug 2026 | No GDN checkpoints. Replays a 5–10% tail from a zero state and keeps 92.8–99.9% of quality. | **No** | Reject. |
| DASC: decay-aware state compression | 2608.30386 [A], Aug 2026 | Compresses GDN and KDA checkpoints using per-head decay. | **No** | Reject (same reasoning as for oMLX's int8 sidecars). |
| LinearKV / HYPIC | 2608.11231, 2607.01299 [A], Jul–Aug 2026 | Position-independent caching for hybrids. Mamba-2 reaches 86.8% of full quality. | **No** | Reject. |
| Gated DeltaNet / GDN-2 | 2412.06464 (ICLR 2025), 2605.22791 [A] | The GDN layer behind Qwen3-Next/3.5+. GDN-2 decouples erase from write. | n/a | Watch. A state layout change would change the checkpoint format. |

Implication [I]: the exact options for hybrids are checkpoints (Marconi admission plus 2605.05219 placement) and recomputation of the gap. No exact shortcut exists, so mrun's design (checkpoints at chunk ends planned in advance) is the state of the art for exact reuse.

## 5. Apple Silicon and unified memory

| Paper | arXiv / date | Key result | Verdict |
|---|---|---|---|
| SiliconBench (NeurIPS 2026 poster) | 2609.19169 [A], Sep 2026 | Benchmarks 9–10 Apple stacks on speed, **memory footprint** and **fidelity** (weighted F1 against an NVIDIA reference), with multi-turn agent prompts of about 4K tokens. "Only llama.cpp, vllm-metal and omlx pass all gates" [S]. | **Adopt now:** add fidelity and footprint cells to M5's `bench`, and expect mrun to be audited this way. |
| FusionML: prefill, not decode | 2607.22785 [A], Jul 2026 | AMX/GPU row-split linears give 1.18–1.25× TTFT on Qwen2.5-7B in mlx-lm with token-identical output. MLX's lazy scheduler serializes cross-stream work unless evaluation is eager. Decode gains nothing. | **Reject now** (dense fp16 only; one author). Use the scheduling insight in M1 profiling. |
| BaseRT on M5 Neural Accelerators | 2607.19438 [A], Jul 2026 | Neural Accelerators speed up prefill but do not raise the bandwidth ceiling for decode. | Note for M5 hardware: prefill gains come from MLX upgrades, not from mrun code [I]. |

## 6. MoE on memory-constrained devices

| Paper | arXiv / date | Key result | Verdict |
|---|---|---|---|
| FlashMoE | 2601.17063 [A], Jan 2026 | SSD expert cache with a learned replacement policy: hit rate up to 51% above LRU/LFU, and 2.6× against prior systems. | Reject for 48 GB. |
| SpecMD (Apple, ICML 2026) | 2602.03921 [A], Feb 2026 | Expert access defeats LRU and LFU. "Least-Stale" eviction: 88% hit rate with 5% cache on OLMoE. | Reject for now. Use it if a smaller-RAM tier is ever offered. |
| Reproducible Evaluation of MoE Expert Caching | 2608.07911 [A], Aug 2026 | Replay semantics and workload contamination confound policy comparisons. | Methodology note for any future offload work. |
| SSD Offloading for MoE Weights Considered Harmful (energy) | 2508.06978 [A], Aug 2025 | SSD offload is energy-inefficient. | Supports rejection on laptops. |
| ZipMoE | 2601.21198 [A], Jan 2026 | Lossless expert compression; edge I/O stalls of 38.5–80.1%. | Reject. |

## 7. Benchmarks and evaluation for agentic serving

| Paper | arXiv / date | Key result | Verdict |
|---|---|---|---|
| XPerf | 2608.20370 [A], Aug 2026 | Replays **recorded input token IDs** and can **force the recorded output**. End-to-end latency MAPE is under 3%. Diagnoses prompts that fail to use prefix caching and KV thrashing. | **Adopt now (M5):** use the same replay contract in `mrun bench`. |
| AgentReplay | 2609.32283 [A], Sep 2026 | Length-only replay drops information that affects prefix caching and **MoE routing**, which matters for Coder-30B-A3B. Small models can replay long-horizon traces. | **Adopt now:** replay token IDs, not lengths. |
| AgentServeSim | 2606.09613 [A], Jun 2026 | Replays SWE-bench and BFCL traces with per-turn tokens and **tool gaps**. | **Adopt:** keep tool-gap timing in traces, since human and tool pauses drive eviction. |
| AgentPerfBench | 2609.34683 [A], Sep 2026 | At C=80, coding planning calls have 2.8× higher median TTFT than chat. | Low fit (high concurrency). |
| SWE-Serve (NVIDIA) | 2609.26777 [A], Sep 2026 | Measures agents writing SGLang patches, not serving performance. | **Irrelevant** despite the name. |
| LLM-42: determinism by verify and rollback | 2601.17768 [A], Jan 2026 | Fast nondeterministic path plus verification and rollback. No batch-invariant kernels. | Roadmap: use it for the exactness gate once batching exists. |

## Hype flags and open caveats

- Large speedups (AgSpec 4.37×, Stateful Inference 4.2×, UltraQuant 4.38×) all compare against
  weak or unlike baselines: cache-off, autoregressive decoding, or BF16 at high concurrency.
  Realistic batch-1 code-editing gains are 1.5–2.3× (2604.26469).
- "Position-independent", "lossless splice" and "replay" methods report quality percentages, not
  equality. None of them passes mrun's exact-reuse rule.
- The SiliconBench "three stacks pass" result and the CacheScout attribution are second-hand [S].
- Not found: any study of KV quantization on long-context *code* agents, any MLX MTP or EAGLE-3
  paper for `qwen3_moe`, or any arXiv study of chunked, interruptible prefill latency on Apple GPUs.
  The cheapest experiments for mrun are still the ones listed under "Measure first" in
  next-features-2026-10.md.
