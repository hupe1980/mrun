# Vision

## One sentence

mrun is the local inference server built for coding agents on Apple Silicon: your agent's context
is computed once, reused exactly, and stays inside a memory budget fixed at startup.

## The problem

Coding agents (Claude Code, Codex, OpenCode, Aider, Cline, Zed) resend 15–60K tokens of system
prompt, tool schemas and history on every turn and append a little. On a laptop, reading that
prompt dominates. As an estimate to be replaced by `mrun bench` numbers: Qwen3-30B-A3B 4-bit
prefills ≈1,750 tok/s at 2K tokens on an M4 Max (mlx-lm BENCHMARKS.md), so an M4 Pro with half the
GPU cores should manage several hundred tok/s at depth: about a minute for a 30K-token
re-prefill. Agents also branch (subagents, title and background requests, retries, edited history) and cancel constantly. General
local servers optimize single-prompt chat; agent loops expose their weak spots: cache misses
nobody can explain, conversations lost when a subagent runs, memory blowups at long context,
clients that time out during long prefills, and slow interruption.

## Users

- **Primary**: a developer running a coding agent locally on an Apple Silicon Mac (first target:
  M4 Pro, 48 GB), privacy-sensitive or offline, who wants cloud-like responsiveness.
- **Secondary**: tool builders who need a fast, predictable local endpoint speaking the OpenAI and
  Anthropic APIs.

## What makes mrun best-in-class

Competitors ship prefix caching too: llama.cpp server, Ollama's MLX engine, oMLX, mlx-lm, LM
Studio, and SGLang's MLX backend with a radix cache. oMLX even reports divergence points. mrun has
to win on what agents feel, and prove it with a published, fair benchmark:

1. **Time to first token on repeated and branching turns.** Exact reuse across turns, subagents
   and tool loops, for attention and hybrid (linear-attention) models. Switching between a main
   conversation and a subagent moves state instead of discarding or duplicating it.
2. **Explainable caching.** Every response reports cached and processed tokens; every miss names
   the cause and the divergence point, in-band (headers or SSE comments) and in `mrun ps`.
3. **Predictable memory.** Weights, attention state and cache live inside a budget fixed at
   startup. Prompts that cannot fit are refused with the numbers; output limits are clamped, never
   silently truncated; memory pressure sheds cache first.
4. **Interruption that keeps progress.** Cancellation takes effect within a few hundred
   milliseconds at any context length, and work already done is kept for the retry.
5. **All three agent APIs, faithfully.** Anthropic Messages, OpenAI Chat Completions and stateless
   Responses, including MCP tools, Codex namespaces and deferred tool loading, from one binary.
6. **Delightful DX.** `mrun launch claude-code` takes a fresh install to a working agent: it offers
   to pull the model and start the server, then configures the client for that process.
7. **The best model that fits.** Default: Qwen3.6-35B-A3B (hybrid MoE, about 20 KiB of attention
   state per token, so its full 262K context fits a 48 GB Mac); Qwen3-Coder-30B-A3B as the
   attention-only alternative.

## Non-goals

Web UI, server-side tool execution (MCP, shell, search), multi-model routing, distributed or
multi-GPU serving, training or fine-tuning, a custom quantization format, Ollama's `/api` surface,
stateful Responses (`previous_response_id` storage), approximate or lossy context reuse,
Linux/CUDA performance work.

## Principles in practice

See the [constitution](../.specify/memory/constitution.md). In short: exactness before speed,
memory budgeted by construction, local and private by default, simple over clever.
