# Vision

## One sentence

mrun is the local inference server built for coding agents on Apple Silicon: your agent's context
is computed once, reused exactly, and never pushes your Mac into swap.

## The problem

Coding agents (Claude Code, OpenCode, Codex, Aider, Cline, Zed) send the same 10–60K tokens of
system prompt, tool schemas and history on every turn, then append a little. On a laptop, prefill
dominates: a 30K-token re-prefill can cost tens of seconds before the first useful token. Agents
also branch (subagents, retries, edited history) and cancel constantly. General local servers
optimize single-prompt chat; agent loops expose their weak spots: cache misses nobody can explain,
memory blowups at long context, and slow interruption.

## Users

- **Primary**: a developer running a coding agent locally on an Apple Silicon Mac (first target:
  M4 Pro, 48 GB), privacy-sensitive or offline, who wants cloud-like responsiveness.
- **Secondary**: tool builders who need a fast, predictable local endpoint speaking the OpenAI and
  Anthropic APIs.

## What makes mrun best-in-class

Competitors already ship prefix caching (llama.cpp server, Ollama's MLX engine, oMLX, mlx-lm).
mrun has to be better on the dimensions agents feel, and prove it with a published benchmark:

1. **Time to first useful token on repeated agent turns.** Exact radix-tree prefix reuse across
   turns, branches and tool loops; restore is a memory copy, never a recompute.
2. **Explainable caching.** Every response reports cached vs processed tokens; every miss says
   where and why the prefix diverged (template drift, volatile header, model change). Users can
   fix their setup instead of guessing.
3. **Predictable memory.** Weights, KV and cache are budgeted at startup. Requests that cannot fit
   are rejected up front with the numbers. Pressure sheds cache, never swaps.
4. **Instant interruption.** Cancellation takes effect at the next prefill chunk or decode step.
5. **All three agent APIs, natively.** OpenAI Chat Completions, Anthropic Messages and stateless
   OpenAI Responses from one binary, so Claude Code, Codex and OpenCode all work.
6. **Delightful DX.** `mrun pull`, `mrun serve`, `mrun connect claude-code`, `mrun doctor`. One
   binary, zero config, clear errors.
7. **Fast engine.** MLX: the fastest Apple Silicon kernels, including MoE and quantized models.

## Non-goals

Web UI, server-side tool execution (MCP, shell, search), multi-model routing, distributed or
multi-GPU serving, training or fine-tuning, a custom quantization format, Ollama's `/api` surface,
stateful Responses (`previous_response_id` storage), Linux/CUDA performance work.

## Principles in practice

See the [constitution](../.specify/memory/constitution.md). In short: exactness before speed,
memory budgeted by construction, local and private by default, simple over clever.
