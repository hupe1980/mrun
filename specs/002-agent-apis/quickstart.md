# Quickstart: Agent APIs

## Automated

```sh
cargo test -p mrun --test api_conformance     # ~100 cases on ScriptedEngine (any OS)
```

## Manual (macOS, real model)

```sh
M=mlx-community/Qwen3.6-35B-A3B-4bit     # repeat everything with mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit
mrun serve --model $M               # → listening on http://127.0.0.1:8484
curl -s localhost:8484/v1/chat/completions -d '{"model":"'$M'","messages":[{"role":"user","content":"hi"}],"stream":true}'
curl -s localhost:8484/v1/messages -H 'anthropic-version: 2023-06-01' -d '{"model":"'$M'","max_tokens":64,"messages":[{"role":"user","content":"hi"}]}'
curl -s localhost:8484/v1/responses -d '{"model":"'$M'","input":"hi","stream":false}'
curl -s localhost:8484/v1/models    # note context_window (CAP) and max_output_tokens (OUT)
```

## Client smoke tests (SC-001)

In `bench/tasks/smoke-fix-test/` (a tiny repo with a failing test), once with
`M=mlx-community/Qwen3.6-35B-A3B-4bit` and once with
`M=mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit` served. `CAP` and `OUT` are the values from
`/v1/models`; `mrun launch` (004) sets the same variables automatically.
Repeat steps 1–3 with gpt-oss-20b (Harmony syntax) and Devstral Small 2 (Mistral syntax) for
SC-007: protocol errors are gated, task completion is recorded. Repo ids: 004 research R4.

1. **Claude Code**:

   ```sh
   ANTHROPIC_BASE_URL=http://127.0.0.1:8484 ANTHROPIC_API_KEY= ANTHROPIC_AUTH_TOKEN=x \
   ANTHROPIC_MODEL=$M ANTHROPIC_DEFAULT_HAIKU_MODEL=$M \
   CLAUDE_CODE_MAX_CONTEXT_TOKENS=$CAP CLAUDE_CODE_MAX_OUTPUT_TOKENS=$OUT \
   ENABLE_TOOL_SEARCH=auto:10 \
   claude -p "make the tests pass"
   ```

   → tests pass, no protocol errors.
2. **Claude Code, crossing capacity**: same environment, Qwen3-Coder (`CAP` ≈105K on 48 GB), and
   a task that reads large files until the conversation exceeds `CAP`
   (`bench/tasks/smoke-long-session/`); → Claude Code auto-compacts, no "prompt is too long" error
   reaches the user, and the task completes.
3. **Codex**: a hand-written `[model_providers.mrun]` with `wire_api = "responses"` in a temporary
   `CODEX_HOME`, then `codex exec --profile mrun "make the tests pass"`. Also run once with
   another request holding the server so Codex waits in the queue for more than 300 s; → no idle
   timeout.
4. **OpenCode**: a hand-written `opencode.json` provider (`@ai-sdk/openai-compatible`,
   `baseURL`), then `opencode run "make the tests pass"`.

Repeat 5× each per model; record client versions.
