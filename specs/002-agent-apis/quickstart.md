# Quickstart: Agent APIs

## Automated

```sh
cargo test -p mrun --test api_conformance     # named conformance cases on ScriptedEngine (any OS)
```

## Manual (macOS, real model)

```sh
M=qwen3.6                           # repeat with qwen3-coder and gpt-oss
mrun serve --model $M               # → listening on http://127.0.0.1:8484
curl -s localhost:8484/v1/chat/completions -d '{"model":"'$M'","messages":[{"role":"user","content":"hi"}],"stream":true}'
curl -s localhost:8484/v1/messages -H 'anthropic-version: 2023-06-01' -d '{"model":"'$M'","max_tokens":64,"messages":[{"role":"user","content":"hi"}]}'
curl -s localhost:8484/v1/responses -d '{"model":"'$M'","input":"hi","stream":false}'
curl -s localhost:8484/v1/models    # note context_window (CAP) and max_output_tokens (OUT)
```

## Client gate (SC-001)

In `bench/tasks/smoke-fix-test/` (a tiny repo with a failing test), for each alias `qwen3.6`,
`qwen3-coder` and `gpt-oss` and each client, 3 runs each:

```sh
mrun launch claude-code -- -p "make the tests pass"
mrun launch codex -- exec "make the tests pass"
mrun launch opencode -- run "make the tests pass"
```

→ tests pass, no protocol errors. Also:

- **Claude Code crossing capacity**: `qwen3-coder` (`CAP` ≈105K on 48 GB) on
  `bench/tasks/smoke-long-session/`, which reads large files until the conversation exceeds `CAP`;
  → Claude Code compacts, no "prompt is too long" error reaches the user, the task completes.
- **Codex in the queue**: another request holds the server so Codex waits > 300 s; → no idle
  timeout.

Before `mrun launch` (004) lands, the same runs use hand-written settings:

1. **Claude Code**:
   ```sh
   ANTHROPIC_BASE_URL=http://127.0.0.1:8484 ANTHROPIC_API_KEY= ANTHROPIC_AUTH_TOKEN=x \
   ANTHROPIC_MODEL=$M ANTHROPIC_DEFAULT_HAIKU_MODEL=$M \
   CLAUDE_CODE_MAX_CONTEXT_TOKENS=$CAP CLAUDE_CODE_MAX_OUTPUT_TOKENS=$OUT \
   ENABLE_TOOL_SEARCH=auto:10 claude -p "make the tests pass"
   ```
2. **Codex**: a hand-written `[model_providers.mrun]` with `wire_api = "responses"` in a temporary
   `CODEX_HOME`, `-c web_search="disabled"`, then `codex exec --profile mrun "make the tests pass"`.
3. **OpenCode**: a hand-written `opencode.json` provider (`@ai-sdk/openai-compatible`,
   `baseURL`), then `opencode run "make the tests pass"`.

Record client versions. Before a release, repeat once with the latest Claude Code (npm `stable`
and `latest`), Codex and OpenCode (T022a).
