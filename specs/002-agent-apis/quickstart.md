# Quickstart: Agent APIs

## Automated

```sh
cargo test -p mrun --test api_conformance     # ~95 cases on ScriptedEngine (any OS)
```

## Manual (macOS, real model)

```sh
mrun serve --model qwen3-coder      # → listening on http://127.0.0.1:8484
curl -s localhost:8484/v1/chat/completions -d '{"model":"qwen3-coder","messages":[{"role":"user","content":"hi"}],"stream":true}'
curl -s localhost:8484/v1/messages -H 'anthropic-version: 2023-06-01' -d '{"model":"qwen3-coder","max_tokens":64,"messages":[{"role":"user","content":"hi"}]}'
curl -s localhost:8484/v1/responses -d '{"model":"qwen3-coder","input":"hi","stream":false}'
```

## Client smoke tests (SC-001)

In `bench/tasks/smoke-fix-test/` (a tiny repo with a failing test):

1. **Claude Code**: `mrun connect claude-code` env, then `claude -p "make the tests pass"` → tests
   pass, no protocol errors.
2. **Codex**: `mrun connect codex` profile, then `codex exec --profile mrun "make the tests pass"`.
3. **OpenCode**: `mrun connect opencode`, then `opencode run "make the tests pass"`.

Repeat 5× each; record client versions.
