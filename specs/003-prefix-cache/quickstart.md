# Quickstart: Prefix Cache

## Automated

```sh
cargo test -p mrun --test prefix_tree_props --test prefix_mutations --test prefix_branches   # any OS
MRUN_TEST_MODELS=qwen3-8b,qwen3.6,gpt-oss cargo test -p mrun-engine --test prefix_equality -- --ignored  # macOS
```

## Manual (target Mac)

1. `mrun serve --model mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit`; run a Claude Code session for 10+ turns. `mrun ps` shows
   cached tokens per turn and no cause on plain appended turns.
2. Grow the main conversation past 80K tokens, let Claude Code start a subagent, then continue the
   main session: the returning turn's `x-mrun-cache-cached` (or `: mrun` comment) is ≥ the parent
   length (SC-002). Repeat with a title request.
3. Edit an earlier message and resend: cause `edit`, divergence at the edited message.
4. `mrun serve --model mlx-community/Qwen3.6-35B-A3B-4bit`; repeat 1–3. Plain turns show gap 0 or
   a gap within the previous reply (SC-002); an edited last message resumes at its start, also
   after a run of ≥ 10 tool turns (SC-006).
5. `curl -H 'x-mrun-cache: off' …` during a session: cause `disabled`; the next session turn still
   reuses its state.
6. `memory_pressure -S -l warn` while a long prefill runs: prefill yields at the next chunk end,
   branch bytes drop within 1 s and the request completes.
7. `mrun cache clear` while idle: `mrun ps` shows zero branches and an empty live path.
8. `mrun bench --suite agent --trace agent/opencode-refactor` with and without `--no-cache`
   (reported with SC-001).
