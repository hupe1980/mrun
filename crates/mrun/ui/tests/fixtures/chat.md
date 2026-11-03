# Why the parser fails

qwen3.6 · 2026-11-01

> Answer briefly.

**You**

`parser.rs`:
```rust
fn parse(s: &str) -> char { s.chars().next().unwrap() }
```

Why does this fail on empty input?

**qwen3.6**

<details><summary>Thought for 1.2 s</summary>

Empty string, so the iterator is empty.

</details>

`next()` returns `None` for an empty string, and `unwrap` panics.

*0 of 120 reused · 0.31 s to first token · 41.2 tok/s · finished*

**You**

Fix it.
