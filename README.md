# mrun

A planned Rust local LLM runtime focused on low-latency coding-agent work on Apple Silicon.
Primary qualification target: M4 Pro with 48 GB unified memory.

**Status: foundations only.** The repository contains Rust configuration, identity, accounting,
owner-lifecycle and qualification-test scaffolding. No inference backend, CLI or qualified model
is available, and no performance results are claimed. Candle is the planned tensor foundation;
it is not integrated yet.

## Development

Use Rust 1.94.0 with rustfmt and Clippy, as pinned in the repository.

```sh
cargo test --locked --lib
cargo check --locked --all-targets --all-features
```

The portable foundation suite, including the admission policy and qualification-report contracts,
passes. Real system observations remain unavailable, so admission fails closed until a backend and
its resource controls are implemented; no model profile is eligible for serving.

The `mrun` package at version `0.0.1` has no stable public API.
The project repository is [hupe1980/mrun](https://github.com/hupe1980/mrun).

## License

Project licensing intent is MIT OR Apache-2.0; see [LICENSE-MIT](LICENSE-MIT) and
[LICENSE-APACHE](LICENSE-APACHE). Model weights and third-party engines retain their own licenses.