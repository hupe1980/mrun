# Contributing to mrun

mrun is 0.x: minor versions may break.

You need Rust (the version in `rust-toolchain.toml`; rustup installs it) and, on a Mac, Xcode with
its Metal toolchain (`xcodebuild -downloadComponent MetalToolchain`), which compiles the engine's
GPU kernels into the binary. Nothing else: no C++ build. Without the Metal toolchain mrun
still builds and compiles its kernels when it starts (`mrun doctor` says so).

```sh
cargo build --release      # the binary, with the engine (nuvec)
cargo test --workspace     # every Rust suite, CPU and GPU (a Mac)
```

Before a pull request, also run what CI runs:

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo clippy --workspace --no-default-features --all-targets -- -D warnings   # the Linux agent
cargo test --workspace --no-default-features
```

Remote GPU mode: its tests run on loopback with a stub upstream and need no account. A live
check needs a RunPod key in `RUNPOD_API_KEY` or `./.env` (git-ignored; never commit or log it) and
a Linux build of the agent, cross-built on the Mac:

```sh
cargo install cargo-zigbuild && rustup target add x86_64-unknown-linux-gnu
cargo zigbuild --release --target x86_64-unknown-linux-gnu.2.28 --no-default-features -p mrun
```

Run the transport checks on a CPU pod first (`mrun cloud up --cpu --model stub`, $0.06/h), name test pods
`mrun-test-*`, and finish with `mrun cloud down`.

## Client gate

The client gate checks that Claude Code, Codex and OpenCode complete real
tasks through `mrun launch`. It needs a Mac with the Metal build, the three clients on `PATH` and
the model pulled:

```sh
cargo build --release
python3 scripts/client_gate.py --self-test         # checks the Wilson bound, no clients needed
python3 scripts/client_gate.py --model qwen3.6     # 10 tasks × 3 runs × 3 clients, a few hours
python3 scripts/client_gate.py --model qwen3.5 --clients codex --runs 1 --tasks 01,09
```

[`scripts/client_gate.py`](scripts/client_gate.py) uses only the Python standard library. Each
run copies a task from [`crates/mrun/tests/clients/tasks/`](crates/mrun/tests/clients/tasks/)
into a fresh temporary git repository, starts the client non-interactively, and lets the task's
`check.sh` decide pass or fail. It prints pass rates with Wilson 95 % intervals per client
(release needs a lower bound ≥ 0.8), writes every run to `client-gate.jsonl` (`--out`) and keeps
client logs in a temporary directory. A new task is a directory with `PROMPT.txt`, `check.sh`
and its fixture files; `check.sh` must also fail when the agent edits files it was told not to.

## Hardware reports

mrun is qualified on one Mac: the maintainer's M4 Pro with 48 GB. Smaller Macs are
simulated there with `--budget-gib` set to their measured working set, which reproduces fit, the
default model, the 64K floor and admission. What simulation cannot show is published as
unmeasured until someone measures it:

- cancellation latency on a slower GPU (M1, M2, M3),
- speed with the NAX kernels on an M5 (nuvec keeps them off until measured: `NUVEC_NAX=on` turns
  them on),
- real memory pressure on a 16 or 24 GB Mac (other apps, swap).

If you have one of these Macs, a report helps everyone:

1. Build mrun (see the [README](README.md#quick-start)) and pull the model this Mac defaults to
   (`mrun ls` shows it).
2. Run `mrun doctor --report`. It prints a Markdown block with the mrun version and numerics
   class, chip and GPU family, physical memory, Metal's recommended working set, macOS version, NAX
   availability, the pulled models with their identity, the calibration table and the doctor
   findings. It contains no prompts and shows your home directory as `~`; read it before you
   paste it.
3. Optionally run `mrun bench` (for example `mrun bench --suite agent`) and keep its
   `result.json` and `report.md`. Note whether the Mac was otherwise idle.
4. Open a [hardware report](https://github.com/hupe1980/mrun/issues/new?template=hardware-report.yml)
   issue with the block, the bench files and which models you ran.

Contributed results are listed on the
[benchmarks page](https://hupe1980.github.io/mrun/docs/benchmarks/) by contributor, chip and date,
with your handle if you want credit. They are community data, not release gates: a contributed
result never blocks a release, and it is never presented as the qualification measurement.
