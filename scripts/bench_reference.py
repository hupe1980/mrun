"""mlx-lm 0.32.0 reference speeds for `bench_engine` (R16, SC-002, SC-003); run by
`scripts/bench_reference.sh`.

Feeds the same prompt ids as `crates/mrun-engine/examples/bench_engine.rs` (`prompt_ids`) through
`mlx_lm.generate.stream_generate` with `prefill_step_size` 2048 and greedy sampling, and reports
mlx-lm's own prompt speed and its generation speed (best of `--runs`) per depth as JSON on
stdout. mlx-lm divides n generated tokens by the n − 1 decode steps after the first; the decode
speed here is rescaled to steps per second, as `bench_engine` counts it.
"""

import argparse
import json
import platform
import subprocess
import sys

import mlx.core as mx
import mlx_lm
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.utils import load


def prompt_ids(n):
    return [1000 + (i * 7919 + 13) % 30000 for i in range(n)]


def power():
    try:
        out = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout
        return out.splitlines()[0].split("'")[1]
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--depths", default="1024,16384")
    parser.add_argument("--decode", type=int, default=128)
    parser.add_argument("--runs", type=int, default=2)
    args = parser.parse_args()
    if mx.__version__ != "0.32.4" or mlx_lm.__version__ != "0.32.0":
        sys.exit(f"MLX 0.32.4 and mlx-lm 0.32.0 required, found {mx.__version__}, {mlx_lm.__version__}")
    model, tokenizer = load(args.model)
    sampler = make_sampler(temp=0.0)
    results = []
    for depth in [int(d) for d in args.depths.split(",")]:
        best_p, best_d = 0.0, 0.0
        for _ in range(args.runs):
            last = None
            for response in stream_generate(
                model,
                tokenizer,
                prompt_ids(depth),
                max_tokens=args.decode,
                sampler=sampler,
                prefill_step_size=2048,
            ):
                last = response
            best_p = max(best_p, last.prompt_tps)
            # mlx-lm reports (n + 1) tokens over n decode steps; bench_engine counts the steps.
            n = last.generation_tokens
            best_d = max(best_d, last.generation_tps * (n - 1) / n)
            mx.clear_cache()
        print(f"depth {depth}: prompt {best_p:.1f} tok/s, decode {best_d:.1f} tok/s", file=sys.stderr)
        results.append({"depth": depth, "prompt_tps": best_p, "decode_tps": best_d, "decode_tokens": args.decode})
    info = mx.device_info()
    print(json.dumps({
        "tool": "mlx-lm",
        "machine": {
            "chip": info.get("device_name"),
            "memory_bytes": info.get("memory_size"),
            "macos": platform.mac_ver()[0],
            "power": power(),
            "mlx": mx.__version__,
            "mlx_lm": mlx_lm.__version__,
        },
        "model": args.model,
        "prefill_step_size": 2048,
        "results": results,
    }, indent=2))


if __name__ == "__main__":
    main()
