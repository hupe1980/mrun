"""Greedy parity references from mlx-lm for `crates/mrun-engine/tests/conformance.rs` (SC-001).

Run with mlx-lm 0.32.0 on MLX 0.32.4 (the versions mrun pins), from the repository root:

    python -I scripts/gen_reference.py mlx-community/Qwen3-8B-4bit [--out DIR] [--tokens 256]

Writes `crates/mrun-engine/tests/reference/<name>.json` (`<name>` is the repo's last path part;
a Hugging Face cache snapshot directory names its repo):

- `prompts`: 22 prompts of 24 to 8,192 tokens: the model's chat template around a slice of this
  repository's specification text (`specs/**/*.md`, sorted), lengths chosen around the chunk
  boundaries (1,536, 2,048) and up to 8K;
- per prompt, `tokens`: 256 greedy ids as mlx-lm's `generate_step` produces them
  (`prefill_step_size` 2048: every prompt token but the last prefilled in chunks of 2,048, the
  last fed alone, end tokens not stopping generation), and `top2`: the two most likely tokens and
  their log-probabilities (`logits - logsumexp(logits)`) at every generated position;
- `eos`: mlx-lm's end-token set (`TokenizerWrapper.eos_token_ids`);
- `numerics`: the class of the MLX it ran on (GPU architecture; NAX is not exposed by MLX's
  Python API, so it is recorded as the build's: off below macOS 26.2 SDKs), with the versions.
"""

import argparse
import json
import platform
import sys
from pathlib import Path

import mlx.core as mx
import mlx_lm
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler
from mlx_lm.utils import load

ROOT = Path(__file__).resolve().parent.parent
LENGTHS = [
    24, 64, 200, 500, 1000, 1535, 1536, 1537, 2047, 2048, 2049, 2500, 3000, 3500, 4096,
    4350, 5000, 6000, 6500, 7000, 7800, 8192,
]
MARKER = "@@@MRUN-PROMPT@@@"


def corpus_ids(tokenizer):
    text = "\n\n".join(
        p.read_text(encoding="utf-8") for p in sorted((ROOT / "specs").rglob("*.md"))
    )
    return tokenizer.encode(text, add_special_tokens=False)


def frame(tokenizer):
    """Template token ids before and after the user's text."""
    messages = [{"role": "user", "content": "Summarize this text.\n\n" + MARKER}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    before, after = text.split(MARKER)
    return (
        tokenizer.encode(before, add_special_tokens=False),
        tokenizer.encode(after, add_special_tokens=False),
    )


def prompts(tokenizer):
    ids = corpus_ids(tokenizer)
    before, after = frame(tokenizer)
    out = []
    offset = 0
    for length in LENGTHS:
        n = length - len(before) - len(after)
        assert n > 0, length
        if offset + n > len(ids):
            offset = 0
        out.append(before + ids[offset : offset + n] + after)
        offset += n
    return out


def run(model, prompt, tokens):
    sampler = make_sampler(temp=0.0)
    ids, top2 = [], []
    for token, logprobs in generate_step(
        mx.array(prompt), model, max_tokens=tokens, sampler=sampler, prefill_step_size=2048
    ):
        order = mx.argsort(-logprobs.astype(mx.float32))[:2]
        values = logprobs[order].astype(mx.float32)
        ids.append(int(token))
        top2.append([[int(t), float(v)] for t, v in zip(order.tolist(), values.tolist())])
    return ids, top2


def reference_name(model):
    """`org/name` → `name`; a Hugging Face cache snapshot directory → its repo's name."""
    parts = Path(model.rstrip("/")).parts
    for part in parts:
        if part.startswith("models--"):
            return part.split("--")[-1]
    return parts[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Hugging Face repo id or local directory")
    parser.add_argument("--out", type=Path, default=ROOT / "crates/mrun-engine/tests/reference")
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None, help="only the first N prompts")
    args = parser.parse_args()
    if mx.__version__ != "0.32.4" or mlx_lm.__version__ != "0.32.0":
        sys.exit(f"MLX 0.32.4 and mlx-lm 0.32.0 required, found {mx.__version__}, {mlx_lm.__version__}")

    model, tokenizer = load(args.model)
    cases = []
    for i, prompt in enumerate(prompts(tokenizer)[: args.limit]):
        tokens, top2 = run(model, prompt, args.tokens)
        cases.append({"prompt": prompt, "tokens": tokens, "top2": top2})
        print(f"prompt {i}: {len(prompt)} tokens -> {len(tokens)} ids", file=sys.stderr)
        mx.clear_cache()
    info = mx.device_info()
    result = {
        "model": args.model,
        "mlx": mx.__version__,
        "mlx_lm": mlx_lm.__version__,
        "prefill_step_size": 2048,
        "numerics": {"gpu_arch": info.get("architecture", "cpu"), "nax": False},
        "machine": {"chip": info.get("device_name"), "macos": platform.mac_ver()[0]},
        "eos": sorted(int(t) for t in tokenizer.eos_token_ids),
        "cases": cases,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{reference_name(args.model)}.json"
    path.write_text(json.dumps(result, separators=(",", ":")))
    print(f"wrote {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
