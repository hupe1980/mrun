"""Tiny random-weight checkpoints and mlx-lm reference outputs for the MoE and hybrid families.

Run with mlx-lm 0.32.0 on MLX 0.32.4 (the versions mrun pins), from any directory:

    python -I crates/mrun-engine/tests/fixtures/gen_tiny.py [fixture ...]

For each fixture it writes `<name>/config.json`, `<name>/model.safetensors` (built by mlx-lm's own
model classes with a fixed seed, quantized like `mlx_lm.convert` unless dense) and
`<name>/reference.json`: the prompt, and per device (`cpu`, and `gpu` when Metal is available)
the log-probabilities mlx-lm computes, as mlx-lm's `generate_step` drives the model (prefill all
but the last prompt token, then one-token steps):

- `single`: prefill in one call, then the last prompt token and 6 greedy steps (7 rows);
- `chunked`: prefill split before positions 5 and 11, then the last prompt token;
- `stepped`: every prompt token fed alone;
- `greedy`: the 6 greedy tokens of `single`.

Log-probabilities are `logits - logsumexp(logits)` in the logits dtype, as f32.
"""

import json
import shutil
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.utils import _get_classes, load_model, quantize_model, save_config

HERE = Path(__file__).resolve().parent
PROMPT = [(i * 37 + 11) % 128 for i in range(24)]
CHUNK_ENDS = [5, 11]
GREEDY_STEPS = 6

QWEN3_MOE = {
    "model_type": "qwen3_moe",
    "hidden_size": 64,
    "num_hidden_layers": 3,
    "intermediate_size": 96,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "num_experts": 8,
    "num_experts_per_tok": 4,
    "decoder_sparse_step": 1,
    "mlp_only_layers": [1],
    "moe_intermediate_size": 32,
    "rms_norm_eps": 1e-6,
    "vocab_size": 128,
    "rope_theta": 10000.0,
    "tie_word_embeddings": False,
    "max_position_embeddings": 4096,
    "norm_topk_prob": True,
    "torch_dtype": "bfloat16",
    "eos_token_id": 1,
}

QWEN3_5_TEXT = {
    "model_type": "qwen3_5_moe_text",
    "hidden_size": 64,
    "num_hidden_layers": 4,
    "full_attention_interval": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 32,
    "linear_num_key_heads": 2,
    "linear_num_value_heads": 4,
    # Dk = 128 takes the packed kernel on Metal, as Qwen3.6 does.
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 16,
    "linear_conv_kernel_dim": 4,
    "num_experts": 8,
    "num_experts_per_tok": 4,
    "moe_intermediate_size": 32,
    "shared_expert_intermediate_size": 32,
    "norm_topk_prob": True,
    "rms_norm_eps": 1e-6,
    "vocab_size": 128,
    "max_position_embeddings": 4096,
    "tie_word_embeddings": False,
    "rope_parameters": {
        "rope_type": "default",
        "rope_theta": 1000000,
        "partial_rotary_factor": 0.25,
        "mrope_section": [11, 11, 10],
        "mrope_interleaved": True,
    },
    "dtype": "bfloat16",
}

# The dense hybrid (Qwen3.5-9B's family): the same text config without experts.
QWEN3_5_TEXT_DENSE = {
    k: v
    for k, v in QWEN3_5_TEXT.items()
    if k
    not in (
        "num_experts",
        "num_experts_per_tok",
        "moe_intermediate_size",
        "shared_expert_intermediate_size",
        "norm_topk_prob",
    )
}
QWEN3_5_TEXT_DENSE.update({"model_type": "qwen3_5_text", "intermediate_size": 96})

QWEN3_5 = {
    "model_type": "qwen3_5",
    "text_config": QWEN3_5_TEXT_DENSE,
    "tie_word_embeddings": False,
    "eos_token_id": 1,
}

QWEN3_5_MOE = {
    "model_type": "qwen3_5_moe",
    "text_config": QWEN3_5_TEXT,
    "tie_word_embeddings": False,
    "eos_token_id": 1,
}


def build(config, seed, dtype=mx.bfloat16):
    """mlx-lm's model for `config`, randomly initialised from `seed`, in `dtype` except what
    `cast_predicate` keeps in f32 (A_log)."""
    mx.random.seed(seed)
    model_class, args_class = _get_classes(config)
    model = model_class(args_class.from_dict(config))
    # Small random weights keep activations in a sane range; norms around 1.
    params = []
    for name, p in tree_flatten(model.parameters()):
        if name.endswith("norm.weight") or name.endswith("layernorm.weight"):
            value = 1.0 + 0.1 * mx.random.normal(p.shape)
        elif name.endswith("A_log"):
            value = mx.log(mx.random.uniform(1.0, 16.0, p.shape))
        else:
            value = 0.25 * mx.random.normal(p.shape)
        keep = getattr(model, "cast_predicate", None)
        params.append((name, value.astype(dtype if keep is None or keep(name) else mx.float32)))
    model.load_weights(params)
    mx.eval(model.parameters())
    return model


def save(model, config, out):
    out.mkdir(parents=True, exist_ok=True)
    weights = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(str(out / "model.safetensors"), weights, metadata={"format": "mlx"})
    save_config(dict(config), out / "config.json")


def logprobs(logits):
    last = logits[:, -1, :]
    return (last - mx.logsumexp(last, axis=-1, keepdims=True)).astype(mx.float32)


def prefill(model, cache, tokens, ends=()):
    start = 0
    for end in list(ends) + [len(tokens)]:
        if end > start:
            model(mx.array(tokens[start:end])[None], cache=cache)
            mx.eval([c.state for c in cache])
            start = end


def reference(model):
    body, last = PROMPT[:-1], PROMPT[-1]

    cache = make_prompt_cache(model)
    prefill(model, cache, body)
    rows, greedy, token = [], [], last
    for step in range(GREEDY_STEPS + 1):
        lp = logprobs(model(mx.array([[token]]), cache=cache))
        rows.append(lp[0].tolist())
        if step < GREEDY_STEPS:
            token = mx.argmax(lp, axis=-1).item()
            greedy.append(token)

    cache = make_prompt_cache(model)
    prefill(model, cache, body, CHUNK_ENDS)
    chunked = logprobs(model(mx.array([[last]]), cache=cache))[0].tolist()

    cache = make_prompt_cache(model)
    for t in body:
        model(mx.array([[t]]), cache=cache)
    stepped = logprobs(model(mx.array([[last]]), cache=cache))[0].tolist()
    return {"single": rows, "greedy": greedy, "chunked": chunked, "stepped": stepped}


def write_reference(out):
    """Loads the saved fixture as `mlx_lm.load` would and records its outputs per device."""
    devices = {"cpu": mx.cpu}
    if mx.metal.is_available():
        devices["gpu"] = mx.gpu
    result = {"prompt": PROMPT, "chunk_ends": CHUNK_ENDS}
    for name, device in devices.items():
        mx.set_default_device(device)
        model, _ = load_model(out)
        result[name] = reference(model)
    mx.set_default_device(mx.gpu if mx.metal.is_available() else mx.cpu)
    (out / "reference.json").write_text(json.dumps(result))


def quantized(config, seed, out, strip_named=False):
    model = build(config, seed)
    model, qconfig = quantize_model(model, config, group_size=32, bits=4)
    if strip_named:
        # Like the published Qwen3.6 conversion: per-path settings only under
        # `quantization_config`, so a loader must infer them from the shapes.
        qconfig["quantization"] = {
            k: v for k, v in qconfig["quantization"].items() if not isinstance(v, dict)
        }
        qconfig["quantization_config"] = dict(qconfig["quantization_config"])
        save(model, qconfig, out)
        written = json.loads((out / "config.json").read_text())
        written["quantization_config"] = qconfig["quantization_config"]
        (out / "config.json").write_text(json.dumps(written, indent=4, sort_keys=True))
    else:
        save(model, qconfig, out)
    return model


def with_skipped_tensors(out):
    """Adds vision tower and MTP tensors (as the published Qwen3.5 repos carry) that a loader
    must skip."""
    path = out / "model.safetensors"
    weights = mx.load(str(path))
    weights["vision_tower.blocks.0.attn.proj.weight"] = mx.zeros((4, 4), dtype=mx.bfloat16)
    weights["vision_tower.patch_embed.proj.weight"] = mx.zeros((4, 2), dtype=mx.bfloat16)
    weights["mtp.fc.weight"] = mx.zeros((4, 4), dtype=mx.bfloat16)
    weights["mtp.layers.0.input_layernorm.weight"] = mx.ones((64,), dtype=mx.bfloat16)
    mx.save_safetensors(str(path), weights, metadata={"format": "mlx"})


def hf_layout(config, seed, out):
    """A dense checkpoint in the Hugging Face layout mlx-lm's `sanitize` converts: names under
    `model.language_model.`, conv kernels `[C, 1, K]`, decoder norms stored as `w - 1`, fused
    `experts.gate_up_proj`/`experts.down_proj`, plus vision and MTP tensors to drop. f32,
    because MLX's dense `gather_mm` runs only f32 on the CPU."""
    config = json.loads(json.dumps(config))
    config["text_config"]["dtype"] = "float32"
    model = build(config, seed, mx.float32)
    norms = (
        ".input_layernorm.weight",
        ".post_attention_layernorm.weight",
        "model.norm.weight",
        ".q_norm.weight",
        ".k_norm.weight",
    )
    weights = {}
    for name, p in tree_flatten(model.parameters()):
        if name.startswith("language_model.model."):
            name = "model.language_model." + name[len("language_model.model.") :]
        elif name.startswith("language_model."):
            name = name[len("language_model.") :]
        if "conv1d.weight" in name:
            p = p.moveaxis(2, 1)
        if any(name.endswith(s) for s in norms):
            p = p - 1.0
        weights[name] = p
    layers = config["text_config"]["num_hidden_layers"]
    for l in range(layers):
        prefix = f"model.language_model.layers.{l}.mlp"
        gate = weights.pop(f"{prefix}.switch_mlp.gate_proj.weight")
        up = weights.pop(f"{prefix}.switch_mlp.up_proj.weight")
        weights[f"{prefix}.experts.gate_up_proj"] = mx.concatenate([gate, up], axis=-2)
        weights[f"{prefix}.experts.down_proj"] = weights.pop(
            f"{prefix}.switch_mlp.down_proj.weight"
        )
    weights["model.visual.blocks.0.attn.proj.weight"] = mx.zeros((4, 4))
    weights["mtp.fc.weight"] = mx.zeros((4, 4))
    out.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(out / "model.safetensors"), weights, metadata={"format": "pt"})
    (out / "config.json").write_text(json.dumps(config, indent=4, sort_keys=True))


def main():
    if mx.__version__ != "0.32.4":
        sys.exit(f"MLX 0.32.4 required, found {mx.__version__}")
    fixtures = {
        "tiny_qwen3_5": lambda out: (quantized(QWEN3_5, 4, out), with_skipped_tensors(out)),
        "tiny_qwen3_moe": lambda out: quantized(QWEN3_MOE, 1, out),
        "tiny_qwen3_5_moe": lambda out: quantized(QWEN3_5_MOE, 2, out, strip_named=True),
        "tiny_qwen3_5_moe_hf": lambda out: hf_layout(QWEN3_5_MOE, 3, out),
    }
    # `gen_tiny.py name ...` regenerates only those fixtures.
    wanted = sys.argv[1:] or list(fixtures)
    for name, make in fixtures.items():
        if name not in wanted:
            continue
        out = HERE / name
        shutil.rmtree(out, ignore_errors=True)
        make(out)
        write_reference(out)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
