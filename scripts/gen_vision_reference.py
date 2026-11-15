"""Vision tower reference from mlx-vlm for `crates/mrun-engine/tests/vision.rs` (009 SC-001).

Run with mlx-vlm 0.7.12 on MLX 0.32.4, from the repository root:

    python -I scripts/gen_vision_reference.py <model dir> --out <file.safetensors>

It feeds mlx-vlm's `VisionModel` a fixed synthetic patch tensor (8 x 12 patches, non-square on
purpose) and writes `patches` in mrun's layout (`[n, t*p*p*c]`, channels last), `grid` (`[gh, gw]`)
and `features` (the tower's output, float32).
"""

import argparse
import glob
import json
import os

import mlx.core as mx
from mlx_vlm.models.qwen3_vl.config import VisionConfig
from mlx_vlm.models.qwen3_vl.vision import VisionModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gh", type=int, default=8)
    ap.add_argument("--gw", type=int, default=12)
    args = ap.parse_args()

    config = json.load(open(os.path.join(args.model, "config.json")))["vision_config"]
    vc = VisionConfig.from_dict(config)
    model = VisionModel(vc)
    weights = {}
    for f in sorted(glob.glob(os.path.join(args.model, "*.safetensors"))):
        for k, v in mx.load(f).items():
            if k.startswith("vision_tower."):
                weights[k[len("vision_tower.") :]] = v
    model.load_weights(list(model.sanitize(weights).items()))
    model.eval()

    t, p, c = vc.temporal_patch_size, vc.patch_size, vc.in_channels
    n = args.gh * args.gw
    i = mx.arange(n, dtype=mx.float32)[:, None]
    j = mx.arange(t * p * p * c, dtype=mx.float32)[None, :]
    mine = mx.sin(i * 0.37 + j * 0.011)  # [n, t*p*p*c] in (t, y, x, c) order, within [-1, 1]
    # mlx-vlm takes (c, t, y, x) rows.
    theirs = mine.reshape(n, t, p, p, c).transpose(0, 4, 1, 2, 3).reshape(n, -1)
    dtype = model.patch_embed.proj.weight.dtype
    grid = mx.array([[1, args.gh, args.gw]])
    features, _ = model(theirs.astype(dtype), grid)
    mx.save_safetensors(
        args.out,
        {
            "patches": mine,
            "grid": mx.array([args.gh, args.gw], dtype=mx.int32),
            "features": features.astype(mx.float32),
        },
    )
    print(f"{args.out}: features {features.shape}")


if __name__ == "__main__":
    main()
