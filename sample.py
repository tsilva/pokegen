#!/usr/bin/env python3
"""Generate new Pokemon from pure noise: z ~ N(0, I) -> VAE decoder -> PNG."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from vae import load_decoder, pick_device, save_grid


def get_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--ckpt", type=Path, default=Path("checkpoints/decoder.pt"))
    p.add_argument("--n", type=int, default=64, help="how many Pokemon to generate")
    p.add_argument("--out", type=Path, default=Path("data/generated"))
    p.add_argument("--seed", type=int, default=0, help="change for a fresh batch from the same decoder")
    p.add_argument("--truncate", type=float, default=0.85, help="<1 = safer/less varied samples")
    p.add_argument("--no-calibrate", action="store_true",
                   help="sample plain z ~ N(0, I) instead of the decoder's trained latent spread")
    p.add_argument("--scale", type=int, default=4, help="upscale factor for saved PNGs")
    p.add_argument("--keep-bg", action="store_true", help="keep the white background (no alpha cutout)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="auto", help="auto | mps | cuda | cpu")
    return p.parse_args(argv)


def to_rgba(img: torch.Tensor, keep_bg: bool) -> Image.Image:
    arr = ((img.detach().cpu().float().permute(1, 2, 0).numpy() + 1) * 127.5).round().clip(0, 255).astype(np.uint8)
    if keep_bg:
        return Image.fromarray(arr)
    rgb = arr.astype(np.float32)
    whitest = rgb.min(axis=2)
    alpha = np.clip((250.0 - whitest) / 10.0, 0.0, 1.0) * 255.0
    return Image.fromarray(np.dstack([arr, alpha.astype(np.uint8)]), "RGBA")


def main(argv=None):
    args = get_args(argv)
    device = pick_device(args.device)
    decoder, ckpt = load_decoder(args.ckpt, device)
    config = ckpt["config"]
    stats = ckpt.get("latent_stats")
    print(f"decoder from {args.ckpt} | latent {config['latent_dim']} | img {config['img_size']}px | device {device}")

    gen = torch.Generator().manual_seed(args.seed)
    eps = torch.randn(args.n, config["latent_dim"], generator=gen)
    if stats and not args.no_calibrate:
        mean = torch.tensor(stats["mean"], dtype=torch.float32)
        std = torch.tensor(stats["std"], dtype=torch.float32)
        z = mean + std * eps * args.truncate
        mode = f"calibrated noise x{args.truncate:g}"
    else:
        z = eps * args.truncate
        mode = f"plain N(0, I) x{args.truncate:g}"
    print(f"sampling {args.n} latents: {mode}")

    args.out.mkdir(parents=True, exist_ok=True)
    scale = max(1, args.scale)
    items = []
    for start in range(0, args.n, args.batch_size):
        chunk = z[start:start + args.batch_size].to(device)
        images = decoder(chunk)
        for i, img in enumerate(images, start=start + 1):
            im = to_rgba(img, args.keep_bg)
            if scale > 1:
                im = im.resize((im.width * scale, im.height * scale), Image.LANCZOS)
            name = f"{i:03d}.png"
            im.save(args.out / name)
            items.append({"index": i, "file": name})

    save_grid(decoder(z[:64].to(device)), args.out / "grid.png", nrow=8, scale=scale)

    manifest = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "checkpoint": str(args.ckpt),
        "latent_dim": config["latent_dim"],
        "img_size": config["img_size"],
        "seed": args.seed,
        "truncate": args.truncate,
        "mode": mode,
        "count": args.n,
        "items": items,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {args.n} PNGs + grid.png + manifest.json in {args.out}")
    print("reload the Pokedex and switch to the Generated tab to see them")


if __name__ == "__main__":
    main()
