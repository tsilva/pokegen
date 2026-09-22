#!/usr/bin/env python3
"""Train an unconditional VAE on Pokemon images: after training, pure noise decodes to Pokemon."""

import argparse
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from vae import (
    VAE,
    PerceptualLoss,
    augment,
    kl_per_sample,
    latent_stats,
    load_images,
    pick_device,
    save_decoder_checkpoint,
    save_full_checkpoint,
    save_grid,
)


def get_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data", type=Path, default=Path("data/artwork"), help="directory of PNGs")
    p.add_argument("--out", type=Path, default=Path("checkpoints"), help="checkpoint directory")
    p.add_argument("--sample-dir", type=Path, default=Path("logs/samples"))
    p.add_argument("--img-size", type=int, default=128)
    p.add_argument("--start-size", type=int, default=64,
                   help="train at this resolution first (whole image, downscaled), then grow to --img-size")
    p.add_argument("--grow-frac", type=float, default=0.5, help="fraction of epochs to spend at --start-size")
    p.add_argument("--cache-pad", type=int, default=136, help="cache resolution; randomly cropped to --img-size")
    p.add_argument("--epochs", type=int, default=600)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--latent-dim", type=int, default=256)
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--l1-weight", type=float, default=1.0, help="L1 term added to summed MSE; sharpens edges")
    p.add_argument("--perceptual-weight", type=float, default=0.02,
                   help="weight on VGG16 feature L1 (per-image sum); 0 disables")
    p.add_argument("--beta", type=float, default=0.5, help="final KL weight (recon is per-pixel squared error summed)")
    p.add_argument("--warmup-epochs", type=int, default=10, help="ramp beta 0 -> beta over this many epochs")
    p.add_argument("--free-bits", type=float, default=0.1, help="per-dim KL floor (nats); 0 disables")
    p.add_argument("--jitter", type=float, default=0.08, help="brightness/contrast augmentation strength")
    p.add_argument("--sample-every", type=int, default=25, help="epochs between sample grids")
    p.add_argument("--sample-count", type=int, default=36)
    p.add_argument("--truncate", type=float, default=0.85, help="latent noise scale for preview samples")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", help="auto | mps | cuda | cpu")
    p.add_argument("--amp", default="auto", choices=["auto", "off", "bf16", "fp16"],
                   help="mixed precision; auto = bf16 on CUDA, off elsewhere")
    p.add_argument("--channels-last", action="store_true", help="channels_last memory format (faster convs on CUDA)")
    p.add_argument("--compile", action="store_true", help="torch.compile the model (CUDA first run pays compile time)")
    return p.parse_args(argv)


def _setup_backend(device: torch.device):
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")


def _amp_dtype(choice: str, device: torch.device):
    if choice == "off":
        return None
    if choice == "auto":
        return torch.bfloat16 if device.type == "cuda" else None
    return torch.bfloat16 if choice == "bf16" else torch.float16


def main(argv=None):
    args = get_args(argv)
    device = pick_device(args.device)
    _setup_backend(device)
    amp_dtype = _amp_dtype(args.amp, device)
    scaler = torch.amp.GradScaler(device.type, enabled=amp_dtype == torch.float16)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)

    images, names = load_images(args.data, args.cache_pad)
    n = images.shape[0]
    print(f"{n} images from {args.data} | cache {args.cache_pad}px -> crop {args.img_size}px | device {device}")

    model = VAE(args.img_size, args.base_channels, args.latent_dim).to(device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    params = sum(p.numel() for p in model.parameters())
    print(f"VAE with {params / 1e6:.2f}M params, latent dim {args.latent_dim}")
    if amp_dtype is not None:
        print(f"amp: {str(amp_dtype).replace('torch.', '')}")
    if args.channels_last:
        print("channels_last enabled")

    perceptual = PerceptualLoss().to(device).eval() if args.perceptual_weight > 0 else None
    if perceptual is not None:
        if args.channels_last:
            perceptual = perceptual.to(memory_format=torch.channels_last)
        print(f"perceptual loss: VGG16 relu2_2+relu3_3 @ {perceptual.size}px, weight {args.perceptual_weight}")

    if args.compile:
        model = torch.compile(model)
        print("torch.compile enabled (first steps are slow)")
    raw_model = getattr(model, "_orig_mod", model)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.min_lr)
    steps = max(1, math.ceil(n / args.batch_size))
    amp_ctx = lambda: torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None)

    def sample(epoch, size):
        stats = latent_stats(raw_model, images, size)
        eps = torch.randn(args.sample_count, args.latent_dim, generator=gen)
        z = torch.tensor(stats["mean"]) + torch.tensor(stats["std"]) * args.truncate * eps
        save_grid(raw_model.decode(z.to(device), size=size), args.sample_dir / f"epoch-{epoch:04d}.png", nrow=6)
        print(f"  wrote {args.sample_dir / f'epoch-{epoch:04d}.png'}")

    t0 = time.time()
    grow_at = math.ceil(args.epochs * args.grow_frac)
    for epoch in range(1, args.epochs + 1):
        size = args.start_size if epoch <= grow_at else args.img_size
        beta = args.beta * min(1.0, epoch / args.warmup_epochs) if args.warmup_epochs > 0 else args.beta
        order = torch.randperm(n, generator=gen)
        recon_sum = l1_sum = kl_sum = perc_sum = 0.0
        model.train()
        for b in range(steps):
            idx = order[b * args.batch_size:(b + 1) * args.batch_size]
            x = augment(images[idx], size, args.jitter, gen).to(device, non_blocking=True)
            if args.channels_last:
                x = x.contiguous(memory_format=torch.channels_last)
            with amp_ctx():
                xhat, mu, logvar = model(x)
                mse = F.mse_loss(xhat.float(), x.float(), reduction="sum") / x.shape[0]
                l1 = F.l1_loss(xhat.float(), x.float(), reduction="sum") / x.shape[0]
                recon = mse + args.l1_weight * l1
                if perceptual is not None:
                    perc = perceptual(xhat, x)
                    recon = recon + args.perceptual_weight * perc
                    perc_sum += perc.item()
                kl = kl_per_sample(mu.float(), logvar.float(), args.free_bits)
                loss = recon + beta * kl
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            recon_sum += mse.item()
            l1_sum += l1.item()
            kl_sum += kl.item()
        sched.step()

        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            perc_str = f"  perc {perc_sum / steps:8.0f}" if perceptual is not None else ""
            print(
                f"epoch {epoch:3d}/{args.epochs}  {size:3d}px  mse {recon_sum / steps:.4f}  l1 {l1_sum / steps:.4f}"
                f"{perc_str}  kl {kl_sum / steps:6.1f}  beta {beta:.2f}  lr {sched.get_last_lr()[0]:.1e}"
                f"  {time.time() - t0:5.0f}s"
            )
        if epoch == 1 or epoch % args.sample_every == 0 or epoch == args.epochs:
            sample(epoch, size)

    save_full_checkpoint(args.out / "vae.pt", raw_model)
    stats = latent_stats(raw_model, images, args.img_size)
    save_decoder_checkpoint(args.out / "decoder.pt", raw_model, stats)
    use = sum(1 for s in stats["std"] if s > 0.2)
    print(f"aggregate posterior: {use}/{args.latent_dim} dims active (std > 0.2)")
    print(f"saved {args.out / 'vae.pt'} and {args.out / 'decoder.pt'} ({time.time() - t0:.0f}s total)")
    print("generate new Pokemon: python3 sample.py --n 64")


if __name__ == "__main__":
    main()
