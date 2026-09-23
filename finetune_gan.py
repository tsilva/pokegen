#!/usr/bin/env python3
"""Adversarial decoder fine-tune: teaches a trained VAE's decoder to synthesize sharp detail.

The encoder stays frozen; the decoder trains against a small PatchGAN critic plus the
original reconstruction terms (so latents keep their meaning). Sampling code is unchanged.
"""

import argparse
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from vae import (
    VAE,
    PerceptualLoss,
    augment,
    latent_stats,
    load_images,
    pick_device,
    save_decoder_checkpoint,
    save_full_checkpoint,
    save_grid,
)


def get_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--ckpt", type=Path, default=Path("checkpoints/deep2/vae.pt"), help="trained VAE to fine-tune")
    p.add_argument("--data", type=Path, default=Path("data/artwork"))
    p.add_argument("--out", type=Path, default=Path("checkpoints/gan"))
    p.add_argument("--sample-dir", type=Path, default=Path("logs/samples-gan"))
    p.add_argument("--cache-pad", type=int, default=136)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4, help="decoder learning rate")
    p.add_argument("--d-lr", type=float, default=2e-4, help="critic learning rate")
    p.add_argument("--adv-weight", type=float, default=0.1, help="hinge GAN loss weight (mean-based losses)")
    p.add_argument("--l1-weight", type=float, default=0.5)
    p.add_argument("--perceptual-weight", type=float, default=0.04)
    p.add_argument("--jitter", type=float, default=0.08)
    p.add_argument("--sample-every", type=int, default=50)
    p.add_argument("--sample-count", type=int, default=36)
    p.add_argument("--truncate", type=float, default=0.85)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", help="auto | mps | cuda | cpu")
    p.add_argument("--amp", default="auto", choices=["auto", "off", "bf16", "fp16"])
    p.add_argument("--channels-last", action="store_true")
    return p.parse_args(argv)


class Discriminator(nn.Module):
    """PatchGAN critic: 128px in -> 8x8 patch logits, spectral-normalized."""

    def __init__(self, base: int = 64):
        super().__init__()
        widths = [3, base, base * 2, base * 4, base * 8]
        layers = []
        for cin, cout in zip(widths, widths[1:]):
            layers += [spectral_norm(nn.Conv2d(cin, cout, 4, stride=2, padding=1)), nn.LeakyReLU(0.2, inplace=True)]
        layers.append(nn.Conv2d(widths[-1], 1, 3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def diff_augment(x: torch.Tensor, jitter: float = 0.1, max_shift: int = 6, cutout: bool = True) -> torch.Tensor:
    """Differentiable, identical-in-kind augmentation for real and fake critic inputs."""
    if random.random() < 0.5:
        x = torch.flip(x, dims=[3])
    b = 1 + (torch.rand(x.shape[0], 1, 1, 1, device=x.device) * 2 - 1) * jitter
    c = 1 + (torch.rand(x.shape[0], 1, 1, 1, device=x.device) * 2 - 1) * jitter
    s = 1 + (torch.rand(x.shape[0], 1, 1, 1, device=x.device) * 2 - 1) * jitter
    x = x * b
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - mean) * c + mean
    gray = x.mean(dim=1, keepdim=True)
    x = gray + (x - gray) * s
    if max_shift > 0 and random.random() < 0.3:
        dy = int(torch.randint(-max_shift, max_shift + 1, (1,)))
        dx = int(torch.randint(-max_shift, max_shift + 1, (1,)))
        h, w = x.shape[-2:]
        x = F.pad(x, (max_shift, max_shift, max_shift, max_shift), value=1.0)
        x = x[:, :, max_shift + dy:max_shift + dy + h, max_shift + dx:max_shift + dx + w]
    if cutout and random.random() < 0.3:
        h, w = x.shape[-2:]
        size = h // 8
        top = int(torch.randint(0, h - size + 1, (1,)))
        left = int(torch.randint(0, w - size + 1, (1,)))
        x = x.clone()
        x[:, :, top:top + size, left:left + size] = 1.0
    return x.clamp(-1, 1)


def _amp_dtype(choice: str, device: torch.device):
    if choice == "off":
        return None
    if choice == "auto":
        return torch.bfloat16 if device.type == "cuda" else None
    return torch.bfloat16 if choice == "bf16" else torch.float16


def main(argv=None):
    args = get_args(argv)
    device = pick_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    amp_dtype = _amp_dtype(args.amp, device)
    scaler = torch.amp.GradScaler(device.type, enabled=amp_dtype == torch.float16)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    amp_ctx = lambda: torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    cfg = ckpt["config"]
    size = cfg["img_size"]
    model = VAE(**cfg).to(device)
    model.load_state_dict(ckpt["model"])
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    print(f"loaded {args.ckpt} | {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params | {size}px | {device}")

    images, names = load_images(args.data, args.cache_pad)
    print(f"{images.shape[0]} images from {args.data}")
    stats = latent_stats(model, images, size)
    z_mean = torch.tensor(stats["mean"]).to(device)
    z_std = torch.tensor(stats["std"]).to(device)

    perceptual = PerceptualLoss(size=0).to(device).eval() if args.perceptual_weight > 0 else None
    disc = Discriminator().to(device)
    if args.channels_last:
        disc = disc.to(memory_format=torch.channels_last)
    print(f"critic with {sum(p.numel() for p in disc.parameters()) / 1e6:.2f}M params")

    opt_g = torch.optim.AdamW(model.decoder.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.AdamW(disc.parameters(), lr=args.d_lr, betas=(0.5, 0.999))
    sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=args.epochs, eta_min=1e-6)
    sched_d = torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=args.epochs, eta_min=2e-6)
    steps = max(1, math.ceil(images.shape[0] / args.batch_size))

    def sample(epoch):
        eps = torch.randn(args.sample_count, cfg["latent_dim"], generator=gen)
        z = z_mean + z_std * args.truncate * eps.to(device)
        save_grid(model.decode(z, size=size), args.sample_dir / f"epoch-{epoch:04d}.png", nrow=6)
        print(f"  wrote {args.sample_dir / f'epoch-{epoch:04d}.png'}")

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        order = torch.randperm(images.shape[0], generator=gen)
        mse_sum = l1_sum = perc_sum = adv_sum = d_sum = 0.0
        model.train()
        for b in range(steps):
            idx = order[b * args.batch_size:(b + 1) * args.batch_size]
            x = augment(images[idx], size, args.jitter, gen).to(device, non_blocking=True)
            if args.channels_last:
                x = x.contiguous(memory_format=torch.channels_last)
            with torch.no_grad(), amp_ctx():
                mu, logvar = model.encoder(x)
                z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
            with amp_ctx():
                xhat = model.decoder(z, size=size)
                real_logits = disc(diff_augment(x))
                fake_logits = disc(diff_augment(xhat.detach()))
                d_loss = F.relu(1 - real_logits).mean() + F.relu(1 + fake_logits).mean()
            opt_d.zero_grad(set_to_none=True)
            scaler.scale(d_loss).backward()
            scaler.unscale_(opt_d)
            nn.utils.clip_grad_norm_(disc.parameters(), 5.0)
            scaler.step(opt_d)
            scaler.update()

            with amp_ctx():
                xhat = model.decoder(z, size=size)
                adv = -disc(diff_augment(xhat)).mean()
                mse = F.mse_loss(xhat.float(), x.float())
                l1 = F.l1_loss(xhat.float(), x.float())
                rec = mse + args.l1_weight * l1
                perc = torch.zeros((), device=device)
                if perceptual is not None:
                    perc = perceptual(xhat, x)
                    rec = rec + args.perceptual_weight * perc
                g_loss = rec + args.adv_weight * adv
            opt_g.zero_grad(set_to_none=True)
            scaler.scale(g_loss).backward()
            scaler.unscale_(opt_g)
            nn.utils.clip_grad_norm_(model.decoder.parameters(), 5.0)
            scaler.step(opt_g)
            scaler.update()
            mse_sum += mse.item()
            l1_sum += l1.item()
            perc_sum += perc.item()
            adv_sum += adv.item()
            d_sum += d_loss.item()
        sched_g.step()
        sched_d.step()

        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(
                f"epoch {epoch:3d}/{args.epochs}  mse {mse_sum / steps:.4f}  l1 {l1_sum / steps:.4f}"
                f"  perc {perc_sum / steps:.3f}  adv {adv_sum / steps:+.3f}  d {d_sum / steps:.3f}"
                f"  lr {sched_g.get_last_lr()[0]:.1e}  {time.time() - t0:5.0f}s"
            )
        if epoch == 1 or epoch % args.sample_every == 0 or epoch == args.epochs:
            sample(epoch)

    save_full_checkpoint(args.out / "vae.pt", model)
    save_decoder_checkpoint(args.out / "decoder.pt", model, stats)
    print(f"saved {args.out / 'vae.pt'} and {args.out / 'decoder.pt'} ({time.time() - t0:.0f}s total)")
    print(f"generate: python3 sample.py --ckpt {args.out / 'decoder.pt'} --n 64")


if __name__ == "__main__":
    main()
