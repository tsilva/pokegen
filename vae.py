#!/usr/bin/env python3
"""Shared pieces for the Pokemon VAE: image cache, augmentation, model, checkpoints."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

GROUPS = 8


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _to_rgb(path: Path) -> Image.Image:
    im = Image.open(path)
    if "A" in im.getbands():
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    return im.convert("RGB")


def load_images(data_dir: Path, pad: int, cache_dir: Path | None = None):
    """Load every PNG in data_dir, composited over white and resized to pad x pad.

    Returns (uint8 tensor (N, pad, pad, 3), names). Results are cached under
    data/cache/ so training epochs never decode the 475px originals again.
    """
    data_dir = Path(data_dir)
    cache_dir = Path(cache_dir) if cache_dir else data_dir.parent / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    arr_path = cache_dir / f"{data_dir.name}-{pad}.npy"
    names_path = cache_dir / f"{data_dir.name}-{pad}.json"

    files = sorted(data_dir.glob("*.png"))
    if not files:
        raise SystemExit(f"no PNGs found in {data_dir}")

    names = [f.stem for f in files]
    if arr_path.exists() and names_path.exists():
        cached = json.loads(names_path.read_text())
        if cached == names:
            return torch.from_numpy(np.load(arr_path)), names

    print(f"caching {len(files)} images from {data_dir} at {pad}px -> {arr_path}")
    arr = np.zeros((len(files), pad, pad, 3), dtype=np.uint8)
    for i, f in enumerate(files):
        arr[i] = np.asarray(_to_rgb(f).resize((pad, pad), Image.LANCZOS))
    np.save(arr_path, arr)
    names_path.write_text(json.dumps(names))
    return torch.from_numpy(arr), names


def augment(batch: torch.Tensor, crop: int, jitter: float, gen: torch.Generator) -> torch.Tensor:
    """(N, S, S, 3) uint8 -> (N, 3, crop, crop) float in [-1, 1] with resize/crop/flip/jitter.

    Images are resized to crop+8 first, so training at a smaller resolution than the
    cache (progressive growing) still sees the whole image, just downscaled.
    """
    x = batch.permute(0, 3, 1, 2).float()
    if x.shape[-1] != crop + 8:
        x = F.interpolate(x, size=crop + 8, mode="bilinear", align_corners=False, antialias=True)
    x = x.div_(127.5).sub_(1.0)
    n, _, h, w = x.shape
    tops = torch.randint(0, h - crop + 1, (n,), generator=gen)
    lefts = torch.randint(0, w - crop + 1, (n,), generator=gen)
    x = torch.stack([x[i, :, tops[i]:tops[i] + crop, lefts[i]:lefts[i] + crop] for i in range(n)])
    flip = torch.rand(n, generator=gen) < 0.5
    x[flip] = torch.flip(x[flip], dims=[3])
    if jitter > 0:
        gain = 1 + (torch.rand(n, 1, 1, 1, generator=gen) * 2 - 1) * jitter
        bias = (torch.rand(n, 1, 1, 1, generator=gen) * 2 - 1) * jitter
        x = torch.clamp(x * gain + bias, -1.0, 1.0)
    return x


def _block(cin: int, cout: int, up: bool = False, depth: int = 1):
    layers = []
    if up:
        layers.append(nn.Upsample(scale_factor=2, mode="nearest"))
    for i in range(depth):
        layers += [
            nn.Conv2d(cin if i == 0 else cout, cout, 3, stride=1 if (up or i > 0) else 2, padding=1),
            nn.GroupNorm(GROUPS, cout),
            nn.SiLU(),
        ]
    return layers


def _channels(base: int, stages: int):
    """Encoder widths per stage: base, base*2, ... capped at base*8."""
    return [min(base * 2 ** i, base * 8) for i in range(stages)]


class Encoder(nn.Module):
    def __init__(self, img_size: int = 64, base_channels: int = 32, latent_dim: int = 128, depth: int = 1):
        super().__init__()
        stages = int(math.log2(img_size)) - 2
        channels = _channels(base_channels, stages)
        blocks = []
        cin = 3
        for cout in channels:
            blocks.append(nn.Sequential(*_block(cin, cout, depth=depth)))
            cin = cout
        self.blocks = nn.ModuleList(blocks)
        self.mu = nn.Linear(channels[-1] * 16, latent_dim)
        self.logvar = nn.Linear(channels[-1] * 16, latent_dim)

    def forward(self, x: torch.Tensor):
        stages = int(math.log2(x.shape[-1])) - 2
        h = x
        for block in self.blocks[:stages]:
            h = block(h)
        h = h.flatten(1)
        return self.mu(h), self.logvar(h).clamp(-8.0, 8.0)


class Decoder(nn.Module):
    def __init__(self, img_size: int = 64, base_channels: int = 32, latent_dim: int = 128, depth: int = 1):
        super().__init__()
        stages = int(math.log2(img_size)) - 2
        top = _channels(base_channels, stages)[-1]
        self.img_size = img_size
        self.shape = (top, 4, 4)
        self.fc = nn.Linear(latent_dim, top * 16)
        blocks = []
        cin, width, res = top, top, 4
        while res < img_size:
            blocks.append(nn.Sequential(*_block(cin, width, up=True, depth=depth)))
            cin, res = width, res * 2
            width = max(base_channels, width // 2)
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Sequential(nn.Conv2d(cin, 3, 3, padding=1), nn.Tanh())

    def forward(self, z: torch.Tensor, size: int | None = None) -> torch.Tensor:
        stages = int(math.log2(size or self.img_size)) - 2
        h = self.fc(z).view(z.shape[0], *self.shape)
        for block in self.blocks[:stages]:
            h = block(h)
        return self.head(h)


class VAE(nn.Module):
    def __init__(self, img_size: int = 64, base_channels: int = 32, latent_dim: int = 128, depth: int = 1):
        super().__init__()
        self.config = dict(img_size=img_size, base_channels=base_channels, latent_dim=latent_dim, depth=depth)
        self.encoder = Encoder(**self.config)
        self.decoder = Decoder(**self.config)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encoder(x)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        return self.decoder(z, size=x.shape[-1]), mu, logvar

    @torch.no_grad()
    def decode(self, z: torch.Tensor, size: int | None = None) -> torch.Tensor:
        return self.decoder(z, size=size)


def kl_per_sample(mu: torch.Tensor, logvar: torch.Tensor, free_bits: float = 0.0) -> torch.Tensor:
    """KL(q(z|x) || N(0,I)) per sample. free_bits floors each latent dim's KL to keep it alive."""
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    if free_bits > 0:
        kl = kl.clamp(min=free_bits)
    return kl.sum(dim=1).mean()


class PerceptualLoss(nn.Module):
    """VGG16 feature L1; targets the structure a pixel loss averages away.

    size=0 compares at the native resolution (needed to guide detail); a smaller
    size is cheaper but blind to detail above that resolution.
    """

    def __init__(self, layers=(8, 15), size: int = 0):
        super().__init__()
        import torchvision

        vgg = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
        self.slice = vgg.features[: max(layers) + 1].eval()
        for p in self.slice.parameters():
            p.requires_grad_(False)
        self.layers = set(layers)
        self.size = size or None
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _prep(self, x: torch.Tensor) -> torch.Tensor:
        x = (x + 1) / 2
        if self.size is not None:
            x = F.interpolate(x, size=self.size, mode="bilinear", align_corners=False)
        return (x - self.mean) / self.std

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Mean L1 in VGG feature space (unitless); scale by pixel count to compare with pixel losses."""
        x, target = self._prep(x), self._prep(target)
        total = x.new_zeros(())
        for i, layer in enumerate(self.slice):
            x, target = layer(x), layer(target)
            if i in self.layers:
                total = total + F.l1_loss(x, target)
        return total


def save_grid(images: torch.Tensor, path: Path, nrow: int = 8, gap: int = 2, scale: int = 1):
    """(N, 3, H, W) in [-1, 1] -> contact sheet PNG."""
    x = ((images.detach().cpu().float() + 1) / 2).clamp(0, 1)
    n, _, h, w = x.shape
    rows = (n + nrow - 1) // nrow
    canvas = torch.ones(rows * (h + gap) - gap, nrow * (w + gap) - gap, 3)
    for i, img in enumerate(x):
        r, c = divmod(i, nrow)
        canvas[r * (h + gap):r * (h + gap) + h, c * (w + gap):c * (w + gap) + w] = img.permute(1, 2, 0)
    sheet = Image.fromarray((canvas.numpy() * 255).astype(np.uint8))
    if scale > 1:
        sheet = sheet.resize((sheet.width * scale, sheet.height * scale), Image.LANCZOS)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def save_full_checkpoint(path: Path, model: VAE):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": model.config}, path)


def save_decoder_checkpoint(path: Path, model: VAE, latent_stats=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"decoder": model.decoder.state_dict(), "config": model.config, "latent_stats": latent_stats}, path)


def load_decoder(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    decoder = Decoder(**ckpt["config"]).to(device)
    decoder.load_state_dict(ckpt["decoder"])
    decoder.eval()
    return decoder, ckpt


@torch.no_grad()
def latent_stats(model: VAE, images: torch.Tensor, size: int, batch_size: int = 64):
    """Per-dim mean/std of the aggregate posterior E_x[q(z|x)] over the dataset.

    The decoder is only trustworthy on the latent region the encoder actually uses.
    Sampling z = mean + std * N(0, I) keeps generation inside that region.
    """
    model.eval()
    device = next(model.parameters()).device
    off = 4
    total = 0
    s = torch.zeros(model.config["latent_dim"])
    ss = torch.zeros_like(s)
    for start in range(0, images.shape[0], batch_size):
        batch = images[start:start + batch_size].permute(0, 3, 1, 2).float()
        if batch.shape[-1] != size + 8:
            batch = F.interpolate(batch, size=size + 8, mode="bilinear", align_corners=False, antialias=True)
        x = batch[:, :, off:off + size, off:off + size].div_(127.5).sub_(1.0).to(device)
        mu, _ = model.encoder(x)
        mu = mu.cpu()
        s += mu.sum(0)
        ss += mu.pow(2).sum(0)
        total += mu.shape[0]
    mean = s / total
    std = (ss / total - mean.pow(2)).clamp_min(1e-6).sqrt()
    return {"mean": mean.tolist(), "std": std.tolist()}
