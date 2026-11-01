#!/usr/bin/env python3
"""Vajra Image Guard - the strong version: encoder attack against a real VAE.

The surrogate encoder in the app is a stand-in. This script does what the
PhotoGuard paper does: it backpropagates through the actual Stable Diffusion VAE
encoder and runs PGD so that the latent of the protected image is dragged towards
the latent of a plain grey target image. That is the attack with real teeth, and
it needs a CUDA GPU, `torch` and `diffusers` with a downloaded VAE checkpoint.

    pip install torch diffusers transformers pillow
    python3 sd_encoder_attack.py photo.jpg -o photo-vajra-strong.png --steps 200

IMPORTANT - read before you trust it: this file is NOT exercised by the test
suite, because this repository is built on machines without a GPU and without
torch installed. The maths is the same PGD loop the app uses (random init, sign
steps, decaying step size, L-infinity projection, masked region support); only the
encoder differs. Treat the output as research-grade, not as a shipped feature.

This is an independent implementation of the idea in
"Raising the Cost of Malicious AI-Powered Image Editing" (arXiv:2302.06588).
No code from that repository was copied.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import torch
from PIL import Image, ImageOps


def load_vae(model_id: str, device: str):
    """VAE only - the full pipeline is not needed for an encoder attack."""
    from diffusers import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(model_id, torch_dtype=torch.float16)
    return vae.to(device).eval()


def to_tensor(image: Image.Image) -> torch.Tensor:
    """RGB image -> [-1, 1] tensor, resized to a multiple of 32 (VAE requirement)."""
    w, h = image.size
    w, h = w - w % 32, h - h % 32
    image = image.resize((max(32, w), max(32, h)), Image.LANCZOS)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)[None]
    return 2.0 * tensor - 1.0


def latent_of(vae, x: torch.Tensor) -> torch.Tensor:
    with torch.autocast("cuda", dtype=torch.float16):
        return vae.encode(x.half()).latent_dist.mean


def encoder_attack(
    vae,
    x: torch.Tensor,
    target: torch.Tensor,
    steps: int = 200,
    eps: float = 0.12,
    step_size: float = 0.01,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """PGD towards the target latent, projected into an L-infinity ball around x.

    x and target are [-1, 1] tensors on the device. mask (1x1xHxW, 1 = keep) limits
    the perturbation to the region the caller wants to protect.
    """
    device = x.device
    x_adv = (x + (torch.rand_like(x) * 2 - 1) * eps).detach()
    tgt = latent_of(vae, target).detach()

    for i in range(steps):
        # step size decays linearly, exactly like the paper's schedule
        current = step_size - (step_size - step_size / 100) * (i / max(1, steps))
        x_adv.requires_grad_(True)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = (latent_of(vae, x_adv) - tgt).norm()
        grad, = torch.autograd.grad(loss, [x_adv])

        x_adv = x_adv.detach() - grad.sign() * current
        x_adv = torch.minimum(torch.maximum(x_adv, x - eps), x + eps)
        x_adv = x_adv.clamp(-1, 1)
        if mask is not None:
            x_adv = x * (1 - mask) + x_adv * mask
        if i % 25 == 0 or i == steps - 1:
            print(f"  step {i + 1:3d}/{steps}  latent distance {loss.item():.4f}", flush=True)
    return x_adv.detach()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Strong encoder attack against a Stable Diffusion VAE.")
    ap.add_argument("input")
    ap.add_argument("-o", "--output", default=None)
    ap.add_argument("--model", default="stabilityai/sd-vae-ft-mse", help="VAE checkpoint")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--eps", type=float, default=0.12, help="L-infinity budget in [-1,1] units")
    ap.add_argument("--step-size", type=float, default=0.01)
    ap.add_argument("--target-grey", type=float, default=0.5, help="target image grey level, 0-1")
    args = ap.parse_args(argv)

    if not torch.cuda.is_available():
        print("error: this script needs a CUDA GPU; use surrogate_pgd.py otherwise", file=sys.stderr)
        return 2

    device = "cuda"
    vae = load_vae(args.model, device)

    image = ImageOps.exif_transpose(Image.open(args.input)).convert("RGB")
    x = to_tensor(image).to(device)
    target = torch.full_like(x, args.target_grey * 2 - 1)

    print(f"attacking {image.size[0]}x{image.size[1]} with {args.steps} PGD steps")
    adv = encoder_attack(vae, x, target, steps=args.steps, eps=args.eps, step_size=args.step_size)

    out = ((adv[0].float().clamp(-1, 1) + 1) / 2).permute(1, 2, 0).cpu().numpy()
    out = (out * 255).round().astype(np.uint8)
    dest = args.output or (args.input.rsplit(".", 1)[0] + "-vajra-strong.png")
    Image.fromarray(out).save(dest)

    diff = np.abs(out.astype(np.int16) - np.asarray(Image.fromarray(out)))
    print(f"wrote {dest}")
    print(f"  output size {out.shape[1]}x{out.shape[0]} (multiple of 32, VAE constraint)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
