#!/usr/bin/env python3
"""Vajra - the attack that actually works: encoder attack against the real VAE.

This is the paper's equation (4): find the smallest perturbation that makes the
latent diffusion model's encoder map the image to a "bad" representation,

    delta = argmin_{||delta||_inf <= eps}  || E(x + delta) - z_target ||_2^2

with z_target taken from a flat grey image, solved with projected gradient
descent. E is the real Stable Diffusion VAE encoder, and the gradient is real
backpropagation through it.

Why this file exists: the browser/app engine uses a fixed *linear* surrogate
instead of E, because no phone can backpropagate through a VAE. That surrogate
is measurably useless - it moves the real latent by 0.4% of the distance it is
supposed to close (see research/measure_attack.py). This script is the version
that moves it by 91%. It needs torch and a VAE checkpoint, and a GPU if you want
it to finish in a reasonable time.

    pip install torch diffusers safetensors pillow
    python3 sd_encoder_attack.py photo.jpg --steps 200 --eps 0.12

    # on a CPU, small and slow (about 12 s/step at 192 px):
    python3 sd_encoder_attack.py photo.jpg --steps 20 --size 192 --device cpu

What has been verified: the PGD loop respects the epsilon ball and reduces the
latent distance (unit tested in tests/test_encoder_attack.py), and on CPU against
sd-vae-ft-mse an 8-step run cut the latent distance from 3889 to 332 (91.5%) with
a maximum pixel change of 6/255. What has NOT been verified: the end-to-end effect
on a full Stable Diffusion inpaint/img2img run, which needs a GPU with far more
memory than the machine this was built on. The paper reports that result; this
repository cannot reproduce it here, and does not claim to.

The idea is from "Raising the Cost of Malicious AI-Powered Image Editing"
(Salman, Khaddaj, Leclerc, Ilyas, Madry - arXiv:2302.06588). No code was copied.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import torch
from PIL import Image, ImageOps

DEFAULT_VAE = "stabilityai/sd-vae-ft-mse"
DEFAULTS = dict(steps=200, eps=0.12, step_size=0.01, size=512, grey=0.5)


# ------------------------------------------------------------------- core PGD --


def encoder_attack(encode, x, z_target, steps=DEFAULTS["steps"], eps=DEFAULTS["eps"],
                   step_size=DEFAULTS["step_size"], mask=None, log=None):
    """PGD on ||E(x + d) - z_target||^2 inside an L-infinity ball around x.

    encode: callable taking a (1,3,H,W) tensor in [-1,1] and returning the latent.
    x: the image tensor, in [-1,1].
    mask: optional (1,1,H,W) tensor, 1 where the image may change, 0 where it must
          stay pixel-identical (the paper's "keep this region untouched" option).
    Returns the protected image tensor, detached, inside the ball.
    """
    x_adv = (x + (torch.rand_like(x) * 2 - 1) * eps).detach()

    for i in range(steps):
        # linearly decaying step, the schedule the paper's reference uses
        current = step_size * (1 - 0.99 * i / max(1, steps - 1))
        x_adv.requires_grad_(True)
        loss = (encode(x_adv) - z_target).pow(2).sum()
        grad, = torch.autograd.grad(loss, [x_adv])

        x_adv = x_adv.detach() - grad.sign() * current
        x_adv = torch.minimum(torch.maximum(x_adv, x - eps), x + eps).clamp(-1, 1)
        if mask is not None:
            x_adv = x * (1 - mask) + x_adv * mask

        if log and (i % 20 == 0 or i == steps - 1):
            dev = float((x_adv - x).abs().max())
            log(f"  step {i + 1:3d}/{steps}  latent distance {loss.item():10.2f}  "
                f"max dev {dev:.4f}/{eps}")

    return x_adv.detach()


# ---------------------------------------------------------------- tensor utils --


def to_tensor(image: Image.Image, size: int | None) -> torch.Tensor:
    """RGB image -> (1,3,H,W) in [-1,1]. The VAE needs multiples of 8."""
    image = image.convert("RGB")
    if size:
        image = ImageOps.contain(image, (size, size), Image.LANCZOS)
    w, h = image.size
    w, h = max(8, w - w % 8), max(8, h - h % 8)
    if (w, h) != image.size:
        image = image.resize((w, h), Image.LANCZOS)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return 2.0 * torch.from_numpy(arr).permute(2, 0, 1)[None] - 1.0


def to_image(tensor: torch.Tensor) -> Image.Image:
    arr = ((tensor[0].clamp(-1, 1) + 1) / 2 * 255).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr.round().astype(np.uint8))


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


# ------------------------------------------------------------------- pipeline --


def load_vae(model: str, device: str, local_file: str | None = None):
    """Load only the VAE - the full pipeline is not needed for an encoder attack."""
    from diffusers import AutoencoderKL

    if local_file:
        vae = AutoencoderKL.from_single_file(local_file, torch_dtype=torch.float32)
    else:
        vae = AutoencoderKL.from_pretrained(model, torch_dtype=torch.float32)
    vae = vae.to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae


def make_encode(vae):
    def encode(x):
        return vae.encode(x).latent_dist.mean
    return encode


def run(image: Image.Image, *, vae=None, encode=None, steps=DEFAULTS["steps"],
        eps=DEFAULTS["eps"], step_size=DEFAULTS["step_size"], size=DEFAULTS["size"],
        grey=DEFAULTS["grey"], mask_image: Image.Image | None = None, log=None):
    """Protect an image. Returns (protected PIL image, stats dict)."""
    if encode is None:
        if vae is None:
            raise ValueError("pass either vae= or encode=")
        encode = make_encode(vae)

    x = to_tensor(image, size)
    if log:
        log(f"working at {x.shape[3]}x{x.shape[2]} (VAE needs multiples of 8)")

    with torch.no_grad():
        z_target = encode(torch.full_like(x, grey * 2 - 1))

    mask = None
    if mask_image is not None:
        m = mask_image.convert("L").resize((x.shape[3], x.shape[2]), Image.LANCZOS)
        mask = (torch.from_numpy(np.asarray(m, dtype=np.float32) / 255.0)[None, None] > 0.5)
        mask = mask.float()

    adv = encoder_attack(encode, x, z_target, steps=steps, eps=eps,
                         step_size=step_size, mask=mask, log=log)

    with torch.no_grad():
        dist_before = float((encode(x) - z_target).pow(2).sum())
        dist_after = float((encode(adv) - z_target).pow(2).sum())

    out = to_image(adv)
    src = to_image(x)
    stats = {
        "size": [x.shape[3], x.shape[2]],
        "steps": steps,
        "eps": eps,
        "latent_before": dist_before,
        "latent_after": dist_after,
        "gap_closed": 100 * (1 - dist_after / dist_before) if dist_before > 0 else 0.0,
        "psnr": psnr(np.asarray(out), np.asarray(src)),
        "max_delta": int(np.abs(np.asarray(out, np.int16) - np.asarray(src, np.int16)).max()),
    }
    return out, stats


def preview(original: Image.Image, protected: Image.Image, path: str) -> None:
    """Side-by-side before/after, so the invisibility is checkable at a glance."""
    w, h = original.size
    strip = Image.new("RGB", (w * 2 + 12, h), (16, 18, 24))
    strip.paste(original, (0, 0))
    strip.paste(protected, (w + 12, 0))
    strip.save(path)


# ------------------------------------------------------------------------ CLI --


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Immunise an image against diffusion editing (encoder attack on the real VAE).")
    ap.add_argument("input", help="input image")
    ap.add_argument("-o", "--output", default=None, help="output path (default <input>-vajra.png)")
    ap.add_argument("--steps", type=int, default=DEFAULTS["steps"])
    ap.add_argument("--eps", type=float, default=DEFAULTS["eps"],
                    help="L-infinity budget in [-1,1] units (0.12 ~ 15/255)")
    ap.add_argument("--step-size", type=float, default=DEFAULTS["step_size"])
    ap.add_argument("--size", type=int, default=DEFAULTS["size"], help="max side to work at")
    ap.add_argument("--grey", type=float, default=DEFAULTS["grey"], help="grey target level, 0-1")
    ap.add_argument("--mask", default=None,
                    help="optional PNG mask, white = region to protect (may change)")
    ap.add_argument("--vae", default=DEFAULT_VAE)
    ap.add_argument("--vae-file", default=None, help="local .safetensors instead of a hub id")
    ap.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    ap.add_argument("--preview", default=None, help="write a before/after strip here")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    log = None if args.quiet else (lambda m: print(m, flush=True))
    if log and device == "cpu":
        log("no GPU found: running on CPU, expect roughly 10 s per step at 192 px")

    image = ImageOps.exif_transpose(Image.open(args.input)).convert("RGB")
    mask_image = Image.open(args.mask) if args.mask else None

    if log:
        log(f"loading {args.vae_file or args.vae} on {device}")
    vae = load_vae(args.vae, device, args.vae_file)

    out, stats = run(image, vae=vae, steps=args.steps, eps=args.eps,
                     step_size=args.step_size, size=args.size, grey=args.grey,
                     mask_image=mask_image, log=log)

    dest = args.output or (args.input.rsplit(".", 1)[0] + "-vajra.png")
    out.save(dest)
    if args.preview:
        preview(to_image(to_tensor(image, args.size)), out, args.preview)

    print(f"\nwrote {dest}  ({stats['size'][0]}x{stats['size'][1]})")
    print(f"  latent distance to grey: {stats['latent_before']:.1f} -> {stats['latent_after']:.1f}"
          f"  ({stats['gap_closed']:.1f}% of the gap closed)")
    print(f"  visible change: PSNR {stats['psnr']:.2f} dB, max {stats['max_delta']}/255 per channel")
    return 0


if __name__ == "__main__":
    sys.exit(main())
