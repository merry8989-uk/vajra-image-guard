#!/usr/bin/env python3
"""Measure what actually happens inside the model - the harness behind the README table.

Compares three versions of the same picture in the Stable Diffusion VAE's latent
space, which is the only space that decides whether an AI edit will come out
normal or broken:

  1. the original, untouched
  2. the image the browser/app engine produces (docs/engine.js, via node)
  3. the image the real encoder attack produces (research/sd_encoder_attack.py)

For each one it reports how far the latent sits from the grey target, how visible
the change is (PSNR, max per-channel shift), and how much the VAE's decoded *view*
of the image moved - the last number is the one that predicts whether an editor
will still see your photo.

    pip install torch diffusers safetensors pillow
    python3 research/measure_attack.py --steps 8 --size 192

Numbers measured with this harness on a 2-core CPU box, 192 px, eps 0.12:

    original            latent 3889   psnr  inf     max 0/255   view shift  0.00/255
    app (surrogate)     latent 3864   psnr 43.85    max 7/255   view shift  2.18/255
    encoder attack (8)  latent  332   psnr 35.96    max 6/255   view shift 33.78/255

That is the whole story: for the same invisible pixel budget, the surrogate moves
the model's view by 2/255 and the real attack by 34/255.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sd_encoder_attack as sa  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENGINE = os.path.join(REPO, "docs", "engine.js")


def synthetic(size: int) -> Image.Image:
    """A structured stand-in photo: gradients, hard edges, a flat region."""
    y, x = np.mgrid[0:size, 0:size].astype(np.float64) / size
    r = 0.20 + 0.55 * x + 0.10 * np.sin(x * 40)
    g = 0.25 + 0.45 * y + 0.08 * np.cos(y * 33)
    b = 0.30 + 0.40 * (x + y) / 2 + 0.06 * np.sin((x + y) * 55)
    r[(x > 0.62) & (y < 0.35)] = 0.92
    g[(x > 0.62) & (y < 0.35)] = 0.88
    b[(x > 0.62) & (y < 0.35)] = 0.80
    flat = (x > 0.65) & (y > 0.62)
    r[flat], g[flat], b[flat] = 0.58, 0.62, 0.70
    return Image.fromarray((np.clip(np.stack([r, g, b], -1), 0, 1) * 255).round().astype(np.uint8))


def run_shipped_engine(img: Image.Image, workdir: str, epsilon_levels: float = 12.0) -> Image.Image:
    """Run docs/engine.js over the same pixels, so the comparison is like for like."""
    os.makedirs(workdir, exist_ok=True)
    arr = np.asarray(img.convert("RGB"))
    h, w = arr.shape[:2]
    src = os.path.join(workdir, "src.raw")
    out = os.path.join(workdir, "sur.raw")
    open(src, "wb").write(arr.tobytes())
    script = os.path.join(workdir, "sur.mjs")
    open(script, "w").write(f"""
import fs from 'node:fs';
import Vajra from {json.dumps(ENGINE)};
const raw = fs.readFileSync({json.dumps(src)});
const d = new Uint8ClampedArray({w} * {h} * 4);
for (let i = 0, p = 0; i < raw.length; i += 3, p += 4) {{
  d[p] = raw[i]; d[p+1] = raw[i+1]; d[p+2] = raw[i+2]; d[p+3] = 255;
}}
const res = Vajra.protect({{ data: d, width: {w}, height: {h} }},
                          {{ epsilon: {epsilon_levels}/255, iterations: 40 }});
const o = Buffer.alloc({w} * {h} * 3);
for (let i = 0, p = 0; i < res.data.length; i += 4, p += 3) {{
  o[p] = res.data[i]; o[p+1] = res.data[i+1]; o[p+2] = res.data[i+2];
}}
fs.writeFileSync({json.dumps(out)}, o);
fs.writeFileSync({json.dumps(os.path.join(workdir, "sur.json"))}, JSON.stringify(res.stats));
""")
    subprocess.run(["node", script], check=True)
    raw = np.frombuffer(open(out, "rb").read(), np.uint8).reshape(h, w, 3)
    return Image.fromarray(raw)


def montage(items, path):
    scale = 2
    pad, label_h = 8, 22
    w, h = items[0][0].size
    total = (w * scale + pad) * len(items) + pad
    canvas = Image.new("RGB", (total, h * scale + pad + label_h), (16, 18, 24))
    draw = ImageDraw.Draw(canvas)
    x = pad
    for im, label in items:
        canvas.paste(im.resize((w * scale, h * scale), Image.NEAREST), (x, label_h))
        draw.text((x, 5), label, fill=(220, 225, 235))
        x += w * scale + pad
    canvas.save(path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", default=None, help="image to use (default: a synthetic one)")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--size", type=int, default=192)
    ap.add_argument("--eps", type=float, default=0.12)
    ap.add_argument("--vae", default=sa.DEFAULT_VAE)
    ap.add_argument("--vae-file", default=None)
    ap.add_argument("--out", default="/scratch/work/attack-exp")
    args = ap.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    os.makedirs(args.out, exist_ok=True)

    img = Image.open(args.input).convert("RGB") if args.input else synthetic(args.size)
    img = img.resize((img.width - img.width % 8, img.height - img.height % 8), Image.LANCZOS)
    print(f"test image {img.width}x{img.height}, device {device}")

    print("running the shipped browser/app engine ...")
    shipped = run_shipped_engine(img, args.out)

    print(f"loading {args.vae_file or args.vae} ...")
    vae = sa.load_vae(args.vae, device, args.vae_file)
    encode = sa.make_encode(vae)

    def enc(x):
        with torch.no_grad():
            return encode(x)

    x = sa.to_tensor(img, None)
    z_targ = enc(torch.zeros_like(x))
    dec_orig = sa.to_image(enc_decode(vae, x))

    rows = []

    def measure(name, pil):
        t = sa.to_tensor(pil, None)
        z = enc(t)
        dist = float((z - z_targ).pow(2).sum())
        view = sa.to_image(enc_decode(vae, t))
        shift = float(np.abs(np.asarray(view, np.int16) - np.asarray(dec_orig, np.int16)).mean())
        arr, src = np.asarray(pil), np.asarray(img)
        rows.append(dict(name=name, latent_dist=dist,
                         psnr=sa.psnr(arr, src),
                         max_delta=int(np.abs(arr.astype(np.int16) - src.astype(np.int16)).max()),
                         view_shift=shift))
        r = rows[-1]
        print(f"  {name:30} latent-to-grey {dist:9.1f}  psnr {r['psnr']:6.2f} dB  "
              f"max {r['max_delta']:3d}/255  VAE-view shift {shift:6.2f}/255")
        return view

    print("\nmeasuring in the real VAE latent space:")
    measure("original (no protection)", img)
    measure("shipped app engine (surrogate)", shipped)
    print(f"\nrunning the real encoder attack: {args.steps} steps, eps {args.eps}, on {device}")
    adv, stats = sa.run(img, vae=vae, steps=args.steps, eps=args.eps,
                        size=None, log=lambda m: print(m, flush=True))
    adv_view = measure("real encoder attack (PGD)", adv)

    montage([(img, "original"), (shipped, "app engine (surrogate)"), (adv, "real encoder attack"),
             (dec_orig, "VAE view: original"), (adv_view, "VAE view: attacked")],
            os.path.join(args.out, "comparison.png"))
    json.dump({"rows": rows, "attack": stats}, open(os.path.join(args.out, "results.json"), "w"), indent=2)
    print(f"\nwrote {args.out}/comparison.png and results.json")

    base = rows[0]["latent_dist"]
    for r in rows[1:]:
        print(f"  {r['name']:30} closes {100 * (1 - r['latent_dist'] / base):5.1f}% of the latent gap")
    return 0


def enc_decode(vae, x):
    with torch.no_grad():
        return vae.decode(vae.encode(x).latent_dist.mean / vae.config.scaling_factor).sample


if __name__ == "__main__":
    sys.exit(main())
