#!/usr/bin/env python3
"""Vajra Image Guard - desktop reference implementation.

This is the same procedure as docs/engine.js, written against numpy so it can run
as a batch CLI on a laptop or a server:

    python3 surrogate_pgd.py photo.jpg -o photo-vajra.png --strength 6

It is *not* bit-identical to the JavaScript engine (different PRNG, float64 vs
float32 arithmetic), but every step is the same: band-limited random projections
as a fixed linear surrogate encoder, projected gradient descent with sign steps
inside an L-infinity ball around the original, a neutral (zero) target, a
perceptual budget mask, and a spectral projection so the result survives JPEG.

The paper this idea comes from optimises against the real Stable Diffusion VAE
with network gradients. That is what sd_encoder_attack.py does, and it needs a
GPU. This file deliberately does not.

Self-test (gradient check, budget check, determinism, loss decrease):

    python3 surrogate_pgd.py --self-test
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

DEFAULTS = dict(
    epsilon=6 / 255,
    iterations=18,
    seed=1337,
    basis_count=4,
    working_side=512,
    luma_factor=0.5,
    objective="suppress",
    mask_floor=0.35,
    spectral_radius=1,
    bands=(3, 6, 12),
)


# --------------------------------------------------------------------- utils --


def _box_blur(a: np.ndarray, r: int, axis: int) -> np.ndarray:
    """Box blur with edge clamping along one axis, via a running sum."""
    if r <= 0:
        return a.copy()
    pad = [(0, 0)] * a.ndim
    pad[axis] = (r, r)
    padded = np.pad(a, pad, mode="edge")
    csum = np.cumsum(padded, axis=axis, dtype=np.float64)
    zero_shape = list(csum.shape)
    zero_shape[axis] = 1
    csum = np.concatenate([np.zeros(zero_shape), csum], axis=axis)
    n = a.shape[axis]
    hi = np.take(csum, np.arange(2 * r + 1, 2 * r + 1 + n), axis=axis)
    lo = np.take(csum, np.arange(0, n), axis=axis)
    return (hi - lo) / (2 * r + 1)


def blur(a: np.ndarray, r: int) -> np.ndarray:
    return _box_blur(_box_blur(a, r, 0), r, 1)


def make_basis(w: int, h: int, seed: int, count: int, bands) -> list[np.ndarray]:
    """Band-passed random patterns, mean removed and scaled to unit RMS."""
    rng = np.random.default_rng(seed)
    out = []
    for k in range(count):
        r = bands[k % len(bands)]
        noise = rng.random((h, w)) * 2 - 1
        band = blur(noise, r) - blur(noise, 2 * r)
        band = band - band.mean()
        rms = np.sqrt((band ** 2).mean()) or 1.0
        out.append(band / rms)
    return out


def perceptual_mask(y: np.ndarray, floor: float) -> np.ndarray:
    """Smoothed |Laplacian| -> per-pixel share of the budget, in [floor, 1]."""
    lap = np.zeros_like(y)
    lap[1:-1, 1:-1] = np.abs(
        4 * y[1:-1, 1:-1] - y[:-2, 1:-1] - y[2:, 1:-1] - y[1:-1, :-2] - y[1:-1, 2:]
    )
    act = blur(lap, 2)
    ref = act.mean() + 2 * act.std() or 1e-6
    m = np.clip(act / ref, 0.0, 1.0)
    return floor + (1 - floor) * np.sqrt(m)


def channel_budgets(epsilon: float, luma_factor: float) -> dict[str, float]:
    """Split the RGB budget so the worst-case RGB shift is exactly epsilon."""
    lf = min(0.95, max(0.0, luma_factor))
    luma = epsilon * lf
    rest = epsilon - luma
    return {"luma": luma, "cr": rest / 1.402, "cb": rest / 1.772}


# ------------------------------------------------------------ surrogate model --


def embedding(x: np.ndarray, basis: list[np.ndarray]) -> np.ndarray:
    return np.array([float((x * w).mean()) for w in basis])


def loss(x: np.ndarray, basis: list[np.ndarray], objective: str = "suppress") -> float:
    e = embedding(x, basis)
    s = float((e ** 2).sum())
    return -s if objective == "displace" else s


def gradient(x: np.ndarray, basis: list[np.ndarray], objective: str = "suppress") -> np.ndarray:
    """Analytic d(loss)/dx: the surrogate is linear in x."""
    n = x.size
    e = embedding(x, basis)
    sign = -2.0 if objective == "displace" else 2.0
    g = np.zeros_like(x)
    for ek, wk in zip(e, basis):
        if ek:
            g += (sign * ek / n) * wk
    return g


def solve_channel(x, basis, eps, iters, mask, objective, seed) -> np.ndarray:
    """Sign-step PGD with an L-infinity projection, returning the perturbation."""
    rng = np.random.default_rng(seed)
    adv = x + (rng.random(x.shape) * 2 - 1) * eps
    direction = -1.0 if objective == "displace" else 1.0
    for it in range(iters):
        g = gradient(adv, basis, objective)
        frac = it / (iters - 1) if iters > 1 else 1.0
        step = eps * 0.5 * (1 - 0.85 * frac)
        adv = adv - direction * step * np.sign(g) * mask
        adv = np.clip(adv, x - eps * mask, x + eps * mask)
        adv = np.clip(adv, 0.0, 1.0)
    return adv - x


def spectral_shape(d, mask, eps, radius) -> np.ndarray:
    """Re-project the perturbation onto the band-limited subspace, then refill."""
    if radius <= 0:
        return d
    smooth = blur(d, radius)
    room = eps * mask
    ratio = np.abs(smooth) / np.maximum(room, 1e-12)
    peak = float(ratio.max())
    gain = 1.0 / peak if peak > 0 else 1.0
    return np.clip(smooth * gain, -room, room)


# ------------------------------------------------------------------ pipeline --


def _downsample(rgb: np.ndarray, ww: int, wh: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = rgb.shape[:2]
    ys = (np.arange(wh) * h // wh)
    xs = (np.arange(ww) * w // ww)
    small = rgb[np.ix_(ys, xs)].astype(np.float64) / 255.0
    r, g, b = small[..., 0], small[..., 1], small[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
    return y, cb, cr


def protect(rgb: np.ndarray, **options) -> tuple[np.ndarray, dict]:
    """rgb: HxWx3 uint8. Returns (protected uint8, stats)."""
    opts = dict(DEFAULTS)
    opts.update({k: v for k, v in options.items() if v is not None})

    h, w = rgb.shape[:2]
    scale = min(1.0, opts["working_side"] / max(w, h))
    ww = max(8, int(round(w * scale)))
    wh = max(8, int(round(h * scale)))

    y, cb, cr = _downsample(rgb, ww, wh)
    mask = perceptual_mask(y, opts["mask_floor"])
    budget = channel_budgets(opts["epsilon"], opts["luma_factor"])

    seed = int(opts["seed"])
    bands = tuple(opts["bands"])
    basis_y = make_basis(ww, wh, seed + 1, opts["basis_count"], bands)
    basis_b = make_basis(ww, wh, seed + 2, opts["basis_count"], bands)
    basis_r = make_basis(ww, wh, seed + 3, opts["basis_count"], bands)

    loss_before = loss(y, basis_y, opts["objective"])
    dy = solve_channel(y, basis_y, budget["luma"], opts["iterations"], mask, opts["objective"], seed + 11)
    dcb = solve_channel(cb, basis_b, budget["cb"], opts["iterations"], mask, opts["objective"], seed + 12)
    dcr = solve_channel(cr, basis_r, budget["cr"], opts["iterations"], mask, opts["objective"], seed + 13)

    dy = spectral_shape(dy, mask, budget["luma"], opts["spectral_radius"])
    dcb = spectral_shape(dcb, mask, budget["cb"], opts["spectral_radius"])
    dcr = spectral_shape(dcr, mask, budget["cr"], opts["spectral_radius"])

    adv_y = np.clip(y + dy, 0, 1)
    loss_after = loss(adv_y, basis_y, opts["objective"])
    if abs(loss_before) > 1e-12:
        suppression = ((loss_after - loss_before) / abs(loss_before)
                       if opts["objective"] == "displace"
                       else 1 - loss_after / loss_before)
    else:
        suppression = 0.0

    # upsample the perturbation and apply it in YCbCr at full resolution
    def up(a):
        ys = np.clip(((np.arange(h) + 0.5) * wh / h - 0.5), 0, wh - 1)
        xs = np.clip(((np.arange(w) + 0.5) * ww / w - 0.5), 0, ww - 1)
        y0 = np.floor(ys).astype(int); x0 = np.floor(xs).astype(int)
        y1 = np.minimum(y0 + 1, wh - 1); x1 = np.minimum(x0 + 1, ww - 1)
        ty = (ys - y0)[:, None]; tx = (xs - x0)[None, :]
        return ((a[np.ix_(y0, x0)] * (1 - tx) * (1 - ty) + a[np.ix_(y0, x1)] * tx * (1 - ty)
                 + a[np.ix_(y1, x0)] * (1 - tx) * ty + a[np.ix_(y1, x1)] * tx * ty))

    full = rgb.astype(np.float64) / 255.0
    fr, fg, fb = full[..., 0], full[..., 1], full[..., 2]
    fy = 0.299 * fr + 0.587 * fg + 0.114 * fb + up(dy)
    fcb = -0.168736 * fr - 0.331264 * fg + 0.5 * fb + 0.5 + up(dcb)
    fcr = 0.5 * fr - 0.418688 * fg - 0.081312 * fb + 0.5 + up(dcr)
    fy = np.clip(fy, 0, 1); fcb = np.clip(fcb, 0, 1); fcr = np.clip(fcr, 0, 1)

    b_, r_ = fcb - 0.5, fcr - 0.5
    out = np.stack([
        fy + 1.402 * r_,
        fy - 0.344136 * b_ - 0.714136 * r_,
        fy + 1.772 * b_,
    ], axis=-1)
    out = np.clip(out * 255.0, 0, 255).round().astype(np.uint8)

    delta = np.abs(out.astype(np.int16) - rgb.astype(np.int16))
    mse = float(((out.astype(np.float64) - rgb.astype(np.float64)) ** 2).mean())
    stats = {
        "psnr": float("inf") if mse == 0 else 10 * np.log10(255 ** 2 / mse),
        "max_delta": int(delta.max()),
        "mean_delta": float(delta.mean()),
        "suppression": float(suppression),
        "working": (ww, wh),
        "luma_budget": budget["luma"] * 255,
        "cb_budget": budget["cb"] * 255,
        "cr_budget": budget["cr"] * 255,
    }
    return out, stats


# --------------------------------------------------------------------- tests --


def self_test() -> int:
    failures = 0

    def check(name, condition, detail=""):
        nonlocal failures
        status = "ok  " if condition else "FAIL"
        if not condition:
            failures += 1
        print(f"{status} {name}{(' — ' + detail) if detail else ''}")

    w, h = 24, 20
    basis = make_basis(w, h, 7, 3, (1, 2, 4))
    rng = np.random.default_rng(0)
    x = 0.2 + 0.6 * rng.random((h, w))

    g = gradient(x, basis, "suppress")
    eps = 1e-6
    worst = 0.0
    for (yy, xx) in [(3, 4), (10, 12), (18, 2), (5, 20)]:
        xp = x.copy(); xm = x.copy()
        xp[yy, xx] += eps; xm[yy, xx] -= eps
        num = (loss(xp, basis, "suppress") - loss(xm, basis, "suppress")) / (2 * eps)
        denom = max(1e-12, abs(num) + abs(g[yy, xx]))
        worst = max(worst, abs(num - g[yy, xx]) / denom)
    check("analytic gradient matches finite differences", worst < 1e-4, f"worst rel err {worst:.2e}")

    mask = np.ones((h, w))
    d = solve_channel(x, basis, 0.03, 12, mask, "suppress", 3)
    check("perturbation stays inside the L-infinity ball", np.abs(d).max() <= 0.03 + 1e-9,
          f"max {np.abs(d).max():.5f} vs 0.03000")

    before = loss(x, basis, "suppress")
    after = loss(np.clip(x + d, 0, 1), basis, "suppress")
    check("PGD reduces the surrogate embedding", after < before, f"{before:.3e} -> {after:.3e}")

    b = channel_budgets(0.04, 0.5)
    worst_rgb = max(b["luma"] + 1.402 * b["cr"],
                    b["luma"] + 0.344136 * b["cb"] + 0.714136 * b["cr"],
                    b["luma"] + 1.772 * b["cb"])
    check("channel budgets bound the RGB shift by epsilon", worst_rgb <= 0.04 + 1e-9,
          f"worst {worst_rgb:.5f}")

    img = (rng.random((60, 80, 3)) * 255).astype(np.uint8)
    out1, s1 = protect(img, seed=5, iterations=8)
    out2, _ = protect(img, seed=5, iterations=8)
    check("same seed is reproducible", np.array_equal(out1, out2))
    check("output stays within the epsilon budget", s1["max_delta"] <= 7,
          f"max delta {s1['max_delta']}")
    check("default strength is visually near-lossless", s1["psnr"] > 38, f"psnr {s1['psnr']:.1f} dB")

    out3, _ = protect(img, seed=6, iterations=8)
    check("a different seed changes the result", not np.array_equal(out1, out3))

    print()
    print("all checks passed" if failures == 0 else f"{failures} check(s) failed")
    return 1 if failures else 0


# ----------------------------------------------------------------------- CLI --


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Protect an image with the Vajra surrogate-encoder attack.")
    ap.add_argument("input", nargs="?", help="input image (JPEG/PNG/WebP)")
    ap.add_argument("-o", "--output", help="output path (default: <input>-vajra.png)")
    ap.add_argument("--strength", type=float, default=6.0, help="max shift per channel, in 0-255 levels")
    ap.add_argument("--iterations", type=int, default=DEFAULTS["iterations"])
    ap.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    ap.add_argument("--objective", choices=["suppress", "displace"], default="suppress")
    ap.add_argument("--mask-floor", type=float, default=DEFAULTS["mask_floor"])
    ap.add_argument("--spectral-radius", type=int, default=DEFAULTS["spectral_radius"])
    ap.add_argument("--working-side", type=int, default=DEFAULTS["working_side"])
    ap.add_argument("--self-test", action="store_true", help="run the internal checks and exit")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()
    if not args.input:
        ap.error("input image required (or use --self-test)")

    from PIL import Image

    img = Image.open(args.input).convert("RGB")
    rgb = np.asarray(img)
    out, stats = protect(
        rgb,
        epsilon=args.strength / 255,
        iterations=args.iterations,
        seed=args.seed,
        objective=args.objective,
        mask_floor=args.mask_floor,
        spectral_radius=args.spectral_radius,
        working_side=args.working_side,
    )
    dest = args.output or (args.input.rsplit(".", 1)[0] + "-vajra.png")
    Image.fromarray(out).save(dest)

    print(f"wrote {dest}")
    print(f"  PSNR {stats['psnr']:.2f} dB   max shift {stats['max_delta']}/255   "
          f"mean {stats['mean_delta']:.2f}/255")
    print(f"  embedding suppression {100 * stats['suppression']:.0f}%   "
          f"solved at {stats['working'][0]}x{stats['working'][1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
