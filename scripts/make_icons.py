#!/usr/bin/env python3
"""Generate every icon the project ships: web (PNG + SVG is hand written) and
Android launcher mipmaps. Deterministic, no external assets.

    python3 scripts/make_icons.py
"""
from __future__ import annotations

import os
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SS = 4  # supersampling factor (memory friendly in small containers)

BG_TOP = (13, 15, 21)
BG_BOTTOM = (7, 8, 11)
ACCENT = (94, 234, 212)


def shield_points(size: int, inset: float) -> list[tuple[float, float]]:
    """Shield outline scaled into a size x size box, as a polygon."""
    # Normalised shield: wide shoulders, pointed bottom.
    base = [
        (0.50, 0.10),
        (0.86, 0.235),
        (0.86, 0.545),
        (0.50, 0.90),
        (0.14, 0.545),
        (0.14, 0.235),
    ]
    span = 1.0 - 2 * inset
    return [((x - inset) / span * size, (y - inset) / span * size) for x, y in base]


def rounded_gradient(size: int, radius_ratio: float, gradient: bool) -> Image.Image:
    big = size * SS
    img = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    if gradient:
        grad = Image.new("RGB", (1, big))
        px = grad.load()
        for y in range(big):
            t = y / max(1, big - 1)
            px[0, y] = (
                round(BG_TOP[0] + (BG_BOTTOM[0] - BG_TOP[0]) * t),
                round(BG_TOP[1] + (BG_BOTTOM[1] - BG_TOP[1]) * t),
                round(BG_TOP[2] + (BG_BOTTOM[2] - BG_TOP[2]) * t),
            )
        plate = grad.resize((big, big))
    else:
        plate = Image.new("RGB", (big, big), BG_TOP)

    mask = Image.new("L", (big, big), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        [0, 0, big - 1, big - 1], radius=int(big * radius_ratio), fill=255
    )
    img.paste(plate, (0, 0), mask)
    return img


def draw_shield(size: int, inset: float, width: float, check: bool) -> Image.Image:
    big = size * SS
    layer = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    pts = shield_points(big, inset)
    d.line(pts + [pts[0]], fill=ACCENT + (255,), width=int(big * width), joint="curve")
    if check:
        w = int(big * width * 1.15)
        d.line(
            [(big * 0.355, big * 0.505), (big * 0.465, big * 0.615), (big * 0.665, big * 0.395)],
            fill=ACCENT + (255,),
            width=w,
            joint="curve",
        )
    return layer


def make_icon(size: int, rounded: bool = True, padding: float = 0.0) -> Image.Image:
    """Full icon: dark plate plus shield, optionally padded for maskable use."""
    big = size * SS
    canvas = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    if rounded:
        plate = rounded_gradient(size, 0.22, gradient=True)
    else:
        plate = rounded_gradient(size, 0.5, gradient=True)
    if padding > 0:
        inner = int(big * (1 - 2 * padding))
        plate = plate.resize((inner, inner), Image.LANCZOS)
        off = (big - inner) // 2
        canvas.paste(plate, (off, off), plate)
        shield = draw_shield(inner, 0.10, 0.052, check=True)
        canvas.alpha_composite(shield, (off, off))
    else:
        canvas.alpha_composite(plate)
        shield = draw_shield(big, 0.10, 0.052, check=True)
        canvas.alpha_composite(shield)
    return canvas.resize((size, size), Image.LANCZOS)


def main() -> None:
    web = os.path.join(ROOT, "docs", "icons")
    os.makedirs(web, exist_ok=True)
    for size in (192, 512):
        make_icon(size).save(os.path.join(web, f"icon-{size}.png"))
    make_icon(512, rounded=False, padding=0.12).save(os.path.join(web, "icon-maskable-512.png"))
    make_icon(180).save(os.path.join(web, "apple-touch-icon.png"))

    android = os.path.join(ROOT, "android", "app", "src", "main", "res")
    densities = {"mdpi": 48, "hdpi": 72, "xhdpi": 96, "xxhdpi": 144, "xxxhdpi": 192}
    for name, size in densities.items():
        out = os.path.join(android, f"mipmap-{name}")
        os.makedirs(out, exist_ok=True)
        make_icon(size).save(os.path.join(out, "ic_launcher.png"))

    print("icons written to docs/icons and android/app/src/main/res/mipmap-*")


if __name__ == "__main__":
    main()
