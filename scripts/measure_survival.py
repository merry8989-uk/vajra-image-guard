#!/usr/bin/env python3
"""Second half of the survival measurement: JPEG round trip and table.

    node scripts/measure_survival.mjs && python3 scripts/measure_survival.py
"""
import json
import io
import os
import sys

import numpy as np
from PIL import Image

OUT = os.environ.get("SURVIVAL_OUT", "/tmp/vajra-survival")
QUALITIES = (95, 80, 60)


def luma(a):
    return 0.299 * a[:, :, 0] + 0.587 * a[:, :, 1] + 0.114 * a[:, :, 2]


def round_trip(img, quality):
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality, subsampling=2)  # 4:2:0, like most sites
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"), dtype=np.float64)


def correlation(a, b):
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def main():
    meta_path = os.path.join(OUT, "meta.json")
    if not os.path.exists(meta_path):
        sys.exit(f"run scripts/measure_survival.mjs first (no {meta_path})")
    meta = json.load(open(meta_path))
    w, h = meta["width"], meta["height"]

    header = f"{'image':10}{'config':16}{'psnr':>7}{'ssim':>9}{'supp':>6}"
    header += "".join(f"{'q' + str(q):>7}" for q in QUALITIES) + f"{'ms':>7}"
    print(header)

    totals = {}
    for row in meta["rows"]:
        kind = row["kind"]
        src = np.frombuffer(open(os.path.join(OUT, f"src_{kind}.raw"), "rb").read(),
                            np.uint8).reshape(h, w, 4)[:, :, :3].astype(np.float64)
        prot = np.frombuffer(open(os.path.join(OUT, f"{kind}_{row['name']}.raw"), "rb").read(),
                             np.uint8).reshape(h, w, 4)[:, :, :3].astype(np.float64)
        applied = luma(prot) - luma(src)
        pimg = Image.fromarray(prot.astype(np.uint8))
        simg = Image.fromarray(src.astype(np.uint8))

        kept = [100 * max(0.0, correlation(applied, luma(round_trip(pimg, q)) - luma(round_trip(simg, q))))
                for q in QUALITIES]

        print(f"{kind:10}{row['name']:16}{row['psnr']:7.1f}{row['ssim']:9.4f}"
              f"{100 * row['suppression']:5.0f}%" + "".join(f"{v:6.0f}%" for v in kept)
              + f"{row['ms']:7d}")

        acc = totals.setdefault(row["name"], {"psnr": 0.0, "supp": 0.0, "kept": [0.0] * len(QUALITIES), "n": 0})
        acc["psnr"] += row["psnr"]
        acc["supp"] += row["suppression"]
        acc["n"] += 1
        for i, v in enumerate(kept):
            acc["kept"][i] += v

    print("\nmean over all test images, best JPEG-q80 survival first:")
    for name, acc in sorted(totals.items(), key=lambda kv: -kv[1]["kept"][1]):
        n = acc["n"]
        print(f"{name:16} psnr {acc['psnr'] / n:5.1f}  suppression {100 * acc['supp'] / n:3.0f}%  "
              + "  ".join(f"q{q} {v / n:3.0f}%" for q, v in zip(QUALITIES, acc["kept"])))


if __name__ == "__main__":
    main()
