/**
 * Measures how much of the perturbation survives JPEG re-encoding, for a set of
 * engine configurations. This is the harness the shipped defaults were picked
 * with; see the table in README.md.
 *
 *   node scripts/measure_survival.mjs && python3 scripts/measure_survival.py
 *
 * The Node half runs the engine and dumps raw RGB frames; the Python half does
 * the JPEG round trip with Pillow and prints the table. Splitting it this way
 * keeps the measurement deterministic - no browser, no async decoding.
 */
import fs from 'node:fs';
import path from 'node:path';
import Vajra from '../docs/engine.js';

const OUT = path.join(process.env.SURVIVAL_OUT || '/tmp/vajra-survival');
fs.mkdirSync(OUT, { recursive: true });

const W = 900;
const H = 675;

const CONFIGS = [
  ['A_bands124_r0', { bands: [1, 2, 4], spectralRadius: 0 }],
  ['B_bands248_r0', { bands: [2, 4, 8], spectralRadius: 0 }],
  ['C_bands248_r1', { bands: [2, 4, 8], spectralRadius: 1 }],
  ['D_bands3612_r1', { bands: [3, 6, 12], spectralRadius: 1 }],
  ['E_bands3612_r2', { bands: [3, 6, 12], spectralRadius: 2 }],
  ['F_bands4816_r1', { bands: [4, 8, 16], spectralRadius: 1 }]
];

const KINDS = ['smooth', 'textured', 'photo'];

function makeImage(kind) {
  const data = new Uint8ClampedArray(W * H * 4);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const p = (y * W + x) * 4;
      const t = (x / W + y / H) / 2;
      let r, g, b;
      if (kind === 'smooth') {
        r = 60 + 180 * t; g = 70 + 150 * t; b = 90 + 120 * t;
      } else if (kind === 'textured') {
        r = 128 + 70 * Math.sin(x / 2.1) * Math.cos(y / 2.7);
        g = 110 + 60 * Math.sin((x + y) / 1.9);
        b = 100 + 50 * Math.cos(x / 2.4 + y / 3.3);
      } else {
        r = 40 + 190 * t + 25 * Math.sin(x / 2.6);
        g = 60 + 150 * t + 20 * Math.cos(y / 3.1);
        b = 95 + 110 * t + 18 * Math.sin((x + y) / 4.2);
      }
      data[p] = r; data[p + 1] = g; data[p + 2] = b; data[p + 3] = 255;
    }
  }
  return { data, width: W, height: H };
}

const rows = [];
for (const kind of KINDS) {
  const src = makeImage(kind);
  fs.writeFileSync(path.join(OUT, `src_${kind}.raw`), Buffer.from(src.data.buffer));
  for (const [name, opts] of CONFIGS) {
    const started = performance.now();
    const res = Vajra.protect(src, opts);
    fs.writeFileSync(path.join(OUT, `${kind}_${name}.raw`), Buffer.from(res.data.buffer));
    rows.push({
      kind,
      name,
      ms: Math.round(performance.now() - started),
      psnr: res.stats.psnr,
      ssim: res.stats.ssim,
      suppression: res.stats.suppression
    });
  }
}

fs.writeFileSync(path.join(OUT, 'meta.json'), JSON.stringify({ width: W, height: H, rows }, null, 2));
console.log(`wrote ${rows.length} frames to ${OUT}`);
