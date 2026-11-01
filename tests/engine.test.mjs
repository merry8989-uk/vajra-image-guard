import test from 'node:test';
import assert from 'node:assert/strict';
import Vajra from '../docs/engine.js';

const I = Vajra._internals;

// Deterministic synthetic photo: smooth gradients + fine detail + a flat patch.
function makeImage(w, h) {
  const data = new Uint8ClampedArray(w * h * 4);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const p = (y * w + x) * 4;
      const flat = x > w * 0.6 && y > h * 0.6;
      const r = flat ? 120 : 60 + (200 * x) / w + 30 * Math.sin(x / 3);
      const g = flat ? 130 : 70 + (170 * y) / h + 25 * Math.cos(y / 4);
      const b = flat ? 140 : 90 + 60 * Math.sin((x + y) / 7);
      data[p] = r;
      data[p + 1] = g;
      data[p + 2] = b;
      data[p + 3] = 255;
    }
  }
  return { data, width: w, height: h };
}

test('surrogate encoder gradient matches finite differences (suppress)', () => {
  const w = 24, h = 20, n = w * h;
  const basis = I.makeBasis(w, h, 'grad-test', 3);
  const x = new Float32Array(n);
  const rng = I.makeRng('x');
  for (let i = 0; i < n; i++) x[i] = 0.2 + 0.6 * rng();

  const g = I.gradient(x, basis, n, 'suppress', new Float32Array(n));
  const eps = 1e-5;
  let worst = 0;
  for (let i = 0; i < n; i += 37) {
    const xp = Float32Array.from(x);
    const xm = Float32Array.from(x);
    xp[i] += eps;
    xm[i] -= eps;
    const num = (I.loss(xp, basis, n, 'suppress') - I.loss(xm, basis, n, 'suppress')) / (2 * eps);
    const rel = Math.abs(num - g[i]) / Math.max(1e-8, Math.abs(num) + Math.abs(g[i]));
    if (rel > worst) worst = rel;
  }
  assert.ok(worst < 1e-3, `worst relative gradient error ${worst}`);
});

test('surrogate encoder gradient matches finite differences (displace)', () => {
  const w = 16, h = 16, n = w * h;
  const basis = I.makeBasis(w, h, 'grad-test-2', 2);
  const x = new Float32Array(n).fill(0.5);
  const g = I.gradient(x, basis, n, 'displace', new Float32Array(n));
  const eps = 1e-5;
  for (let i = 0; i < n; i += 29) {
    const xp = Float32Array.from(x);
    const xm = Float32Array.from(x);
    xp[i] += eps;
    xm[i] -= eps;
    const num = (I.loss(xp, basis, n, 'displace') - I.loss(xm, basis, n, 'displace')) / (2 * eps);
    assert.ok(Math.abs(num - g[i]) < 1e-6, `index ${i}: numeric ${num} analytic ${g[i]}`);
  }
});

test('PGD drives the surrogate embedding towards the neutral target', () => {
  const w = 32, h = 32, n = w * h;
  const basis = I.makeBasis(w, h, 'pgd', 4);
  const x = new Float32Array(n);
  const rng = I.makeRng('img');
  for (let i = 0; i < n; i++) x[i] = 0.15 + 0.7 * rng();

  const before = I.loss(x, basis, n, 'suppress');
  const d = I.solveChannel(x, w, h, basis, 0.03, 24, null, 'suppress', 'seed');
  const adv = new Float32Array(n);
  for (let i = 0; i < n; i++) adv[i] = x[i] + d[i];
  const after = I.loss(adv, basis, n, 'suppress');

  assert.ok(after < before * 0.25, `loss should drop sharply: ${before} -> ${after}`);
});

test('PGD respects the L-infinity budget on every channel', () => {
  const w = 40, h = 30, n = w * h;
  const basis = I.makeBasis(w, h, 'budget', 3);
  const x = new Float32Array(n).fill(0.5);
  const eps = 0.02;
  const d = I.solveChannel(x, w, h, basis, eps, 12, null, 'suppress', 'b');
  for (let i = 0; i < n; i++) {
    assert.ok(Math.abs(d[i]) <= eps + 1e-6, `pixel ${i} delta ${d[i]} exceeds ${eps}`);
  }
});

test('protect() output respects epsilon in 8-bit space and leaves input untouched', () => {
  const img = makeImage(96, 80);
  const original = Uint8ClampedArray.from(img.data);
  const res = Vajra.protect(img, { epsilon: 6 / 255, iterations: 12, seed: 7 });

  assert.deepEqual(img.data, original, 'input image must not be mutated');
  assert.equal(res.data.length, original.length);

  let max = 0;
  for (let i = 0; i < res.data.length; i++) {
    const d = Math.abs(res.data[i] - original[i]);
    if (d > max) max = d;
  }
  // epsilon is 6/255 => at most 6 levels in RGB, plus rounding slack of 1.
  assert.ok(max <= 7, `max per-channel delta ${max} exceeds budget`);
  assert.ok(max >= 2, `perturbation looks empty (max delta ${max})`);
  assert.equal(res.stats.width, 96);
  assert.equal(res.stats.workingWidth, 96);
});

test('default strength stays visually near-lossless', () => {
  const img = makeImage(128, 96);
  const res = Vajra.protect(img, {});
  assert.ok(res.stats.psnr > 38, `PSNR too low: ${res.stats.psnr}`);
  assert.ok(res.stats.ssim > 0.97, `SSIM too low: ${res.stats.ssim}`);
  assert.ok(res.stats.meanDelta < 4, `mean delta too high: ${res.stats.meanDelta}`);
});

test('protection is deterministic per seed and changes with the seed', () => {
  const img = makeImage(64, 64);
  const a = Vajra.protect(img, { seed: 42, iterations: 8 });
  const b = Vajra.protect(img, { seed: 42, iterations: 8 });
  const c = Vajra.protect(img, { seed: 43, iterations: 8 });

  assert.deepEqual(Array.from(a.data), Array.from(b.data), 'same seed must be reproducible');
  let diff = 0;
  for (let i = 0; i < a.data.length; i++) if (a.data[i] !== c.data[i]) diff++;
  assert.ok(diff > a.data.length * 0.2, 'a different seed must change the perturbation');
});

test('larger epsilon produces a larger perturbation', () => {
  const img = makeImage(80, 80);
  const low = Vajra.protect(img, { epsilon: 2 / 255, iterations: 10 });
  const high = Vajra.protect(img, { epsilon: 10 / 255, iterations: 10 });
  assert.ok(high.stats.meanDelta > low.stats.meanDelta);
  assert.ok(high.stats.psnr < low.stats.psnr);
});

test('channel budgets bound the worst-case RGB shift by epsilon', () => {
  const b = I.channelBudgets(0.04, 0.5);
  const worstR = b.luma + 1.402 * b.cr;
  const worstG = b.luma + 0.344136 * b.cb + 0.714136 * b.cr;
  const worstB = b.luma + 1.772 * b.cb;
  assert.ok(worstR <= 0.04 + 1e-9, `R ${worstR}`);
  assert.ok(worstG <= 0.04 + 1e-9, `G ${worstG}`);
  assert.ok(worstB <= 0.04 + 1e-9, `B ${worstB}`);
});

test('perceptual mask keeps flat regions below the full budget', () => {
  const img = makeImage(120, 120);
  const res = Vajra.protect(img, { epsilon: 8 / 255, iterations: 10, maskFloor: 0.2 });
  const w = 120;
  let flatMax = 0, busyMax = 0;
  for (let y = 0; y < 120; y++) {
    for (let x = 0; x < 120; x++) {
      const p = (y * w + x) * 4;
      let d = 0;
      for (let k = 0; k < 3; k++) d = Math.max(d, Math.abs(res.data[p + k] - img.data[p + k]));
      if (x > 80 && y > 80) flatMax = Math.max(flatMax, d);
      else if (x < 40 && y < 40) busyMax = Math.max(busyMax, d);
    }
  }
  // The guarantee is per-pixel room, not relative ordering: a flat region may not
  // exceed epsilon * maskFloor, whatever the busy regions happen to use.
  const flatRoom = Math.ceil(8 * 0.2) + 1;
  assert.ok(flatMax <= flatRoom, `flat region ${flatMax} exceeds its ${flatRoom}-level room`);
  assert.ok(busyMax >= flatMax, `busy region ${busyMax} quieter than flat ${flatMax}`);
});

test('psnr/ssim sanity', () => {
  const img = makeImage(64, 64);
  assert.equal(Vajra.psnr(img.data, img.data), Infinity);
  assert.ok(Math.abs(Vajra.ssim(img.data, img.data, 64, 64) - 1) < 1e-6);
  const res = Vajra.protect(img, { epsilon: 12 / 255 });
  assert.ok(Vajra.ssim(img.data, res.data, 64, 64) < 1);
});

test('guards reject bad input', () => {
  assert.throws(() => Vajra.protect(null), TypeError);
  assert.throws(() => Vajra.protect({ data: new Uint8ClampedArray(4), width: 1, height: 1 }, { maxPixels: 0 }), RangeError);
  assert.throws(() => Vajra.protect(makeImage(16, 16), { epsilon: 0 }), RangeError);
});
