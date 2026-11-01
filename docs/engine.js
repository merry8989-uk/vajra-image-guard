/*
 * Vajra Image Guard - protection engine.
 *
 * Idea (not code) borrowed from "Raising the Cost of Malicious AI-Powered Image
 * Editing" (Salman, Khaddaj, Leclerc, Ilyas, Madry, 2023 - the PhotoGuard paper):
 * add a small L-infinity bounded perturbation that pushes an image's embedding
 * away from a neutral target, so a diffusion model conditioned on the image
 * produces broken edits. The paper optimises the perturbation against the real
 * Stable Diffusion VAE encoder with PGD.
 *
 * This engine implements the same *procedure* (random init, projected gradient
 * descent, sign steps, L-infinity projection, neutral target) against a fixed
 * *linear* surrogate encoder: a bank of band-limited random projection patterns.
 * A linear encoder has an analytic Jacobian, so the gradient is exact and the
 * whole solve runs in a few hundred milliseconds on a phone, with no model
 * weights and no network. See docs/index.html ("Honest limits") for what that
 * trade buys and what it costs.
 *
 * No DOM APIs are used here: the module runs in a browser, a WebView and Node.
 */
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && typeof module.exports === 'object') {
    module.exports = api;
  } else {
    root.Vajra = api;
  }
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';

  const VERSION = '1.0.0';

  const DEFAULTS = {
    epsilon: 6 / 255,      // maximum per-channel L-infinity budget, in [0,1] units
    iterations: 18,        // PGD steps
    seed: 1337,            // seed for the surrogate encoder + PGD init
    basisCount: 4,         // number of band-limited projection patterns
    workingSide: 512,      // the PGD is solved at this resolution, then upsampled
    lumaFactor: 0.5,       // luma budget = epsilon * lumaFactor (luma is the visible one)
                           // the rest goes to chroma, see channelBudgets()
    objective: 'suppress', // 'suppress' (towards neutral target) or 'displace' (away)
    maskFloor: 0.35,       // perceptual mask value in flat regions (1.0 = full budget)
    spectralRadius: 1,     // low-pass radius applied to the solved perturbation (0 = off)
    bands: [3, 6, 12],     // box radii that define the band-pass of the surrogate patterns
                           // (chosen by measurement, see scripts/measure_survival.py)
    maxPixels: 24e6        // refuse absurd inputs rather than run out of memory
  };

  /* ---------------------------------------------------------------- random -- */

  function xmur3(str) {
    let h = 1779033703 ^ str.length;
    for (let i = 0; i < str.length; i++) {
      h = Math.imul(h ^ str.charCodeAt(i), 3432918353);
      h = (h << 13) | (h >>> 19);
    }
    return function () {
      h = Math.imul(h ^ (h >>> 16), 2246822507);
      h = Math.imul(h ^ (h >>> 13), 3266489909);
      h ^= h >>> 16;
      return h >>> 0;
    };
  }

  function mulberry32(a) {
    return function () {
      a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function makeRng(seed) {
    const s = xmur3(String(seed));
    return mulberry32(s());
  }

  /* ------------------------------------------------------------ blur utils -- */

  function blurH(src, dst, w, h, r) {
    const inv = 1 / (2 * r + 1);
    for (let y = 0; y < h; y++) {
      const row = y * w;
      let acc = 0;
      for (let i = -r; i <= r; i++) {
        const xi = i < 0 ? 0 : (i > w - 1 ? w - 1 : i);
        acc += src[row + xi];
      }
      for (let x = 0; x < w; x++) {
        dst[row + x] = acc * inv;
        const outX = x - r < 0 ? 0 : x - r;
        const inX = x + r + 1 > w - 1 ? w - 1 : x + r + 1;
        acc += src[row + inX] - src[row + outX];
      }
    }
  }

  function blurV(src, dst, w, h, r) {
    const inv = 1 / (2 * r + 1);
    for (let x = 0; x < w; x++) {
      let acc = 0;
      for (let i = -r; i <= r; i++) {
        const yi = i < 0 ? 0 : (i > h - 1 ? h - 1 : i);
        acc += src[yi * w + x];
      }
      for (let y = 0; y < h; y++) {
        dst[y * w + x] = acc * inv;
        const outY = y - r < 0 ? 0 : y - r;
        const inY = y + r + 1 > h - 1 ? h - 1 : y + r + 1;
        acc += src[inY * w + x] - src[outY * w + x];
      }
    }
  }

  /* --------------------------------------------------- surrogate encoder ---- */

  // Each basis vector is white noise band-passed by a difference of box blurs,
  // then mean-removed and scaled to unit RMS. Band-passing keeps the resulting
  // perturbation in the mid-frequency range, where the eye is least sensitive
  // and where JPEG's quantisation table hurts it least.
  function makeBasis(w, h, seed, count, bandList) {
    const rng = makeRng(seed);
    const n = w * h;
    const bands = bandList && bandList.length ? bandList : [1, 2, 4];
    const out = [];
    const noise = new Float32Array(n);
    const lowR = new Float32Array(n);
    const scratch = new Float32Array(n);
    const low2R = new Float32Array(n);

    for (let k = 0; k < count; k++) {
      const r = bands[k % bands.length];
      for (let i = 0; i < n; i++) noise[i] = rng() * 2 - 1;

      blurH(noise, scratch, w, h, r);
      blurV(scratch, lowR, w, h, r);

      blurH(lowR, scratch, w, h, r * 2);
      blurV(scratch, low2R, w, h, r * 2);

      const wk = new Float32Array(n);
      let mean = 0;
      for (let i = 0; i < n; i++) {
        wk[i] = lowR[i] - low2R[i];
        mean += wk[i];
      }
      mean /= n;

      let ss = 0;
      for (let i = 0; i < n; i++) {
        wk[i] -= mean;
        ss += wk[i] * wk[i];
      }
      const rms = Math.sqrt(ss / n) || 1;
      const inv = 1 / rms;
      for (let i = 0; i < n; i++) wk[i] *= inv;
      out.push(wk);
    }
    return out;
  }

  // e_k = mean(x * w_k). For the neutral target (mid grey) the embedding is the
  // zero vector, because every w_k has zero mean.
  function embedding(x, basis, n, out) {
    const e = out || new Float64Array(basis.length);
    for (let k = 0; k < basis.length; k++) {
      const wk = basis[k];
      let s = 0;
      for (let i = 0; i < n; i++) s += x[i] * wk[i];
      e[k] = s / n;
    }
    return e;
  }

  function loss(x, basis, n, objective) {
    const e = embedding(x, basis, n);
    let s = 0;
    for (let k = 0; k < e.length; k++) s += e[k] * e[k];
    return objective === 'displace' ? -s : s;
  }

  // Analytic gradient of loss() with respect to x, since e_k is linear in x.
  function gradient(x, basis, n, objective, out) {
    const e = embedding(x, basis, n);
    const g = out || new Float32Array(n);
    g.fill(0);
    const sign = objective === 'displace' ? -2 : 2;
    for (let k = 0; k < basis.length; k++) {
      const c = sign * e[k] / n;
      if (c === 0) continue;
      const wk = basis[k];
      for (let i = 0; i < n; i++) g[i] += c * wk[i];
    }
    return g;
  }

  /* ------------------------------------------------------------ perceptual -- */

  // Local activity (smoothed |Laplacian|) -> per-pixel share of the budget.
  // Busy regions hide perturbation; flat skies and skin get a fraction of it.
  function perceptualMask(y, w, h, floor) {
    const n = w * h;
    const act = new Float32Array(n);
    for (let yy = 1; yy < h - 1; yy++) {
      const row = yy * w;
      for (let xx = 1; xx < w - 1; xx++) {
        const i = row + xx;
        const l = 4 * y[i] - y[i - 1] - y[i + 1] - y[i - w] - y[i + w];
        act[i] = l < 0 ? -l : l;
      }
    }
    const t = new Float32Array(n);
    const s = new Float32Array(n);
    blurH(act, t, w, h, 2);
    blurV(t, s, w, h, 2);

    let mean = 0;
    for (let i = 0; i < n; i++) mean += s[i];
    mean /= n;
    let v = 0;
    for (let i = 0; i < n; i++) {
      const d = s[i] - mean;
      v += d * d;
    }
    v /= n;
    const ref = mean + 2 * Math.sqrt(v) || 1e-6;

    const mask = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      let m = s[i] / ref;
      if (m > 1) m = 1;
      if (m < 0) m = 0;
      mask[i] = floor + (1 - floor) * Math.sqrt(m);
    }
    return mask;
  }

  /* ------------------------------------------------------------------ PGD --- */

  // Projected gradient descent with sign steps, exactly the loop shape the
  // paper uses, minus the neural encoder.
  function solveChannel(x, w, h, basis, eps, iters, mask, objective, seed) {
    const n = w * h;
    const rng = makeRng(seed);
    const adv = new Float32Array(n);
    for (let i = 0; i < n; i++) adv[i] = x[i] + (rng() * 2 - 1) * eps;

    const g = new Float32Array(n);
    const dir = objective === 'displace' ? -1 : 1;

    for (let it = 0; it < iters; it++) {
      gradient(adv, basis, n, objective, g);
      const frac = iters > 1 ? it / (iters - 1) : 1;
      const step = eps * 0.5 * (1 - 0.85 * frac);
      for (let i = 0; i < n; i++) {
        const m = mask ? mask[i] : 1;
        const gi = g[i];
        if (gi !== 0) adv[i] -= dir * step * (gi > 0 ? 1 : -1) * m;
        const lo = x[i] - eps * m;
        const hi = x[i] + eps * m;
        let v = adv[i];
        if (v < lo) v = lo; else if (v > hi) v = hi;
        if (v < 0) v = 0; else if (v > 1) v = 1;
        adv[i] = v;
      }
    }

    const d = new Float32Array(n);
    for (let i = 0; i < n; i++) d[i] = adv[i] - x[i];
    return d;
  }

  // Sign steps are what makes PGD strong inside an L-infinity ball, but the sign
  // of a band-limited gradient is itself broadband, and broadband perturbation is
  // what a JPEG encoder throws away first. This projects the solved perturbation
  // back onto the band-limited subspace (a separable low pass) and then rescales
  // it so it uses the budget again. Survival after re-encoding goes up a lot;
  // the suppression it costs is small.
  function spectralShape(d, w, h, mask, eps, radius) {
    const n = w * h;
    const tmp = new Float32Array(n);
    const smooth = new Float32Array(n);
    blurH(d, tmp, w, h, radius);
    blurV(tmp, smooth, w, h, radius);

    let peak = 0;
    for (let i = 0; i < n; i++) {
      const room = eps * (mask ? mask[i] : 1);
      if (room <= 0) continue;
      const ratio = Math.abs(smooth[i]) / room;
      if (ratio > peak) peak = ratio;
    }
    const gain = peak > 0 ? 1 / peak : 1;

    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const room = eps * (mask ? mask[i] : 1);
      let v = smooth[i] * gain;
      if (v > room) v = room; else if (v < -room) v = -room;
      out[i] = v;
    }
    return out;
  }

  /* ------------------------------------------------------------ colour ----- */

  function downsampleChannels(data, w, h, ww, wh) {
    const n = ww * wh;
    const Y = new Float32Array(n);
    const Cb = new Float32Array(n);
    const Cr = new Float32Array(n);
    for (let y = 0; y < wh; y++) {
      const y0 = Math.floor((y * h) / wh);
      let y1 = Math.floor(((y + 1) * h) / wh);
      if (y1 <= y0) y1 = y0 + 1;
      for (let x = 0; x < ww; x++) {
        const x0 = Math.floor((x * w) / ww);
        let x1 = Math.floor(((x + 1) * w) / ww);
        if (x1 <= x0) x1 = x0 + 1;
        let r = 0, g = 0, b = 0, c = 0;
        for (let sy = y0; sy < y1; sy++) {
          let p = (sy * w + x0) * 4;
          for (let sx = x0; sx < x1; sx++) {
            r += data[p];
            g += data[p + 1];
            b += data[p + 2];
            p += 4;
            c++;
          }
        }
        const inv = 1 / c;
        r *= inv; g *= inv; b *= inv;
        const i = y * ww + x;
        Y[i] = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
        Cb[i] = (-0.168736 * r - 0.331264 * g + 0.5 * b) / 255 + 0.5;
        Cr[i] = (0.5 * r - 0.418688 * g - 0.081312 * b) / 255 + 0.5;
      }
    }
    return { Y: Y, Cb: Cb, Cr: Cr, width: ww, height: wh };
  }

  // The L-infinity bound has to hold on what the eye and a re-encoder see: RGB.
  // Going back from YCbCr amplifies chroma (R gains 1.402 x Cr, B gains 1.772 x Cb),
  // so the per-channel budgets are derived from the epsilon that is left after
  // luma has taken its share. Worst case RGB shift then equals epsilon.
  function channelBudgets(epsilon, lumaFactor) {
    const lf = Math.min(0.95, Math.max(0, lumaFactor));
    const luma = epsilon * lf;
    const rest = epsilon - luma;
    return {
      luma: luma,
      cr: rest / 1.402,
      cb: rest / 1.772
    };
  }

  function axisWeights(srcLen, dstLen) {
    const i0 = new Int32Array(dstLen);
    const i1 = new Int32Array(dstLen);
    const t = new Float32Array(dstLen);
    const scale = srcLen / dstLen;
    for (let i = 0; i < dstLen; i++) {
      let f = (i + 0.5) * scale - 0.5;
      if (f < 0) f = 0;
      let a = Math.floor(f);
      if (a > srcLen - 1) a = srcLen - 1;
      let b = a + 1;
      if (b > srcLen - 1) b = srcLen - 1;
      i0[i] = a;
      i1[i] = b;
      t[i] = f - a;
    }
    return { i0: i0, i1: i1, t: t };
  }

  /* ---------------------------------------------------------------- stats --- */

  function psnr(a, b) {
    let se = 0;
    const n = a.length;
    for (let i = 0; i < n; i++) {
      const d = a[i] - b[i];
      se += d * d;
    }
    const mse = se / n;
    if (mse === 0) return Infinity;
    return 10 * Math.log10((255 * 255) / mse);
  }

  function toGray(data, w, h) {
    const n = w * h;
    const g = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const p = i * 4;
      g[i] = 0.299 * data[p] + 0.587 * data[p + 1] + 0.114 * data[p + 2];
    }
    return g;
  }

  // Structural similarity over 8x8 windows, stride 4 (Wang et al. 2004).
  function ssim(a, b, w, h) {
    const ga = toGray(a, w, h);
    const gb = toGray(b, w, h);
    const C1 = 6.5025;   // (0.01 * 255)^2
    const C2 = 58.5225;  // (0.03 * 255)^2
    const win = 8;
    const stride = 4;
    let total = 0;
    let count = 0;
    for (let y = 0; y + win <= h; y += stride) {
      for (let x = 0; x + win <= w; x += stride) {
        let sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
        for (let yy = 0; yy < win; yy++) {
          const row = (y + yy) * w + x;
          for (let xx = 0; xx < win; xx++) {
            const va = ga[row + xx];
            const vb = gb[row + xx];
            sa += va; sb += vb;
            saa += va * va; sbb += vb * vb; sab += va * vb;
          }
        }
        const N = win * win;
        const ma = sa / N, mb = sb / N;
        const va = saa / N - ma * ma;
        const vb = sbb / N - mb * mb;
        const cov = sab / N - ma * mb;
        total += ((2 * ma * mb + C1) * (2 * cov + C2)) /
                 ((ma * ma + mb * mb + C1) * (va + vb + C2));
        count++;
      }
    }
    return count ? total / count : 1;
  }

  /* --------------------------------------------------------------- protect -- */

  function protect(imageData, options) {
    if (!imageData || !imageData.data || !imageData.width || !imageData.height) {
      throw new TypeError('protect() needs {data, width, height}');
    }
    const opts = Object.assign({}, DEFAULTS, options || {});
    const w = imageData.width;
    const h = imageData.height;
    const n = w * h;
    if (n > opts.maxPixels) {
      throw new RangeError('image too large: ' + n + ' px (limit ' + opts.maxPixels + ')');
    }
    if (!(opts.epsilon > 0)) throw new RangeError('epsilon must be > 0');

    const src = imageData.data;
    const out = new Uint8ClampedArray(src.length);
    out.set(src);

    const scale = Math.min(1, opts.workingSide / Math.max(w, h));
    const ww = Math.max(8, Math.round(w * scale));
    const wh = Math.max(8, Math.round(h * scale));

    const ch = downsampleChannels(src, w, h, ww, wh);
    const mask = perceptualMask(ch.Y, ww, wh, opts.maskFloor);

    const budget = channelBudgets(opts.epsilon, opts.lumaFactor);
    const epsLuma = budget.luma;
    const epsCb = budget.cb;
    const epsCr = budget.cr;

    const basisL = makeBasis(ww, wh, opts.seed + ':luma', opts.basisCount, opts.bands);
    const basisB = makeBasis(ww, wh, opts.seed + ':cb', opts.basisCount, opts.bands);
    const basisR = makeBasis(ww, wh, opts.seed + ':cr', opts.basisCount, opts.bands);

    const shapeRadius = opts.spectralRadius | 0;
    const lossBefore = loss(ch.Y, basisL, ww * wh, opts.objective);

    let dY = solveChannel(ch.Y, ww, wh, basisL, epsLuma, opts.iterations, mask, opts.objective, opts.seed + ':py');
    let dCb = solveChannel(ch.Cb, ww, wh, basisB, epsCb, opts.iterations, mask, opts.objective, opts.seed + ':pb');
    let dCr = solveChannel(ch.Cr, ww, wh, basisR, epsCr, opts.iterations, mask, opts.objective, opts.seed + ':pr');

    if (shapeRadius > 0) {
      dY = spectralShape(dY, ww, wh, mask, epsLuma, shapeRadius);
      dCb = spectralShape(dCb, ww, wh, mask, epsCb, shapeRadius);
      dCr = spectralShape(dCr, ww, wh, mask, epsCr, shapeRadius);
    }

    const ax = axisWeights(ww, w);
    const ay = axisWeights(wh, h);

    let maxAbs = 0;
    let sumAbs = 0;

    for (let y = 0; y < h; y++) {
      const r0 = ay.i0[y] * ww;
      const r1 = ay.i1[y] * ww;
      const ty = ay.t[y];
      for (let x = 0; x < w; x++) {
        const c0 = ax.i0[x];
        const c1 = ax.i1[x];
        const tx = ax.t[x];
        const w00 = (1 - tx) * (1 - ty);
        const w10 = tx * (1 - ty);
        const w01 = (1 - tx) * ty;
        const w11 = tx * ty;

        const dy = dY[r0 + c0] * w00 + dY[r0 + c1] * w10 + dY[r1 + c0] * w01 + dY[r1 + c1] * w11;
        const dcb = dCb[r0 + c0] * w00 + dCb[r0 + c1] * w10 + dCb[r1 + c0] * w01 + dCb[r1 + c1] * w11;
        const dcr = dCr[r0 + c0] * w00 + dCr[r0 + c1] * w10 + dCr[r1 + c0] * w01 + dCr[r1 + c1] * w11;

        const p = (y * w + x) * 4;
        const R = src[p];
        const G = src[p + 1];
        const B = src[p + 2];

        let Y = (0.299 * R + 0.587 * G + 0.114 * B) / 255 + dy;
        let Cb = (-0.168736 * R - 0.331264 * G + 0.5 * B) / 255 + 0.5 + dcb;
        let Cr = (0.5 * R - 0.418688 * G - 0.081312 * B) / 255 + 0.5 + dcr;

        if (Y < 0) Y = 0; else if (Y > 1) Y = 1;
        if (Cb < 0) Cb = 0; else if (Cb > 1) Cb = 1;
        if (Cr < 0) Cr = 0; else if (Cr > 1) Cr = 1;

        const cb = Cb - 0.5;
        const cr = Cr - 0.5;
        let nr = Y + 1.402 * cr;
        let ng = Y - 0.344136 * cb - 0.714136 * cr;
        let nb = Y + 1.772 * cb;

        nr = nr < 0 ? 0 : (nr > 1 ? 1 : nr);
        ng = ng < 0 ? 0 : (ng > 1 ? 1 : ng);
        nb = nb < 0 ? 0 : (nb > 1 ? 1 : nb);

        const o = nr * 255;
        const o2 = ng * 255;
        const o3 = nb * 255;
        out[p] = o;
        out[p + 1] = o2;
        out[p + 2] = o3;

        const dr = Math.abs(o - R);
        const dg = Math.abs(o2 - G);
        const db = Math.abs(o3 - B);
        const m = Math.max(dr, dg, db);
        if (m > maxAbs) maxAbs = m;
        sumAbs += dr + dg + db;
      }
    }

    const advY = new Float32Array(ww * wh);
    for (let i = 0; i < advY.length; i++) {
      let v = ch.Y[i] + dY[i];
      advY[i] = v < 0 ? 0 : (v > 1 ? 1 : v);
    }
    const lossAfter = loss(advY, basisL, ww * wh, opts.objective);
    const suppression = lossBefore > 1e-12
      ? (opts.objective === 'displace'
          ? (lossAfter - lossBefore) / Math.abs(lossBefore)
          : 1 - lossAfter / lossBefore)
      : 0;

    const stats = {
      width: w,
      height: h,
      workingWidth: ww,
      workingHeight: wh,
      iterations: opts.iterations,
      epsilon: opts.epsilon,
      objective: opts.objective,
      maxDelta: maxAbs,                          // in 0..255 units
      meanDelta: sumAbs / (n * 3),
      psnr: psnr(src, out),
      ssim: ssim(src, out, w, h),
      lumaBudget: epsLuma * 255,
      cbBudget: epsCb * 255,
      crBudget: epsCr * 255,
      suppression: suppression
    };

    return { data: out, width: w, height: h, stats: stats };
  }

  return {
    VERSION: VERSION,
    DEFAULTS: DEFAULTS,
    protect: protect,
    psnr: psnr,
    ssim: ssim,
    // internals exposed for the test suite
    _internals: {
      makeBasis: makeBasis,
      embedding: embedding,
      loss: loss,
      gradient: gradient,
      solveChannel: solveChannel,
      perceptualMask: perceptualMask,
      downsampleChannels: downsampleChannels,
      channelBudgets: channelBudgets,
      spectralShape: spectralShape,
      makeRng: makeRng,
      blurH: blurH,
      blurV: blurV
    }
  };
});
