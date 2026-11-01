# Vajra Image Guard

**Armour your photos before AI can edit them.** Vajra adds a perturbation you cannot
see — bounded, seeded and computed on your own device — that drags an image's
embedding towards a neutral target, so diffusion models conditioned on it produce
broken edits instead of convincing fakes.

Website (also the app): **https://merry8989-uk.github.io/vajra-image-guard/**
Android APK: **[docs/download/vajra-image-guard-1.0.0.apk](docs/download/vajra-image-guard-1.0.0.apk)**

- Runs entirely in the browser and in the APK. No server, no upload, no account.
- The Android build requests **zero permissions** — not even `INTERNET`.
- Deterministic: the same seed always produces the same protection.
- MIT licensed. Not affiliated with the authors of the paper below.

## What it is, honestly

The idea comes from *Raising the Cost of Malicious AI-Powered Image Editing*
(Salman, Khaddaj, Leclerc, Ilyas, Madry — [arXiv:2302.06588](https://arxiv.org/abs/2302.06588),
reference implementation `MadryLab/photoguard`). That work optimises an L∞-bounded
perturbation against the real Stable Diffusion VAE with gradients from the network.

Vajra reimplements the *procedure* from scratch — random init, sign-step PGD, an
L∞ projection back into a ball around the original, a neutral target — but runs it
against a **fixed linear surrogate encoder** (a bank of band-limited random
projections) instead of the VAE. A linear encoder has an analytic Jacobian, so the
solve needs no weights, no backprop and no GPU, and finishes in about a second on a
phone. That is the whole trade: same shape of attack, far less strength, small
enough to ship in a browser.

So: **this raises the cost of an edit, it does not prevent one.** A determined
person can crop, denoise or re-photograph the image; every one of those costs them
quality, which is the objective. For the strong version, `research/sd_encoder_attack.py`
runs the real encoder attack on a CUDA GPU.

No code, notebooks or assets were copied from the reference implementation.

## Using it

**Browser / PWA.** Open the site and drop an image. It is a static page; you can
also serve `docs/` from anywhere.

**Android.** Install the APK from [docs/download/](docs/download/), or from the
website. Android 7.0 (API 24) and up. Saved images land in `Pictures/Vajra` through
the system media store; the share sheet is wired up.

**Command line (reference implementation, numpy).**

```bash
python3 research/surrogate_pgd.py photo.jpg -o photo-vajra.png --strength 6
```

## How the protection works

1. **Surrogate encoder.** Four random patterns per channel are band-pass filtered
   (box radii 3, 6, 12), mean-removed and scaled to unit RMS. The embedding is the
   projection `e_k = mean(x · w_k)`.
2. **Neutral target.** Every pattern has zero mean, so a flat grey image embeds to
   the zero vector. That is the target — the same "push it towards nothing" idea as
   the paper's grey-image latent target.
3. **PGD.** Minimise `‖e(x)‖²` with sign steps of decaying size, then project each
   pixel back inside `[x - ε·m, x + ε·m]`. The gradient is analytic:
   `∂loss/∂x = 2·Σ e_k·w_k`.
4. **Budget split.** Luma takes half the budget; the remainder is divided by the
   1.402 and 1.772 chroma gains, so the worst-case shift on any *RGB* channel stays
   at or below `ε` while colour absorbs what the eye notices least.
5. **Perceptual mask.** Smoothed |Laplacian| gives each pixel a share of the budget
   between `maskFloor` and 1 — flat sky and skin get a fraction, texture gets all.
6. **Spectral projection.** The solved perturbation is low-passed and rescaled, so
   it lives in the mid band where JPEG's quantisation table does least damage.
7. **Upsampling.** The solve runs at 512 px and is bilinearly upsampled, which is
   what keeps a 12-megapixel photo under a second.

## How the defaults were chosen

Not guessed — measured with `npm run survival`, which runs the engine and then
JPEG round-trips the result at three qualities (4:2:0 subsampling, like most sites)
and correlates what is left of the perturbation with what was applied. Mean over a
smooth, a textured and a photo-like test image, ε = 6/255:

| config | PSNR | embedding suppression | q95 | q80 | q60 |
| --- | --- | --- | --- | --- | --- |
| bands 1,2,4 · no shaping | 47.1 dB | 96% | 73% | 47% | 30% |
| bands 2,4,8 · no shaping | 47.1 dB | 99% | 73% | 48% | 31% |
| bands 2,4,8 · r=1 | 51.4 dB | 96% | 62% | 45% | 34% |
| bands 3,6,12 · r=2 | 51.9 dB | 79% | 66% | 51% | 41% |
| **bands 3,6,12 · r=1 (shipped)** | **50.8 dB** | **91%** | **67%** | **50%** | **38%** |
| bands 4,8,16 · r=1 | 50.9 dB | 67% | 67% | 50% | 38% |

The shipped row keeps ~91% of the achievable suppression while being measurably
harder to see than the fine-band settings, and it survives re-compression better
than the original band choice. The app shows the same two numbers live (PSNR and
embedding suppression), plus a survival probe at whatever quality you choose.

## Repository layout

```
docs/            the website and the PWA (GitHub Pages serves this folder)
  engine.js      the protection engine — no DOM, no dependencies, unit tested
  app.js         UI, canvas work, JPEG survival probe, Android bridge
  download/      the signed APK
android/         WebView shell: one Activity, one bridge, no Gradle
research/        numpy port + the GPU encoder attack (see research/README.md)
scripts/         icon generator, survival measurement harness
tests/           node:test suite, python unittest suite, headless UI self-test
```

## Build the APK

No Gradle, no AndroidX — `aapt2`, `javac`, `d8`, `zipalign`, `apksigner`:

```bash
export ANDROID_SDK_ROOT=/path/to/android-sdk     # needs build-tools 34 and platform 34
export JAVA_HOME=/path/to/jdk-17
bash android/build-apk.sh
```

The script stages `docs/` into the APK's assets, so the app and the site can never
drift apart. The first run generates a signing key under `android/.keystore/`
(git-ignored) and reuses it afterwards — that key is what signs the APK published
here, so rebuilds upgrade cleanly over the download. It is not a Play Store key.

Two environment notes, both learned the hard way:

- d8 defaults to a 2 GB heap. Set `BUILD_HEAP=700M` on a small machine.
- If the container caps the address space (`ulimit -v`), the tool JVMs die with
  `Could not reserve enough space`; the script raises the soft limit when it can.

## Tests

```bash
npm test                    # 12 engine tests: gradient vs finite differences,
                            # ε bound, determinism, mask, PSNR/SSIM, guards
npm run test:py             # 10 reference-implementation tests
python3 research/surrogate_pgd.py --self-test
bash tests/run-ui-selftest.sh   # drives the real UI in headless Chromium
```

What each suite actually covers:

- **Engine (node).** The analytic gradient is checked against central finite
  differences for both objectives; PGD is shown to respect the ε ball per channel
  and to reduce the embedding; the output is shown to stay within ε in 8-bit space
  and not to mutate its input; determinism per seed; flat regions held below their
  budget; `psnr`/`ssim` sanity; bad input rejected.
- **Reference (python).** The same properties for the numpy port, plus a check that
  the channel budgets bound the worst-case RGB shift.
- **UI (headless Chromium).** Loads the real page, pushes a generated PNG through
  the real `<input type=file>`, clicks Protect, then asserts the metrics, the
  amplified-difference view, the JPEG survival probe and the strength slider.

## What has and has not been verified

Verified here, by the commands above: the engine and reference maths, the website
end to end in headless Chromium, and the APK — it builds, is zipaligned, is signed
with v2 **and** v3 schemes, declares `minSdkVersion 24` / `targetSdkVersion 34`, and
contains zero `uses-permission` entries.

**Not** verified: the APK has not been installed on a physical device or emulator
(no emulator in the build environment), and `research/sd_encoder_attack.py` has
never been executed — it needs a CUDA GPU and a downloaded VAE checkpoint. Both are
called out where they live. If you install the APK and something misbehaves, please
open an issue with your Android version.

## Limits and ethics

- A deterrent, not a guarantee. Crop, denoise or screen-photograph and it degrades.
- Re-compression eats part of it — check the survival probe before posting.
- Protected files are for publishing; keep the original. Same seed reproduces the
  same protection.
- Use it on your own photos, or images you have the rights to protect.

## Licence

MIT — see [LICENSE](LICENSE). Built on the idea in arXiv:2302.06588; no code from
that project was copied, and this project is not affiliated with its authors or MIT.
