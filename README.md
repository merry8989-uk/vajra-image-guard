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

## What it is, honestly — and which half works

The idea comes from *Raising the Cost of Malicious AI-Powered Image Editing*
(Salman, Khaddaj, Leclerc, Ilyas, Madry — [arXiv:2302.06588](https://arxiv.org/abs/2302.06588),
reference implementation `MadryLab/photoguard`). That work minimises
`||E(x + δ) − z_target||²` under `||δ||∞ ≤ ε`, where **E is the real Stable
Diffusion VAE encoder** and `z_target` is the latent of a flat grey image.

There are two implementations here, and they are not two strengths of the same
thing. Measured on the same image, at the same invisible pixel budget, in the real
VAE's latent space:

| image | latent distance to grey | gap closed | PSNR | max shift | VAE view shift |
| --- | --- | --- | --- | --- | --- |
| original | 3889 | — | — | 0/255 | 0.00/255 |
| **browser / app engine** (linear surrogate) | 3864 | **0.6%** | 43.9 dB | 7/255 | **2.18/255** |
| **real encoder attack** (8 PGD steps) | 423 | **89.1%** | 29.0 dB | 15/255 | **33.4/255** |

Reproduce with `python3 research/measure_attack.py --steps 8 --size 192`.

Read that table plainly: the browser engine changes the perturbation in a way the
model barely notices — it closes six parts in a thousand of the gap it is aiming
at — so an AI editor will still make a convincing edit of a "protected" image from
the app. **The browser tool and the APK do not protect your images.** They are
shipped because they are honest about their own numbers and because the maths is
interesting, not because they work.

The **real attack works**: it moves the latent by 89% of the gap in eight steps, and
with the paper's 200 steps at 512 px on a GPU it is stronger still. It needs a GPU,
so it runs in Colab or on a desktop:

- **Colab (free):** open `notebooks/vajra_colab.ipynb`, pick a T4 runtime, run all.
- **Locally:** `pip install torch diffusers safetensors pillow` then
  `python3 research/sd_encoder_attack.py photo.jpg --steps 200 --eps 0.12`

Even the real attack is a deterrent, not a guarantee: a determined person can crop,
denoise or re-photograph the image, and each of those costs them quality — which is
the objective. Newer models are also more robust than the Stable Diffusion-era ones
this was designed against.

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
research/        numpy port, the real GPU encoder attack, and the measurement harness
notebooks/       the Colab notebook that runs the real attack on a free GPU
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
python3 -m unittest tests.test_encoder_attack -v   # 7 tests, needs torch
bash tests/run-ui-selftest.sh   # drives the real UI in headless Chromium
```

The encoder-attack tests are skipped when torch is not installed, and need no GPU
and no VAE download: the PGD loop is tested against a tiny convolution standing in
for an encoder, which pins down the ε ball, the loss decreasing, masking and
determinism.

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

Verified here, by the commands above: the engine and reference maths; the website
end to end in headless Chromium; the APK — it builds, is zipaligned, is signed with
v2 **and** v3 schemes, declares `minSdkVersion 24` / `targetSdkVersion 34`, and
contains zero `uses-permission` entries; and the real encoder attack — its PGD loop
is unit tested, and it was run against `sd-vae-ft-mse` on CPU, cutting the latent
distance from 3889 to 423 (89.1%) in 8 steps with a maximum pixel change of 15/255.

**Not** verified: the APK has never been installed on a physical device or emulator
(no emulator in the build environment); and the end-to-end effect of the real attack
on a full Stable Diffusion inpaint/img2img run, which needs a GPU with far more
memory than the machine this was built on. The paper reports that result — this
repository does not reproduce it, and does not claim to. If you install the APK and
something misbehaves, please open an issue with your Android version.

## Limits and ethics

- A deterrent, not a guarantee. Crop, denoise or screen-photograph and it degrades.
- Re-compression eats part of it — check the survival probe before posting.
- Protected files are for publishing; keep the original. Same seed reproduces the
  same protection.
- Use it on your own photos, or images you have the rights to protect.

## Licence

MIT — see [LICENSE](LICENSE). Built on the idea in arXiv:2302.06588; no code from
that project was copied, and this project is not affiliated with its authors or MIT.
