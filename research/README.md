# Research notes

Two things live here, and they are very different in strength.

## `surrogate_pgd.py` — what the app ships

A numpy port of the engine in `docs/engine.js`: band-limited random projections
stand in for the VAE encoder, PGD pushes the image's projection towards the zero
vector, the step is projected back into an L-infinity ball around the original.

It is fast, it needs no weights, it runs on a phone — and it is a **deterrent**,
not the paper's attack. Its self-test (`--self-test`) checks the analytic gradient
against finite differences, the budget, determinism and the quality floor, and the
same checks run in CI from `tests/test_surrogate.py`.

## `sd_encoder_attack.py` — the real thing

Backpropagates through an actual Stable Diffusion VAE (`stabilityai/sd-vae-ft-mse`)
and minimises the distance between the protected image's latent and the latent of a
flat grey image, which is what makes diffusion edits collapse into nonsense.

**It has not been executed in this repository.** It needs a CUDA GPU, `torch`,
`diffusers` and a downloaded checkpoint; the machines this project is built on have
none of those, so the file is shipped as a documented reference implementation and
nothing more. Do not treat it as verified code. The PGD loop is the same shape as
the one that *is* verified in `surrogate_pgd.py`, but the encoder is real, and real
means you have to test it on your own hardware.

## The idea, and where it comes from

"Raising the Cost of Malicious AI-Powered Image Editing" — Hadi Salman, Alaa
Khaddaj, Guillaume Leclerc, Andrew Ilyas, Aleksander Madry (arXiv:2302.06588). The
reference implementation is `MadryLab/photoguard`. Vajra reimplements the
*procedure* — random init, sign-step PGD, L-infinity projection, neutral target —
and nothing else: no code, assets or notebooks were copied from that repository.

## What the surrogate buys and what it costs

| | paper's encoder attack | Vajra's surrogate |
| --- | --- | --- |
| gradient source | real VAE, backprop | fixed linear projection, analytic |
| hardware | CUDA GPU | any browser, any phone |
| cost per image | ~200 forward/backward passes | ~50 ms – 1 s |
| strength against a determined editor | strong | raises the cost, does not prevent |

The honest summary is the one on the website: this raises the price of an edit. It
does not make one impossible, and no consumer tool does.
