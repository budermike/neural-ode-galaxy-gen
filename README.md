# neural-ode-galaxy-gen

> A step-by-step exploration of Continuous Normalizing Flows applied to galaxy image generation.
> Starting from Neural ODE fundamentals on synthetic datasets, the project progresses to 64×64-resolution
> generation on the Galaxy10 DECaLS dataset. It includes a comparison against a VAE baseline and lays out a
> planned LLM/CNF hybrid extension: a text-conditioning pipeline for controlled morphology synthesis.

---

## Overview

This project builds and studies **Continuous Normalizing Flows (CNFs)** powered by **Neural ODEs** as a generative model for galaxy morphology.
Rather than jumping straight to a large training run, the project is structured as a deliberate learning progression across three stages, each one grounded in theory before moving to implementation.

The dataset used in Stages 2 and 3 is **Galaxy10 DECaLS**, the same dataset from a prior VAE-based image generation final exam project.
This overlap is intentional: it allows a clean, apples-to-apples comparison between the VAE and CNF paradigms on identical data.

---

## Motivation

A Variational Autoencoder encodes images into a Gaussian latent space and reconstructs via a learned decoder.
The KL regularization that keeps this latent space well-behaved and the lower dimensional latent space tends to smooth away high-frequency detail — the characteristic softness of VAE outputs.

A CNF instead learns a continuous vector field whose flow transforms a simple base distribution (e.g. a Gaussian) into the data distribution.
The transformation is exact and invertible, avoiding the need for a low-dimensional latent space bottleneck.

**The central question of this project:** can a CNF trained directly in pixel space generate sharper, more morphologically faithful galaxy images than the VAE baseline — and is effective text-conditioned generation possible using automatically generated captions as conditioning signals?

Stage 2 answers the first half (yes); the text-conditioning half is the postponed Stage 3.

---

## Project Structure

```
.
├── 01_foundations/                   # Stage 1: PyTorch + Neural ODE / CNF from scratch (2D toys, MNIST)
│   ├── pytorch_introduction.ipynb
│   ├── circle-spiral-flow.ipynb
│   ├── circle-spiral-flow-wrong-MLE.ipynb    # earlier MLE version (incorrect formulation, kept as record)
│   ├── neural-ode-mnist-gen.ipynb            # CNF via MLE
│   ├── neural-ode-mnist-flowmatching.ipynb   # CNF via Flow Matching
│   ├── images/
│   ├── data/
│   └── README.md
│
├── 02_galaxy_cnf/                    # Stage 2: Flow-Matching CNF on Galaxy10 DECaLS + VAE comparison
│   ├── neural_ode_galaxy_gen.ipynb           # CNF (Flow Matching)
│   ├── vae_galaxy_gen.ipynb                  # VAE baseline
│   ├── CNF_vs_VAE_evaluation.ipynb           # power spectrum, FID, UMAP
│   ├── images/
│   ├── data/
│   └── README.md
│
├── 03_llm_cnf_hybrid (postponed)/    # Stage 3 (postponed): text-conditioned CNF
│   ├── llm_cnf_hybrid.ipynb
│   ├── CNN-galaxy-classifier.ipynb           # morphology classifier (from exam), reused for evaluation
│   └── README.md
│
├── requirements.txt
├── LICENSE                           # MIT License
└── README.md
```

---

## Stages

### Stage 1 — Neural ODE Fundamentals

Building intuition for continuous flows from first principles, without any galaxy data. A structured path from PyTorch fundamentals to CNF-based image generation on a simple benchmark.

- PyTorch fundamentals and ODE solvers (Euler, RK4) implemented from scratch
- Neural ODE and adjoint-based backpropagation from scratch
- Learning the flow from a standard Gaussian to a circle distribution, then a spiral distribution
- Scaling to MNIST digit generation as a bridge toward the galaxy pipeline
- **Key finding:** MLE-based CNF training is intractable on a single GPU at image scale (trace cheating and mode collapse); Flow Matching resolves this structurally by replacing the likelihood objective with direct regression on the vector field

**Status:** Complete

---

### Stage 2 — CNF on Galaxy10 DECaLS

Scaling the CNF to real image data with Flow Matching and systematically comparing it against the VAE baseline.

- Flow-Matching CNF with a U-Net vector field (ResBlocks, GroupNorm, SiLU, skip connections), `[-1, 1]` preprocessing, time `t` as an extra input channel
- Evaluation: radially-averaged power spectrum (domain-appropriate primary metric), FID, and UMAP of the learned manifold
- VAE comparison on identical data: sharpness, high-frequency content, manifold coverage
- **Result:** the CNF clearly outperforms the VAE — FID 55.66 vs. 108.08, a flat power-spectrum log-difference vs. the VAE's high-frequency falloff, and broad manifold coverage without wholesale mode collapse

**Status:** Complete

---

### (Optional) Stage 3 — Text-Conditioned Image Generation via CNF

Training a Continuous Normalizing Flow directly in pixel space, conditioned on semantic text embeddings.

- Generate per-image natural language captions from raw galaxy images using **Qwen3-VL 7B**
- Encode captions into dense **384-dimensional** embedding vectors via **Snowflake Arctic Embed S** (33M parameters)
- Train a Flow-Matching CNF conditioned on the text embedding via **FiLM** injection into the vector field
- Evaluate prompt-following with the reused **CNN morphology classifier**: does an image generated from a caption classify as the intended morphology class?

**Status:** Postponed — will be done when time allows

---

## Results

**Stage 2 (complete).** A Flow-Matching CNF trained directly in pixel space clearly outperforms the VAE baseline on Galaxy10 DECaLS. The CNF reproduces the DECaLS texture — spiral arms, edge-on disks, colour variation, and the noisy, star-speckled background — while the VAE recovers recognizable morphology but softens it and denoises the background, losing high-frequency detail.

![CNF-generated galaxies](02_galaxy_cnf/images/CNF_generated_cropped.jpeg)
*CNF (Flow Matching) samples.*

![VAE-generated galaxies](02_galaxy_cnf/images/VAE_generated_cropped.jpeg)
*VAE baseline samples.*

The gap is quantitative, not just visual:

- **FID:** 55.66 (CNF) vs. 108.08 (VAE) — the CNF roughly halves the score.
- **Radial power spectrum:** the CNF tracks the real spectrum across frequencies; the VAE falls off at high frequency — the frequency-domain signature of its softness.

![Radially-averaged power spectrum](02_galaxy_cnf/images/power_spectrum.png)

- **UMAP:** generated samples spread across the real manifold (no wholesale mode collapse), with dense clusters dominated by classes 2, 3, 7 and 9; finer-morphology classes are generated less often — a learnability limit, and the clearest place for a future version to improve.

**Stage 1 (complete).** The key outcome is a diagnostic rather than a score: MLE-based CNF training is intractable on a single GPU at image scale, and Flow Matching resolves it structurally.

See each stage's `README.md` for the full write-up and additional figures.

**Status:** Stages 1–2 complete; Stage 3 postponed.

---

## References

- Chen et al. (2019) — [Neural Ordinary Differential Equations](https://arxiv.org/abs/1806.07366)
- Lipman et al. (2023) — [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
- Lipman et al. (2024) — [Flow Matching Guide and Code](https://arxiv.org/abs/2412.06264)
- Grathwohl et al. (2018) — [FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models](https://arxiv.org/abs/1810.01367)
- Finlay et al. (2020) — [How to Train Your Neural ODE: the World of Jacobian and Kinetic Regularization](https://arxiv.org/abs/2002.02798)

---

## About

BSc project in Applied Mathematics & Machine Learning at the University of Zurich.
A hands-on investigation into continuous normalizing flows for image generation — combining ODE theory with applied deep learning on real astrophysical image data.
