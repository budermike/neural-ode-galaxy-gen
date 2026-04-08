# neural-ode-galaxy-gen

> A step-by-step exploration of Continuous Normalizing Flows applied to galaxy image generation.
> Starting from Neural ODE fundamentals on synthetic datasets, the project progresses to 64x64-resolution
> generation on Galaxy10 DECaLS dataset. It includes a comparison with VAE baselines and implements a LLM/CNF hybrid model featuring a text conditioning pipeline for controlled morphology synthesis.

---

## Overview

This project builds and studies **Continuous Normalizing Flows (CNFs)** powered by **Neural ODEs** as a generative model for galaxy morphology.
Rather than jumping straight to a large training run, the project is structured as a deliberate learning progression across three stages, each one grounded in theory before moving to implementation.

The dataset used in Stages 2 and 3 is **Galaxy10 DECaLS**, the same dataset from a prior VAE-based image generation final exam project.
This overlap is intentional: it allows a clean, apples-to-apples comparison between the VAE and CNF paradigms on identical data.

---

## Motivation

A Variational Autoencoder encodes images into a Gaussian latent space and reconstructs via a learned decoder.
The KL regularization that keeps this latent space well-behaved also smears reconstructions, the characteristic blurriness of VAE outputs.

A CNF instead learns a continuous vector field whose flow transforms a simple base distribution (e.g. a Gaussian) into the data distribution.
The transformation is exact and invertible, avoiding the need for a low-dimensional latent space bottleneck.

*The central question of this project:* can a CNF trained directly in pixel space generate sharper, more morphologically faithful galaxy images than the VAE baseline and is effective text-conditioned generation possible using automatically generated captions as conditioning signals?

---

## Project Structure

```
.
├── 01_foundations/          # Stage 1: Introduction to Pytorch and Neural ODE from scratch on toy datasets
│   ├── circle_spiral_flow.ipynb
│   ├── pytorch_introduction.ipynb
│   └── README.md
│
├── 02_galaxy_cnf/           # Stage 2: CNF on Galaxy10 DECaLS + VAE comparison
│   ├── neural-ode-galaxy-gen.ipynb
│   ├── vae-galaxy-gen.ipynb
│   └── README.md
│
├── 03_llm_cnf_hybrid/       # Stage 3: llm text conditioning + CNF image generation
│   ├── llm_cnf_hybrid.ipynb
│   └── README.md
│
├── results/                 # Generated samples, loss curves, comparisons
└── requirements.txt
```

---

## Stages

### Stage 1 — Neural ODE Fundamentals

<!-- TODO: Fill in once Stage 1 is complete -->

Building intuition for continuous flows from first principles, without any galaxy data.

- Implementing neural ODE and adjoint-based backpropagation from scratch
- Learning the flow from a standard Gaussian to a **circle** distribution
- Extending to a **spiral** distribution to probe expressiveness
- Visualizing the learned vector field and integration trajectories over training

**Status:** In progress

---

### Stage 2 — CNF on Galaxy10 DECaLS

<!-- TODO: Fill in once Stage 2 is complete -->

Scaling the CNF to real image data and systematically comparing it against the VAE baseline.

- Architecture: ODE function depth, solver choice (Euler, Runge-Kutta, Dopri5, fixed-step, etc), time discretization
- Training: NFE (number of function evaluations), loss curves, numerical stability
- Evaluation: FID, visual sharpness, morphology diversity across galaxy classes
- VAE comparison: reconstruction quality, blurriness analysis, latent space geometry

**Status:** Planned

---

### (Optional) Stage 3 — Text-Conditioned Image Generation via CNF

<!-- TODO: Fill in once Stage 3 is complete -->

Training a Continuous Normalizing Flow directly in pixel space, conditioned on semantic text embeddings.

- Generate per-image natural language captions from raw galaxy images using Qwen3-VL 7B, producing a rich text description.
- Encode captions into dense 768-dimensional embedding vectors via Snowflake Arctic Embed 33M, an embedding model that ensures semantically similarity.
- Train a CNF to learn the mapping from a Gaussian prior directly to 64×64 pixel space, conditioned on the text embedding.
- Generate galaxy images by integrating the learned ODE from noise to data, guided by the text conditioning signal.

**Status:** Planned

---

## Results

<!-- TODO: Add generated samples, FID scores, and visual comparisons once available -->

**Status:** In progress

---

## References

- Chen et al. (2019) — [Neural Ordinary Differential Equations](https://arxiv.org/abs/1806.07366)
- Lipman et al. (2024) — [Flow Matching Guide and Code](https://arxiv.org/abs/2412.06264)

---

## About

BSc project in Applied Mathematics & Machine Learning at the University of Zurich.
A hands-on investigation into continuous normalizing flows for image generation — combining ODE theory with applied deep learning on real astrophysical image data.
