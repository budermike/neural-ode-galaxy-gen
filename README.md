# neural-ode-galaxy-gen

> A step-by-step exploration of Continuous Normalizing Flows applied to galaxy image generation.
> Starting from Neural ODE fundamentals on synthetic datasets, the project progresses to 64x64-resolution
> generation on Galaxy10 DECaLS dataset. It includes a comparison with VAE baselines and implements a VAE/CNF hybrid model featuring a latent-space conditioning pipeline for controlled morphology synthesis.

---

## Overview

This project builds and studies **Continuous Normalizing Flows (CNFs)** powered by **Neural ODEs** as a generative model for galaxy morphology.
Rather than jumping straight to a large training run, the project is structured as a deliberate learning progression across three stages, each one grounded in theory before moving to implementation.

The dataset used in Stages 2 and 3 is **Galaxy10 DECaLS**, the same dataset from a prior VAE-based image generation final exam project.
This overlap is intentional: it allows a clean, apples-to-apples comparison between the VAE and CNF paradigms on identical data.
