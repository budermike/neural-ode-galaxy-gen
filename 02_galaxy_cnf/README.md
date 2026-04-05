# 02 — Galaxy CNF: Scaling to Real Image Data

Scaling the continuous normalizing flow from toy 2D distributions to real
64×64 galaxy images from the Galaxy10 DECaLS dataset. The central question:
can a CNF learn the complex, multimodal structure of galaxy morphology —
and does it outperform the VAE baseline trained during the exam project?

---

## Goals

- Scale the Neural ODE architecture from Part 01 to high-dimensional image data
- Systematically study the architectural and numerical tradeoffs that appear at this scale
- make a comparison against the exam VAE

---

## Status

**Planned** — begin after Part 01 adjoint implementation is validated on the spiral experiment.

## References

- Chen et al. (2019) — *Neural Ordinary Differential Equations*
  [`arXiv:1806.07366`](https://arxiv.org/abs/1806.07366)
- Lipman et al. (2024) — *Flow Matching Guide and Code*
  [`arXiv:2412.06264`](https://arxiv.org/abs/2412.06264)