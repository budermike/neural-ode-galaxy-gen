# 01 — Foundations: Neural ODEs from Scratch

Building intuition for continuous normalizing flows from first principles,
implemented entirely from scratch in PyTorch — no high-level ODE libraries,
no galaxy data yet.

This stage is split into three parts: first getting comfortable with PyTorch
itself, then using that foundation to implement Neural ODEs from scratch, and
finally scaling to CNF-based image generation on a simple benchmark dataset.

---

## Part A — Introduction to PyTorch

Before implementing ODE theory, I build familiarity with the tools used
throughout the entire project.

**Topics covered:**

- Tensors, autograd and the computational graph
- Writing custom Model class
- Training loops, optimizers and loss functions
- Dataset, DataLoader, split
- Evaluation
- Simple regression example

The goal here is not a general PyTorch tutorial — it is specifically to get
an understanding of PyTorch and then learn it while implementing the Neural ODE.

---

## Part B — Neural ODEs from Scratch

Starting from the core idea of the Neural ODE paper (Chen et al., 2019):
instead of stacking discrete layers, parameterize the *derivative* of the hidden
state with a neural network and integrate it forward in time.

**Topics implemented from scratch:**

- Euler solver, RK4 solver
- The adjoint sensitivity method for memory-efficient backpropagation through
  an ODE solver (the three coupled ODEs for state, adjoint, and ∂L/∂θ)
- An MLP vector field f(z, t, θ) as the continuous dynamics
- Training a CNF with self written methods to map a standard Gaussian to **circle** distribution
- Extending to a **spiral** distribution to probe expressiveness limits
- Optional: Visualizing the learned vector field and integration trajectories throughout
  training

---

## Part C — MNIST Digit Generation

Scaling the Neural ODE implementation to image data as a bridge toward the
galaxy pipeline.

**Topics covered:**

- Training a CNF on MNIST to generate handwritten digits in pixel space

---

## Status

**Done:** PyTorch introduction, ODE Solvers (RK4 and Euler)
**In progress:** Circle experiment running (Working on adjoint sensitivity method), spiral experiment next.
**Upcoming:** MNIST digit generation.

## References

- Chen et al. (2019) — *Neural Ordinary Differential Equations*
  [`arXiv:1806.07366`](https://arxiv.org/abs/1806.07366)