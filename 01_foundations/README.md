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

**Done:** PyTorch introduction, ODE Solvers (RK4 and Euler), Adjount Method, Model Class, Training loop, Evaluation, Hutchinson Trace Estimate, FFJORD-Regularisation, Circle experiment, Spiral experiment

**In progress:** MNIST digit generation.

---

## Known Issues & Findings

### ~~Trace Cheating~~
~~The biggest current issue is trace cheating: the model exploits the
Jacobian trace to drive delta_log_p artificially large,
causing the NLL loss to diverge toward -∞ instead of learning the target
distribution. Addressed with FFJORD regularization, Hutchinson trace
estimation and gradient clipping, partially resolved but not fully stable.~~
> *Root cause: wrong sign, not trace cheating*

### Circle Distribution: Training Difficulty

The circle distribution proved challenging to learn due to a mismatch between the Gaussian prior and the target geometry. The prior concentrates most of its probability mass near the origin, whereas the target distribution places mass along a ring, precisely where the prior is sparsest. This forces the vector field to transport a large amount of probability mass over long distances, making training slow.

After a sufficient number of epochs the model produced output loosely resembling two arcs, at which point the experiment was stopped. A less concentrated prior (e.g. uniform) would be a better choice for ring-shaped targets in future runs.

### Implementation findings

- **Incorrect MLE formulation**: early runs omitted the delta_log_p term in the change-of-variables formula, causing the model to receive no learning signal. Subsequently, a sign error in the vector field MLE caused the model to push probability mass outward rather than concentrating it.
- **Activation functions**: ReLU replaced with Tanh for better
  differentiability and Lipschitz continuity of the vector field
- **Double Tanh bug**: output layer had Tanh applied twice, artificially
  bounding the vector field to [-1.5, 1.5] and preventing the model from
  transporting mass far enough from the origin
- **First larger PyTorch project**: encountered typical beginner issues with
  batch dimensions propagating correctly through the augmented state vector
  [z, delta_log_p],  particularly when concatenating and slicing the state
  across the ODE solver, adjoint backward pass, and loss calculation

---

## Results & Visualizations

Plots (circle/spiral outputs, training history)
are documented inline in the notebook:
📓 [`circle-spiral-flow.ipynb`](./circle-spiral-flow.ipynb)

## References

- Chen et al. (2019) — *Neural Ordinary Differential Equations*
  [`arXiv:1806.07366`](https://arxiv.org/abs/1806.07366)
