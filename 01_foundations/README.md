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
- Extending to a **spiral** distribution

---

## Part C — MNIST Digit Generation

Scaling the Neural ODE implementation to image data as a bridge toward the
galaxy pipeline.

**Topics covered:**

- Training a CNF on MNIST to generate handwritten digits in pixel space
- Visualize the trained distribution with UMAP and compare it with original MNIST data distribution.

---

## Status

**Done:** PyTorch introduction, ODE Solvers (RK4 and Euler), Adjount Method, Model Class, Training loop, Evaluation, Hutchinson Trace Estimate, Regularisation for Circle/Moon, Circle experiment, Spiral experiment, ResNet-Block, FilM injection of time t, UMAP of original MNIST data, FFJord implementation

**In progress:** CNF with CNN, MNIST digit generation, correct 

---

## Known Issues & Findings

### Circle Distribution: Training Difficulty

The circle distribution proved challenging to learn due to a mismatch between the Gaussian prior and the target geometry. The prior concentrates most of its probability mass near the origin, whereas the target distribution places mass along a ring, precisely where the prior is sparsest. This forces the vector field to transport a large amount of probability mass over long distances, making training slow.

After a sufficient number of epochs the model produced output loosely resembling two arcs, at which point the experiment was stopped. A less concentrated prior (e.g. uniform) would be a better choice for ring-shaped targets in future runs.


### Trace Cheating

TThe biggest current issue is trace cheating: the model exploits the Jacobian trace to drive delta_log_p artificially negative, inflating the log-likelihood without learning a meaningful density.
An attempt to add a regularization penalty on delta_log_p (bounded from below) had limited effect — it slightly reduced trace cheating in the 2D run, but failed to contain it in the MNIST run.
Possible root causes:

1. FiLM scaling (γ): The learned scale parameter γ directly multiplies the diagonal of the Jacobian, scaling the trace at every conditioned layer.
2. UNet architecture: Skip connections add independent Jacobian contributions (additive), while the main encoder–decoder path compounds γ-scaling multiplicatively through the chain rule — amplifying the effect across depth.
3. Hutchinson trace estimator variance: The stochastic trace estimate introduces noise that the optimizer can exploit, pushing estimates further negative.

Next step: Strip back to a simpler architecture close to the original FFJORD paper — a single CNF without skip connections, with time t concatenated as an additional input channel.

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
[`circle_spiral_flow.ipynb`](./circle_spiral_flow.ipynb)

## References

- Chen et al. (2019) — *Neural Ordinary Differential Equations*
  [`arXiv:1806.07366`](https://arxiv.org/abs/1806.07366)
- Grathwohl et al. (2018) — *FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models*
  [`arXiv:1810.01367`](https://arxiv.org/abs/1810.01367)
