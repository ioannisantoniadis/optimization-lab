"""Backpropagation from scratch: implementing the chain rule directly rather than
leaning on JAX autodiff for the one place in this repo where doing so by hand is most
instructive — exactly the "worth deriving by hand at least once, for the pedagogy"
spirit `optimlab.core.Problem`'s own docstring states for gradients in general.
Cross-checked against `optimlab.core._autograd`'s JAX gradient on the identical network
and loss: two completely independent derivations of the same quantity (one
symbolic-by-hand, one automatic) agreeing is real evidence neither has a bug, not just
a plausible-looking number.
"""

from __future__ import annotations

import numpy as np

from optimlab.core import ArrayLike
from optimlab.highdim.nets import MLPShape, unflatten


def manual_mlp_gradient(flat_params: ArrayLike, shape: MLPShape, X: ArrayLike, y: ArrayLike) -> np.ndarray:
    """Gradient of the mean-squared-error loss (matching
    `optimlab.highdim.nets.mlp_training_problem`'s objective exactly) with respect to
    every flat parameter — a hand-implemented forward pass (caching every layer's
    pre-activation `z` and activation `a`) followed by a backward pass applying the
    chain rule through `tanh` hidden activations and a linear output layer, layer by
    layer from the output back to the input.
    """
    layers = [(np.asarray(W), np.asarray(b)) for W, b in unflatten(np.asarray(flat_params), shape)]
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n_layers = len(layers)

    activations = [X]
    pre_activations = []
    a = X
    for i, (W, b) in enumerate(layers):
        z = a @ W + b
        pre_activations.append(z)
        a = z if i == n_layers - 1 else np.tanh(z)
        activations.append(a)

    grad_W: list[np.ndarray] = [None] * n_layers  # type: ignore[list-item]
    grad_b: list[np.ndarray] = [None] * n_layers  # type: ignore[list-item]
    delta_a = 2.0 * (activations[-1] - y) / y.size  # dL/da at the output layer, for mean-squared error

    for i in reversed(range(n_layers)):
        W, _b = layers[i]
        z = pre_activations[i]
        delta_z = delta_a if i == n_layers - 1 else delta_a * (1.0 - np.tanh(z) ** 2)
        grad_W[i] = activations[i].T @ delta_z
        grad_b[i] = delta_z.sum(axis=0)
        delta_a = delta_z @ W.T  # propagate to the previous layer's activation gradient

    flat_grad_parts = []
    for gW, gb in zip(grad_W, grad_b, strict=True):
        flat_grad_parts.append(gW.ravel())
        flat_grad_parts.append(gb)
    return np.concatenate(flat_grad_parts)
