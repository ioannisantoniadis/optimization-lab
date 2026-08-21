"""A minimal multi-layer perceptron whose *entire* parameter vector is a single flat
array, so it plugs directly into `optimlab.core.Problem` — "training a neural network"
becomes literally the same optimization interface used on a two-dimensional test
function throughout Chapters 1-4, just with hundreds of dimensions instead of two. This
is the backbone every other module in `optimlab.highdim` builds on: the Hessian
eigenspectrum, loss-landscape slices, and mode connectivity all need an actual trained
model's flat parameter vector to operate on.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from optimlab.core import ArrayLike, Problem


@dataclass
class MLPShape:
    """Static layer sizes, e.g. `[2, 32, 32, 1]` for a 2-input, 1-output network with
    two 32-unit hidden layers — kept separate from the flat parameter *values* so
    `forward` stays a pure function of the flat array alone (what JAX autodiff traces
    through); this object is closed over in a training `Problem`'s objective, never
    passed as a traced argument.
    """

    layer_sizes: list[int]

    @property
    def _layer_pairs(self) -> list[tuple[int, int]]:
        return list(zip(self.layer_sizes[:-1], self.layer_sizes[1:], strict=True))

    @property
    def n_params(self) -> int:
        return sum(n_in * n_out + n_out for n_in, n_out in self._layer_pairs)


def init_params(shape: MLPShape, *, seed: int = 0, scale: float = 1.0) -> np.ndarray:
    """Flat parameter vector: each layer's weights `N(0, (scale/sqrt(n_in))^2)` (the
    standard "keep activation variance roughly constant through depth" scaling), biases
    at zero.
    """
    rng = np.random.default_rng(seed)
    parts = []
    for n_in, n_out in shape._layer_pairs:
        W = rng.standard_normal((n_in, n_out)) * scale / np.sqrt(n_in)
        b = np.zeros(n_out)
        parts.append(W.ravel())
        parts.append(b)
    return np.concatenate(parts)


def unflatten(flat: ArrayLike, shape: MLPShape) -> list[tuple[ArrayLike, ArrayLike]]:
    """The inverse of `init_params`'s concatenation — a list of `(W, b)` pairs, one per
    layer, sliced out of the flat vector in the same order they were packed in.
    """
    flat = jnp.asarray(flat)
    layers = []
    offset = 0
    for n_in, n_out in shape._layer_pairs:
        w_size = n_in * n_out
        W = flat[offset : offset + w_size].reshape(n_in, n_out)
        offset += w_size
        b = flat[offset : offset + n_out]
        offset += n_out
        layers.append((W, b))
    return layers


def forward(flat: ArrayLike, shape: MLPShape, X: ArrayLike) -> ArrayLike:
    """`tanh` hidden activations, linear output layer. `X` is `(n_samples, n_in)`;
    returns `(n_samples, n_out)`.
    """
    layers = unflatten(flat, shape)
    activation = jnp.asarray(X, dtype=jnp.float64)
    for i, (W, b) in enumerate(layers):
        activation = activation @ W + b
        if i < len(layers) - 1:
            activation = jnp.tanh(activation)
    return activation


def mlp_training_problem(
    shape: MLPShape, X: ArrayLike, y: ArrayLike, *, seed: int = 0, name: str = "mlp_training"
) -> Problem:
    """Mean-squared-error regression loss as an ordinary `Problem` whose `x` *is* the
    network's flat weight vector — any solver in `optimlab.optimizers` already trains
    this network with no changes. `y` must already be shaped `(n_samples, n_out)`
    (matching `forward`'s output shape), the network's own convention rather than the
    1D-per-sample shape plain regression code elsewhere in this repo uses.
    """
    X_j = jnp.asarray(X, dtype=jnp.float64)
    y_j = jnp.asarray(y, dtype=jnp.float64)

    def loss(flat_params: ArrayLike) -> float:
        preds = forward(flat_params, shape, X_j)
        return jnp.mean((preds - y_j) ** 2)

    x0 = init_params(shape, seed=seed)
    return Problem(f=loss, x0=x0, name=name)
