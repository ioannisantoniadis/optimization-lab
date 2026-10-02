"""Loss-landscape visualization at scale (Li et al. 2018): a trained network's weight
space has hundreds to billions of dimensions, far past anything a contour plot can show
directly — but the loss along a random 2D *slice* through that space, evaluated at
`theta* + a*d1 + b*d2` for two random directions `d1, d2`, is still just an ordinary 2D
function, exactly the kind `optimlab.viz.landscape.contour_figure` already draws for
low-dimensional test functions. The one subtlety is *which* random directions: a
component-wise random direction re-scaled uniformly puts far more of its "step size"
into whichever layer happens to have the largest weight norm, which can make the same
step size look sharp in one network's landscape and flat in another's purely from
weight-scale differences that have nothing to do with the loss surface's actual shape.
Filter normalization fixes this by rescaling the random direction one *filter* at a time:
each output neuron's incoming weights in the direction get the same norm as that neuron's
weights in `theta*` (Li et al. 2018, Sec. 4 — for a fully connected layer "the filter
corresponds to the weights that generate one neuron"). Rescaling a whole layer at once is a
different, coarser scheme the paper calls layer normalization. Bias directions are zeroed,
matching the default (`ignore='biasbn'`) of the authors' reference code.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from optimlab.core import ArrayLike, Objective
from optimlab.highdim.nets import MLPShape, unflatten


def filter_normalized_direction(base_params: ArrayLike, shape: MLPShape, *, seed: int = 0) -> np.ndarray:
    """A random direction the same size as `base_params`, rescaled filter-by-filter so
    each neuron's incoming-weight vector in the direction has the same norm as that
    neuron's weights in `base_params` — comparable "step sizes" across neurons and layers
    regardless of how differently-scaled the weights happen to be. Layers here compute
    `a @ W + b` with `W` of shape `(n_in, n_out)`, so one filter is one *column* of `W`.
    Bias directions are zero.
    """
    rng = np.random.default_rng(seed)
    base_layers = unflatten(np.asarray(base_params), shape)
    direction_parts = []
    for W, b in base_layers:
        W, b = np.asarray(W), np.asarray(b)
        dW = rng.standard_normal(W.shape)
        dW *= np.linalg.norm(W, axis=0) / (np.linalg.norm(dW, axis=0) + 1e-10)
        direction_parts.append(dW.ravel())
        direction_parts.append(np.zeros_like(b))
    return np.concatenate(direction_parts)


@dataclass
class LossSlice2D:
    A: np.ndarray
    B: np.ndarray
    Z: np.ndarray


def loss_landscape_slice(
    f: Objective,
    base_params: ArrayLike,
    shape: MLPShape,
    *,
    span: float = 1.0,
    resolution: int = 40,
    seed: int = 0,
) -> LossSlice2D:
    """`f` evaluated on a grid of `base_params + a*d1 + b*d2`, `a, b` ranging over
    `[-span, span]`, `d1`/`d2` independent filter-normalized random directions.
    """
    d1 = filter_normalized_direction(base_params, shape, seed=seed)
    d2 = filter_normalized_direction(base_params, shape, seed=seed + 1)
    base_params = np.asarray(base_params)

    coords = np.linspace(-span, span, resolution)
    A, B = np.meshgrid(coords, coords)
    Z = np.empty_like(A)
    for i in range(resolution):
        for j in range(resolution):
            Z[i, j] = float(f(base_params + A[i, j] * d1 + B[i, j] * d2))
    return LossSlice2D(A=A, B=B, Z=Z)


def linear_interpolation_loss(
    f: Objective, theta_a: ArrayLike, theta_b: ArrayLike, *, n_points: int = 50, extrapolate: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Goodfellow et al. 2015's cheaper diagnostic: loss along the straight line
    `(1-alpha)*theta_a + alpha*theta_b`, `alpha` ranging a bit past `[0, 1]` if
    `extrapolate > 0` — far cheaper than a full 2D slice (one line, not a grid), and
    often enough on its own to see whether two points are separated by a real loss
    barrier or just sit in the same basin.
    """
    theta_a = np.asarray(theta_a)
    theta_b = np.asarray(theta_b)
    alphas = np.linspace(-extrapolate, 1.0 + extrapolate, n_points)
    losses = np.array([float(f((1 - a) * theta_a + a * theta_b)) for a in alphas])
    return alphas, losses
