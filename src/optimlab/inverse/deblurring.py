"""Image deblurring as a linear inverse problem: recover an unknown image `x` from a
blurred, noisy observation `b = Ax + noise`, where `A` is a known blur operator. Unlike
Chapter 3's regression (`A` there came from data, arbitrary and usually well-behaved),
`A` here is a smoothing operator by construction — it destroys exactly the
high-frequency detail a sharp image needs, which is what makes deblurring genuinely
ill-posed rather than merely noisy: without regularization, inverting `A` amplifies
whatever noise happens to live in those same near-annihilated frequencies without
bound. Posed here as an ordinary `optimlab.core.Problem` (mean-squared reconstruction
error plus an L2 penalty, i.e. Tikhonov regularization — the exact same idea as Chapter
3's `ridge_regression`, just solved by gradient descent on an image instead of the
closed-form SVD route) so the existing solvers recover it directly.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from optimlab.core import ArrayLike, Problem


def gaussian_blur_matrix(n: int, sigma: float) -> np.ndarray:
    """An `n x n` circulant (periodic-boundary) Gaussian blur operator. Convolving with
    a Gaussian kernel is linear, so it's exactly representable as a matrix; circulant
    boundary handling keeps every row an identical shifted copy of the same kernel, the
    simplest case to reason about (no special-casing image edges).
    """
    positions = np.arange(n)
    kernel = np.exp(-0.5 * (np.minimum(positions, n - positions) / sigma) ** 2)
    kernel /= kernel.sum()
    return np.stack([np.roll(kernel, i) for i in range(n)])


def blur_image(image: ArrayLike, sigma: float) -> np.ndarray:
    """Apply a separable 2D Gaussian blur (blur every row, then every column) — exactly
    equivalent to a genuine 2D Gaussian convolution, computed as two cheap `n x n`
    matrix multiplies instead of one much larger `(n*m) x (n*m)` operator.
    """
    image = np.asarray(image, dtype=float)
    n_rows, n_cols = image.shape
    A_rows = gaussian_blur_matrix(n_rows, sigma)
    A_cols = gaussian_blur_matrix(n_cols, sigma)
    return A_rows @ image @ A_cols.T


def deblurring_problem(observed: ArrayLike, *, sigma: float, alpha: float) -> Problem:
    """Recover the sharp image as `argmin_x ||blur(x) - observed||^2 + alpha ||x||^2`
    — the mean-squared reconstruction error (how well a re-blurred guess matches what
    was actually observed) plus a Tikhonov penalty discouraging the wild
    high-frequency noise amplification an unregularized inverse would produce.
    `x0` starts from the observed (blurry) image itself, already a much better guess
    than random noise since the true image and its blur share the same low frequencies.
    """
    observed = np.asarray(observed, dtype=float)
    n_rows, n_cols = observed.shape
    A_rows = jnp.asarray(gaussian_blur_matrix(n_rows, sigma))
    A_cols = jnp.asarray(gaussian_blur_matrix(n_cols, sigma))
    observed_j = jnp.asarray(observed)

    def loss(flat_image: ArrayLike) -> float:
        image = flat_image.reshape(n_rows, n_cols)
        blurred = A_rows @ image @ A_cols.T
        return jnp.mean((blurred - observed_j) ** 2) + alpha * jnp.mean(flat_image**2)

    return Problem(f=loss, x0=observed.ravel(), name="image_deblurring")
