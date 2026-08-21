"""Mode connectivity (Garipov et al. 2018): two independently trained minima of the
identical loss surface, joined by a low-loss path that isn't a straight line. The
straight line between two minima routinely crosses a real loss barrier — a ridge the
optimizer had to be steered around during training, not through — but a *curved* path
between the same two points, explicitly chosen to avoid that ridge, often stays close to
each endpoint's own loss the entire way. Parameterize the curve as a quadratic Bezier
`theta(t) = (1-t)^2 theta_A + 2t(1-t) theta_C + t^2 theta_B` with only the single control
point `theta_C` free, and find it the same way every other minimum in this repo is
found: minimize the (Monte-Carlo-sampled) average loss along the curve as an ordinary
`Problem`, handed to the identical solvers used everywhere else.
"""

from __future__ import annotations

import numpy as np

from optimlab.core import ArrayLike, Objective, Problem


def bezier_point(t: float, theta_a: ArrayLike, theta_c: ArrayLike, theta_b: ArrayLike) -> ArrayLike:
    return (1 - t) ** 2 * theta_a + 2 * t * (1 - t) * theta_c + t**2 * theta_b


def bezier_curve_problem(
    f: Objective, theta_a: ArrayLike, theta_b: ArrayLike, *, n_samples: int = 8
) -> Problem:
    """A `Problem` over the single free control point `theta_c`, whose objective is the
    average loss along the Bezier curve at `n_samples` fixed points `t` in `(0, 1)`
    (endpoints excluded — they're pinned at `theta_a`/`theta_b` regardless of `theta_c`,
    so including them would only add a `theta_c`-independent constant to the objective).
    `x0` starts at the curve's most literal "no bend" guess, the straight-line midpoint.
    """
    theta_a = np.asarray(theta_a)
    theta_b = np.asarray(theta_b)
    ts = np.linspace(0.0, 1.0, n_samples + 2)[1:-1]

    def path_loss(theta_c: ArrayLike) -> float:
        return sum(f(bezier_point(t, theta_a, theta_c, theta_b)) for t in ts) / len(ts)

    x0 = 0.5 * (theta_a + theta_b)
    return Problem(f=path_loss, x0=x0, name="bezier_curve")


def evaluate_curve_loss(
    f: Objective, theta_a: ArrayLike, theta_c: ArrayLike, theta_b: ArrayLike, *, n_points: int = 30
) -> tuple[np.ndarray, np.ndarray]:
    """Loss at `n_points` values of `t` evenly spaced over `[0, 1]` along the Bezier
    curve through `theta_a`, `theta_c`, `theta_b` — the curve traced out once
    `theta_c` is fixed (typically the output of `bezier_curve_problem`, solved).
    """
    ts = np.linspace(0.0, 1.0, n_points)
    losses = np.array([float(f(bezier_point(t, theta_a, theta_c, theta_b))) for t in ts])
    return ts, losses
