"""Standard benchmark test functions, owned here (not imported from an external test-
function package) so we can attach the metadata the rest of optimlab actually needs:
known minima, convexity/multimodality/separability tags, and a default viewing domain
for plotting. Every function is written in `jax.numpy` so it is simultaneously usable as
a plain numpy-style function *and* differentiable end to end via `Problem`'s autodiff
fallback (see `optimlab.core`).

Two functions (sphere, rosenbrock) also carry a hand-derived closed-form gradient —
worth doing once by hand for the pedagogy — while the rest lean on autodiff, which is
the realistic default for anything past textbook-toy functions.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from optimlab.core import ArrayLike, GradFn, Objective, Problem


@dataclass
class BenchmarkFunction:
    """A benchmark objective plus the metadata needed to reason about it programmatically
    (e.g. "only run separable-assuming solvers on separable functions") rather than
    hard-coding facts about each function wherever it's used.
    """

    name: str
    f: Objective
    domain: tuple[float, float]
    minima: list[np.ndarray]
    f_min: float
    convex: bool
    multimodal: bool
    separable: bool
    description: str
    n_dim: int | None = None  # None => works for any n; fixed int => only that many dims
    grad: GradFn | None = None

    def problem(self, x0: ArrayLike | None = None, n_dim: int = 2, seed: int = 0) -> Problem:
        """Build a `Problem` from this benchmark function, sampling a random start point
        in-domain if `x0` isn't given.
        """
        dim = self.n_dim if self.n_dim is not None else n_dim
        if x0 is None:
            rng = np.random.default_rng(seed)
            low, high = self.domain
            x0 = rng.uniform(low, high, size=dim)
        return Problem(
            f=self.f,
            x0=np.asarray(x0, dtype=float),
            grad=self.grad,
            name=self.name,
            minimum=self.minima[0] if self.minima else None,
            f_min=self.f_min,
            domain=self.domain,
            reference=self.description,
        )


# --- objectives -------------------------------------------------------------------


def _sphere(x: ArrayLike) -> float:
    return jnp.sum(x**2)


def _sphere_grad(x: ArrayLike) -> ArrayLike:
    return 2.0 * np.asarray(x)


def _rosenbrock(x: ArrayLike) -> float:
    x = jnp.asarray(x)
    return jnp.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)


def _rosenbrock_grad(x: ArrayLike) -> ArrayLike:
    # Hand-derived: d/dx_j of sum_i[100(x_{i+1}-x_i^2)^2 + (1-x_i)^2], see ROADMAP Phase 1.
    x = np.asarray(x, dtype=float)
    n = x.size
    g = np.zeros(n)
    g[:-1] += -400.0 * x[:-1] * (x[1:] - x[:-1] ** 2) - 2.0 * (1.0 - x[:-1])
    g[1:] += 200.0 * (x[1:] - x[:-1] ** 2)
    return g


def _rastrigin(x: ArrayLike) -> float:
    x = jnp.asarray(x)
    n = x.shape[0]
    return 10.0 * n + jnp.sum(x**2 - 10.0 * jnp.cos(2.0 * jnp.pi * x))


def _ackley(x: ArrayLike) -> float:
    x = jnp.asarray(x)
    n = x.shape[0]
    sum_sq = jnp.sum(x**2) / n
    sum_cos = jnp.sum(jnp.cos(2.0 * jnp.pi * x)) / n
    return (
        -20.0 * jnp.exp(-0.2 * jnp.sqrt(sum_sq))
        - jnp.exp(sum_cos)
        + 20.0
        + jnp.e
    )


def _himmelblau(x: ArrayLike) -> float:
    x0, x1 = x[0], x[1]
    return (x0**2 + x1 - 11.0) ** 2 + (x0 + x1**2 - 7.0) ** 2


def _beale(x: ArrayLike) -> float:
    x0, x1 = x[0], x[1]
    return (
        (1.5 - x0 + x0 * x1) ** 2
        + (2.25 - x0 + x0 * x1**2) ** 2
        + (2.625 - x0 + x0 * x1**3) ** 2
    )


def _styblinski_tang(x: ArrayLike) -> float:
    x = jnp.asarray(x)
    return 0.5 * jnp.sum(x**4 - 16.0 * x**2 + 5.0 * x)


def _matyas(x: ArrayLike) -> float:
    x0, x1 = x[0], x[1]
    return 0.26 * (x0**2 + x1**2) - 0.48 * x0 * x1


# --- registry ----------------------------------------------------------------------

ALL_FUNCTIONS: dict[str, BenchmarkFunction] = {
    "sphere": BenchmarkFunction(
        name="sphere",
        f=_sphere,
        grad=_sphere_grad,
        domain=(-5.12, 5.12),
        minima=[np.zeros(1)],
        f_min=0.0,
        convex=True,
        multimodal=False,
        separable=True,
        description="sum(x_i^2) — the textbook convex sanity check every solver should nail.",
    ),
    "rosenbrock": BenchmarkFunction(
        name="rosenbrock",
        f=_rosenbrock,
        grad=_rosenbrock_grad,
        domain=(-5.0, 10.0),
        minima=[np.ones(1)],
        f_min=0.0,
        convex=False,
        multimodal=False,
        separable=False,
        description=(
            "Generalized n-D Rosenbrock 'banana' valley — unimodal but highly "
            "ill-conditioned, the classic test of whether a solver handles curvature well."
        ),
    ),
    "rastrigin": BenchmarkFunction(
        name="rastrigin",
        f=_rastrigin,
        domain=(-5.12, 5.12),
        minima=[np.zeros(1)],
        f_min=0.0,
        convex=False,
        multimodal=True,
        separable=True,
        description="Sinusoidal ripples over a convex bowl — many regularly-spaced local minima.",
    ),
    "ackley": BenchmarkFunction(
        name="ackley",
        f=_ackley,
        domain=(-32.768, 32.768),
        minima=[np.zeros(1)],
        f_min=0.0,
        convex=False,
        multimodal=True,
        separable=False,
        description="Nearly flat outer region, one deep well at the origin riddled with local minima.",
    ),
    "himmelblau": BenchmarkFunction(
        name="himmelblau",
        f=_himmelblau,
        domain=(-5.0, 5.0),
        n_dim=2,
        minima=[
            np.array([3.0, 2.0]),
            np.array([-2.805118, 3.131312]),
            np.array([-3.779310, -3.283186]),
            np.array([3.584428, -1.848126]),
        ],
        f_min=0.0,
        convex=False,
        multimodal=True,
        separable=False,
        description="Four equal-value global minima — good for showing 'which basin did you land in.'",
    ),
    "beale": BenchmarkFunction(
        name="beale",
        f=_beale,
        domain=(-4.5, 4.5),
        n_dim=2,
        minima=[np.array([3.0, 0.5])],
        f_min=0.0,
        convex=False,
        multimodal=False,
        separable=False,
        description="Sharp corners and a flat plateau — stresses fixed-step methods.",
    ),
    "styblinski_tang": BenchmarkFunction(
        name="styblinski_tang",
        f=_styblinski_tang,
        domain=(-5.0, 5.0),
        minima=[np.full(1, -2.903534)],
        f_min=-39.16617,  # per dimension; scale by n_dim for the true f_min
        convex=False,
        multimodal=True,
        separable=True,
        description="Separable quartic with 2^n local minima — good for the dimension-scaling story.",
    ),
    "matyas": BenchmarkFunction(
        name="matyas",
        f=_matyas,
        domain=(-10.0, 10.0),
        n_dim=2,
        minima=[np.array([0.0, 0.0])],
        f_min=0.0,
        convex=True,
        multimodal=False,
        separable=False,
        description="Convex but non-separable quadratic with a narrow curved valley.",
    ),
}


def get(name: str) -> BenchmarkFunction:
    try:
        return ALL_FUNCTIONS[name]
    except KeyError as e:
        available = ", ".join(sorted(ALL_FUNCTIONS))
        raise KeyError(f"Unknown benchmark function {name!r}. Available: {available}") from e


def as_callable(name_or_fn: str | Callable[[ArrayLike], float]) -> Objective:
    """Convenience: accept either a registry name or a raw callable."""
    return get(name_or_fn).f if isinstance(name_or_fn, str) else name_or_fn
