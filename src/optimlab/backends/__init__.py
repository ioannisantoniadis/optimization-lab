"""Thin adapters to scipy / cvxpy (and eventually jax·optax / optuna / nevergrad /
pymoo), each convertible to the same result types as the from-scratch solvers in
`optimlab.optimizers`, so they can be used as correctness oracles and — for problems too
large for the from-scratch versions to handle comfortably — scale-up backends.

`scipy_backend` needs nothing beyond this project's core dependencies. `cvxpy_backend`
needs the `backends` extra (`uv sync --extra backends`); its two functions are simply
absent from this package's namespace if cvxpy isn't installed, rather than raising an
import error the moment any part of `optimlab.backends` is imported.
"""

from optimlab.backends.scipy_backend import scipy_linprog, scipy_nonlinear_least_squares

__all__ = ["scipy_linprog", "scipy_nonlinear_least_squares"]

try:
    from optimlab.backends.cvxpy_backend import cvxpy_linprog, cvxpy_qp

    __all__ += ["cvxpy_linprog", "cvxpy_qp"]
except ImportError:  # pragma: no cover - exercised only without the `backends` extra
    pass
