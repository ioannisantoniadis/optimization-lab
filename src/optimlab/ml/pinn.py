"""Physics-informed neural networks (PINNs): training a network to satisfy a
differential equation directly — the loss is the equation's own residual at a set of
sample ("collocation") points plus the initial condition, not a table of precomputed
solution values the network is asked to imitate. The network never sees "the answer"
anywhere; it only ever sees the equation it must satisfy. Reuses
`optimlab.highdim.nets`' MLP-as-`Problem` machinery directly, with one addition: the
loss now needs a derivative of the network's own output with respect to its *input*
(not just its parameters) — computed the same way, `jax.grad`, just differentiating
through a different argument.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from optimlab.core import ArrayLike, Problem
from optimlab.highdim.nets import MLPShape, forward, init_params


def ode_pinn_problem(
    decay_rate: float,
    y0: float,
    x_range: tuple[float, float],
    shape: MLPShape,
    *,
    n_collocation: int = 50,
    seed: int = 0,
    ic_weight: float = 10.0,
) -> Problem:
    """`dy/dx = -decay_rate * y`, `y(0) = y0` — a network `y_NN(x)` trained so both the
    ODE residual (`dy_NN/dx + decay_rate * y_NN`, evaluated at `n_collocation` points
    spread across `x_range`) and the initial-condition residual (`y_NN(0) - y0`)
    vanish. The true solution `y0 * exp(-decay_rate * x)` is never referenced anywhere
    in this loss — it exists only to check the trained network against afterward.
    """
    xs = jnp.linspace(x_range[0], x_range[1], n_collocation)

    def net_scalar(params: ArrayLike, x: float) -> float:
        return forward(params, shape, jnp.array([[x]]))[0, 0]

    dnet_dx = jax.vmap(jax.grad(net_scalar, argnums=1), in_axes=(None, 0))

    def loss(params: ArrayLike) -> float:
        y_vals = jax.vmap(lambda x: net_scalar(params, x))(xs)
        dy_dx = dnet_dx(params, xs)
        ode_residual = dy_dx + decay_rate * y_vals
        ic_residual = net_scalar(params, 0.0) - y0
        return jnp.mean(ode_residual**2) + ic_weight * ic_residual**2

    x0 = init_params(shape, seed=seed)
    return Problem(f=loss, x0=x0, name="ode_pinn")


def predict(params: ArrayLike, shape: MLPShape, xs: ArrayLike) -> ArrayLike:
    """The trained network's output at each point in `xs` — reshaping `xs` to match
    `forward`'s `(n_samples, n_in)` convention and flattening the `(n_samples, 1)`
    result back down for easy comparison against a 1D array of true solution values.
    """
    return forward(params, shape, jnp.asarray(xs).reshape(-1, 1)).reshape(-1)
