"""Per-parameter adaptive step sizes (book §2.3): Adagrad, RMSProp, and Adam. Where
momentum reshapes the update using gradient *history*, these reshape it using gradient
*magnitude* — each parameter gets its own effective learning rate, shrunk by how large
its past gradients have been. This is what makes Adam usable straight out of the box
across wildly different parameter scales, which is exactly the setting a many-million-
parameter neural net puts you in.
"""

from __future__ import annotations

import numpy as np

from optimlab.core import OptimizeResult, Problem, track_iterations


def adagrad(
    problem: Problem,
    *,
    lr: float = 0.1,
    eps: float = 1e-8,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> OptimizeResult:
    """Adagrad: divide each coordinate's step by the running sqrt-sum of its squared
    gradients. Directions with large/frequent gradients get slowed down, sparse/rare
    directions keep a large effective step. Monotonically accumulating `G` also means
    the effective learning rate only ever shrinks — the well-known failure mode RMSProp
    and Adam both fix.
    """
    x = problem.x0.copy()
    G = np.zeros_like(x)
    f_x = float(problem.f(x))
    g = problem.grad(x)
    x_hist, f_hist, g_hist = [x.copy()], [f_x], [float(np.linalg.norm(g))]

    converged = False
    for _ in range(max_iter):
        if np.linalg.norm(g) < tol:
            converged = True
            break
        G += g**2
        x = x - lr * g / (np.sqrt(G) + eps)
        f_x = float(problem.f(x))
        g = problem.grad(x)
        x_hist.append(x.copy())
        f_hist.append(f_x)
        g_hist.append(float(np.linalg.norm(g)))

    return track_iterations(
        x_hist, f_hist, g_hist, converged=converged, solver_name="adagrad",
        message="gradient norm below tol" if converged else "max_iter reached",
    )


def rmsprop(
    problem: Problem,
    *,
    lr: float = 0.01,
    decay: float = 0.9,
    eps: float = 1e-8,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> OptimizeResult:
    """RMSProp: replace Adagrad's ever-growing sum with an exponential moving average of
    squared gradients, so old gradient magnitudes decay away instead of permanently
    shrinking the step size — lets the optimizer keep adapting on non-stationary /
    long-running problems (this is the piece Hinton added specifically to fix Adagrad's
    "learning rate decays to zero too fast" problem).
    """
    x = problem.x0.copy()
    E = np.zeros_like(x)
    f_x = float(problem.f(x))
    g = problem.grad(x)
    x_hist, f_hist, g_hist = [x.copy()], [f_x], [float(np.linalg.norm(g))]

    converged = False
    for _ in range(max_iter):
        if np.linalg.norm(g) < tol:
            converged = True
            break
        E = decay * E + (1 - decay) * g**2
        x = x - lr * g / (np.sqrt(E) + eps)
        f_x = float(problem.f(x))
        g = problem.grad(x)
        x_hist.append(x.copy())
        f_hist.append(f_x)
        g_hist.append(float(np.linalg.norm(g)))

    return track_iterations(
        x_hist, f_hist, g_hist, converged=converged, solver_name="rmsprop",
        message="gradient norm below tol" if converged else "max_iter reached",
    )


def adam(
    problem: Problem,
    *,
    lr: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> OptimizeResult:
    """Adam: momentum (first-moment EMA `m`) combined with RMSProp-style per-coordinate
    scaling (second-moment EMA `v`), plus bias correction for both because `m` and `v`
    start at zero and are otherwise biased toward zero for the first several iterations
    — the `/(1 - beta^t)` terms are exactly compensating for that startup bias, not an
    arbitrary tweak.
    """
    x = problem.x0.copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    f_x = float(problem.f(x))
    g = problem.grad(x)
    x_hist, f_hist, g_hist = [x.copy()], [f_x], [float(np.linalg.norm(g))]

    converged = False
    for t in range(1, max_iter + 1):
        if np.linalg.norm(g) < tol:
            converged = True
            break
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g**2
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        x = x - lr * m_hat / (np.sqrt(v_hat) + eps)
        f_x = float(problem.f(x))
        g = problem.grad(x)
        x_hist.append(x.copy())
        f_hist.append(f_x)
        g_hist.append(float(np.linalg.norm(g)))

    return track_iterations(
        x_hist, f_hist, g_hist, converged=converged, solver_name="adam",
        message="gradient norm below tol" if converged else "max_iter reached",
    )
