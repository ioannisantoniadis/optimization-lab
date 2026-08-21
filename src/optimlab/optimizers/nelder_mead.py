"""Nelder-Mead downhill simplex (book §5.2): a *derivative-free* method for minimizing
`problem.f` — never touches `problem.grad`, which matters wherever a gradient doesn't
exist (a genuinely nonsmooth objective) or simply isn't available (a black-box
simulation you can only call, not differentiate). Every gradient-based method in this
repo needs a slope to follow; Nelder-Mead instead maintains `n+1` points (a "simplex" —
a triangle in 2D, a tetrahedron in 3D) and reshapes it step by step, always moving the
worst vertex toward better territory.
"""

from __future__ import annotations

import numpy as np

from optimlab.core import OptimizeResult, Problem, track_iterations


def _initial_simplex(x0: np.ndarray, step: float) -> np.ndarray:
    n = x0.size
    simplex = np.tile(x0, (n + 1, 1))
    for i in range(n):
        # scale the step by the coordinate's own magnitude (a common heuristic, shared
        # with scipy's Nelder-Mead) so a component near zero still gets perturbed
        simplex[i + 1, i] += step * max(abs(x0[i]), 1.0)
    return simplex


def nelder_mead(
    problem: Problem,
    *,
    step: float = 0.1,
    max_iter: int = 500,
    tol: float = 1e-8,
    alpha: float = 1.0,
    gamma: float = 2.0,
    rho: float = 0.5,
    sigma: float = 0.5,
) -> OptimizeResult:
    """The standard reflect / expand / contract / shrink loop (Nocedal & Wright,
    Algorithm 9.1-adjacent). `tol` is checked against the *spread* of `f` values across
    the simplex, `std(f(vertices))` — the natural stand-in for `grad_norm` when there's
    no gradient: a simplex that has collapsed onto a single point of nearly-equal
    function value has nowhere better nearby to reflect toward, the same way a
    near-zero gradient means no descent direction remains.
    """
    n = problem.n_dim
    simplex = _initial_simplex(problem.x0.copy(), step)
    f_values = np.array([float(problem.f(x)) for x in simplex])

    def sort_simplex() -> None:
        order = np.argsort(f_values)
        simplex[:] = simplex[order]
        f_values[:] = f_values[order]

    sort_simplex()
    x_hist = [simplex[0].copy()]
    f_hist = [float(f_values[0])]
    g_hist = [float(np.std(f_values))]

    converged = g_hist[0] < tol
    n_iter = 0
    while not converged and n_iter < max_iter:
        centroid = simplex[:-1].mean(axis=0)
        worst, second_worst_f, best_f = simplex[-1], f_values[-2], f_values[0]

        reflected = centroid + alpha * (centroid - worst)
        f_reflected = float(problem.f(reflected))

        if best_f <= f_reflected < second_worst_f:
            simplex[-1], f_values[-1] = reflected, f_reflected
        elif f_reflected < best_f:
            expanded = centroid + gamma * (reflected - centroid)
            f_expanded = float(problem.f(expanded))
            if f_expanded < f_reflected:
                simplex[-1], f_values[-1] = expanded, f_expanded
            else:
                simplex[-1], f_values[-1] = reflected, f_reflected
        else:
            if f_reflected < f_values[-1]:  # outside contraction
                contracted = centroid + rho * (reflected - centroid)
                f_contracted = float(problem.f(contracted))
                shrink_needed = f_contracted >= f_reflected
            else:  # inside contraction
                contracted = centroid + rho * (worst - centroid)
                f_contracted = float(problem.f(contracted))
                shrink_needed = f_contracted >= f_values[-1]

            if not shrink_needed:
                simplex[-1], f_values[-1] = contracted, f_contracted
            else:
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    f_values[i] = float(problem.f(simplex[i]))

        sort_simplex()
        n_iter += 1
        x_hist.append(simplex[0].copy())
        f_hist.append(float(f_values[0]))
        g_hist.append(float(np.std(f_values)))

        if g_hist[-1] < tol:
            converged = True

    return track_iterations(
        x_hist, f_hist, g_hist, converged=converged, solver_name="nelder_mead",
        message="simplex collapsed below tol" if converged else "max_iter reached",
    )
