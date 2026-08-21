"""Equality-constrained quadratic programming (book §4.6): minimize a quadratic subject
only to linear *equality* constraints, which — unlike the general inequality-constrained
case Phase 4's KKT/interior-point chapter will cover — reduces to a single linear solve.
`optimlab.optimizers.projected_gradient` covers the complementary case (box
*inequality* constraints, no equalities), the other easy corner of general QP.
"""

from __future__ import annotations

import numpy as np

from optimlab.core import ArrayLike


def equality_constrained_qp(P: ArrayLike, q: ArrayLike, A_eq: ArrayLike, b_eq: ArrayLike) -> ArrayLike:
    """Solve `min_x 0.5 x^T P x + q^T x` subject to `A_eq x = b_eq`.

    The Lagrangian is `L(x, lambda) = 0.5 x^T P x + q^T x + lambda^T (A_eq x - b_eq)`;
    setting both `grad_x L` and `grad_lambda L` to zero gives the KKT system

        [ P    A_eq^T ] [ x      ]   [ -q   ]
        [ A_eq  0     ] [ lambda ] = [ b_eq ]

    which is linear in `(x, lambda)` *precisely because* the constraints are equalities
    — no case-by-case reasoning about which constraints are "active" (compare
    `optimlab.optimizers.projected_gradient`, which needs exactly that reasoning for
    inequality bounds). Solving this one system gives the exact constrained optimum in
    a single step, the same way `optimlab.optimizers.newton` solves an unconstrained
    quadratic exactly in one step.
    """
    P = np.asarray(P, dtype=float)
    q = np.asarray(q, dtype=float)
    A_eq = np.atleast_2d(np.asarray(A_eq, dtype=float))
    b_eq = np.asarray(b_eq, dtype=float)
    n, m = P.shape[0], A_eq.shape[0]

    kkt = np.block([[P, A_eq.T], [A_eq, np.zeros((m, m))]])
    rhs = np.concatenate([-q, b_eq])
    solution = np.linalg.solve(kkt, rhs)
    return solution[:n]
