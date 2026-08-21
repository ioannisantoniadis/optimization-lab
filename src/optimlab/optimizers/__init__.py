from optimlab.optimizers.adaptive import adagrad, adam, rmsprop
from optimlab.optimizers.barrier_method import ConstrainedProblem, barrier_method
from optimlab.optimizers.conjugate_gradient import conjugate_gradient
from optimlab.optimizers.gauss_newton import NonlinearLeastSquaresProblem, gauss_newton
from optimlab.optimizers.genetic_algorithm import genetic_algorithm
from optimlab.optimizers.gradient_descent import gradient_descent
from optimlab.optimizers.line_search import backtracking_armijo, strong_wolfe_line_search
from optimlab.optimizers.linear_programming import LinearProgram, LPResult, dual, simplex
from optimlab.optimizers.momentum import heavy_ball, nesterov
from optimlab.optimizers.nelder_mead import nelder_mead
from optimlab.optimizers.newton import newton_method
from optimlab.optimizers.particle_swarm import particle_swarm
from optimlab.optimizers.projected_gradient import projected_gradient
from optimlab.optimizers.proximal_gradient import (
    CompositeProblem,
    proximal_gradient,
    soft_threshold,
)
from optimlab.optimizers.quasi_newton import bfgs, lbfgs
from optimlab.optimizers.simulated_annealing import simulated_annealing

#: Every from-scratch solver that speaks the `Problem -> OptimizeResult` interface,
#: name -> callable(problem, **kwargs) -> OptimizeResult. Used by the solver-arena
#: benchmarking harness (ROADMAP Phase 8) to run "all solvers against this problem"
#: without listing them out by hand at each call site — including, as of Phase 3, a mix
#: of gradient-based and gradient-free methods, so a `race_figure`/`convergence_figure`
#: can put e.g. gradient descent getting stuck against simulated annealing escaping on
#: the exact same non-convex landscape. `genetic_algorithm`/`particle_swarm` need a
#: search region (`bounds`, defaulting to `problem.domain`) rather than just a starting
#: point, but that default makes them callable the same uniform way for any
#: `optimlab.landscapes` benchmark. Not here, each for its own reason tied to a
#: differently-shaped problem (see each module's docstring): `simplex` (LinearProgram),
#: `conjugate_gradient` (a plain (A, b) linear system), `gauss_newton`
#: (NonlinearLeastSquaresProblem, residuals + Jacobian), `proximal_gradient`
#: (CompositeProblem, a smooth part plus a proximal operator), and `projected_gradient`
#: (a Problem plus box bounds it needs as extra, required arguments).
ALL_SOLVERS = {
    "gradient_descent": gradient_descent,
    "heavy_ball": heavy_ball,
    "nesterov": nesterov,
    "adagrad": adagrad,
    "rmsprop": rmsprop,
    "adam": adam,
    "newton": newton_method,
    "bfgs": bfgs,
    "lbfgs": lbfgs,
    "nelder_mead": nelder_mead,
    "simulated_annealing": simulated_annealing,
    "genetic_algorithm": genetic_algorithm,
    "particle_swarm": particle_swarm,
}

__all__ = [
    "ALL_SOLVERS",
    "CompositeProblem",
    "ConstrainedProblem",
    "LPResult",
    "LinearProgram",
    "NonlinearLeastSquaresProblem",
    "adagrad",
    "adam",
    "backtracking_armijo",
    "barrier_method",
    "bfgs",
    "conjugate_gradient",
    "dual",
    "gauss_newton",
    "genetic_algorithm",
    "gradient_descent",
    "heavy_ball",
    "lbfgs",
    "nelder_mead",
    "nesterov",
    "newton_method",
    "particle_swarm",
    "projected_gradient",
    "proximal_gradient",
    "rmsprop",
    "simplex",
    "simulated_annealing",
    "soft_threshold",
    "strong_wolfe_line_search",
]
