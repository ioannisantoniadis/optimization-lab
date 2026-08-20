from optimlab.optimizers.adaptive import adagrad, adam, rmsprop
from optimlab.optimizers.gradient_descent import gradient_descent
from optimlab.optimizers.line_search import backtracking_armijo, strong_wolfe_line_search
from optimlab.optimizers.momentum import heavy_ball, nesterov
from optimlab.optimizers.newton import newton_method
from optimlab.optimizers.quasi_newton import bfgs, lbfgs

#: Every from-scratch solver, name -> callable(problem, **kwargs) -> OptimizeResult.
#: Used by the solver-arena benchmarking harness (ROADMAP Phase 8) to run "all solvers
#: against this problem" without listing them out by hand at each call site.
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
}

__all__ = [
    "ALL_SOLVERS",
    "adagrad",
    "adam",
    "backtracking_armijo",
    "bfgs",
    "gradient_descent",
    "heavy_ball",
    "lbfgs",
    "nesterov",
    "newton_method",
    "rmsprop",
    "strong_wolfe_line_search",
]
