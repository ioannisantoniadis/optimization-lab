# optimization-lab

From-scratch implementations of core optimization algorithms, plus tooling for
comparing solvers, visualizing landscapes, and applying them to problems across
domains (physics, economics, machine learning).

- Docs site: **[ioannisantoniadis.github.io/optimization-lab](https://ioannisantoniadis.github.io/optimization-lab/)**
- Roadmap / current status: [`ROADMAP.md`](ROADMAP.md)

> Local folder note: this checkout may still be named `optimization/` rather than
> `optimization-lab/`. Rename it whenever convenient — nothing else depends on the name.

## Quickstart

```bash
uv sync --extra viz --extra dev --extra docs   # install everything into .venv
uv run pytest                                  # run the test suite
```

[`uv`](https://docs.astral.sh/uv/) manages the Python environment; `uv run <cmd>` runs
a command inside it without activating a venv by hand. Add `--extra backends` for the
cvxpy-based correctness oracle (`optimlab.backends.cvxpy_backend`) — everything else,
including the scipy-based oracles, works without it.

## Repository layout

```
src/optimlab/
  core.py         Problem / OptimizeResult / Solver — the interface everything else uses
  optimizers/     From-scratch solvers: gradient descent, momentum, Adam family, Newton,
                  BFGS/L-BFGS, line search, simplex (LP), conjugate gradient, Gauss-Newton
                  (nonlinear least squares), projected/proximal gradient (box constraints,
                  LASSO), Nelder-Mead, simulated annealing, genetic algorithm, particle
                  swarm (all four gradient-free methods share ALL_SOLVERS with the rest)
  linalg/         SVD / condition number, least squares (+ minimum-norm, rank-deficient),
                  ridge regression, equality-constrained least squares / QP
  landscapes/     Benchmark test functions (sphere, Rosenbrock, Rastrigin, ...) with
                  metadata (known minima, convexity, domain)
  viz/            Plotly figure helpers: landscapes, solver-comparison plots, LP feasible
                  regions, regression fit/residuals, SVD conditioning, ridge/LASSO paths
  backends/       Correctness-oracle adapters: scipy (core dep) and cvxpy (needs the
                  `backends` extra) for LP, least squares, and QP
  problems/       Cross-domain problems (physics, economics, ML, ...) — not yet populated
docs/             Quarto site: theory + write-ups, with figures executed from real code
notebooks/marimo/ Interactive apps (see below)
tests/            Correctness/convergence tests for every solver
```

## Interactive exploration with marimo

[marimo](https://marimo.io) is a reactive Python notebook: cells re-run automatically
when their inputs change, and it has built-in UI widgets (sliders, dropdowns) that
trigger that re-run live in the browser. It's how this repo does hands-on
experimentation, as opposed to a static plot.

```bash
uv run marimo edit notebooks/marimo/gradient_descent_explorer.py
```

Opens in your browser. Pick a landscape and one or more solvers, drag the
learning-rate / momentum / iteration-count sliders, and the trajectory and
convergence plots redraw live.

## Building the docs site

The site is a [Quarto](https://quarto.org) book that executes real `optimlab` code at
render time, so its figures are regenerated from the actual solvers on every render —
not hand-copied images. This needs the [Quarto CLI](https://quarto.org/docs/get-started/)
installed separately (it isn't a Python package):

```bash
# One-time: register this project's venv as a Jupyter kernel, so Quarto executes
# code cells against it (with optimlab, jax, plotly available):
uv run python -m ipykernel install --user --name optimlab --display-name "optimlab (uv)"

quarto preview docs   # live-reloading local preview
quarto render docs    # one-shot build to docs/_site/
```

The docs page for the marimo app is a screenshot (`docs/images/`), not a live embed —
`optimlab` depends on JAX, which has no WebAssembly build, so the app can't run
client-side in a browser-only export. Regenerate the screenshot after changing that
notebook's UI: `uv run marimo edit notebooks/marimo/gradient_descent_explorer.py`,
screenshot it, save over `docs/images/gradient-descent-explorer-screenshot.jpg`.

## Sources

This project draws on standard optimization references and the primary literature
rather than following any single source — see `ROADMAP.md` for what's cited where, and
`docs/references.bib` for everything cited on the docs site.
