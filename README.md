# optimization-lab

*A living lab for applied mathematical optimization: from-scratch algorithms, real solver
comparisons, cross-domain problems, and — the part most textbooks skip — actual intuition
for what a million- or billion-dimensional non-convex loss surface looks like.*

Live at [github.com/ioannisantoniadis/optimization-lab](https://github.com/ioannisantoniadis/optimization-lab)
· docs: [ioannisantoniadis.github.io/optimization-lab](https://ioannisantoniadis.github.io/optimization-lab/)

> The local folder is still `optimization/` even though the repo and package
> (`optimlab`) are already named `optimization-lab`. Rename it whenever convenient:
> `mv ~/GitHub/optimization ~/GitHub/optimization-lab`.

## Why this exists

I've used SGD, Adam, and friends as black boxes throughout a machine learning career
without ever really digging into *why* they work, especially on the wildly non-convex,
billion-parameter loss surfaces of modern neural nets. The spark for finally doing that
digging was working through Steven Brunton's [*Optimization Bootcamp*](https://www.youtube.com/@eigensteve)
(book + lecture series) — but this repo isn't a transcription of it. Classical treatments
(Brunton's included) tend to stop short of the high-dimensional non-convex regime where
the interesting open questions live, and don't reach into the domains (physics, econ,
sociology, personal decision-making) where these techniques quietly do real work. The
structure and content here draw on standard references (Nocedal & Wright, Boyd &
Vandenberghe), the primary literature (cited per-topic as it's used — see `ROADMAP.md`),
existing open-source tooling, and original material, with Brunton's book as one starting
point among several rather than the blueprint.

This repo is the place to close both gaps: relearn the classical toolkit by building it
from scratch, then push past what any single textbook covers.

Three commitments shape everything here:

1. **Build it, don't just import it.** Every core classical algorithm — gradient
   descent, momentum, Adam, Newton, BFGS, simplex, ADMM, proximal methods, simulated
   annealing, genetic algorithms — is implemented from scratch in `optimlab`, because
   implementing an algorithm is how you actually learn it. Mature libraries (`scipy`,
   `cvxpy`, `jax`/`optax`, `optuna`, `nevergrad`) are used deliberately as *correctness
   oracles* and *scale-up backends*, never as a substitute for the from-scratch version.
2. **Port a problem in, get solvers for free.** Every problem — a toy 2D benchmark
   function, a physics system, a portfolio, a tiny neural net, a personal-life decision
   model — implements one small `Problem` interface. Every solver in the repo, from-scratch
   or backend, can then run against it, and the comparison/benchmarking harness ("solver
   arena") works out of the box.
3. **See it, don't just prove it.** Every technique ships with an interactive visual:
   optimizer trajectories racing across a landscape, Hessian eigenvalue spectra morphing
   as dimension grows, the geometry a KKT condition or a duality gap actually looks like.
   See [`docs/`](docs/) (static reference site) and [`notebooks/marimo/`](notebooks/marimo/)
   (drag-a-slider, watch-it-update reactive apps) for the two visual mediums this repo uses.

## Repository layout

```
src/optimlab/
  core.py            Problem / OptimizeResult / Solver — the common interface everything speaks
  landscapes/         Benchmark test functions + tools for exploring high-dimensional,
                       non-convex landscapes (Hessian eigenspectra, filter-normalized slices,
                       concentration-of-measure demos, critical-point statistics)
  optimizers/          From-scratch solvers: gradient descent, momentum/Nesterov, Adam family,
                       Newton, BFGS/L-BFGS, line search, (more arriving per the roadmap)
  backends/           Thin adapters to scipy / cvxpy / jax·optax / optuna / nevergrad / pymoo,
                       used as correctness oracles and scale-up paths, sharing the same
                       Problem → OptimizeResult interface as the from-scratch solvers
  viz/                Shared Plotly figure helpers (surfaces, trajectories, spectra, "solver races")
  problems/           Cross-domain problem library: physics, economics, sociology/networks,
                       machine learning, and a personal "life as optimization" case study
docs/                 Static reference site — theory, derivations, live-rendered figures
notebooks/marimo/     Reactive, git-diffable exploration apps (sliders, live re-solving)
tests/                Correctness + convergence tests for every solver, against known optima
```

## Status

Early build. See [`ROADMAP.md`](ROADMAP.md) for the phased plan — **Phase 1** (core
framework + classical foundations: convexity, gradients, gradient-based solvers) is now
complete: the `Problem` / solver framework, 9 from-scratch gradient-based solvers, 8
benchmark landscapes, a colorblind-validated Plotly viz layer, a first reactive marimo
app, and a rendered [Quarto](https://quarto.org) docs site with live figures, deployed to
GitHub Pages.

## Getting started

```bash
uv sync --extra viz --extra dev --extra docs   # add --extra backends once Phase 2 lands
uv run pytest                                  # run the solver correctness/convergence tests

# Interactively explore solvers racing across a landscape (opens in your browser):
uv run marimo edit notebooks/marimo/gradient_descent_explorer.py
```

### Building the docs site

The site is a [Quarto](https://quarto.org) book that executes real `optimlab` code at
render time (via a Jupyter kernel backed by this project's own venv), so figures are
never hand-copied images — they're regenerated from the actual solvers each render.

```bash
# One-time: install Quarto (https://quarto.org/docs/get-started/), then register the
# venv as a Jupyter kernel so Quarto executes code cells against it:
uv run python -m ipykernel install --user --name optimlab --display-name "optimlab (uv)"

# Regenerate the embedded marimo snapshot (see docs/chapters/01-foundations.qmd for why
# this is a static export, not a live WASM app — optimlab depends on JAX, which has no
# WebAssembly build, so it can't run client-side in a browser-only marimo export):
uv run marimo export html notebooks/marimo/gradient_descent_explorer.py \
    -o docs/embeds/gradient_descent_explorer.html -f

quarto preview docs   # live-reloading local preview
quarto render docs    # one-shot build to docs/_site/
```

## References

No single source drives this repo's structure or content — the reading list is
deliberately plural:

- Steven L. Brunton, *Optimization: A Bootcamp for Machine Learning, Inverse Problems,
  and Control* (draft, Cambridge University Press) — the spark for this project, and
  still a good on-ramp for the classical toolkit. Companion code:
  [github.com/dynamicslab/optimizationbook](https://github.com/dynamicslab/optimizationbook).
- Standard references for the material Brunton's book compresses: Nocedal & Wright,
  *Numerical Optimization*; Boyd & Vandenberghe, *Convex Optimization*.
- Primary literature, cited per-topic as it's actually used rather than gathered into one
  bibliography — see `ROADMAP.md`'s Phase 6 for the papers behind the high-dimensional
  non-convexity module specifically, and `docs/references.bib` for everything cited on
  the docs site.
- Existing open-source tooling used deliberately as correctness oracles / scale-up
  backends (`scipy`, `cvxpy`, `jax`, `optuna`, `nevergrad`, `pymoo`) rather than vendored
  wholesale — see the "Build it, don't just import it" commitment above.
