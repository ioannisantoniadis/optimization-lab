# optimization-lab

*A living lab for applied mathematical optimization: from-scratch algorithms, real solver
comparisons, cross-domain problems, and — the part most textbooks skip — actual intuition
for what a million- or billion-dimensional non-convex loss surface looks like.*

> Working title for now — the local folder is still `optimization/`. Rename it and the
> GitHub repo to `optimization-lab` whenever convenient; the package itself is already
> named `optimlab`.

## Why this exists

I've used SGD, Adam, and friends as black boxes throughout a machine learning career
without ever really digging into *why* they work, especially on the wildly non-convex,
billion-parameter loss surfaces of modern neural nets. Separately, I've been working
through Steven Brunton's [*Optimization Bootcamp*](https://www.youtube.com/@eigensteve)
(book + lecture series), which covers the classical theory well — convexity, gradients,
Lagrangians, duality — but like most textbooks, stops short of the high-dimensional
non-convex regime where the interesting open questions live.

This repo is the place to close both gaps at once: relearn the classical toolkit by
building it from scratch, then push past it into the regime that classical textbooks
don't cover, using the tools we just built.

Three commitments shape everything here:

1. **Build it, don't just import it.** Every core algorithm the book teaches — gradient
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
docs/                 Static reference site (the "book" — theory, derivations, references)
notebooks/marimo/     Reactive, git-diffable exploration apps (sliders, live re-solving)
tests/                Correctness + convergence tests for every solver, against known optima
```

## Status

Early build. See [`ROADMAP.md`](ROADMAP.md) for the phased plan — currently finishing
**Phase 1** (core framework + Chapter 1–2 foundations). Done so far: the `Problem` /
solver framework, 9 from-scratch gradient-based solvers, 8 benchmark landscapes, a
colorblind-validated Plotly viz layer (contour/surface/"solver race"/convergence plots),
and a first reactive marimo app. Still open: the Chapter 1–2 docs page.

## Getting started

```bash
uv sync --extra viz --extra dev   # install deps (add --extra backends once Phase 2 lands)
uv run pytest                     # run the solver correctness/convergence tests

# Interactively explore solvers racing across a landscape (opens in your browser):
uv run marimo edit notebooks/marimo/gradient_descent_explorer.py
```

## References

- Steven L. Brunton, *Optimization: A Bootcamp for Machine Learning, Inverse Problems,
  and Control* (draft, Cambridge University Press) — the primary text motivating this repo's
  structure. Companion code: [github.com/dynamicslab/optimizationbook](https://github.com/dynamicslab/optimizationbook).
- Per-topic further-reading lists (papers, blog posts, alternate implementations) live
  next to each module/notebook rather than in one giant bibliography — see `ROADMAP.md`
  for the key papers behind the high-dimensional non-convexity module specifically.
