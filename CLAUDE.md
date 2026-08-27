# optimization-lab — context for AI assistants

## What this is

A "living lab" for applied mathematical optimization: **from-scratch**
implementations (numpy + JAX for autodiff, no scipy/cvxpy solvers in the
core algorithms) of ~30 classical optimization algorithms, a shared
`Problem -> OptimizeResult` interface that lets any solver run against any
problem, a solver-arena comparison harness, cross-domain applications
(physics, economics, sociology, ML), and a docs site whose figures are
executed from the real code at render time.

- **Docs site**: https://ioannisantoniadis.github.io/optimization-lab/ (a
  9-chapter Quarto book, auto-deployed to GitHub Pages on push to `main` via
  `.github/workflows/docs.yml`, only when `docs/**`, `src/optimlab/**`,
  `pyproject.toml`, or `uv.lock` changed)
- **CI**: `.github/workflows/ci.yml` runs `ruff check .` + `pytest` on every
  push to `main` and every PR.

**This repo already documents itself extensively — read before duplicating.**
Two files carry almost all of the "why," not just "what":
- **`README.md`** — user-facing quickstart, full `src/optimlab/` layout with
  one line per subpackage, marimo/benchmarks/docs-build instructions.
- **`ROADMAP.md`** (36KB) — the real design document. Organized as 8 phases,
  **all marked `done`**. Each phase's bullets are a decision log: what was
  built, what broke and how it was actually caught (not hypothetically —
  e.g. a disabled contour-line flag that made a landscape plot read as
  "flat," a solver that silently hadn't converged within its iteration
  budget, a JAX/WebAssembly incompatibility that killed a planned live embed).
  **Before touching an area of the repo, check whether `ROADMAP.md` already
  explains a past decision or a bug that was fixed there** — reversing one
  without reading the rationale is a real risk in a repo this
  self-documenting. A "Non-goals" section at the very end states what this
  repo deliberately isn't trying to be.

Every module's own docstring is written with the same density (see
`core.py`, `arena.py`, `viz/theme.py`, `optimizers/__init__.py` for
examples) — reading the top-of-file docstring before editing a module is
usually faster than inferring intent from the code alone.

## Architecture

```
src/optimlab/
  core.py         Problem / OptimizeResult / Solver — the interface everything else uses
  arena.py         run_arena: register a Problem, get a standardized report across
                    every solver in optimizers.ALL_SOLVERS
  optimizers/      From-scratch solvers (see below)
  linalg/          SVD/condition number, least squares, ridge, equality-constrained QP
  landscapes/      Benchmark test functions (sphere, Rosenbrock, Rastrigin, ...) as
                    BenchmarkFunction objects carrying minima/convexity/domain metadata
  inference/       MLE/MAP, Laplace approximation, Metropolis-Hastings MCMC, GMM via EM
  highdim/         MLP-as-Problem, saddle-point statistics, Lanczos Hessian
                    eigenspectrum, filter-normalized loss landscapes, mode connectivity, NTK
  inverse/         Image deblurring (Tikhonov deconvolution), nonlinear system ID
  control/         LQR (Riccati), direct-shooting nonlinear optimal control, grid-world
                    value iteration
  ml/              Manual backprop (cross-checked against JAX autodiff), PINNs
  viz/              Plotly figure builders, one shared theme (viz/theme.py)
  backends/         Correctness-oracle adapters: scipy (core dep), cvxpy/optuna (need
                    the `backends` extra)
  problems/         Cross-domain problems: economics (Markowitz), sociology (fair
                    allocation) — physics/ML applications live in optimizers/control instead
docs/               Quarto book — theory + write-ups, figures executed from real code
notebooks/marimo/   8 reactive notebooks, one per phase
benchmarks/         Wall-clock timing harness + generated BENCHMARKS.md
tests/              One test file per solver/module, pytest
```

### The core interface (`src/optimlab/core.py`)

Everything hinges on three things:

- **`Problem`** — a dataclass: `f` (objective), `x0`, optional closed-form
  `grad`/`hess` (falls back to JAX autodiff, then finite differences if JAX
  isn't installed), `jit_f` (opt-in `jax.jit` on the objective itself —
  `grad`/`hess` are *always* JIT-compiled via JAX regardless of this flag),
  plus `name`/`minimum`/`f_min`/`domain`/`reference` metadata used by tests
  and plots.
- **`OptimizeResult`** — standardized solver output: `x`, `f`, `n_iter`,
  `converged`, plus `trajectory`/`f_trajectory`/`grad_norm_trajectory` for
  plotting convergence.
- **`Solver` protocol** — any callable `solver(problem: Problem, **kwargs) ->
  OptimizeResult`. This is the *entire* contract that lets "port a new
  problem in, get every applicable solver for free" work (`arena.py`).

`optimizers/__init__.py` maintains **`ALL_SOLVERS`**, a `name -> callable`
registry used by the solver arena and comparison plots. Not every solver is
in it — `simplex` (needs a `LinearProgram`, not a `Problem`),
`conjugate_gradient` (plain `(A, b)`), `gauss_newton`
(`NonlinearLeastSquaresProblem`), `proximal_gradient` (`CompositeProblem`),
and `projected_gradient` (needs extra required `bounds`) each have a
differently-shaped call signature tied to their problem type — see that
module's docstring for the full explanation before assuming a new solver
belongs in `ALL_SOLVERS`. `genetic_algorithm`/`particle_swarm`/
`bayesian_optimize` *are* in the registry despite needing a search region,
because they default `bounds` to `problem.domain`.

`track_iterations(...)` in `core.py` is the shared helper every from-scratch
solver uses to package its accumulated per-iteration history lists into a
final `OptimizeResult` — use it rather than constructing one by hand when
writing a new iterative solver.

## Key conventions and gotchas

- **JAX float64 is enabled as an import-time side effect** in `core.py`
  (`jax.config.update("jax_enable_x64", True)`, wrapped in `try/except
  ImportError`). This must happen before any JAX array is created — JAX
  defaults to float32, which is not precise enough for this repo's
  convergence tolerances or Hessian eigenvalue work. Import `optimlab.core`
  (directly or transitively) before creating JAX arrays elsewhere; don't
  assume float64 is on if you're working with JAX outside that import path.
- **`jit_f`'s implementation order is deliberate, not incidental**: `_autof`
  (the optional `jax.jit` wrapper on the raw objective) is built and applied
  *separately* from `_autograd`/`_autohess`, which always trace the
  **unwrapped** `f`. A jitted `f` forces its output to a concrete Python
  `float`; tracing `jax.grad`/`jax.hessian` through that concrete cast is
  exactly the `ConcretizationTypeError` this ordering avoids. Don't refactor
  `Problem.__post_init__` to build `grad`/`hess` from the already-jitted `f`.
- **Objectives must be written in `jax.numpy`, not plain Python control
  flow/`float()` casts on traced values**, if they're meant to support
  autodiff or `jit_f=True` — see `landscapes/testfunctions.py` for the
  pattern (every benchmark function is `jax.numpy`-native so it's usable
  both as a plain callable and as an autodiff target).
- **`viz/theme.py`** is the single source of chart color/style truth —
  every `optimlab.viz.*` figure builder goes through it rather than picking
  colors ad hoc. Its `CATEGORICAL_LIGHT`/`CATEGORICAL_DARK` palettes are a
  **fixed, colorblind-validated order** (sourced from Anthropic's `dataviz`
  skill reference palette, deltaE-checked) — never re-cycle or reassign per
  plot, and if you change a hex, re-validate with that skill's
  `scripts/validate_palette.js`. **`MAX_RELIABLE_CATEGORICAL = 4`**: beyond
  4 series, per-pair colorblind separation can't be guaranteed by hue alone,
  so any figure that can show more than 4 solvers (convergence comparisons,
  solver races) must keep the legend visible and rely on Plotly's hover
  tooltip for identity — never color-alone past that point. Slot 1 (blue)
  doubles as the sequential ramp's hue, so a figure that also shows a blue
  sequential surface (e.g. a contour plot with an overlaid trajectory)
  should start assigning categorical colors from slot 2 onward.
- **"Verified by actually running it" is a strong, repeated project norm**
  (see `ROADMAP.md` throughout) — a visualization fix isn't done because it
  "should" work; it's done after rendering to PNG/HTML/a live browser tab
  and reviewing the actual output. Concrete precedent: a `showlines=False`
  bug that made a landscape plot read as "flat" was only caught by
  rendering and looking; a solver's `.converged` flag was checked directly
  rather than eyeballing a plot to see if a demo had actually converged;
  marimo notebooks are verified with `marimo check` + `marimo export html`
  loaded in a real browser, not just "the source looks right." Follow this
  norm for any viz/notebook/docs change in this repo.
- **The marimo docs embed is a screenshot, not a live app.** `optimlab`
  depends on JAX, which has no WebAssembly build, so a marimo notebook can't
  run client-side in a browser-only (Pyodide) docs export. `docs/images/`
  holds a screenshot instead of a live embed; regenerate it by hand after
  changing that notebook's UI (`uv run marimo edit
  notebooks/marimo/gradient_descent_explorer.py`, screenshot, overwrite the
  matching file in `docs/images/`). An earlier version embedded a *static
  HTML export* instead, which was strictly worse (visible sliders that
  silently did nothing) — don't reintroduce that.
- **Benchmark numbers in `benchmarks/BENCHMARKS.md` are generated, not
  hand-written** — regenerate with `uv run python benchmarks/run_benchmarks.py`
  after changing a solver or adding a problem. Every timing is a *median of
  several warmed-up repeats on a single reused `Problem` instance*: a
  `Problem`'s JAX-compiled `grad`/`hess` (and `jit_f` if set) compile lazily
  on first call and are cached per-instance, so an unwarmed/cold timing
  mostly measures JIT tracing overhead, not steady-state solver cost.
- **`backends` extra is optional and most of the repo doesn't need it.**
  `cvxpy`/`optuna`/`nevergrad`/`pymoo` are only used as correctness-oracle
  adapters in `optimlab.backends` (e.g. comparing `simplex`/`barrier_method`
  results against `cvxpy`, or `bayesian_optimize` against Optuna) — `scipy`
  is the only always-available oracle dependency. Don't add a hard
  dependency on a `backends`-extra package outside `optimlab/backends/`.

## Dev workflow

```bash
uv sync --extra viz --extra dev --extra docs   # install everything
uv run pytest                                  # run the test suite
uv run ruff check .                            # lint (line-length 100, py311 target)
```

- Add `--extra backends` only when you actually need the cvxpy/optuna/
  nevergrad/pymoo oracle adapters — CI's `test` job installs only `viz dev`.
- `tests/` is one file per solver/module (`test_gradient_descent.py`-style
  naming, e.g. `test_optimizers.py`, `test_admm.py`, `test_lqr.py`, ...) —
  add a new solver's/module's tests as a matching new file rather than
  appending to an unrelated one. Tests check real convergence against known
  optima (see `test_optimizers.py`'s `test_solver_converges_on_convex_sphere`
  for the pattern), not just "doesn't crash."
- **Docs site** (needs the [Quarto CLI](https://quarto.org/docs/get-started/)
  installed separately — not a Python package):
  ```bash
  uv run python -m ipykernel install --user --name optimlab --display-name "optimlab (uv)"
  quarto preview docs   # live-reloading local preview
  quarto render docs    # one-shot build to docs/_site/
  ```
- **marimo notebooks** (need the `viz` extra):
  ```bash
  uv run marimo edit notebooks/marimo/<phase>_explorer.py
  ```
  One entry point per phase (`gradient_descent_explorer.py`,
  `lp_polytope_explorer.py`, `regularization_path_explorer.py`,
  `central_path_explorer.py`, `bayesian_posterior_explorer.py`,
  `loss_landscape_explorer.py`, `control_explorer.py`,
  `solver_arena_explorer.py`) — see README.md's "Interactive exploration"
  section for what each one demonstrates.

## Current status

All 8 roadmap phases are complete (`ROADMAP.md`). There is no in-flight
"next phase" recorded as of this writing — before starting new algorithmic
work, check `ROADMAP.md`'s end and the git log for anything more recent than
what's summarized here, and check with the owner what's actually wanted
next (new algorithms, more cross-domain problems, or refinement of what
exists) rather than assuming a phase 9 is expected.
