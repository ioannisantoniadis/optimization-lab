# Roadmap

Phased plan, organized around the standard topic progression of applied optimization
(the same rough order most treatments of the field use — Brunton's *Optimization
Bootcamp*, Nocedal & Wright's *Numerical Optimization*, Boyd & Vandenberghe's *Convex
Optimization* all agree on convex-before-nonconvex, unconstrained-before-constrained),
plus three repo-specific threads no single one of those covers on its own: the
`Problem`/solver-arena framework, the high-dimensional non-convex intuition module, and
cross-domain (including personal life-planning) case studies. No phase is a
transcription of any one source — see `README.md`'s References section for the full
reading list feeding each phase, and each phase's own bullets for the specific papers
behind it. Each phase should leave the repo in a working, tested, documented state — no
half-finished chapters.

## Phase 1 — Core framework + classical gradient-based optimization — **done**

- [x] `optimlab.core`: `Problem`, `OptimizeResult`, the solver call signature every
      algorithm (from-scratch or backend) shares. Autodiff-backed gradients/Hessians via
      JAX, with a finite-difference fallback.
- [x] `optimlab.landscapes.testfunctions`: sphere, Rosenbrock, Rastrigin, Ackley,
      Himmelblau, Beale, Styblinski–Tang, Matyas — each with metadata (known minima,
      convexity, separability, standard domain) so later modules can reason about them
      programmatically rather than hard-coding numbers.
- [x] `optimlab.optimizers`: gradient descent (fixed-step and backtracking-Armijo),
      heavy-ball momentum, Nesterov accelerated gradient, Adagrad/RMSProp/Adam, Newton's
      method (with Levenberg-style damping for indefinite Hessians), BFGS and L-BFGS,
      standalone Armijo/Wolfe line search.
- [x] Convergence tests against known optima; ill-conditioning demo (why GD zig-zags,
      why Newton doesn't).
- [x] `optimlab.viz`: shared Plotly theme (colorblind-validated categorical palette +
      sequential blue ramp, light/dark chart chrome) + contour/surface landscape plots +
      "solver race" and convergence-comparison plots. Colors and layout checked by
      rendering to PNG and reviewing, not just built-without-erroring (a first pass had a
      real colorbar/legend collision, caught this way and fixed).
- [x] First `notebooks/marimo/` app (`gradient_descent_explorer.py`): pick a landscape, a
      starting point, and one or more solvers; drag learning-rate/momentum/iteration
      sliders and watch the solver-race and convergence plots redraw. Verified by actually
      running it (`marimo check` + `marimo export html`, then loaded in a real browser),
      not just eyeballing the source.
- [x] First `docs/` page covering convexity, gradients, and the gradient-based solver
      zoo, with embedded figures and a link to the marimo app. Built as a
      [Quarto](https://quarto.org) book (`docs/`) that executes real `optimlab` code at
      render time via a Jupyter kernel bound to this project's venv — figures are
      regenerated from the actual solvers on every render, never hand-copied images.
      Verified by rendering and reviewing the real output in a browser, the same way as
      the Phase 1 visualization slice. One real constraint surfaced and resolved this
      way: `optimlab` depends on JAX for autodiff, and JAX has no WebAssembly build, so
      the marimo app can't run live inside a browser-only (Pyodide) export — the docs
      page embeds a static snapshot instead and tells readers to run the app locally for
      genuine interactivity, rather than silently shipping something broken.
      A GitHub Actions workflow (`.github/workflows/docs.yml`) builds and deploys to
      GitHub Pages on push to `main` — live at
      [ioannisantoniadis.github.io/optimization-lab](https://ioannisantoniadis.github.io/optimization-lab/),
      confirmed by an actual successful run, not just a written-and-hoped-for workflow.

## Phase 2 — Convex workhorses: linear programming & least squares — **next up**

- Simplex method from scratch + polytope/vertex visualization in 3D; brief detour into
  why LP vertices are the right mental model for high-dimensional feasible regions.
- Least-squares regression, SVD, condition number, ridge/Tikhonov regularization,
  constrained least-squares, QP, Gauss-Newton, conjugate gradient.
- `optimlab.backends`: scipy.optimize / cvxpy adapters, used here first as correctness
  oracles for the from-scratch simplex and CG implementations.

## Phase 3 — Nonsmooth, gradient-free, and global optimization

- Proximal gradient, Nelder–Mead, simulated annealing, a from-scratch genetic algorithm,
  particle swarm.
- This is the natural home for objectives that are black-box, noisy, or nonsmooth —
  which includes most real personal-decision objectives, so the first pass at the
  **life-as-optimization** case study (see Phase 8) lands here as a running example rather
  than waiting for the full roadmap.

## Phase 4 — Constraints and duality

- Lagrange multipliers, KKT conditions, duality and duality gaps, interior point methods,
  ADMM, a short calculus-of-variations / Euler–Lagrange excursion.
- Visual goal: make a KKT condition and a duality gap *look like something geometrically*,
  not just algebra.

## Phase 5 — Bayesian modeling and estimation

- MLE/MAP, Bayesian inference, EM/GMMs, least squares as statistical estimation.
- Backend link: Optuna/Bayesian-optimization adapters for black-box hyperparameter search.

## Phase 6 — High-dimensional non-convexity: the flagship module

The centerpiece motivating this whole repo: building real intuition for what a
million-to-billion-dimensional non-convex loss surface looks like, and why gradient
descent works on it anyway. Requires the JAX-based autodiff core from Phase 1 and small
trainable networks, so it's scheduled once that backbone is solid — but conceptual pieces
(concentration of measure, random-matrix eigenvalue toys) can start earlier as standalone
demos.

Planned content, with the literature behind each piece:

- **Saddle points dominate, not bad local minima.** Dauphin et al. 2014 (*Identifying and
  Attacking the Saddle Point Problem*, [arXiv:1406.2572](https://arxiv.org/abs/1406.2572))
  and the spin-glass argument in Choromanska et al. 2015 (*The Loss Surfaces of Multilayer
  Networks*, [PMLR v38](https://proceedings.mlr.press/v38/choromanska15.pdf)): as dimension
  grows, a random critical point is exponentially unlikely to have an all-positive-eigenvalue
  Hessian. From-scratch demo: sample random quadratic forms, plot P(local min) vs.
  dimension, next to the empirical eigenvalue-sign histogram.
- **Hessian eigenspectrum of real trained networks.** Sagun et al. 2016
  ([arXiv:1611.07476](https://arxiv.org/abs/1611.07476)) and Ghorbani et al. 2019
  ([PMLR v97](https://proceedings.mlr.press/v97/ghorbani19b/ghorbani19b.pdf)): a small
  number of large outlier eigenvalues sitting on a large near-zero bulk. Implement Lanczos
  / stochastic Hutchinson trace estimation so this is computable without forming the full
  Hessian.
- **Loss landscape visualization at scale.** Li et al. 2018 (*Visualizing the Loss
  Landscape of Neural Nets*, [arXiv:1712.09913](https://arxiv.org/abs/1712.09913)):
  filter-normalized random 2D slices through weight space — reimplemented from scratch
  (~100 lines), not vendored from `tomgoldstein/loss-landscape`. Pair with the cheaper
  Goodfellow et al. 2015 linear-interpolation diagnostic
  ([arXiv:1412.6544](https://arxiv.org/abs/1412.6544)).
- **Mode connectivity.** Garipov et al. 2018
  ([arXiv:1802.10026](https://arxiv.org/pdf/1802.10026)) and Draxler et al. 2018: distinct
  trained minima are joined by simple low-loss paths — "no real barriers" once you stop
  looking only at straight lines.
- **Sharp vs. flat minima and generalization.** Keskar et al. 2016
  ([arXiv:1609.04836](https://arxiv.org/abs/1609.04836)), plus the reparametrization-invariance
  critique (Dinh et al. 2017) and a 2025 revisit
  ([arXiv:2511.03548](https://arxiv.org/pdf/2511.03548)) — present both the appealing
  story and why it's not the whole story.
- **Why high dimensions are weird, geometrically.** Concentration of measure (random
  points concentrate in a thin shell), near-orthogonality of random Gaussian directions.
  No single citation needed — original interactive builds: histogram of pairwise cosine
  similarities vs. dimension, sphere-volume-vs-ball-volume plots.
- **Overparameterization makes things easier, not harder.** Neural Tangent Kernel (Jacot
  et al. 2018) and Du et al. 2019 (*Gradient Descent Provably Optimizes Over-parameterized
  Neural Networks*) — the "lazy training" story for why wide nets behave locally like a
  convex quadratic. Secondary source: Lilian Weng's NTK derivation write-up.

## Phase 7 — Domain applications: inverse problems, control, machine learning

- Inverse problems: medical/computational imaging, PDE-constrained inversion, system ID.
- Control: LQR, nonlinear optimal control, dynamic programming, MPC, a light RL bridge.
- Machine learning: backprop from scratch, training small nets/toy transformers end to
  end with the Phase 1 optimizers (so you watch *your own* Adam implementation train a
  real model), physics-informed ML.

## Phase 8 — Cross-domain problem library and the solver arena

Runs partly in parallel with the phases above, growing as each phase adds solvers that
can be pitted against each other.

- **Solver arena**: register a `Problem`, get a standardized report (convergence plot,
  iterations, wall time, success rate) across every applicable from-scratch and backend
  solver — the mechanism for "port a new problem in, get solvers for free."
- **Domain problems**: one worked, visualized problem per domain — physics (e.g. optimal
  control of a pendulum swing-up), economics (portfolio optimization / market
  equilibrium), sociology/networks (opinion dynamics equilibrium, fair resource
  allocation), machine learning (hyperparameter search via the Optuna backend vs.
  from-scratch Bayesian optimization).
- **Life as optimization** (flagship case study, first pass lands once Phase 3's
  gradient-free tooling exists — real personal objectives are noisy, nonsmooth, and
  multi-objective by nature): a worked example of taking a personal planning problem,
  making the parameterization and independence/stationarity assumptions explicit, framing
  it as multi-objective optimization, and being honest about where the model breaks down.
  The point isn't a "solved" life — it's making the modeling assumptions of applying
  optimization outside of textbook settings legible.

## Non-goals

This repo isn't trying to be a single-purpose optimizer-visualization demo, a
loss-landscape-plotting script, or a transcription of any one book's code or chapter
structure — those already exist in various forms elsewhere. The combination this repo
aims for instead: algorithms built from scratch, a common cross-domain `Problem`
interface, an explicit low-dimension → high-dimension intuition bridge (Phase 6), and
domain reach (Phases 7–8: physics, econ, sociology, personal decision-making) that a
typical optimization text doesn't attempt. Before writing content that closely overlaps
an existing project or text, skim it first so this repo stays complementary rather than
a re-transcription.
