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
      page shows a screenshot of it instead (`docs/images/`) and tells readers to run
      the app locally for genuine interactivity, rather than silently shipping something
      broken (an earlier version embedded a *static HTML export* of the notebook, which
      was itself confusing — visible, unlabeled sliders that silently did nothing — so
      that was dropped in favor of an honestly-non-interactive screenshot).
      A GitHub Actions workflow (`.github/workflows/docs.yml`) builds and deploys to
      GitHub Pages on push to `main` — live at
      [ioannisantoniadis.github.io/optimization-lab](https://ioannisantoniadis.github.io/optimization-lab/),
      confirmed by an actual successful run, not just a written-and-hoped-for workflow.
- [x] Visualization quality pass, prompted by direct feedback that the landscape plots
      read as "flat" and the convergence curves looked like they cut off before reaching
      the optimum. Both were real, not just perceptual: `contour_figure` had contour
      lines disabled (`showlines=False`), so most of a domain far from any minimum
      rendered as a near-uniform dark fill; and the ill-conditioned-quadratic demo's
      `gradient_descent` call genuinely hadn't converged within its `max_iter` budget
      (verified by checking `.converged` directly, not by eyeballing the plot). Fixed:
      contour lines back on with labeled levels; `surface_figure`'s `log_z` now defaults
      to `True` (it defaulted `False`, which flattens a steep function to a spike) with
      a three-quarter camera angle instead of Plotly's near-top-down default; new
      `surface_race_figure`/`add_trajectory_3d` overlay a solver's actual path onto the
      3D surface (`z` from `OptimizeResult.f_trajectory`, run through the same transform
      as the surface it sits on); `convergence_figure` gained `pad_converged=True`
      (default), which extends a solver that converged early with a dotted line at its
      final value out to the longest-running solver in the group, so "converged, holding
      at the optimum" is visually distinguishable from "ran out of budget mid-descent" —
      the latter is deliberately *not* padded, so an unfinished run still looks unfinished.
      Re-verified the same way as everything else in this phase: rendered to PNG and
      looked at it (twice — the first pass had the same colorbar/legend collision as
      before, just in the 3D scene this time), then re-rendered the live docs site and
      reviewed it in a browser.

## Phase 2 — Convex workhorses: linear programming & least squares — **done**

- [x] `optimlab.optimizers.linear_programming`: two-phase full-tableau simplex with
      Bland's rule (provably no cycling). `LinearProgram`/`LPResult` are their own types,
      not `Problem`/`OptimizeResult` — simplex walks polytope vertices, it doesn't follow
      a gradient. Verified against `scipy.optimize.linprog` on 500+ random bounded-
      feasible LPs (0 mismatches) plus hand-verified textbook cases (equality constraints
      needing phase-1 artificials, mixed constraints, infeasible, unbounded, no
      constraints beyond `x >= 0`). One interesting non-bug surfaced by the randomized
      check: a single instance where scipy/HiGHS reported "infeasible" for an LP
      independently confirmed (via an explicit feasible unbounded recession direction) to
      genuinely be unbounded — not a bug here, see the comment in
      `tests/test_linear_programming.py`.
- [x] `optimlab.viz.polytope`: 2D feasible-region shading (pairwise half-plane
      intersection + angular sort around the centroid) with the simplex path overlaid
      vertex to vertex. Scoped down from the original "3D" plan in this bullet once 2D
      turned out to already carry the full pedagogical point (LP optima sit at *vertices*
      — a 2D polygon makes that exactly as visible as a 3D polytope would, for
      meaningfully less implementation risk); 3D stays an option later if a specific demo
      needs it. Verified by rendering to PNG and reviewing it — first pass had needlessly
      zoomed-out default bounds, fixed with a two-pass rough-then-tight approach.
- [x] `optimlab.linalg`: `svd`/`condition_number`, `least_squares` (one pseudoinverse
      formula covering ordinary, minimum-norm-underdetermined, and rank-deficient cases),
      `ridge_regression` (same SVD, shrunk by `s/(s^2+alpha)` instead of divided by `s`),
      `equality_constrained_least_squares` / `equality_constrained_qp` (exact KKT solve).
      Verified against `numpy.linalg.lstsq`, closed-form ridge, and
      `scipy.optimize.minimize`.
- [x] `optimlab.optimizers.conjugate_gradient`: linear CG for SPD systems, framed as
      minimizing the equivalent quadratic so it shares `OptimizeResult` with every other
      solver. Verified: exact solution within the textbook `n`-step bound on random SPD
      systems, and (in `docs/chapters/03-least-squares.qmd`) against Chapter 1's own
      ill-conditioned quadratic — 2 steps for CG vs. ~900 for gradient descent on the
      identical system.
- [x] `optimlab.optimizers.gauss_newton`: nonlinear least squares via linearized-residual
      steps (`NonlinearLeastSquaresProblem`, JAX-Jacobian by default, finite-difference
      fallback). Verified: exact parameter recovery on noiseless exponential-decay data,
      matches `scipy.optimize.least_squares` on noisy data, one-step-exact on a linear
      residual.
- [x] `optimlab.optimizers.projected_gradient`: box-constrained gradient descent (a
      general `Problem` plus bounds, not QP-only), using the *projected*-gradient norm as
      its convergence criterion since the raw gradient needn't vanish at a boundary
      optimum. Verified against `scipy.optimize.minimize(method="L-BFGS-B", bounds=...)`.
- [x] `optimlab.backends`: `scipy_linprog` / `scipy_nonlinear_least_squares` (core deps
      only) and `cvxpy_linprog` / `cvxpy_qp` (the optional `backends` extra — absent from
      `optimlab.backends`'s namespace entirely, not just erroring, when cvxpy isn't
      installed). Used as correctness oracles for simplex and Gauss-Newton, and — for
      general QP with both equality *and* inequality constraints together, the one case
      neither from-scratch QP function covers alone — as the reach-for-it option.
      Verified: scipy and cvxpy each independently match the from-scratch solvers, and
      cross-checked against each other directly on a harder mixed-constraint LP.
- [x] `optimlab.viz.regression`: `regression_fit_figure` + `residuals_figure` (a
      diagnostic that works regardless of feature count), `svd_conditioning_figure` (the
      unit circle becoming an ellipse under a 2x2 matrix — condition number made
      literal), `ridge_path_figure` (coefficient shrinkage vs. `alpha`). Verified by
      rendering to PNG and reviewing it; the ridge path's log-axis had cluttered minor
      ticks at this figure's width, fixed to decade-only ticks.
- [x] Two docs chapters (`02-linear-programming.qmd`, `03-least-squares.qmd`), executing
      real code the same way Chapter 1 does. Caught and fixed three real issues by
      actually rendering and reading the output rather than trusting the source: LaTeX
      accidentally wrapped in code backticks instead of math delimiters (twice); a
      cross-reference chapter number that leaked the *book's* chapter numbering
      (`Chapter 4`/`Chapter 6`) instead of this site's own (fixed to `Chapter 3` and a
      phase-based reference respectively — exactly the book-centering slip this repo is
      trying to avoid); and a genuine bug in the conjugate-gradient demo, where the
      quadratic-minus-linear objective `0.5 x^T A x - b^T x` legitimately goes negative
      near the optimum, which `convergence_figure`'s log-scale clipping (`max(series,
      1e-16)`) flattened into a useless flat line for *both* solvers — switched that one
      figure to the gradient-norm metric, which stays positive and is the natural
      residual metric for `Ax=b` regardless.

## Phase 3 — Nonsmooth, gradient-free, and global optimization — **done**

- [x] `optimlab.optimizers.proximal_gradient`: minimize `g(x)+h(x)` for smooth `g` and
      possibly-nonsmooth `h` via alternating gradient/proximal steps, with
      `soft_threshold` as `h`'s proximal operator for LASSO. Direct continuation of
      Phase 2's ridge regression: same data, `optimlab.viz.lasso_path_figure` next to
      `ridge_path_figure` makes the qualitative difference (LASSO hits exact zero at a
      finite `alpha`; ridge only approaches it) visible, not just asserted.
      `optimlab.optimizers.projected_gradient` (Phase 2) turned out to already be a
      proximal method in disguise — box constraints are the special case where the
      proximal operator is exactly clipping. Verified: exact match with cvxpy's LASSO
      solution, exact recovery of a sparse ground truth's support, reduces to plain
      gradient descent when the nonsmooth term is zero.
- [x] `optimlab.optimizers.nelder_mead`: reflect/expand/contract/shrink simplex search,
      never touches `problem.grad` (checked directly in a test). Verified against
      `scipy.optimize.minimize(method="Nelder-Mead")`.
- [x] `optimlab.optimizers.simulated_annealing`: temperature-scaled random walk with
      Metropolis acceptance of worse moves, tracking best-found (not current) point.
      Verified: escapes a Rastrigin local minimum that traps gradient descent from the
      identical start.
- [x] `optimlab.optimizers.genetic_algorithm` / `optimlab.optimizers.particle_swarm`:
      population-based search (blend crossover + tournament selection; velocity toward
      personal/global best). Both default their search region to `problem.domain`, so
      they're callable the same uniform way as every gradient-based solver. Verified:
      both reliably reach the global optimum on Rastrigin/Ackley/Himmelblau — problems a
      purely local method can't solve from an arbitrary start.
- [x] All four `Problem`-based methods above added to `ALL_SOLVERS` — the first
      gradient-free entries there, so `race_figure`/`convergence_figure`/
      `surface_race_figure` immediately support mixed gradient-based vs. gradient-free
      comparisons (gradient descent visibly stuck vs. simulated annealing/particle swarm
      escaping, on the identical landscape) with no new visualization code, the direct
      payoff of Phase 1's `Problem`/`OptimizeResult` interface. Verified by rendering
      that exact comparison and reviewing it, the same way as every other figure in this
      repo — which surfaced and led to fixing a real, pre-existing bug:
      `BenchmarkFunction.problem()` passed an any-dimension benchmark's minimum (e.g.
      Rastrigin's) straight through as a length-1 placeholder instead of broadcasting it
      to the problem's actual dimension, silently breaking any 2D viz code indexing
      `minimum[0]`/`minimum[1]`.
- [x] Docs chapter (`04-nonsmooth-and-global-optimization.qmd`) covering all five
      methods, plus the first pass at the **life-as-optimization** case study this phase
      promised: a small, explicitly-labeled *toy* example (allocating a week's
      discretionary hours across sleep/work/exercise/social to maximize an invented,
      deliberately nonsmooth "wellbeing" utility). Genetic algorithm and simulated
      annealing were run on the identical problem and *disagree* on the answer — kept in
      the chapter and explained (population blending biases toward compromise points;
      a single random walker can commit to a narrow region) rather than cherry-picking a
      seed where they'd agree, in keeping with this phase's stated goal of making
      optimization's modeling assumptions and limits legible rather than presenting a
      falsely clean success story.

## Phase 4 — Constraints and duality — **done**

- [x] `optimlab.optimizers.barrier_method`: interior point / log-barrier method
      (Boyd & Vandenberghe, Algorithm 11.1) — a sequence of damped-Newton solves on
      `t*f(x) - sum log(-g_i(x))` with `t` driven up geometrically, feasibility-aware
      backtracking (a trial step is rejected outright the moment it leaves the strictly
      feasible region, before Armijo is even checked). `ConstrainedProblem` autodiff-fills
      gradients/Hessians for `f` and every constraint, and validates `x0`'s strict
      feasibility up front, naming which constraints are violated if it isn't. Verified
      against known symmetric optima, scipy SLSQP, and cvxpy across randomized seeds.
- [x] `optimlab.optimizers.linear_programming.dual`: the Lagrangian dual of a `<=`-only
      `LinearProgram`, reusing Phase 2's `simplex` on both sides. The sign relationship
      it settles on (`primal.objective == -dual.objective`) was derived by hand *and*
      confirmed empirically against 30+ random LPs before being trusted — an earlier,
      more "elegant"-looking direct-equality construction was tried first and was simply
      wrong.
- [x] `optimlab.optimizers.admm`: Alternating Direction Method of Multipliers — alternate
      proximal steps on `f`/`g` with a scaled dual variable enforcing `x=z`, never
      touching a gradient of either piece. Cross-verified against cvxpy and against
      Phase 3's `proximal_gradient` on an identical LASSO instance (agree to ~1e-8).
      Verifying this surfaced a real correctness bug, not just a test bug: the shipped
      primal-residual-only stopping check could declare false convergence — confirmed
      concretely at `rho=50` on the LASSO test problem, where `||x-z||` transiently hits
      ~1e-17 at iteration 5 while the objective is still ~50% off the true minimum, then
      rises again before properly decaying. Fixed with the standard combined
      primal-and-dual-residual stopping rule (Boyd et al. 2011, §3.3.1); regression test
      reproduces the exact failure this replaced.
- [x] Three visualizations making the phase's explicit "look like something
      geometrically" goal concrete: `central_path_figure` (the feasible region shaded,
      each constraint's boundary drawn, the barrier method's central path threading
      through the interior to the optimum), `kkt_geometry_figure` (`-grad f(x*)` and the
      active constraints' `lambda_i`-scaled gradients drawn as vectors at the optimum —
      verified to cancel exactly, both for a single active constraint and a
      two-active-constraint corner case), and `duality_gap_figure` (the gap on a log
      axis, showing the geometric shrink rate directly, not just the eventual
      guarantee). All three rendered and visually reviewed before committing.
- [x] Short calculus-of-variations / Euler-Lagrange excursion, done as a docs-chapter
      example rather than new library code (`optimlab.problems` stays reserved for
      Phase 8 per its own docstring): the minimal-surface-of-revolution problem (a soap
      film between two coaxial rings), whose Euler-Lagrange solution is a closed-form
      catenary. Discretizing the same surface-area functional onto a grid and handing it
      to plain BFGS recovers the analytic catenary to 5 decimal places — direct
      discretize-then-optimize and solving the ODE by hand land on the same answer.
- [x] Docs chapter (`05-constraints-and-duality.qmd`) covering the KKT conditions, the
      barrier method with the central-path and KKT-geometry figures, LP strong duality
      (primal vs. dual solved by the same simplex code), ADMM (including the
      rho-sensitivity numbers and the false-convergence fix, told honestly rather than
      smoothed over), and the catenary excursion. Full site rendered and every new
      section/figure reviewed in a real browser tab; caught and fixed a real Pandoc
      footgun along the way — a fig-cap containing bare `^2` text (not LaTeX) was
      silently parsed as superscript markup, mangling the caption.

## Phase 5 — Bayesian modeling and estimation — **next up**

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
- **Life as optimization** (flagship case study — expand the first pass already in
  `docs/chapters/04-nonsmooth-and-global-optimization.qmd`, a small weekly-hours
  allocation toy problem solved with `genetic_algorithm`/`simulated_annealing`, into
  something closer to a real worked example): make the parameterization and
  independence/stationarity assumptions explicit, frame it as genuinely multi-objective
  (the Phase 3 version collapses "wellbeing" into one scalar utility — an assumption
  worth relaxing here), and stay honest about where the model breaks down. The point
  isn't a "solved" life — it's making the modeling assumptions of applying optimization
  outside of textbook settings legible, which the Phase 3 pass already showed concretely
  (two solvers disagreeing on the same toy problem, not papered over).

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
