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

## Phase 5 — Bayesian modeling and estimation — **done**

- [x] `optimlab.inference.mle_fit` / `map_fit`: MLE and MAP as ordinary `Problem`
      minimizations (negative log-likelihood, optionally plus a negative log-prior),
      with a pluggable `solver` so a boundary-constrained parameter (a probability, a
      variance) can use `projected_gradient` instead of plain BFGS — needed in practice,
      not hypothetically: unconstrained BFGS overshoots wildly on a Beta-Binomial
      log-likelihood's unbounded boundary curvature.
- [x] Least squares as statistical estimation: verified, not just asserted — fitting the
      identical data through `optimlab.linalg.least_squares` (Chapter 3's SVD-derived
      solver) and through `mle_fit` with a Gaussian log-likelihood lands on the same
      optimum to `1e-6`, and likewise `ridge_regression` vs. `map_fit` with a matching
      Gaussian prior. Regularization and a prior turn out to be the same object, not
      just an analogy.
- [x] `optimlab.inference.laplace_approximation`: Gaussian approximation to a posterior
      at its MAP mode, covariance from the inverse Hessian there (reusing
      `optimlab.core._autohess`). Verified exact on a conjugate Gaussian-Gaussian model
      (where the true posterior already is Gaussian), and shown breaking down on a
      skewed Beta-Binomial posterior — its Gaussian shape assigns real probability mass
      past the parameter's valid boundary, and its mean lands on the mode rather than
      the true posterior mean.
- [x] `optimlab.inference.metropolis_hastings`: from-scratch random-walk MCMC sampler.
      Verified against the same two conjugate models — matches the closed-form
      Gaussian-Gaussian posterior's mean/std, and (unlike Laplace) correctly recovers
      the skewed Beta-Binomial posterior's true mean rather than its mode.
- [x] `optimlab.inference.em_gmm`: Expectation-Maximization for Gaussian mixture models
      — the one genuinely new algorithm this phase adds, alternating closed-form E/M
      steps rather than taking a gradient step at all. Verified two ways: the true
      log-likelihood is checked directly to be monotonically non-decreasing across
      iterations (EM's one guarantee, a consequence of Jensen's inequality — no
      step-size condition needed the way every gradient method in this repo has one),
      and three well-separated synthetic cluster means are recovered to within 0.1.
- [x] Backend link: `optimlab.backends.optuna_minimize` — a fourth black-box optimizer
      (Optuna's TPE sampler) alongside Phase 3's from-scratch `genetic_algorithm`/
      `particle_swarm`/`simulated_annealing`, following the same try/except-absent
      pattern as `cvxpy_backend` if the `backends` extra isn't installed.
- [x] Three visualizations (`optimlab/viz/inference.py`): `posterior_figure` (true
      density / Laplace's Gaussian / MCMC's histogram overlaid — the figure that makes
      the Beta-Binomial skew mismatch immediately visible), `mcmc_trace_figure` (chain
      trace + marginal histogram, the standard first-look mixing diagnostic), and
      `gmm_figure` (fitted data colored by hard cluster assignment, with each
      component's covariance drawn as an n-std ellipse via its eigendecomposition). All
      three rendered and visually reviewed before committing.
- [x] Docs chapter (`06-bayesian-modeling-and-estimation.qmd`) covering all of the
      above: MLE, least-squares-as-MLE and ridge-as-MAP (both verified live in the
      rendered chapter, not just claimed), the Laplace approximation, MCMC (with the
      Beta-Binomial skew comparison and a trace-diagnostic figure), EM/GMM, and the
      optuna backend comparison (four methods landing in the same basin, reported
      honestly rather than as a "the new backend wins" story). Full site rendered and
      every section/figure reviewed in a real browser tab.

## Phase 6 — High-dimensional non-convexity: the flagship module — **done**

The centerpiece motivating this whole repo: building real intuition for what a
million-to-billion-dimensional non-convex loss surface looks like, and why gradient
descent works on it anyway.

- [x] `optimlab.highdim.nets`: a minimal MLP whose entire parameter vector is a single
      flat array, wrapped as an ordinary `optimlab.core.Problem` — the backbone every
      other piece below builds on. Chapters 1–4's unmodified solvers already train it
      (verified: a 673-parameter network fitting noisy sine data down to the injected
      noise floor with plain `adam`, no special-casing anywhere for "this happens to be
      a neural network").
- [x] **Saddle points dominate, not bad local minima**
      [@dauphin2014identifying; @choromanska2015loss]: sample random GOE matrix
      eigenvalues and measure P(local min) vs. dimension directly — collapses from a
      coin flip at dimension 1 to 0/20,000 samples by dimension 6.
- [x] **Hessian eigenspectrum of real trained networks**
      [@sagun2016eigenvalues; @ghorbani2019investigation]: Hessian-vector products via
      forward-over-reverse autodiff (`jax.jvp` of `jax.grad`, verified against a dense
      Hessian to `1e-8`) feeding a from-scratch Lanczos algorithm with full
      reorthogonalization (Ritz values verified against a dense eigendecomposition's
      extremes to `1e-6`, and checked to never exceed the true spectrum's range). Plain
      Lanczos rather than the originally-scoped stochastic Hutchinson trace estimation
      — Lanczos alone was already enough for the "few outliers over a near-zero bulk"
      shape this phase actually needed to show.
- [x] **Loss landscape visualization at scale** [@li2018visualizing]: filter-normalized
      random 2D slices through weight space (verified: each layer's random direction
      matches that layer's own weight norm exactly), plus the cheaper
      [@goodfellow2015qualitatively] linear-interpolation diagnostic.
- [x] **Mode connectivity** [@garipov2018loss]: two independently trained minima joined
      by a quadratic Bezier curve whose one free control point is found by minimizing
      the average loss along it — another ordinary `Problem`, solved by the same `bfgs`
      used everywhere else. Verified: straight-line interpolation between two trained
      minima crosses a real barrier (~50x either endpoint's own loss at its peak); the
      optimized curve stays flat near each endpoint's loss the entire way.
- [x] **Sharp vs. flat minima** [@keskar2016large]: reuses the Lanczos tool above rather
      than new code — comparing plain vs. L2-regularized training's top Hessian
      eigenvalue on the identical problem shows regularization biasing toward flatter
      minima at a real cost in fit quality, reported honestly (a modest gap for this
      specific setup, not oversold as dramatic).
- [x] **Why high dimensions are weird, geometrically**: pairwise cosine similarity of
      random directions (concentrates toward 0 as dimension grows) and the ball-shell
      volume fraction closed form (a high-dim ball's volume lives almost entirely in a
      thin shell near its surface) — no citation needed, both original from-scratch
      demos.
- [x] **Overparameterization makes training easier, not harder**
      [@jacot2018neural; @du2019gradient]: the empirical neural tangent kernel and a
      concentration experiment — independent random initializations' NTKs become more
      similar to each other as width grows (averaged over seed pairs per width to
      smooth small-width noise; verified monotonically decreasing across widths
      4→16→64→256→1024).
- [x] Six visualizations (`optimlab/viz/highdim.py`): the saddle-point collapse,
      cosine-similarity concentration, a Hessian eigenspectrum scree plot (chosen over
      a histogram, which blurs the "few outliers" shape), a filter-normalized loss
      slice, a shared curve-comparison plot (linear interpolation and mode connectivity
      are both just "loss along a 1D path"), and NTK concentration with a shaded std
      band. Rendering and reviewing them surfaced a real bug: the shaded-band trace
      didn't set `mode="lines"`, so Plotly's default drew stray markers at the band's
      own concatenated corner points instead of a clean fill.
- [x] Docs chapter (`07-high-dimensional-non-convexity.qmd`) covering all of the above
      end to end on one trained network, reviewed in a real browser tab. Caught and
      corrected two numbers before committing that a first draft's prose overclaimed
      relative to what actually rendered: the sharp-vs-flat gap ("measurably flatter...
      slightly worse fit" corrected to state the real, several-times-larger MSE cost),
      and the NTK trend's small-width noise (rather than asserting clean monotonicity
      the first draft's number happened not to have).

## Phase 7 — Domain applications: inverse problems, control, machine learning — **done**

- [x] `optimlab.inverse.deblurring`: image deblurring posed as an ordinary `Problem`
      (mean-squared reconstruction error + Tikhonov/L2 penalty on a separable Gaussian
      blur operator) — Chapter 3's `ridge_regression` idea, solved by gradient descent
      on an image instead of the closed-form SVD route. Verified a genuine
      bias-variance tradeoff, not just a plausible-looking recovery: too little
      regularization (`alpha=1e-4`) amplifies noise past the blurry observation's own
      MSE, `alpha=0.01` cuts MSE by ~40% below it, and `alpha=0.3` over-smooths back
      toward blur.
- [x] `optimlab.inverse.system_id`: recovering a damped oscillator's physical
      parameters from noisy trajectory data via `NonlinearLeastSquaresProblem` /
      `gauss_newton` — the RK4 integrator is written entirely in `jax.numpy` so
      autodiff flows through the simulation loop itself, giving Gauss-Newton's Jacobian
      as a byproduct rather than a hand-derived sensitivity equation. Verified:
      recovers `omega`/`zeta` to within 1% of their true values from noisy data.
- [x] `optimlab.control.lqr`: LQR via the closed-form backward Riccati recursion.
      Cross-checked against the identical finite-horizon problem posed instead as an
      ordinary `Problem` over the flattened control sequence and solved by `bfgs` —
      closed-form and iterative routes agree on cost to 1e-6 relative and on the
      controls themselves to 1e-6 absolute.
- [x] `optimlab.control.trajectory_optimization`: nonlinear optimal control via direct
      shooting — a pendulum swing-up posed as an ordinary `Problem` over the entire
      control sequence, an RK4-simulated nonlinear dynamics model differentiated
      through directly (no linearization). Verified: swings from hanging straight down
      to upright within 0.12 degrees using only `bfgs`, no new optimizer.
- [x] `optimlab.control.dynamic_programming`: value iteration for a grid-world MDP —
      the one genuinely new algorithm in this phase, a discrete Bellman fixed-point
      iteration rather than continuous optimization. Verified: value increases
      monotonically toward the goal and the greedy policy navigates around obstacles to
      reach it. MPC and a full RL bridge scoped out for now — value iteration already
      carries this phase's "a genuinely different algorithm shape" content.
- [x] `optimlab.ml.backprop`: hand-implemented forward + backward pass for the
      `optimlab.highdim.nets` MLP architecture — explicit chain rule through `tanh`
      hidden activations and a linear output layer, no autodiff anywhere in the
      function. Cross-checked against `optimlab.core`'s JAX-autodiff gradient on the
      identical network and loss: agrees to `8e-17` (essentially machine precision)
      across two different architectures.
- [x] `optimlab.ml.pinn`: physics-informed neural networks — a network trained purely
      from an ODE's own residual plus its initial condition, never shown the true
      solution anywhere in its loss. Verified: matches the analytic solution to within
      `0.0002` absolute error across the domain after training. A toy transformer
      scoped out for now as a separate, larger undertaking.
- [x] Six visualizations (`optimlab/viz/{inverse,control,ml}.py`): the
      true/observed/recovered image triptych, the observed-vs-fitted trajectory
      comparison, a shared state/control plot (reused for both LQR and the pendulum),
      a grid-world value-function-plus-policy heatmap, and a PINN-vs-analytic-solution
      overlay. Rendering and reviewing them surfaced a real layout bug: a
      `scaleanchor`'d 1:1 aspect constraint across three side-by-side image subplots
      fought the shared figure width and padded a 24x24 image out to several times its
      own size — fixed with explicit per-panel axis ranges instead.
- [x] Docs chapter (`08-domain-applications.qmd`) covering all of the above end to end,
      reviewed in a real browser tab. Caught and fixed a second real Pandoc footgun
      along the way (after Phase 4's `^`-as-superscript one): a bare `*` used for
      multiplication in a fig-cap (`y=2*exp(-0.5x)`) was parsed as markdown emphasis
      syntax, italicizing an entire unrelated span of the caption — fixed by wrapping
      the math in backticks.

## Phase 8 — Cross-domain problem library and the solver arena — **done**

The final phase — deliberately no new algorithms, just the payoff of every solver
since Phase 1 sharing the identical `Problem -> OptimizeResult` interface.

- [x] `optimlab.arena.run_arena`: register a `Problem`, get a standardized report
      (final objective, iterations, wall time, converged flag) across every solver in
      `ALL_SOLVERS`. Catches each solver's own exceptions rather than letting one
      inapplicable solver stop the rest — verified on a `Problem` with no domain set:
      the three solvers needing `bounds` (`genetic_algorithm`, `particle_swarm`,
      `bayesian_optimize`) fail with a clear message while the other 11 succeed
      normally. `arena_figure` draws the result as one bar per solver, colored by
      outcome.
- [x] **Domain problems** — physics already had its worked example (Chapter 7's
      pendulum swing-up), not duplicated:
    - *Economics*: `optimlab.problems.economics`, Markowitz portfolio optimization —
      an equality-constrained QP solved by Chapter 1's `equality_constrained_qp`.
      Cross-checked against `cvxpy_qp` to `1e-6`; the efficient frontier traces the
      classic Markowitz hyperbola (risk falls to a single global-minimum-variance
      point, then rises on either side).
    - *Sociology/networks*: `optimlab.problems.sociology`, fair resource allocation
      via proportional fairness (Kelly 1997 / Network Utility Maximization), solved by
      Chapter 4's `barrier_method`. Verified against the one closed-form case (n
      identical users sharing one resource split exactly equally) and an asymmetric
      case (a user contending for two constrained resources at once gets squeezed
      relative to users contending for only one).
    - *Machine learning*: `optimlab.optimizers.bayesian_optimization`, a from-scratch
      Bayesian optimizer (closed-form Gaussian process regression + Expected
      Improvement) — a genuinely new algorithm, maintaining a probabilistic surrogate
      instead of following a gradient or evolving a population. Compared against the
      Optuna backend on Himmelblau and Rastrigin; came out ahead on both at the
      specific low-dimensional, small-budget comparison run (not claimed as a general
      result — Optuna's TPE sampler targets much higher-dimensional search spaces
      where a GP's `O(n^3)` cost per fit becomes the bottleneck).
- [x] **Life as optimization, revisited**: Chapter 4's single-weighted-sum version
      reframed as genuinely multi-objective — two aggregate objectives ("obligations":
      sleep+work, "fulfillment": exercise+social), a weighted-sum sweep over their
      trade-off (reusing `genetic_algorithm`, no new solver), and the non-dominated
      points identified directly rather than asserted. Independence, stationarity, and
      the four-to-two grouping simplification stated explicitly as modeling
      assumptions, plus the weighted-sum method's own blind spot (it can only trace
      the convex part of a Pareto front) — the actual output is the shape of a
      real trade-off, not a single "solved" answer.
- [x] Docs chapter (`09-cross-domain-problems-and-the-solver-arena.qmd`) covering all
      of the above end to end, reviewed in a real browser tab.

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
