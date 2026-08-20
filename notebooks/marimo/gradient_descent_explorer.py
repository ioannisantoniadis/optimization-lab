# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.24.0",
# ]
# ///
import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Gradient Descent Explorer")


@app.cell
def _():
    import marimo as mo
    import numpy as np

    from optimlab.landscapes import ALL_FUNCTIONS
    from optimlab.optimizers import ALL_SOLVERS
    from optimlab.viz import convergence_figure, race_figure

    return ALL_FUNCTIONS, ALL_SOLVERS, convergence_figure, mo, np, race_figure


@app.cell
def _(mo):
    mo.md(r"""
    # Gradient Descent Explorer

    Pick a landscape, pick a starting point, pick one or more solvers, and drag the
    hyperparameter sliders — the trajectories and convergence curves redraw live.

    A few things worth trying:

    - Start `gradient_descent` from a corner of **rosenbrock** and watch it crawl
      along the curved valley floor — then compare against `newton` or `bfgs` from
      the same point.
    - Push `gradient_descent`'s learning rate up on **sphere** until it stops
      converging — that threshold is exactly `2 / L`, where `L` is the sphere's
      curvature (its Hessian is `2·I`, so its largest eigenvalue is `2`).
    - Start from `(-4, 4)` on **himmelblau** (four equal global minima) with two or
      three solvers at once and see whether they land in the same basin.
    """)
    return


@app.cell
def _(ALL_FUNCTIONS, ALL_SOLVERS, mo):
    function_dropdown = mo.ui.dropdown(
        options=sorted(ALL_FUNCTIONS), value="himmelblau", label="landscape"
    )
    solver_picker = mo.ui.multiselect(
        options=sorted(ALL_SOLVERS),
        value=["gradient_descent", "newton", "bfgs"],
        label="solvers to race",
    )
    x0_slider = mo.ui.slider(-5.0, 5.0, value=-4.0, step=0.1, label="start x0")
    x1_slider = mo.ui.slider(-5.0, 5.0, value=4.0, step=0.1, label="start x1")
    lr_slider = mo.ui.slider(
        0.0001, 1.0, value=0.01, step=0.0001, label="learning rate (gd / momentum / adaptive)"
    )
    beta_slider = mo.ui.slider(0.0, 0.999, value=0.9, step=0.01, label="momentum beta")
    max_iter_slider = mo.ui.slider(10, 2000, value=300, step=10, label="max iterations")

    mo.hstack(
        [
            mo.vstack([function_dropdown, solver_picker]),
            mo.vstack([x0_slider, x1_slider]),
            mo.vstack([lr_slider, beta_slider, max_iter_slider]),
        ],
        justify="start",
        gap=2,
    )
    return (
        beta_slider,
        function_dropdown,
        lr_slider,
        max_iter_slider,
        solver_picker,
        x0_slider,
        x1_slider,
    )


@app.cell
def _(ALL_FUNCTIONS, function_dropdown, np, x0_slider, x1_slider):
    benchmark = ALL_FUNCTIONS[function_dropdown.value]
    start = np.array([x0_slider.value, x1_slider.value])
    problem_template = benchmark.problem(x0=start, n_dim=2)
    return (problem_template,)


@app.cell
def _():
    # Hyperparameters each solver actually accepts — a slider only affects the solvers
    # that use it (e.g. dragging `beta` does nothing for `newton`, which takes none).
    SOLVER_KWARGS = {
        "gradient_descent": {"lr"},
        "heavy_ball": {"lr", "beta"},
        "nesterov": {"lr", "beta"},
        "adagrad": {"lr"},
        "rmsprop": {"lr"},
        "adam": {"lr"},
        "newton": set(),
        "bfgs": set(),
        "lbfgs": set(),
    }
    return (SOLVER_KWARGS,)


@app.cell
def _(
    ALL_SOLVERS,
    SOLVER_KWARGS,
    beta_slider,
    lr_slider,
    max_iter_slider,
    mo,
    problem_template,
    solver_picker,
):
    all_kwargs = {"lr": lr_slider.value, "beta": beta_slider.value}

    results = {}
    for name in solver_picker.value:
        solver = ALL_SOLVERS[name]
        kwargs = {k: v for k, v in all_kwargs.items() if k in SOLVER_KWARGS[name]}
        kwargs["max_iter"] = max_iter_slider.value
        # Every solver shares the same Problem (same objective, same start point) — each
        # solver only ever *copies* problem.x0 internally, never mutates it, so reusing
        # the instance across solvers is safe and skips rebuilding grad/hess each time.
        results[name] = solver(problem_template, **kwargs)

    status = mo.md(
        "Pick at least one solver above." if not results else
        "  \n".join(f"**{n}** — {r}" for n, r in results.items())
    )
    status
    return (results,)


@app.cell
def _(problem_template, race_figure, results):
    race_figure(problem_template, results) if results else None
    return


@app.cell
def _(convergence_figure, results):
    convergence_figure(results) if results else None
    return


if __name__ == "__main__":
    app.run()
