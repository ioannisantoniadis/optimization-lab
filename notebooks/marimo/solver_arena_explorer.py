# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.24.0",
# ]
# ///
import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Solver Arena Explorer")


@app.cell
def _():
    import marimo as mo
    import numpy as np

    from optimlab.arena import run_arena
    from optimlab.landscapes import ALL_FUNCTIONS
    from optimlab.viz import arena_figure

    return ALL_FUNCTIONS, arena_figure, mo, np, run_arena


@app.cell
def _(mo):
    mo.md(r"""
    # Solver Arena Explorer

    Pick a landscape and a starting point — every solver in `ALL_SOLVERS` runs
    against it at once, and the bar chart redraws live, sorted best-to-worst.

    A few things worth trying:

    - Start `rastrigin` from a point far from the origin — the gradient-based
      solvers (green/orange, top of a well-behaved run) can get stuck in a nearby
      ripple, while `bayesian_optimize`/`genetic_algorithm`/`particle_swarm` are
      more likely to still find the global basin.
    - Compare `sphere` (no local minima anywhere) against `rastrigin` (many) from
      the *same* starting point — the ranking reshuffles completely.
    - Watch which solvers report `converged` (green) vs. `max_iter reached`
      (orange) even when their final objective value is excellent — a different
      stopping rule, not a worse answer.
    """)
    return


@app.cell
def _(ALL_FUNCTIONS, mo):
    function_dropdown = mo.ui.dropdown(options=sorted(ALL_FUNCTIONS), value="himmelblau", label="landscape")
    x0_slider = mo.ui.slider(-5.0, 5.0, value=-4.0, step=0.1, label="start x0")
    x1_slider = mo.ui.slider(-5.0, 5.0, value=4.0, step=0.1, label="start x1")
    mo.hstack([function_dropdown, x0_slider, x1_slider], justify="start", gap=2)
    return function_dropdown, x0_slider, x1_slider


@app.cell
def _(ALL_FUNCTIONS, function_dropdown, np, run_arena, x0_slider, x1_slider):
    benchmark = ALL_FUNCTIONS[function_dropdown.value]
    problem = benchmark.problem(x0=np.array([x0_slider.value, x1_slider.value]), n_dim=2)
    report = run_arena(problem)
    return (report,)


@app.cell
def _(arena_figure, mo, report):
    best = report.ranked_by_objective()[0]
    mo.vstack(
        [
            mo.md(f"**best**: `{best.name}`, f = {best.result.f:.4g} in {best.result.n_iter} iterations"),
            arena_figure(report),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
