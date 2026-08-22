# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.24.0",
# ]
# ///
import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Central Path Explorer")


@app.cell
def _():
    import marimo as mo
    import numpy as np

    from optimlab.optimizers.barrier_method import ConstrainedProblem, barrier_method
    from optimlab.viz import central_path_figure

    return ConstrainedProblem, barrier_method, central_path_figure, mo, np


@app.cell
def _(mo):
    mo.md(r"""
    # Central Path Explorer

    `minimize (x0-3)^2 + (x1-3)^2` subject to `x0 <= k1`, `x1 <= k2` — pulled toward
    an infeasible target `(3, 3)`, fenced in by two box constraints. Drag `k1`/`k2`
    and watch the barrier method's central path re-curve toward wherever the new
    constrained corner sits.

    A few things worth trying:

    - Set `k1 = k2` — the corner sits on the diagonal, and the central path should
      approach it symmetrically.
    - Push `k1` and `k2` both above 3 — the target becomes feasible, the corner
      disappears, and the path should walk essentially straight to `(3, 3)`.
    - Shrink one bound far below the other — the path bends hard toward whichever
      constraint is actually binding, barely touching the other.
    """)
    return


@app.cell
def _(mo):
    k1_slider = mo.ui.slider(0.2, 6.0, value=1.0, step=0.1, label="k1  (x0 <= k1)")
    k2_slider = mo.ui.slider(0.2, 6.0, value=1.0, step=0.1, label="k2  (x1 <= k2)")
    mo.vstack([k1_slider, k2_slider])
    return k1_slider, k2_slider


@app.cell
def _(ConstrainedProblem, barrier_method, k1_slider, k2_slider, np):
    k1, k2 = k1_slider.value, k2_slider.value
    x0_start = np.array([0.4 * k1, 0.4 * k2])

    problem = ConstrainedProblem(
        f=lambda x: (x[0] - 3.0) ** 2 + (x[1] - 3.0) ** 2,
        x0=x0_start,
        inequality_constraints=[lambda x: x[0] - k1, lambda x: x[1] - k2],
        name="interactive_box_corner",
    )
    result = barrier_method(problem)
    return problem, result


@app.cell
def _(central_path_figure, mo, problem, result):
    mo.vstack(
        [
            mo.md(
                f"**optimum**: x = {result.x.round(4)}, f = {result.f:.4f}, "
                f"{result.n_iter} outer iterations, converged = {result.converged}"
            ),
            central_path_figure(problem, result),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
