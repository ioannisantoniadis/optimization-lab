# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.24.0",
# ]
# ///
import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="LP Polytope Explorer")


@app.cell
def _():
    import marimo as mo

    from optimlab.optimizers.linear_programming import LinearProgram, simplex
    from optimlab.viz import polytope_figure

    return LinearProgram, mo, polytope_figure, simplex


@app.cell
def _(mo):
    mo.md(r"""
    # LP Polytope Explorer

    A 2-variable linear program: `maximize c0*x0 + c1*x1` subject to
    `x0 <= k1`, `2*x1 <= k2`, `3*x0 + 2*x1 <= k3`, `x >= 0`. Drag the constraint
    bounds `k1`/`k2`/`k3` to reshape the feasible region, or the objective
    direction `c0`/`c1` to change which vertex is optimal — simplex re-solves and
    the polytope + vertex path redraw live.

    A few things worth trying:

    - Shrink `k3` until it cuts through the polytope's interior — watch which
      vertex simplex lands on change discontinuously as the feasible region's shape
      changes.
    - Set `c0` and `c1` equal — the optimal vertex should sit wherever the
      constraint `3x0+2x1<=k3` is tightest, since both objectives pull equally.
    - Push `k1` or `k2` far out until it stops touching the polytope's optimal
      corner at all — the optimum stops moving once a constraint is no longer
      binding.
    """)
    return


@app.cell
def _(mo):
    k1_slider = mo.ui.slider(1.0, 15.0, value=4.0, step=0.5, label="k1  (x0 <= k1)")
    k2_slider = mo.ui.slider(1.0, 20.0, value=12.0, step=0.5, label="k2  (2*x1 <= k2)")
    k3_slider = mo.ui.slider(1.0, 30.0, value=18.0, step=0.5, label="k3  (3*x0+2*x1 <= k3)")
    c0_slider = mo.ui.slider(-5.0, 5.0, value=3.0, step=0.25, label="c0  (objective weight on x0)")
    c1_slider = mo.ui.slider(-5.0, 5.0, value=5.0, step=0.25, label="c1  (objective weight on x1)")

    mo.hstack(
        [mo.vstack([k1_slider, k2_slider, k3_slider]), mo.vstack([c0_slider, c1_slider])],
        justify="start",
        gap=2,
    )
    return c0_slider, c1_slider, k1_slider, k2_slider, k3_slider


@app.cell
def _(LinearProgram, c0_slider, c1_slider, k1_slider, k2_slider, k3_slider, simplex):
    # simplex always minimizes -- negate the objective to "maximize" c0*x0 + c1*x1.
    lp = LinearProgram(
        name="interactive_lp",
        c=[-c0_slider.value, -c1_slider.value],
        A_ub=[[1, 0], [0, 2], [3, 2]],
        b_ub=[k1_slider.value, k2_slider.value, k3_slider.value],
    )
    result = simplex(lp)
    return lp, result


@app.cell
def _(lp, mo, polytope_figure, result):
    (
        mo.md(f"**status: {result.status}** -- no polytope to draw for this constraint combination.")
        if result.status != "optimal"
        else mo.vstack(
            [
                mo.md(
                    f"**optimum**: x = {result.x.round(3)}, "
                    f"objective = {-result.objective:.3f}, {result.n_iter} pivots"
                ),
                polytope_figure(lp, result),
            ]
        )
    )
    return


if __name__ == "__main__":
    app.run()
