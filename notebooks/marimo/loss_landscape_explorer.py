# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.24.0",
# ]
# ///
import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Loss Landscape Explorer")


@app.cell
def _():
    import marimo as mo
    import numpy as np

    from optimlab.highdim import (
        MLPShape,
        lanczos_eigenvalues,
        loss_landscape_slice,
        mlp_training_problem,
    )
    from optimlab.optimizers import adam
    from optimlab.viz import hessian_spectrum_figure, loss_landscape_figure

    return (
        MLPShape,
        adam,
        hessian_spectrum_figure,
        lanczos_eigenvalues,
        loss_landscape_figure,
        loss_landscape_slice,
        mlp_training_problem,
        mo,
        np,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Loss Landscape Explorer

    A small MLP is trained once, at startup, to fit noisy `sin(x)` data. Everything
    below is interactive *around that fixed trained minimum* — a filter-normalized
    2D slice through its (hundreds-of-dimensional) weight space, and a Lanczos
    estimate of its Hessian's eigenspectrum, both re-drawn from the same trained
    network as you drag the controls.

    A few things worth trying:

    - Drag `span` out wide — the bowl around the minimum should eventually give way
      to a much rougher-looking surface further out, in directions the training run
      never had to be well-behaved in.
    - Change `direction seed` — a genuinely different random 2D slice through the
      same 673-dimensional space, usually still bowl-shaped near the minimum (this
      is what makes "just look at a random slice" a reasonable diagnostic at all).
    - Push `Lanczos iterations` down to 5 or so — the top eigenvalue estimate should
      already be close to its value at 80 iterations; the bulk near zero is what
      actually needs many more iterations to resolve accurately.
    """)
    return


@app.cell
def _(MLPShape, adam, mlp_training_problem, np):
    rng = np.random.default_rng(0)
    X = rng.uniform(-3, 3, size=(150, 1))
    y = (np.sin(X) + 0.05 * rng.standard_normal(X.shape)).reshape(-1, 1)

    shape = MLPShape(layer_sizes=[1, 24, 24, 1])
    problem = mlp_training_problem(shape, X, y, seed=0)
    trained = adam(problem, lr=0.01, max_iter=3000)
    return problem, shape, trained


@app.cell
def _(mo):
    span_slider = mo.ui.slider(0.1, 5.0, value=1.0, step=0.1, label="span")
    seed_slider = mo.ui.slider(0, 20, value=0, step=1, label="direction seed")
    n_iter_slider = mo.ui.slider(2, 80, value=40, step=1, label="Lanczos iterations")
    mo.vstack([span_slider, seed_slider, n_iter_slider])
    return n_iter_slider, seed_slider, span_slider


@app.cell
def _(
    hessian_spectrum_figure,
    lanczos_eigenvalues,
    loss_landscape_figure,
    loss_landscape_slice,
    mo,
    n_iter_slider,
    problem,
    seed_slider,
    shape,
    span_slider,
    trained,
):
    loss_slice = loss_landscape_slice(
        problem.f, trained.x, shape, span=span_slider.value, resolution=40, seed=seed_slider.value
    )
    lanczos_result = lanczos_eigenvalues(problem.f, trained.x, n_iter=n_iter_slider.value, seed=0)

    mo.hstack(
        [loss_landscape_figure(loss_slice), hessian_spectrum_figure(lanczos_result.ritz_values)],
        justify="start",
        gap=2,
    )
    return


if __name__ == "__main__":
    app.run()
