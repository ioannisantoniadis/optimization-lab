# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.24.0",
# ]
# ///
import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Regularization Path Explorer")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go

    from optimlab.linalg import ridge_regression
    from optimlab.optimizers.proximal_gradient import (
        CompositeProblem,
        proximal_gradient,
        soft_threshold,
    )
    from optimlab.viz.theme import contrasting_categorical, layout_template

    return (
        CompositeProblem,
        contrasting_categorical,
        go,
        layout_template,
        mo,
        np,
        proximal_gradient,
        ridge_regression,
        soft_threshold,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Regularization Path Explorer

    The same synthetic regression data (only features 0 and 2 actually matter;
    features 1 and 3 are pure noise), fit two ways at whatever `alpha` you drag to
    — ridge (Chapter 3's closed-form SVD shrinkage) and LASSO (Chapter 4's
    proximal-gradient soft-thresholding).

    Watch feature 1 and 3's LASSO bars hit *exactly* zero once `alpha` crosses a
    threshold, while ridge's only ever approach zero asymptotically — the
    qualitative difference the whole regularization-path story is about, made
    interactive instead of a single static path plot.
    """)
    return


@app.cell
def _(mo):
    seed_slider = mo.ui.slider(0, 20, value=0, step=1, label="data seed")
    noise_slider = mo.ui.slider(0.0, 0.5, value=0.05, step=0.01, label="noise std")
    log_alpha_slider = mo.ui.slider(-2.0, 2.5, value=0.0, step=0.05, label="log10(alpha)")

    mo.vstack([seed_slider, noise_slider, log_alpha_slider])
    return log_alpha_slider, noise_slider, seed_slider


@app.cell
def _(np, noise_slider, seed_slider):
    rng = np.random.default_rng(seed_slider.value)
    A = rng.standard_normal((30, 4))
    x_true = np.array([2.0, 0.0, -1.5, 0.0])
    b = A @ x_true + noise_slider.value * rng.standard_normal(30)
    return A, b, x_true


@app.cell
def _(A, CompositeProblem, b, log_alpha_slider, np, proximal_gradient, ridge_regression, soft_threshold):
    alpha = 10.0**log_alpha_slider.value

    ridge_x = ridge_regression(A, b, alpha).x

    L = np.linalg.eigvalsh(A.T @ A).max()
    lasso_problem = CompositeProblem(
        grad_smooth=lambda x: A.T @ (A @ x - b),
        prox_nonsmooth=lambda v, t: soft_threshold(v, alpha * t),
        x0=np.zeros(A.shape[1]),
    )
    lasso_x = proximal_gradient(lasso_problem, lr=1.0 / L, max_iter=2000, tol=1e-10).x
    return alpha, lasso_x, ridge_x


@app.cell
def _(alpha, contrasting_categorical, go, layout_template, lasso_x, mo, ridge_x, x_true):
    categories = ["x0", "x1", "x2", "x3"]
    colors = contrasting_categorical()

    fig = go.Figure(
        [
            go.Bar(x=categories, y=x_true, name="true coefficient", marker={"color": "#898781", "opacity": 0.5}),
            go.Bar(x=categories, y=ridge_x, name="ridge", marker={"color": colors[0]}),
            go.Bar(x=categories, y=lasso_x, name="LASSO", marker={"color": colors[1]}),
        ]
    )
    fig.update_layout(
        **layout_template(title=f"Coefficients at alpha={alpha:.3g}", yaxis_title="coefficient value"),
        barmode="group",
    )
    mo.vstack([mo.md(f"**LASSO exact zeros**: {int((abs(lasso_x) < 1e-8).sum())} of {len(lasso_x)} coefficients"), fig])
    return


if __name__ == "__main__":
    app.run()
