# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.24.0",
# ]
# ///
import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="LQR Control Explorer")


@app.cell
def _():
    import marimo as mo
    import numpy as np

    from optimlab.control import simulate_lqr, solve_lqr
    from optimlab.viz import trajectory_and_control_figure

    return mo, np, simulate_lqr, solve_lqr, trajectory_and_control_figure


@app.cell
def _(mo):
    mo.md(r"""
    # LQR Control Explorer

    A double integrator (position + velocity, control = acceleration), driven from
    `(x0, v0)` toward the origin by the closed-form Riccati recursion. Because LQR
    is a closed form rather than an iterative solve, every drag re-solves instantly
    — no waiting for an optimizer to converge.

    A few things worth trying:

    - Push `R` (control cost) way up — the controller gets stingier with
      acceleration, and the trajectory takes longer, gentler swing back to the
      origin.
    - Push `Q_velocity` up relative to `Q_position` — the controller starts caring
      more about killing velocity quickly than about position error, changing the
      whole shape of the approach.
    - Start with a large `v0` in the *same* direction as `x0` (both positive) vs.
      opposite signs — one overshoots the origin before correcting, the other
      doesn't.
    """)
    return


@app.cell
def _(mo):
    x0_slider = mo.ui.slider(-3.0, 3.0, value=1.0, step=0.1, label="x0 (initial position)")
    v0_slider = mo.ui.slider(-3.0, 3.0, value=0.0, step=0.1, label="v0 (initial velocity)")
    q_pos_slider = mo.ui.slider(0.01, 10.0, value=1.0, step=0.01, label="Q_position")
    q_vel_slider = mo.ui.slider(0.01, 10.0, value=0.1, step=0.01, label="Q_velocity")
    r_slider = mo.ui.slider(0.01, 5.0, value=0.1, step=0.01, label="R (control cost)")
    n_steps_slider = mo.ui.slider(10, 60, value=30, step=1, label="n_steps")

    mo.hstack(
        [mo.vstack([x0_slider, v0_slider]), mo.vstack([q_pos_slider, q_vel_slider, r_slider]), mo.vstack([n_steps_slider])],
        justify="start",
        gap=2,
    )
    return n_steps_slider, q_pos_slider, q_vel_slider, r_slider, v0_slider, x0_slider


@app.cell
def _(n_steps_slider, np, q_pos_slider, q_vel_slider, r_slider, simulate_lqr, solve_lqr, v0_slider, x0_slider):
    dt = 0.1
    A = np.array([[1.0, dt], [0.0, 1.0]])
    B = np.array([[0.0], [dt]])
    Q = np.diag([q_pos_slider.value, q_vel_slider.value])
    R = np.array([[r_slider.value]])
    Q_f = np.diag([10.0, 10.0])
    n_steps = n_steps_slider.value

    lqr_result = solve_lqr(A, B, Q, R, Q_f, n_steps)
    states, controls = simulate_lqr(A, B, lqr_result.gains, x0=np.array([x0_slider.value, v0_slider.value]))
    t = np.arange(n_steps + 1) * dt
    return controls, states, t


@app.cell
def _(controls, mo, states, t, trajectory_and_control_figure):
    mo.vstack(
        [
            mo.md(f"**final state**: position = {states[-1, 0]:.4f}, velocity = {states[-1, 1]:.4f}"),
            trajectory_and_control_figure(
                t, states, controls, state_labels=["position", "velocity"], control_labels=["accel"],
                title="LQR: double integrator",
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
