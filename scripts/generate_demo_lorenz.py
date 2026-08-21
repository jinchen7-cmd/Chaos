"""Third demo in the gallery: free-running (closed-loop) Lorenz forecasting.

Unlike the food-chain and Ikeda demos, the classic Lorenz system (rho=28) has
no critical transition to predict -- it's just sustained chaos. The story
here is different: an Echo State Network is trained on one-step-ahead
prediction, then cut loose to generate its own trajectory with no further
access to ground truth (its own output becomes its next input). Pointwise
divergence from the true trajectory is inevitable in a chaotic system, but a
well-trained closed-loop ESN keeps landing back on the same butterfly-shaped
attractor instead of blowing up or collapsing to a fixed point.

Run:
    python scripts/generate_demo_lorenz.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

from rc_prediction import ESN
from rc_prediction.utils import lorenz_system, standardize

OUT_PATH = Path(__file__).resolve().parent.parent / "assets" / "lorenz_freerun.gif"
BURN_IN = 1000
N_TRAIN = 3000
N_STEPS = 400
N_FRAMES = 110


def main() -> None:
    raw = lorenz_system(BURN_IN + N_TRAIN + N_STEPS + 10, dt=0.02)[BURN_IN:]
    data, _, _ = standardize(raw)

    X, y = data[:-1], data[1:]
    X_train, y_train = X[:N_TRAIN], y[:N_TRAIN]

    model = ESN(
        n_units=500,
        spectral_radius=1.05,
        leaking_rate=0.3,
        input_scaling=0.6,
        ridge=1e-7,
        warmup=200,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Reservoir state right after fit already encodes the training sequence up
    # to X_train[-1]; run closed loop from there with no further teacher forcing.
    state = model.reservoir.state.copy()
    pred = np.empty((N_STEPS, 3))
    for k in range(N_STEPS):
        y_pred = model.readout.predict(state.reshape(1, -1))[0]
        pred[k] = y_pred
        state = model.reservoir.step(y_pred)

    true_future = X[N_TRAIN : N_TRAIN + N_STEPS]

    steps = np.arange(N_STEPS)
    frame_idx = np.unique(np.linspace(0, N_STEPS - 1, N_FRAMES).astype(int))

    fig, (ax_ts, ax_phase) = plt.subplots(
        2, 1, figsize=(6.0, 6.4), height_ratios=[1, 1.3], constrained_layout=True, dpi=78
    )
    fig.suptitle(
        "A reservoir computer, running blind, still finds the butterfly",
        fontsize=11,
        fontweight="bold",
    )

    TRUE_COLOR = "#1f2933"
    PRED_COLOR = "#e8590c"

    ax_ts.set_xlim(0, N_STEPS)
    ylim = max(np.abs(true_future[:, 0]).max(), np.abs(pred[:, 0]).max()) * 1.15
    ax_ts.set_ylim(-ylim, ylim)
    ax_ts.set_xlabel("time step  (closed loop: own output feeds back as next input)", fontsize=8.5)
    ax_ts.set_ylabel("x (standardized)", fontsize=9)
    ax_ts.tick_params(labelsize=8)
    (line_true_ts,) = ax_ts.plot([], [], color=TRUE_COLOR, lw=1.4, label="Ground truth (ODE)")
    (line_pred_ts,) = ax_ts.plot(
        [], [], color=PRED_COLOR, lw=1.2, ls="--", label="ESN closed-loop forecast"
    )
    ax_ts.legend(loc="upper right", frameon=False, fontsize=8)

    def ptp(arr: np.ndarray) -> float:
        return float(np.max(arr) - np.min(arr))

    pad = 0.08 * max(ptp(true_future[:, 0]), ptp(pred[:, 0]), 1e-6)
    ax_phase.set_xlim(
        min(true_future[:, 0].min(), pred[:, 0].min()) - pad,
        max(true_future[:, 0].max(), pred[:, 0].max()) + pad,
    )
    ax_phase.set_ylim(
        min(true_future[:, 2].min(), pred[:, 2].min()) - pad,
        max(true_future[:, 2].max(), pred[:, 2].max()) + pad,
    )
    ax_phase.set_xlabel("x", fontsize=9)
    ax_phase.set_ylabel("z", fontsize=9)
    ax_phase.tick_params(labelsize=8)
    ax_phase.set_title("phase portrait (x, z)", fontsize=9.5)
    (line_true_phase,) = ax_phase.plot([], [], color=TRUE_COLOR, lw=1.0, alpha=0.8)
    (line_pred_phase,) = ax_phase.plot([], [], color=PRED_COLOR, lw=0.9, ls="--", alpha=0.8)
    (dot_true,) = ax_phase.plot([], [], "o", color=TRUE_COLOR, ms=5)
    (dot_pred,) = ax_phase.plot([], [], "^", color=PRED_COLOR, ms=5)

    def init():
        for artist in (line_true_ts, line_pred_ts, line_true_phase, line_pred_phase, dot_true, dot_pred):
            artist.set_data([], [])
        return line_true_ts, line_pred_ts, line_true_phase, line_pred_phase, dot_true, dot_pred

    def update(i: int):
        line_true_ts.set_data(steps[: i + 1], true_future[: i + 1, 0])
        line_pred_ts.set_data(steps[: i + 1], pred[: i + 1, 0])
        line_true_phase.set_data(true_future[: i + 1, 0], true_future[: i + 1, 2])
        line_pred_phase.set_data(pred[: i + 1, 0], pred[: i + 1, 2])
        dot_true.set_data([true_future[i, 0]], [true_future[i, 2]])
        dot_pred.set_data([pred[i, 0]], [pred[i, 2]])
        return line_true_ts, line_pred_ts, line_true_phase, line_pred_phase, dot_true, dot_pred

    anim = animation.FuncAnimation(
        fig, update, frames=frame_idx, init_func=init, interval=60, blit=False
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    anim.save(OUT_PATH, writer=animation.PillowWriter(fps=14))
    plt.close(fig)

    rmse_final = float(np.sqrt(np.mean((pred[-1] - true_future[-1]) ** 2)))
    print(f"pointwise RMSE at final step: {rmse_final:.3f} (divergence is expected -- it's chaos)")
    print(f"saved animation to: {OUT_PATH}")


if __name__ == "__main__":
    main()
