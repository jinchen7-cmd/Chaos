"""Generate the hero animation for the chaos-ml showcase.

Trains rc_prediction's ParameterAwareRC only on *pre-collapse* trajectories of the
Hastings-Powell / McCann-Yodzis food chain (carrying capacity K = 0.97, 0.98, 0.99),
then rolls the model forward in closed loop from a shared starting state at a
held-out, post-critical K = 1.01 it never saw during training. The ground-truth
ODE is integrated from the same starting state at the same K for comparison. Both
trajectories necessarily diverge pointwise (the system is chaotic), but the model
anticipates the predator collapse within the same regime as the true collapse.

Run:
    python scripts/generate_demo.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

from rc_prediction import FoodChain, ParameterAwareRC
from rc_prediction.systems.food_chain import DEFAULT_TRAINING_K

OUT_PATH = Path(__file__).resolve().parent.parent / "assets" / "collapse_prediction.gif"
K_TEST = 1.01
N_STEPS = 420
N_FRAMES = 110  # subsample N_STEPS down to this many animation frames
TAIL = 40  # steps after both trajectories collapse, kept for visual settle
FREEZE_MARGIN = 15  # steps past its own collapse a trajectory keeps drawing, then freezes


def build_model() -> tuple[FoodChain, ParameterAwareRC, np.ndarray]:
    system = FoodChain()
    training_data = system.simulate_training_set(
        DEFAULT_TRAINING_K,
        t_max=1200.0,
        dt=1.0,
        burn_in=600.0,
        random_state=42,
    )
    model = ParameterAwareRC(
        n_units=400,
        average_degree=4.0,
        spectral_radius=2.3,
        input_scaling=3.6,
        param_gain=0.50,
        param_bias=-2.2,
        leaking_rate=0.30,
        ridge=3e-5,
        random_state=3,
    )
    model.fit(
        training_data,
        parameter_name="K",
        train_length=400,
        predator_index=system.predator_index,
    )
    warmup = training_data[max(DEFAULT_TRAINING_K)][-50:]
    return system, model, warmup


def first_collapse_index(predator: np.ndarray, threshold: float = 0.01) -> int | None:
    below = np.where(predator < threshold)[0]
    return int(below[0]) if below.size else None


def main() -> None:
    system, model, warmup = build_model()

    _, true_from_start = system.simulate(
        K_TEST, t_max=float(N_STEPS + 1), dt=1.0, burn_in=0.0, initial_state=warmup[-1]
    )
    true_future = true_from_start[1 : N_STEPS + 1]

    result = model.predict_closed_loop(K_TEST, n_steps=N_STEPS, warmup=warmup)
    pred = result.trajectory

    true_collapse = first_collapse_index(true_future[:, 2])
    pred_collapse = result.collapse_step
    last_event = max(true_collapse or 0, pred_collapse or 0)
    n_show = min(N_STEPS, last_event + TAIL)

    true_future = true_future[:n_show]
    pred = pred[:n_show]
    steps = np.arange(n_show)
    frame_idx = np.unique(np.linspace(0, n_show - 1, N_FRAMES).astype(int))

    fig, (ax_ts, ax_phase) = plt.subplots(
        2, 1, figsize=(6.4, 6.8), height_ratios=[1, 1.3], constrained_layout=True, dpi=85
    )
    fig.suptitle(
        "Reservoir computing predicts a collapse it never trained on",
        fontsize=11.5,
        fontweight="bold",
    )

    TRUE_COLOR = "#1f2933"
    PRED_COLOR = "#e8590c"

    ax_ts.set_xlim(0, n_show)
    ax_ts.set_ylim(0, max(true_future[:, 2].max(), pred[:, 2].max()) * 1.15)
    ax_ts.set_xlabel("time step  (test K = 1.01, held out from training)", fontsize=9)
    ax_ts.set_ylabel("predator density P", fontsize=9)
    ax_ts.tick_params(labelsize=8)
    (line_true_ts,) = ax_ts.plot([], [], color=TRUE_COLOR, lw=1.6, label="Ground truth (ODE)")
    (line_pred_ts,) = ax_ts.plot(
        [], [], color=PRED_COLOR, lw=1.4, ls="--", label="RC prediction (trained on K ≤ 0.99)"
    )
    ax_ts.legend(loc="upper right", frameon=False, fontsize=8)

    def ptp(arr: np.ndarray) -> float:
        return float(np.max(arr) - np.min(arr))

    pad = 0.05 * max(ptp(true_future[:, 0]), ptp(pred[:, 0]), 1e-6)
    ax_phase.set_xlim(
        min(true_future[:, 0].min(), pred[:, 0].min()) - pad,
        max(true_future[:, 0].max(), pred[:, 0].max()) + pad,
    )
    ax_phase.set_ylim(
        min(true_future[:, 2].min(), pred[:, 2].min()) - pad,
        max(true_future[:, 2].max(), pred[:, 2].max()) + pad,
    )
    ax_phase.set_xlabel("resource R", fontsize=9)
    ax_phase.set_ylabel("predator P", fontsize=9)
    ax_phase.tick_params(labelsize=8)
    ax_phase.set_title("phase portrait (R, P)", fontsize=9.5)
    (line_true_phase,) = ax_phase.plot([], [], color=TRUE_COLOR, lw=1.2, alpha=0.85)
    (line_pred_phase,) = ax_phase.plot([], [], color=PRED_COLOR, lw=1.0, ls="--", alpha=0.85)
    (dot_true,) = ax_phase.plot([], [], "o", color=TRUE_COLOR, ms=5)
    (dot_pred,) = ax_phase.plot([], [], "^", color=PRED_COLOR, ms=5)

    collapse_text = ax_ts.text(
        0.02, 0.06, "", transform=ax_ts.transAxes, fontsize=8, color="#555", va="bottom"
    )

    true_stop = (true_collapse + FREEZE_MARGIN) if true_collapse is not None else n_show
    pred_stop = (pred_collapse + FREEZE_MARGIN) if pred_collapse is not None else n_show

    def collapse_note(i: int) -> str:
        parts = []
        if true_collapse is not None and i >= true_collapse:
            parts.append(f"true collapse at t={true_collapse}")
        if pred_collapse is not None and i >= pred_collapse:
            parts.append(f"RC predicted collapse at t={pred_collapse}")
        return "  |  ".join(parts)

    def init():
        for artist in (line_true_ts, line_pred_ts, line_true_phase, line_pred_phase, dot_true, dot_pred):
            artist.set_data([], [])
        collapse_text.set_text("")
        return line_true_ts, line_pred_ts, line_true_phase, line_pred_phase, dot_true, dot_pred, collapse_text

    def update(i: int):
        i_true = min(i, true_stop)
        i_pred = min(i, pred_stop)
        line_true_ts.set_data(steps[: i_true + 1], true_future[: i_true + 1, 2])
        line_pred_ts.set_data(steps[: i_pred + 1], pred[: i_pred + 1, 2])
        line_true_phase.set_data(true_future[: i_true + 1, 0], true_future[: i_true + 1, 2])
        line_pred_phase.set_data(pred[: i_pred + 1, 0], pred[: i_pred + 1, 2])
        dot_true.set_data([true_future[i_true, 0]], [true_future[i_true, 2]])
        dot_pred.set_data([pred[i_pred, 0]], [pred[i_pred, 2]])
        collapse_text.set_text(collapse_note(i))
        return line_true_ts, line_pred_ts, line_true_phase, line_pred_phase, dot_true, dot_pred, collapse_text

    anim = animation.FuncAnimation(
        fig, update, frames=frame_idx, init_func=init, interval=70, blit=False
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    anim.save(OUT_PATH, writer=animation.PillowWriter(fps=14))
    plt.close(fig)

    print(f"true collapse step:      {true_collapse}")
    print(f"RC predicted collapse:   {pred_collapse}")
    print(f"frames shown:            {n_show}")
    print(f"saved animation to:      {OUT_PATH}")


if __name__ == "__main__":
    main()
