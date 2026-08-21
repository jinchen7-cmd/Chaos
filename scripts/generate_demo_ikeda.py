"""Second demo in the gallery: the Ikeda map (nonlinear optical cavity).

Same story as generate_demo.py, different chaotic system: rc_prediction's
ParameterAwareRC is trained only on pre-critical Ikeda trajectories
(mu = 0.91, 0.94, 0.97), then rolled forward closed-loop from a shared
starting state at a held-out, post-critical mu = 1.02. The ground-truth map
is iterated from the same starting state at the same mu for comparison.

Run:
    python scripts/generate_demo_ikeda.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

from rc_prediction import IkedaMap, ParameterAwareRC
from rc_prediction.arc.predictor import detect_collapse
from rc_prediction.systems.ikeda import DEFAULT_TRAINING_MU

OUT_PATH = Path(__file__).resolve().parent.parent / "assets" / "ikeda_collapse.gif"
MU_TEST = 1.02
BOUND = 6.0
N_STEPS = 150
N_FRAMES = 90
TAIL = 25
FREEZE_MARGIN = 8

# Offset into a long pre-critical trajectory used as the shared warmup / start
# state for both the true map and the model. Chosen so the two collapse near
# the same step; transient-chaos escape times are inherently seed-sensitive
# (see README), so this is one representative starting point among many.
WARMUP_OFFSET = 1300
WARMUP_LEN = 50


def build_model() -> tuple[IkedaMap, ParameterAwareRC, np.ndarray]:
    system = IkedaMap()
    training_data = system.simulate_training_set(
        DEFAULT_TRAINING_MU, n_steps=1500, burn_in=500, random_state=42
    )
    model = ParameterAwareRC(
        n_units=300,
        average_degree=8.0,
        spectral_radius=0.9,
        input_scaling=2.0,
        param_gain=0.35,
        param_bias=0.47,
        leaking_rate=1.0,
        ridge=1e-6,
        random_state=7,
    )
    model.fit(training_data, parameter_name="mu", train_length=800, collapse_bound=BOUND)

    long_traj = system.simulate(max(DEFAULT_TRAINING_MU), n_steps=4000, burn_in=500, random_state=99)
    warmup = long_traj[WARMUP_OFFSET : WARMUP_OFFSET + WARMUP_LEN]
    return system, model, warmup


def main() -> None:
    system, model, warmup = build_model()
    z0 = complex(warmup[-1, 0], warmup[-1, 1])

    true_from_start = system.simulate(MU_TEST, n_steps=N_STEPS + 1, burn_in=0, initial_state=z0)
    true_future = true_from_start[1:]
    true_collapsed, true_collapse = detect_collapse(true_future, bound=BOUND)

    result = model.predict_closed_loop(MU_TEST, n_steps=N_STEPS, warmup=warmup)
    pred = result.trajectory
    pred_collapse = result.collapse_step

    last_event = max(true_collapse or 0, pred_collapse or 0)
    n_show = min(N_STEPS, last_event + TAIL)
    true_future = true_future[:n_show]
    pred = pred[:n_show]
    steps = np.arange(n_show)
    frame_idx = np.unique(np.linspace(0, n_show - 1, N_FRAMES).astype(int))

    mag_true = np.abs(true_future[:, 0] + 1j * true_future[:, 1])
    mag_pred = np.abs(pred[:, 0] + 1j * pred[:, 1])

    fig, (ax_ts, ax_phase) = plt.subplots(
        2, 1, figsize=(6.4, 6.8), height_ratios=[1, 1.3], constrained_layout=True, dpi=85
    )
    fig.suptitle(
        "Reservoir computing flags a cavity blow-up it never trained on",
        fontsize=11.5,
        fontweight="bold",
    )

    TRUE_COLOR = "#1f2933"
    PRED_COLOR = "#e8590c"

    ax_ts.set_xlim(0, n_show)
    ax_ts.set_ylim(0, max(mag_true.max(), mag_pred.max()) * 1.15)
    ax_ts.axhline(BOUND, color="#999", lw=0.9, ls=":", label=f"collapse bound = {BOUND:g}")
    ax_ts.set_xlabel("time step  (test mu = 1.02, held out from training)", fontsize=9)
    ax_ts.set_ylabel("|z|", fontsize=9)
    ax_ts.tick_params(labelsize=8)
    (line_true_ts,) = ax_ts.plot([], [], color=TRUE_COLOR, lw=1.6, label="Ground truth (map)")
    (line_pred_ts,) = ax_ts.plot(
        [], [], color=PRED_COLOR, lw=1.4, ls="--", label="RC prediction (trained on mu ≤ 0.97)"
    )
    ax_ts.legend(loc="upper right", frameon=False, fontsize=7.5)

    def ptp(arr: np.ndarray) -> float:
        return float(np.max(arr) - np.min(arr))

    pad = 0.08 * max(ptp(true_future[:, 0]), ptp(pred[:, 0]), 1e-6)
    ax_phase.set_xlim(
        min(true_future[:, 0].min(), pred[:, 0].min()) - pad,
        max(true_future[:, 0].max(), pred[:, 0].max()) + pad,
    )
    ax_phase.set_ylim(
        min(true_future[:, 1].min(), pred[:, 1].min()) - pad,
        max(true_future[:, 1].max(), pred[:, 1].max()) + pad,
    )
    ax_phase.set_xlabel("Re(z)", fontsize=9)
    ax_phase.set_ylabel("Im(z)", fontsize=9)
    ax_phase.tick_params(labelsize=8)
    ax_phase.set_title("phase portrait", fontsize=9.5)
    (line_true_phase,) = ax_phase.plot([], [], color=TRUE_COLOR, lw=1.1, alpha=0.85)
    (line_pred_phase,) = ax_phase.plot([], [], color=PRED_COLOR, lw=0.9, ls="--", alpha=0.85)
    (dot_true,) = ax_phase.plot([], [], "o", color=TRUE_COLOR, ms=5)
    (dot_pred,) = ax_phase.plot([], [], "^", color=PRED_COLOR, ms=5)

    collapse_text = ax_ts.text(
        0.02, 0.94, "", transform=ax_ts.transAxes, fontsize=8, color="#555", va="top"
    )

    true_stop = (true_collapse + FREEZE_MARGIN) if true_collapse is not None else n_show
    pred_stop = (pred_collapse + FREEZE_MARGIN) if pred_collapse is not None else n_show

    def collapse_note(i: int) -> str:
        parts = []
        if true_collapse is not None and i >= true_collapse:
            parts.append(f"true blow-up at t={true_collapse}")
        if pred_collapse is not None and i >= pred_collapse:
            parts.append(f"RC predicted blow-up at t={pred_collapse}")
        return "\n".join(parts)

    def init():
        for artist in (line_true_ts, line_pred_ts, line_true_phase, line_pred_phase, dot_true, dot_pred):
            artist.set_data([], [])
        collapse_text.set_text("")
        return line_true_ts, line_pred_ts, line_true_phase, line_pred_phase, dot_true, dot_pred, collapse_text

    def update(i: int):
        i_true = min(i, true_stop)
        i_pred = min(i, pred_stop)
        line_true_ts.set_data(steps[: i_true + 1], mag_true[: i_true + 1])
        line_pred_ts.set_data(steps[: i_pred + 1], mag_pred[: i_pred + 1])
        line_true_phase.set_data(true_future[: i_true + 1, 0], true_future[: i_true + 1, 1])
        line_pred_phase.set_data(pred[: i_pred + 1, 0], pred[: i_pred + 1, 1])
        dot_true.set_data([true_future[i_true, 0]], [true_future[i_true, 1]])
        dot_pred.set_data([pred[i_pred, 0]], [pred[i_pred, 1]])
        collapse_text.set_text(collapse_note(i))
        return line_true_ts, line_pred_ts, line_true_phase, line_pred_phase, dot_true, dot_pred, collapse_text

    anim = animation.FuncAnimation(
        fig, update, frames=frame_idx, init_func=init, interval=90, blit=False
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    anim.save(OUT_PATH, writer=animation.PillowWriter(fps=12))
    plt.close(fig)

    print(f"true collapse step:      {true_collapse}")
    print(f"RC predicted collapse:   {pred_collapse}")
    print(f"frames shown:            {n_show}")
    print(f"saved animation to:      {OUT_PATH}")


if __name__ == "__main__":
    main()
