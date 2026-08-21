# chaos-ml

**Can a neural network see a collapse coming before it happens — in a system it never trained on?**

![Reservoir computing predicting a predator population collapse it was never trained on](assets/collapse_prediction.gif)

A reservoir-computing model is trained on three *stable* simulations of a three-species food chain (resource → consumer → predator). It never sees a single collapse during training. Then it's handed a new, harder parameter setting and left to run freely — no more ground truth, just its own predictions feeding back into itself. It anticipates the predator's extinction almost as early as the real equations do.

This repo is the visual front door to a small ecosystem of chaos + machine-learning projects. Start here for the "why," then follow the links below for the code and the paper.

---

## What the animation shows

The system is a Hastings–Powell / McCann–Yodzis food chain — a classic chaotic ecological model — with resource carrying capacity `K` as the control parameter:

- **Training:** the model only ever sees `K = 0.97, 0.98, 0.99` — all pre-collapse, all sustained chaos.
- **Test:** it's run at `K = 1.01`, a value it has never seen, where the true dynamics eventually drive the predator extinct.
- **Black line:** the real system, integrated from the governing ODEs.
- **Orange dashed line:** the model's own closed-loop forecast — every step feeds back in as the next input, with no correction from the truth.

Because the system is chaotic, the two trajectories can't stay identical forever — that's what chaos *means*. What's notable is that they track each other closely at first, and the model still calls the collapse in the same regime as the real thing, using nothing but a parameter-aware reservoir and ridge regression.

This is a reproduction of the method from [Kong, Fan, Grebogi & Lai (2021), *Physical Review Research* 3, 013090](https://doi.org/10.1103/PhysRevResearch.3.013090) — "Machine learning prediction of critical transition and system collapse."

## Regenerate it yourself

```bash
git clone https://github.com/jinchen7-cmd/chaos-ml.git
cd chaos-ml
pip install -r requirements.txt
python scripts/generate_demo.py
```

Takes under a minute on a laptop CPU. Tweak `K_TEST`, `N_STEPS`, or the reservoir hyperparameters in [`scripts/generate_demo.py`](scripts/generate_demo.py) to explore other regimes.

## The rest of the ecosystem

| Repo | What it is |
|---|---|
| [**Reservoir-Computing**](https://github.com/jinchen7-cmd/Reservoir-Computing) | `rc_prediction` — the actual Python package behind this demo. Published on [PyPI](https://pypi.org/project/rc-prediction/), tested, documented. Implements parameter-aware reservoir computing for critical-transition prediction across the Ikeda map, this food chain, and the Kuramoto–Sivashinsky equation. |
| [**RC-SINDy-Ecosystem**](https://github.com/jinchen7-cmd/RC-SINDy-Ecosystem) | A research write-up comparing reservoir computing against SINDy (sparse identification of nonlinear dynamics) for discovering the governing equations of chaotic ecological systems — Lorenz, Lotka–Volterra, Hastings–Powell, Beddington–DeAngelis. |

## Why this matters

Real systems don't hand you their equations. Ecosystems collapsing under overharvesting, power grids approaching blackout, climate subsystems nearing tipping points — in each case you'd like to know a transition is coming *before* you have data from the other side of it. Reservoir computing offers a model-free way to extrapolate past what a system has shown you, trained purely on the time series it left behind.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
