# Chaos

Eight things to play with, then the research they're standing in for. Ecology, statistical physics, celestial mechanics, complex dynamics, cellular automata — different corners of the same subject: what happens when a system is simple to describe and impossible to predict.

| | |
|---|---|
| 🦋 **[The Butterfly Effect](https://jinchen7-cmd.github.io/Chaos/butterfly-effect.html)** | Sixteen double pendulums released a hair's width apart fan out into total disagreement in seconds. Sound, presets, a live divergence readout, a race-the-chaos stopwatch. |
| 🌳 **[Where Order Breaks](https://jinchen7-cmd.github.io/Chaos/bifurcation.html)** | The logistic map's bifurcation diagram: `x ← r·x·(1−x)` going from boring to chaotic as one knob turns. Zoom in anywhere — the fig-tree pattern repeats forever. |
| 🔺 **[The Chaos Game](https://jinchen7-cmd.github.io/Chaos/chaos-game.html)** | Pick a random corner, move halfway there, repeat forever with no memory. A perfect fractal falls out of pure randomness. |
| ⬛ **[The Simplest Chaos](https://jinchen7-cmd.github.io/Chaos/rule30.html)** | An 8-line lookup table (Wolfram's Rule 30) turns a single black cell into a pattern irregular enough to generate random numbers — deterministically. |
| 💧 **[One More Drop](https://jinchen7-cmd.github.io/Chaos/percolation.html)** | Fill a grid at random, cell by cell. Below a sharp threshold: scattered puddles. Above it: one cluster suddenly spans the whole grid. A phase transition you can click through. |
| 🦊 **[Boom and Bust](https://jinchen7-cmd.github.io/Chaos/boom-and-bust.html)** | The actual food-chain model behind the research below, live and draggable. Push the carrying capacity up and watch stable cycles turn chaotic, then collapse. |
| 🪐 **[The Three-Body Problem](https://jinchen7-cmd.github.io/Chaos/three-body.html)** | Real gravity, RK4-integrated — the problem that led Poincaré to discover chaos in 1887. Click to nudge the bodies and watch a stable dance fall apart. |
| 🌀 **[The Mandelbrot Set](https://jinchen7-cmd.github.io/Chaos/mandelbrot.html)** | Drag to zoom into the most famous fractal in complex dynamics. The boundary never simplifies, at any depth. |

That unpredictability — and the order hiding underneath it — is the whole problem this repo is about, and the rest of it is about clawing some of it back with machine learning.

---

**Can a neural network see a collapse coming before it happens — in a system it never trained on?**

![Reservoir computing predicting a predator population collapse it was never trained on](assets/collapse_prediction.gif)

A reservoir-computing model is trained on three *stable* simulations of a three-species food chain (resource → consumer → predator). It never sees a single collapse during training. Then it's handed a new, harder parameter setting and left to run freely — no more ground truth, just its own predictions feeding back into itself. It anticipates the predator's extinction almost as early as the real equations do.

Start with the toy above, then read on for how the prediction actually works.

---

## What the animation shows

The system is a Hastings–Powell / McCann–Yodzis food chain — a classic chaotic ecological model — with resource carrying capacity `K` as the control parameter:

- **Training:** the model only ever sees `K = 0.97, 0.98, 0.99` — all pre-collapse, all sustained chaos.
- **Test:** it's run at `K = 1.01`, a value it has never seen, where the true dynamics eventually drive the predator extinct.
- **Black line:** the real system, integrated from the governing ODEs.
- **Orange dashed line:** the model's own closed-loop forecast — every step feeds back in as the next input, with no correction from the truth.

Because the system is chaotic, the two trajectories can't stay identical forever — that's what chaos *means*. What's notable is that they track each other closely at first, and the model still calls the collapse in the same regime as the real thing, using nothing but a parameter-aware reservoir and ridge regression.

This is a reproduction of the method from [Kong, Fan, Grebogi & Lai (2021), *Physical Review Research* 3, 013090](https://doi.org/10.1103/PhysRevResearch.3.013090) — "Machine learning prediction of critical transition and system collapse."

## Gallery: the same method, different chaos

The food-chain demo above is one instance of a general recipe. Here it is again on two more classic chaotic systems.

<table>
<tr>
<td width="50%">

**Ikeda map** — a nonlinear optical cavity, iterated as a discrete map instead of an ODE. Trained on `mu ≤ 0.97`, tested at `mu = 1.02`: the model flags the cavity's blow-up almost as early as the true map does.

![RC flags an Ikeda map blow-up it never trained on](assets/ikeda_collapse.gif)

</td>
<td width="50%">

**Lorenz system** — the original chaos icon, no critical transition here, just sustained chaos. An Echo State Network trained on one-step prediction is cut loose to generate its *own* trajectory with no ground truth at all. It can't track the true path forever (that's chaos), but it keeps landing back on the same two-winged butterfly attractor instead of blowing up or flatlining.

![A reservoir computer freely generating the Lorenz butterfly attractor](assets/lorenz_freerun.gif)

</td>
</tr>
</table>

## Regenerate it yourself

```bash
git clone https://github.com/jinchen7-cmd/Chaos.git
cd Chaos
pip install -r requirements.txt
python scripts/generate_demo.py          # food chain
python scripts/generate_demo_ikeda.py    # Ikeda map
python scripts/generate_demo_lorenz.py   # Lorenz butterfly
```

Each takes well under a minute on a laptop CPU. Tweak the test parameter, step count, or reservoir hyperparameters at the top of each script to explore other regimes.

A note on the timing match in the food-chain and Ikeda demos: both are *transient chaos* — post-critical trajectories eventually collapse, but chaos makes the exact collapse step for any one run highly sensitive to the starting state. The paper's actual claim (and the right way to evaluate this) is statistical — matching the *distribution* of collapse times over many runs, via `scan_critical_point` and `ensemble_predict` in the underlying package — not a single-trajectory race. The animations pick one representative starting point each; run the scripts with a different offset and the exact step numbers will shift.

## Why this matters

Real systems don't hand you their equations. Ecosystems collapsing under overharvesting, power grids approaching blackout, climate subsystems nearing tipping points — in each case you'd like to know a transition is coming *before* you have data from the other side of it. Reservoir computing offers a model-free way to extrapolate past what a system has shown you, trained purely on the time series it left behind.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
