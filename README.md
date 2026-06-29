# Sample-Efficient Imitation Learning

**Do the data-efficiency tricks that work for language-model pre-training also work for robot imitation learning?**

A controlled study: under a small budget of human demonstrations, do aggressive regularization, ensembling, and distillation buy data-efficiency for imitation-learning policies (ACT on LeRobot / PushT) — as Konwoo et al. showed for data-constrained LM pre-training?

## Question

Imitation learning is data-scarce by nature: every demonstration is a human teleoperating the robot, which is slow and expensive. The field would gain a lot from policies that reach the same task performance with **fewer** demonstrations. This repo tests whether a recipe proven in a different domain (data-constrained LM pre-training, Konwoo et al. 2025) transfers to embodied imitation learning.

## Method

We orchestrate LeRobot's battle-tested trainer (`lerobot-train`, which handles training + simulated evaluation in one run) and sweep over **number of demonstrations × recipe × seed**.

- **standard** — ACT with default hyperparameters.
- **enhanced** — aggressive regularization: dropout 0.2, weight decay 1e-3, image-transform data augmentation. (Ensembling and distillation are planned extensions; regularization alone already yields a defensible two-curve result.)

**Deliverable — one figure:**
- X axis = number of demonstrations (10, 25, 50, 100, 200)
- Y axis = success rate (PushT simulation)
- Two curves: `standard` vs `enhanced`
- Plus a single number: the **data-efficiency multiplier** — how many fewer demonstrations the enhanced recipe needs for the same success rate.

## Status

**Validated end-to-end locally (June 26, 2026, Apple M5 / MPS).** The full pipeline runs: LeRobot 0.5.1 installed, the `lerobot/pusht` dataset loads (206 demos / 25,650 frames), an ACT policy trains, simulated PushT evaluation produces a success flag, and the harness parses the success rate into a CSV + figure (exit 0). Scaling to a full sweep is a matter of compute: same command with `--steps 8000 --demos 10 25 50 100 200 --device cuda`.

## Run

```bash
# Full sweep (GPU)
python data_efficiency.py --demos 10 25 50 100 200 --recipes standard enhanced \
    --seeds 0 1 --steps 8000 --eval-episodes 50 --device cuda
# -> results.csv + results.png

# Local smoke test (Apple MPS)
python data_efficiency.py --demos 10 --recipes standard --seeds 0 --steps 200 \
    --eval-episodes 10 --device mps
```

Dependencies: `lerobot[pusht]` (0.5.1) + PyTorch. Training is GPU-bound; CPU/MPS is only for smoke tests.

## Research grounding

- **Konwoo et al., "Pre-training under infinite compute"** (arXiv:2509.14786, Stanford) — in data-constrained LM pre-training, aggressive regularization + ensembling + distillation yield ~5.17× data-efficiency; distilling an 8-model ensemble into one keeps ~83% of the gain. This repo asks whether the same levers transfer to imitation learning.
- **Wilson, "Deep Learning is Not So Mysterious or Different"** — inductive biases (i.e. regularization) are a primary lever of sample efficiency; theoretical backing for why the enhanced recipe should help.

## License

MIT (see `LICENSE`).
