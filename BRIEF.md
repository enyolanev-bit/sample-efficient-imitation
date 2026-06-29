# Technical Brief — Sample-Efficient Imitation Learning

**Author:** Nevil Enyola · [github.com/enyolanev-bit](https://github.com/enyolanev-bit) · [enyolanev-bit.github.io](https://enyolanev-bit.github.io)
**For:** AMD University Program — Learning Cloud access discussion

## One line

Does a data-efficiency recipe proven for language-model pre-training (aggressive regularization + ensembling + distillation) transfer to **robot imitation learning**, letting a policy reach the same task success with far fewer human demonstrations?

## Why it matters

Imitation learning is bottlenecked by demonstration cost: every demo is a human teleoperating the robot. Cutting the number of demos needed for a given success rate is one of the most practically valuable levers in robot learning. Konwoo et al. (Stanford, arXiv:2509.14786) showed ~5.17× data-efficiency from this recipe in data-constrained LM pre-training. **No public result tests whether it transfers to embodied imitation learning** — that is the gap this study fills.

## Method

Orchestrate LeRobot's `lerobot-train` (training + simulated PushT evaluation in one run) and sweep **demonstrations × recipe × seed**:

| Recipe | Description |
|---|---|
| `standard` | ACT, default hyperparameters |
| `enhanced` | aggressive regularization (dropout 0.2, weight decay 1e-3, image-transform augmentation); ensembling + distillation as extensions |

**Deliverable = one figure:** success rate (Y) vs number of demonstrations — 10, 25, 50, 100, 200 — (X), `standard` vs `enhanced`, plus the **data-efficiency multiplier** (fewer demos for equal success).

## Status — already validated, compute-bound

The full pipeline was **validated end-to-end locally (June 26, 2026, Apple M5 / MPS)**: LeRobot 0.5.1, the `lerobot/pusht` dataset (206 demos / 25,650 frames), ACT training, simulated evaluation, and the CSV+figure harness all run (exit 0). **This is not a proposal — the harness exists and works.** What it needs is GPU to run the full sweep at a meaningful step budget.

## Compute profile (why GPU-heavy)

Full sweep = 5 demo budgets × 2 recipes × ≥2 seeds × 8,000 steps each = ~40 ACT training+eval runs, scaling further with ensembling. This is squarely GPU-bound training — an ideal fit for the AUP Learning Cloud. LeRobot/ACT is PyTorch-based; AMD has run a LeRobot hackathon, so the ROCm path is supported.

## Reproducibility

- Harness: `data_efficiency.py` (writes `results.csv` + `results.png`, streams results to disk to survive crashes).
- Repo: clean, MIT-licensed, single-command sweep.
- Each run logged independently; success rate parsed from `lerobot-train` evaluation output.

## References

- Konwoo et al., *Pre-training under infinite compute*, arXiv:2509.14786 (Stanford).
- Wilson, *Deep Learning is Not So Mysterious or Different* — inductive biases as the lever of sample efficiency.
- LeRobot (Hugging Face) — training stack and `lerobot/pusht` dataset.
