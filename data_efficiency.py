#!/usr/bin/env python3
"""Data-efficiency sweep for imitation learning (ACT on PushT / LeRobot).

QUESTION:
  "Under a small budget of demonstrations, do aggressive regularization
   (+ ensembling + distillation) buy data-efficiency for imitation-learning
   policies, as Konwoo et al. showed for data-constrained LM pre-training?"

DELIVERABLE = one figure: success rate (Y) vs number of demos (X), two curves
  (STANDARD vs ENHANCED recipe), plus a data-efficiency multiplier.

This script ORCHESTRATES LeRobot's battle-tested trainer (lerobot-train), which
handles training + simulated eval (success_rate) in one run. We only vary:
demos x recipe x seed.

Usage (sur les GPU du hacka) :
  python data_efficiency.py --demos 10 25 50 100 200 --recipes standard enhanced \
      --seeds 0 1 --steps 8000 --eval-episodes 50 --device cuda

Itération rapide locale (Mac MPS, smoke) :
  python data_efficiency.py --demos 10 --recipes standard --seeds 0 --steps 200 \
      --eval-episodes 10 --device mps
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable  # use the interpreter that launched this script (venv or system)
RUNS = HERE / "runs"

# Recettes : la STANDARD = défauts ACT. ENHANCED = régularisation agressive.
# (ensembling + distillation = leviers jour 2, voir README — la reg seule donne déjà 2 courbes.)
RECIPES = {
    "standard": [],
    "enhanced": [
        "--policy.dropout=0.2",            # dropout agressif
        "--optimizer.weight_decay=1e-3",   # weight decay agressif
        "--dataset.image_transforms.enable=true",  # data augmentation
    ],
}

# regex de parsing du success_rate dans les logs lerobot-train (à confirmer jour 1 sur 1 run).
SUCCESS_RE = re.compile(r"(?:success_rate|pc_success|eval/success)[^0-9]*([0-9]+\.?[0-9]*)", re.I)


def episodes_arg(n: int) -> str:
    return "[" + ",".join(str(i) for i in range(n)) + "]"


def run_one(n_demos: int, recipe: str, seed: int, steps: int, eval_eps: int, device: str) -> float | None:
    import shutil
    job = f"de_{recipe}_n{n_demos}_s{seed}"
    out_dir = RUNS / job
    RUNS.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(out_dir, ignore_errors=True)  # lerobot exige un output_dir inexistant
    cmd = [
        PY, "-m", "lerobot.scripts.lerobot_train",
        "--dataset.repo_id=lerobot/pusht",
        f"--dataset.episodes={episodes_arg(n_demos)}",
        "--dataset.video_backend=pyav",  # self-contained ffmpeg; avoids torchcodec/libavutil gaps on some nodes
        "--policy.type=act",
        f"--policy.device={device}",
        "--policy.push_to_hub=false",
        "--env.type=pusht",
        f"--steps={steps}",
        f"--seed={seed}",
        f"--eval.n_episodes={eval_eps}",
        f"--eval.batch_size={min(eval_eps, 10)}",  # batch <= n_episodes (sinon ParsingError)
        f"--eval_freq={steps}",          # eval à la fin
        f"--output_dir={out_dir}",
        f"--job_name={job}",
        "--save_checkpoint=true",
        "--wandb.enable=false",
    ] + RECIPES[recipe]
    log = RUNS / f"{job}.log"  # hors de out_dir (que lerobot crée lui-même)
    print(f"\n▶ {job}  (démos={n_demos} recette={recipe} seed={seed} steps={steps})")
    with open(log, "w") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
    txt = log.read_text(errors="replace")
    # cherche le dernier success_rate loggé
    hits = SUCCESS_RE.findall(txt)
    succ = float(hits[-1]) if hits else None
    if succ is not None and succ > 1.5:  # parfois en %
        succ /= 100.0
    print(f"  → success={succ}  (exit {p.returncode}, log {log})")
    return succ


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", type=int, nargs="+", default=[10, 25, 50, 100, 200])
    ap.add_argument("--recipes", nargs="+", default=["standard", "enhanced"], choices=list(RECIPES))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--eval-episodes", type=int, default=50)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=str(HERE / "results.csv"))
    a = ap.parse_args()

    rows = []
    for recipe in a.recipes:
        for n in a.demos:
            for seed in a.seeds:
                succ = run_one(n, recipe, seed, a.steps, a.eval_episodes, a.device)
                rows.append({"recipe": recipe, "demos": n, "seed": seed, "success": succ})
                # écrit au fil de l'eau (anti-perte si crash)
                with open(a.out, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=["recipe", "demos", "seed", "success"])
                    w.writeheader(); w.writerows(rows)
    print(f"\n✅ Résultats: {a.out}")
    try:
        make_figure(a.out)
    except Exception as e:
        print(f"(figure: {e} — lance plot séparément)")
    return 0


def make_figure(csv_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from collections import defaultdict
    import statistics as st
    agg = defaultdict(list)  # (recipe, demos) -> [success]
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            if r["success"] not in ("", "None", None):
                agg[(r["recipe"], int(r["demos"]))].append(float(r["success"]))
    plt.figure(figsize=(7, 5))
    for recipe in sorted({k[0] for k in agg}):
        xs = sorted({k[1] for k in agg if k[0] == recipe})
        ys = [st.mean(agg[(recipe, x)]) for x in xs]
        plt.plot(xs, ys, marker="o", label=recipe)
    plt.xlabel("Nb de démonstrations"); plt.ylabel("Taux de succès (sim pusht)")
    plt.title("Data-efficiency : standard vs enhanced (ACT, pusht)")
    plt.legend(); plt.grid(alpha=.3)
    out = str(Path(csv_path).with_suffix(".png"))
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"📊 Figure: {out}")


if __name__ == "__main__":
    sys.exit(main())
