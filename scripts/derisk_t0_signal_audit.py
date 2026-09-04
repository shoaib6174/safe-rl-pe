"""T0.1 signal audit (derisk plan v2). Rolls the gated SAC clone for >= 1M
env steps in BOTH the stochastic (training-time) and deterministic regimes
and writes results/derisk/t0_signal_audit.{json,png}.

`signal_audit` seeds torch's global generator from SEED0 at the start of
each regime and records it as `torch_seed`, so the stochastic regime — the
training-time one the kill rule is about — is reproducible from the seed
this file records rather than from the seed baked into the clone's zip.

Usage: ./venv/bin/python scripts/derisk_t0_signal_audit.py [--role pursuer] [--min-steps 1000000]
"""

import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import SAC

from pe2.config import EnvConfig
from pe2.derisk import signal_audit
from pe2.scripted import GreedyEvader, GreedyPursuer

OUT = pathlib.Path("results/derisk")
CLONE = {
    "pursuer": pathlib.Path("results/visual_gates/step_05/sac_bc_pursuer.zip"),
    "evader": pathlib.Path("results/visual_gates/step_05/sac_bc_evader.zip"),
}
OPPONENT = {"pursuer": GreedyEvader, "evader": GreedyPursuer}
SEED0 = 790000


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--role", choices=["pursuer", "evader"], default="pursuer")
    p.add_argument("--min-steps", type=int, default=1_000_000)
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = EnvConfig()
    model = SAC.load(CLONE[args.role], device="cpu")
    report = {}
    for det in (False, True):
        rep = signal_audit(
            model,
            cfg,
            args.role,
            OPPONENT[args.role],
            args.min_steps,
            SEED0,
            deterministic=det,
        )
        key = "deterministic" if det else "stochastic"
        report[key] = rep
        print(
            f"[{key}] steps={rep['total_steps']} episodes={rep['n_episodes']} "
            f"captures={rep['captures']} positives/1M={rep['positives_per_1M']:.1f} "
            f"visible={rep['visible_fraction']:.3f} var(G)={rep['return_variance_overall']:.4f} "
            f"torch_seed={rep['torch_seed']}",
            flush=True,
        )
    report["kill_rule"] = {
        "threshold_positives_per_1M": 50,
        "note": "evaluated on the P-A pilot's dense-reward training episodes; "
        "this file gives the terminal-only baseline",
    }
    with open(OUT / f"t0_signal_audit_{args.role}.json", "w") as f:
        json.dump(report, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for key, style in (("stochastic", "-"), ("deterministic", "--")):
        b = report[key]["return_variance_by_bucket"]
        axes[0].plot(
            [x["t_lo"] for x in b], [x["variance"] for x in b], style, label=key
        )
        axes[1].hist(
            report[key]["capture_lengths"],
            bins=24,
            range=(0, cfg.max_steps),
            histtype="step",
            label=f"{key} (n={report[key]['captures']})",
        )
    axes[0].set(
        xlabel="step index t", ylabel="var(G_t)", title="MC-return variance by step"
    )
    axes[1].set(
        xlabel="capture length (steps)", ylabel="count", title="capture lengths"
    )
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / f"t0_signal_audit_{args.role}.png", dpi=120)
    print(f"wrote {OUT}/t0_signal_audit_{args.role}.{{json,png}}")


if __name__ == "__main__":
    main()
