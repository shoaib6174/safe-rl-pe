"""Step 5 gate figure: trained clone vs the D76 baseline battery.

One bar chart, JSON-only provenance (no env access): the D76 battery
(4 heuristics + K random inits, N=200, gate seeds 720000+) with Wilson 95%
CI whiskers, the pooled random-init distribution the gate actually tests
against, and the gated trained model. Numbers are cross-checked against
the two source JSONs; re-renders must be byte-identical (md5 asserted).

Checklist for sign-off (hard rule 9):
1. trained bar sits above every heuristic and the pooled-init band, with
   non-overlapping CIs (the D76 criteria)
2. the random-init spread (seed-4 outlier) is visible — the reason the
   gate uses the pooled distribution, not one init
3. md5 stable across re-render (asserted, printed)

Reads  results/visual_gates/step_05/step_05_baselines_pursuer.json
       results/visual_gates/step_05/step_05_bc_sanity.json (or --gate)
Writes results/visual_gates/step_05/step_05_gate.png

Usage: ./venv/bin/python scripts/pe2_step05_gate_figure.py [--role pursuer] [--gate PATH]
"""

import argparse
import hashlib
import json
import pathlib

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

OUT_DIR = pathlib.Path("results/visual_gates/step_05")


def md5(path: pathlib.Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def build_figure(
    baselines_path: pathlib.Path, gate_path: pathlib.Path, out: pathlib.Path
):
    with open(baselines_path) as f:
        battery = json.load(f)["baselines"]
    with open(gate_path) as f:
        gate = json.load(f)["gate"]

    names = list(battery)
    rates = [battery[nm]["win_rate"] for nm in names]
    lo = [battery[nm]["win_ci"][0] for nm in names]
    hi = [battery[nm]["win_ci"][1] for nm in names]
    colors = [
        "#4c72b0" if battery[nm]["kind"] == "heuristic" else "#c0c0c0" for nm in names
    ]

    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=100)
    x = list(range(len(names)))
    ax.bar(x, rates, color=colors, width=0.7)
    ax.errorbar(
        x,
        rates,
        yerr=[
            [r - l for r, l in zip(rates, lo, strict=True)],
            [h - r for h, r in zip(hi, rates, strict=True)],
        ],
        fmt="none",
        ecolor="black",
        capsize=3,
        lw=1,
    )

    pooled = gate["pooled_random_init"]
    ax.axhspan(pooled["win_ci"][0], pooled["win_ci"][1], color="#c0c0c0", alpha=0.25)
    ax.axhline(pooled["win_rate"], color="#808080", ls="--", lw=1)
    ax.text(
        len(names) - 0.4,
        pooled["win_ci"][1] + 0.005,
        f"pooled random-init {pooled['win_rate']:.3f} "
        f"[{pooled['win_ci'][0]:.3f}, {pooled['win_ci'][1]:.3f}]",
        ha="right",
        fontsize=8,
        color="#606060",
    )

    t = gate["trained_br"]
    ax.bar([len(names)], [t["win_rate"]], color="#dd8452", width=0.7)
    ax.errorbar(
        [len(names)],
        [t["win_rate"]],
        yerr=[[t["win_rate"] - t["win_ci"][0]], [t["win_ci"][1] - t["win_rate"]]],
        fmt="none",
        ecolor="black",
        capsize=3,
        lw=1.5,
    )

    crit = gate["criteria"]
    ax.set_title(
        f"Step 5 D76 gate: {pathlib.Path(gate_path).stem} — pass={crit['pass']} "
        f"(heuristics {crit['beats_every_heuristic']}, pooled-init "
        f"{crit['beats_pooled_random_init']}, McNemar {crit['mcnemar_vs_iid_random']})",
        fontsize=10,
    )
    ax.set_xticks(x + [len(names)])
    ax.set_xticklabels(
        [nm.replace("random_init_", "init_") for nm in names] + ["TRAINED"],
        rotation=30,
        ha="right",
        fontsize=8,
    )
    ax.set_ylabel("win rate (capture)")
    ax.set_ylim(0, max(hi + [t["win_ci"][1]]) * 1.15)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=["pursuer", "evader"], default="pursuer")
    parser.add_argument(
        "--gate", type=pathlib.Path, default=OUT_DIR / "step_05_bc_sanity_pursuer.json"
    )
    args = parser.parse_args()

    baselines_path = OUT_DIR / f"step_05_baselines_{args.role}.json"
    out = OUT_DIR / "step_05_gate.png"

    build_figure(baselines_path, args.gate, out)
    first = md5(out)
    build_figure(baselines_path, args.gate, out)  # re-render over it
    second = md5(out)
    assert first == second, f"gate figure not deterministic: {first} != {second}"
    print(f"determinism: md5 {first} stable across re-render OK")

    with open(args.gate) as f:
        gate = json.load(f)["gate"]
    t = gate["trained_br"]
    print(
        f"cross-check: TRAINED {t['win_rate']:.3f} [{t['win_ci'][0]:.3f}, "
        f"{t['win_ci'][1]:.3f}] vs pooled-init "
        f"{gate['pooled_random_init']['win_rate']:.3f}; criteria {gate['criteria']}"
    )
    print(f"wrote {out}")
    print("\nchecklist for sign-off:")
    print("1. TRAINED bar clears every heuristic + pooled-init band, CIs disjoint")
    print("2. random-init spread (seed-4 outlier) visible — why the gate pools")
    print("3. md5 stable across re-render (asserted above)")


if __name__ == "__main__":
    main()
