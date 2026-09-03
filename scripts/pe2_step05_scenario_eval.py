"""Step 5 scenario eval: per-scenario performance of a trained BR model.

Runs the saved SAC BR (sac_br_pursuer.zip) over the six canonical
scenarios (pe2.scenarios, D70) and reports CR + Wilson CI per stratum —
the "where does it fail" complement to the random-seed battery's "how
much".

Writes results/visual_gates/step_05/step_05_scenario_eval.json.

Usage: ./venv/bin/python scripts/pe2_step05_scenario_eval.py [--n 30] [--model PATH]
"""

import argparse
import json
import pathlib

from pe2.config import EnvConfig
from pe2.eval import summarize
from pe2.scenarios import default_scenarios, evaluate_scenarios
from pe2.scripted import GreedyEvader

OUT_DIR = pathlib.Path("results/visual_gates/step_05")
DEFAULT_MODEL = OUT_DIR / "sac_br_pursuer.zip"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--model", type=pathlib.Path, default=DEFAULT_MODEL)
    args = parser.parse_args()

    from stable_baselines3 import SAC  # lazy: optional dependency

    cfg = EnvConfig()
    scenarios = default_scenarios(cfg)
    model = SAC.load(args.model)

    results = evaluate_scenarios(
        model, cfg, "pursuer", GreedyEvader, scenarios, n_variants=args.n
    )

    print(f"scenario eval: {args.model.name}, N={args.n} variants per scenario")
    print(f"{'scenario':<16} {'CR':>6} {'Wilson 95% CI':>18} {'mean steps':>10}")
    report = {}
    for name, logs in results.items():
        s = summarize(logs)
        report[name] = {"summary": s, "logs": logs}
        print(
            f"{name:<16} {s['cr']:>6.3f} "
            f"[{s['cr_ci'][0]:.3f}, {s['cr_ci'][1]:.3f}]{'':>4}"
            f"{s['mean_steps']:>10.1f}"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "step_05_scenario_eval.json", "w") as f:
        json.dump({"model": args.model.name, "n": args.n, "scenarios": report}, f)
    print(f"wrote {OUT_DIR / 'step_05_scenario_eval.json'}")


if __name__ == "__main__":
    main()
