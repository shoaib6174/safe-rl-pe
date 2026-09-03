"""Step 5 episode grid: 3x3 storyboard of the GATED BR model vs frozen Ge.

Renders the first 9 paired eval seeds (720000..720008) played by the exact
SAC model that produced the gate numbers (default `sac_br_pursuer.zip`;
pass --model/--json for another gated model, e.g. the BC clone), each cell
a full-trajectory scene at the terminal step. Every rendered episode is
asserted equal to its row in the gate JSON's trained_br logs (outcome,
steps) — same cross-check discipline as the Step 3 gate.

Deterministic: model.predict(deterministic=True) + seeded env, so the PNG
is bit-stable across re-renders (md5 asserted).

Checklist for sign-off (hard rule 9):
1. pursuer trails visibly HOME IN on the evader (converging trails), not
   the wandering/scanning of the random-init policy
2. capture cells show trail contact at episode end
3. outcomes/steps match the gate JSON (asserted)
4. md5 stable across re-render (asserted, printed)

Reads  results/visual_gates/step_05/sac_br_pursuer.zip (or --model)
       results/visual_gates/step_05/step_05_br_sanity.json (or --json)
Writes results/visual_gates/step_05/step_05_episodes.png

Usage: ./venv/bin/python scripts/pe2_step05_episode_grid.py [--model PATH] [--json PATH]
"""

import argparse
import hashlib
import json
import pathlib

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from pe2.config import EnvConfig
from pe2.exploitability import SingleAgentWrapper
from pe2.render import Renderer, Trails
from pe2.scripted import GreedyEvader

OUT_DIR = pathlib.Path("results/visual_gates/step_05")
OUT_PNG = OUT_DIR / "step_05_episodes.png"
GRID_SEEDS = list(range(720000, 720009))  # first 9 paired eval seeds (D57)


def md5(path: pathlib.Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def render_episode(model, cfg, seed, renderer):
    """Play one deterministic episode; return (frame, outcome, steps)."""
    wrapper = SingleAgentWrapper(cfg, "pursuer", GreedyEvader)
    obs, _ = wrapper.reset(seed=seed)
    trails = Trails()
    trails.append("pursuer", *wrapper.env.pursuer_pose[:2])
    trails.append("evader", *wrapper.env.evader_pose[:2])
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = wrapper.step(action)
        trails.append("pursuer", *wrapper.env.pursuer_pose[:2])
        trails.append("evader", *wrapper.env.evader_pose[:2])
        if terminated or truncated:
            break
    frame = renderer.draw(wrapper.env, trails)
    return frame, info["outcome"], wrapper.env.current_step


def build_grid(model_path: pathlib.Path, json_path: pathlib.Path) -> None:
    from stable_baselines3 import SAC  # lazy: optional dependency

    cfg = EnvConfig()
    model = SAC.load(model_path)
    with open(json_path) as f:
        report = json.load(f)
    # gate JSONs nest trained_br either at top level or under "gate"
    trained = report.get("trained_br") or report["gate"]["trained_br"]
    truth = {log["seed"]: log for log in trained["logs"]}

    renderer = Renderer(cfg, dpi=50, figsize=(3.0, 3.0))
    cells = []
    for seed in GRID_SEEDS:
        frame, outcome, steps = render_episode(model, cfg, seed, renderer)
        log = truth[seed]  # KeyError -> grid seed not in the gated eval range
        assert (outcome, steps) == (log["outcome"], log["steps"]), (
            f"seed {seed}: rendered ({outcome}, {steps}) != gate log "
            f"({log['outcome']}, {log['steps']})"
        )
        cells.append((seed, frame, outcome, steps))
    renderer.close()

    fig, axes = plt.subplots(3, 3, figsize=(9, 9), dpi=100)
    for ax, (seed, frame, outcome, steps) in zip(axes.flat, cells, strict=True):
        ax.imshow(frame)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"seed {seed} — {outcome} @ {steps}", fontsize=9)
    fig.suptitle(f"Step 5: {model_path.stem} vs frozen GreedyEvader")
    fig.tight_layout()
    fig.savefig(OUT_PNG)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=pathlib.Path, default=OUT_DIR / "sac_br_pursuer.zip"
    )
    parser.add_argument(
        "--json", type=pathlib.Path, default=OUT_DIR / "step_05_br_sanity.json"
    )
    args = parser.parse_args()

    build_grid(args.model, args.json)
    first = md5(OUT_PNG)
    build_grid(args.model, args.json)  # re-render over it
    second = md5(OUT_PNG)
    assert first == second, f"episode grid not deterministic: {first} != {second}"
    print(f"determinism: md5 {first} stable across re-render OK")
    print(f"cross-check: all 9 rendered episodes match {args.json.name} OK")
    print(f"wrote {OUT_PNG}")
    print("\nchecklist for sign-off:")
    print("1. pursuer trails home in on the evader (converging, not wandering)")
    print("2. capture cells show trail contact at episode end")
    print("3. outcomes/steps match the sanity JSON (asserted above)")
    print("4. md5 stable across re-render (asserted above)")


if __name__ == "__main__":
    main()
