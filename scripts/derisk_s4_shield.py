"""S4 — Shielded Games on the learned pursuer (plan v3 Step 8-alt).
Spec: docs/pathway/shielded_learned_pursuer_prereg_2026-09-14.md. Nothing here
may change a threshold, a seed, a stream, a rule or a budget after a number is
seen.

Deployment cells (one process each): POLICY:OPPONENT:ARM with
  POLICY   STR14 (D111 artefact, hold 10) | TEACHER (OrbitSearch, hold 1 — V1-identical to SearchingPursuer)
  OPPONENT GREEDY12 (primary) FOV HUG OPEN GREEDY10 GREEDY14
  ARM      NONE (NoOpShield) | SHIELD (CBFShield, delta 0) | ROBUST (CBFShield, delta 0.05)
TEACHER runs on GREEDY12 only -> 6*3 + 3 = 21 cells, N = 500 paired episodes each.
Training arms: STRSH16/17/18 = S3's structured learner with the shield INSIDE the
loop (R3-lite), 10M env steps, gated shielded on the primary stream.

Usage:
  ./venv/bin/python scripts/derisk_s4_shield.py --cell STR14:GREEDY12:NONE   (one per process)
  ./venv/bin/python scripts/derisk_s4_shield.py --arms STRSH16
  ./venv/bin/python scripts/derisk_s4_shield.py --summarize
  ./venv/bin/python scripts/derisk_s4_shield.py --gif
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib

from pe2.config import EnvConfig
from pe2.derisk import gate_pass
from pe2.eval import mcnemar_pvalue, wilson_ci, win_count
from pe2.metrics import summarize_shield
from pe2.scripted import (
    FovLimitedEvader,
    GreedyEvader,
    ObstacleHuggingEvader,
    OpenSpaceEvader,
)
from pe2.structured import HOLD

OUT = pathlib.Path("results/derisk")
STR14_CKPT = OUT / "ckpts_STR14" / "STR14_10000000.zip"

# --- pre-registered (spec sections 3-4). Do not tune. ---
N_CELL = 500
TOTAL_LONG = 10_000_000
EVERY_LONG = 1_000_000
EVAL_N = 200
PROBE_SEED0 = 1_100_000
SEEDS = (16, 17, 18)
PASS_MIN = 2
P_GATE = 0.05
ROBUST_DELTA = 0.05
GAMMA_S = 0.99
OPPONENTS = {
    "GREEDY12": {"factory": GreedyEvader, "speed": 1.2, "seed0": 1_090_000},
    "FOV": {"factory": FovLimitedEvader, "speed": 1.2, "seed0": 1_091_000},
    "HUG": {"factory": ObstacleHuggingEvader, "speed": 1.2, "seed0": 1_092_000},
    "OPEN": {"factory": OpenSpaceEvader, "speed": 1.2, "seed0": 1_093_000},
    "GREEDY10": {"factory": GreedyEvader, "speed": 1.0, "seed0": 1_094_000},
    "GREEDY14": {"factory": GreedyEvader, "speed": 1.4, "seed0": 1_095_000},
}
SHIELD_ARMS = ("NONE", "SHIELD", "ROBUST")
POLICIES = {"STR14": {"hold": HOLD}, "TEACHER": {"hold": 1}}
CELLS = [f"STR14:{o}:{a}" for o in OPPONENTS for a in SHIELD_ARMS] + [
    f"TEACHER:GREEDY12:{a}" for a in SHIELD_ARMS
]
ARMS = {f"STRSH{s}": {"seed": s} for s in SEEDS}


def cfg_for(opp: str) -> EnvConfig:
    return dataclasses.replace(EnvConfig(), evader_speed_ratio=OPPONENTS[opp]["speed"])


def make_shield(arm: str, cfg: EnvConfig):
    from pe2.shield import CBFShield, NoOpShield

    if arm == "NONE":
        return NoOpShield(cfg)
    if arm == "SHIELD":
        return CBFShield(cfg)
    if arm == "ROBUST":
        return CBFShield(cfg, margin=ROBUST_DELTA)
    raise ValueError(arm)


def _ci(k: int, n: int) -> dict:
    lo, hi = wilson_ci(k, n)
    return {"k": k, "n": n, "rate": k / n, "wilson": [lo, hi]}


def provenance() -> dict:
    import os
    import subprocess

    import stable_baselines3
    import torch

    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        sha = None
    return {
        "git_commit": sha,
        "sb3_version": stable_baselines3.__version__,
        "torch_threads": torch.get_num_threads(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
    }


def _cell_file(cell: str) -> pathlib.Path:
    return OUT / f"s4_cell_{cell.replace(':', '_')}.json"


def require_cell_not_run(cell: str) -> None:
    if _cell_file(cell).exists():
        raise SystemExit(
            f"cell {cell} has already run; the stop rule (spec section 5) forbids re-running it."
        )


def require_arm_not_run(arm: str) -> None:
    if (OUT / f"s4_arm_{arm}.json").exists() or (OUT / f"ckpts_{arm}").exists():
        raise SystemExit(
            f"arm {arm} has already run; the stop rule (spec section 5) forbids re-running it."
        )


# ----------------------------------------------------------------------------
# deployment cells
# ----------------------------------------------------------------------------
def policy_predict(policy: str, cfg: EnvConfig, device: str):
    from stable_baselines3 import PPO

    from pe2.structured import OrbitSearch

    if policy == "STR14":
        return PPO.load(STR14_CKPT, device=device).predict
    if policy == "TEACHER":
        return OrbitSearch(cfg).predict
    raise ValueError(policy)


class _P:
    def __init__(self, f):
        self.predict = f


def run_cell(cell: str, device: str) -> dict:
    from pe2.structured import evaluate_structured

    require_cell_not_run(cell)
    policy, opp, arm = cell.split(":")
    if (
        policy not in POLICIES
        or opp not in OPPONENTS
        or arm not in SHIELD_ARMS
        or cell not in CELLS
    ):
        raise SystemExit(f"unknown cell {cell}")
    cfg = cfg_for(opp)
    logs = evaluate_structured(
        _P(policy_predict(policy, cfg, device)),
        cfg,
        OPPONENTS[opp]["factory"],
        N_CELL,
        OPPONENTS[opp]["seed0"],
        hold=POLICIES[policy]["hold"],
        shield_factory=lambda: make_shield(arm, cfg),
    )
    result = {
        "cell": cell,
        "policy": policy,
        "opponent": opp,
        "arm": arm,
        "n": N_CELL,
        "seed0": OPPONENTS[opp]["seed0"],
        "speed": OPPONENTS[opp]["speed"],
        "hold": POLICIES[policy]["hold"],
        "provenance": provenance(),
        "score": _ci(*win_count(logs, "pursuer")),
        "shield": summarize_shield(logs),
        "logs": logs,
    }
    result["rate"] = result["score"]["rate"]
    _cell_file(cell).write_text(json.dumps(result, indent=2))
    sh = result["shield"]
    print(
        f"[{cell}] capture {result['rate']:.3f}; violation steps {sh['violation_steps']}; "
        f"intervention {sh['intervention_rate']:.3f}; stall {sh['stall_step_rate']:.3f}; "
        f"deadlock eps {sh['deadlock_episode_rate']:.3f}; clearance floor {sh['min_clearance_floor']}",
        flush=True,
    )
    return result


# ----------------------------------------------------------------------------
# shield-in-training arms (R3-lite)
# ----------------------------------------------------------------------------
def run_train_arm(arm: str, device: str, holder: dict | None = None) -> dict:
    from stable_baselines3 import PPO

    from pe2.shield import CBFShield
    from pe2.structured import SearchMDP, evaluate_structured

    require_arm_not_run(arm)
    seed = ARMS[arm]["seed"]
    cfg = EnvConfig()
    d = OUT / f"ckpts_{arm}"
    d.mkdir(parents=True, exist_ok=True)
    # probes are SHIELDED rollouts (spec section 3): StructuredEvalProbe builds its
    # own SearchMDP through evaluate_structured, so pass the shield factory via hold+shield
    probe = ShieldedStructuredProbe(
        cfg,
        GreedyEvader,
        every=EVERY_LONG,
        total=TOTAL_LONG,
        eval_n=EVAL_N,
        seed0=PROBE_SEED0,
        ckpt_dir=d,
        report_path=OUT / f"s4_probe_{arm}.json",
        label=arm,
    )
    if holder is not None:
        holder["rows"] = probe.rows
    env = SearchMDP(
        cfg,
        GreedyEvader,
        seed0=seed * 1000,
        hold=HOLD,
        r_acq=0.0,
        zero_reward=False,
        shield=CBFShield(cfg),
    )
    model = PPO("MlpPolicy", env, seed=seed, gamma=GAMMA_S, verbose=0, device=device)
    model.save(OUT / f"s4_init_{seed}.zip")
    model.learn(total_timesteps=TOTAL_LONG, callback=probe.callback)
    result = {
        "arm": arm,
        "seed": seed,
        "env_steps": int(env.env_steps),
        "decisions": int(model.num_timesteps),
        "train_violations": int(env.violations_total),
        "probe_violations": int(sum(r.get("violations", 0) for r in probe.rows)),
        "checkpoints": probe.rows,
        "provenance": provenance(),
    }
    ckpt = d / f"{arm}_{TOTAL_LONG}.zip"
    if not any(r["mark"] == TOTAL_LONG for r in probe.rows) or not ckpt.exists():
        result["verdict"], result["reason"] = (
            "UNDETERMINED",
            f"final probe at mark {TOTAL_LONG} never fired (env steps {env.env_steps})",
        )
        return result
    # gate: shielded, on the primary stream, vs SHIELD STR14 and vs NONE TEACHER (their paired logs)
    refs = {}
    for name, cell in (
        ("STR14_SHIELD", "STR14:GREEDY12:SHIELD"),
        ("TEACHER_NONE", "TEACHER:GREEDY12:NONE"),
    ):
        f = _cell_file(cell)
        if not f.exists():
            raise SystemExit(
                f"{arm}: reference cell {cell} has not run; run the deployment cells first."
            )
        refs[name] = json.loads(f.read_text())["logs"]
    gcfg = cfg_for("GREEDY12")
    logs = evaluate_structured(
        PPO.load(ckpt, device=device),
        gcfg,
        GreedyEvader,
        N_CELL,
        OPPONENTS["GREEDY12"]["seed0"],
        hold=HOLD,
        shield_factory=lambda: CBFShield(gcfg),
    )
    result["final"] = {
        "ckpt": str(ckpt),
        "score": _ci(*win_count(logs, "pursuer")),
        "violations": int(sum(lg["violations"] for lg in logs)),
        "shield": summarize_shield(logs),
        "gate": {
            name: gate_pass(logs, ref, "pursuer", P_GATE) for name, ref in refs.items()
        },
    }
    g = result["final"]["gate"]["STR14_SHIELD"]
    result["verdict"] = f"final {'PASS' if g['pass'] else 'FAIL'} vs STR14_SHIELD"
    result["reason"] = (
        f"final {g['k']}/{g['n']} = {g['rate']:.3f}, McNemar p={g['p']:.2e} (b={g['b']}, c={g['c']}) "
        f"vs shielded STR14 {g['ref_rate']:.3f}; vs unshielded teacher "
        f"{'PASS' if result['final']['gate']['TEACHER_NONE']['pass'] else 'fail'}; "
        f"training violations {result['train_violations']}, probe violations "
        f"{result['probe_violations']}, gate violations {result['final']['violations']}"
    )
    return result


class ShieldedStructuredProbe:
    """StructuredEvalProbe whose evaluations run SHIELDED (spec section 3) and whose
    rows carry the probe's violation count. Same env-step grid, same stop rule."""

    def __init__(
        self,
        cfg,
        opponent_factory,
        every,
        total,
        eval_n,
        seed0,
        ckpt_dir,
        report_path,
        label,
        hold=HOLD,
    ):
        from stable_baselines3.common.callbacks import BaseCallback

        from pe2.shield import CBFShield
        from pe2.structured import evaluate_structured

        if total % every != 0:
            raise ValueError(f"total ({total}) must be a multiple of every ({every})")
        rows: list[dict] = []

        def evaluate_at(model, env_steps: int, mark: int) -> dict:
            path = ckpt_dir / f"{label}_{mark}.zip"
            model.save(path)
            logs = evaluate_structured(
                model,
                cfg,
                opponent_factory,
                eval_n,
                seed0,
                hold=hold,
                shield_factory=lambda: CBFShield(cfg),
            )
            k, n = win_count(logs, "pursuer")
            row = {
                "steps": env_steps,
                "mark": mark,
                "ckpt": str(path),
                "win_rate": k / n,
                "k": k,
                "n": n,
                "violations": int(sum(lg["violations"] for lg in logs)),
                "intervention_rate": summarize_shield(logs)["intervention_rate"],
            }
            rows.append(row)
            with open(report_path, "w") as f:
                json.dump({"checkpoints": rows}, f, indent=2)
            print(
                f"  @{env_steps:>9} env steps (mark {mark}): shielded win rate "
                f"{k}/{n} = {k / n:.3f}, violations {row['violations']}",
                flush=True,
            )
            return row

        class _CB(BaseCallback):
            def __init__(inner_self) -> None:
                super().__init__()
                inner_self.next_mark = every

            def _on_step(inner_self) -> bool:
                counts = inner_self.training_env.get_attr("env_steps")
                assert len(counts) == 1
                env_steps = int(counts[0])
                while (
                    inner_self.next_mark <= total and env_steps >= inner_self.next_mark
                ):
                    evaluate_at(inner_self.model, env_steps, inner_self.next_mark)
                    inner_self.next_mark += every
                return env_steps < total

        self.rows = rows
        self.callback = _CB()
        self.evaluate_at = evaluate_at


def run_arm(arm: str, device: str) -> dict:
    holder: dict = {}
    try:
        result = run_train_arm(arm, device, holder)
    except Exception as e:
        (OUT / f"s4_arm_{arm}.json").write_text(
            json.dumps(
                {
                    "arm": arm,
                    "verdict": "UNDETERMINED",
                    "reason": f"crashed: {type(e).__name__}: {e}",
                    "checkpoints": holder.get("rows", []),
                    "provenance": provenance(),
                },
                indent=2,
            )
        )
        raise
    (OUT / f"s4_arm_{arm}.json").write_text(json.dumps(result, indent=2))
    print(f"[{arm}] {result['verdict']}: {result['reason']}", flush=True)
    return result


# ----------------------------------------------------------------------------
# verdicts (spec section 4)
# ----------------------------------------------------------------------------
def paired_shift(logs_shielded: list[dict], logs_unshielded: list[dict]) -> dict:
    """Delta = rate(shielded) - rate(unshielded) with a paired CI from the discordant
    counts (b = shielded wins where unshielded lost, c = the reverse): the difference
    of paired proportions is (b - c)/n with variance (b + c - (b - c)^2/n)/n^2."""
    import math

    p, b, c = mcnemar_pvalue(logs_shielded, logs_unshielded, "pursuer")
    ks, n = win_count(logs_shielded, "pursuer")
    ku, _ = win_count(logs_unshielded, "pursuer")
    delta = (ks - ku) / n
    var = (b + c - (b - c) ** 2 / n) / n**2
    half = 1.96 * math.sqrt(max(var, 0.0))
    return {
        "delta": delta,
        "ci": [delta - half, delta + half],
        "p": p,
        "b": b,
        "c": c,
        "rate_shielded": ks / n,
        "rate_unshielded": ku / n,
        "n": n,
    }


def h1_verdict(
    cells: dict, policy: str = "STR14", opp: str = "GREEDY12", arm: str = "SHIELD"
) -> dict:
    s, u = cells.get(f"{policy}:{opp}:{arm}"), cells.get(f"{policy}:{opp}:NONE")
    if s is None or u is None:
        return {"verdict": "INCOMPLETE"}
    sh = paired_shift(s["logs"], u["logs"])
    _lo_s, hi_s = wilson_ci(*win_count(s["logs"], "pursuer"))
    if sh["p"] < P_GATE and sh["c"] > sh["b"] and hi_s < sh["rate_unshielded"]:
        v = "CONFIRMED"
    elif sh["p"] < P_GATE and sh["b"] > sh["c"]:
        v = "REVERSED"
    else:
        v = "NO SHIFT"
    return {"verdict": v, "cell": f"{policy}:{opp}:{arm}"} | sh


def hsafe_verdict(cells: dict, arms: dict) -> dict:
    shielded = {k: v for k, v in cells.items() if v["arm"] in ("SHIELD", "ROBUST")}
    steps = sum(v["shield"]["violation_steps"] for v in shielded.values())
    offenders = [k for k, v in shielded.items() if v["shield"]["violation_steps"] > 0]
    for a, r in arms.items():
        tv = (
            int(r.get("train_violations", 0))
            + int(r.get("probe_violations", 0))
            + int((r.get("final") or {}).get("violations", 0))
        )
        steps += tv
        if tv:
            offenders.append(a)
    none = cells.get("STR14:GREEDY12:NONE")
    none_rate = None
    if none is not None:
        n = none["n"] if "n" in none else len(none["logs"])
        eps = none["shield"].get("episodes_with_violation")
        none_rate = (eps / n) if eps is not None else None
    return {
        "verdict": "HOLDS" if steps == 0 else "FAILS",
        "shielded_violation_steps": steps,
        "offenders": offenders,
        "none_greedy12_violation_episode_rate": none_rate,
        "none_probe_too_benign": (none_rate is not None and none_rate < 0.05),
    }


def h3_verdict(arms: dict) -> dict:
    per = {}
    for a in ARMS:
        r = arms.get(a)
        per[a] = (
            None
            if r is None or "final" not in r
            else bool(r["final"]["gate"]["STR14_SHIELD"]["pass"])
        )
    if any(v is None for v in per.values()):
        return {"verdict": "INCOMPLETE", "per_seed": per}
    n = sum(per.values())
    return {
        "verdict": "RECOVERED" if n >= PASS_MIN else "NOT RECOVERED",
        "passes": n,
        "per_seed": per,
    }


def direction(h: dict) -> str:
    if h["hsafe"] == "FAILS":
        return (
            "H-SAFE FAILS: a shielded arm recorded a contact step. Stop; the census is written; no further round "
            "without a user ruling (Step 6 stop rule)."
        )
    if h["h1"] == "INCOMPLETE":
        return (
            "incomplete: the primary cells STR14:GREEDY12:{NONE,SHIELD} must both run"
        )
    if h["h1"] == "CONFIRMED":
        d = (
            "H1 CONFIRMED at zero violations: R1 is measured on a learned agent with the sign the exact twin predicted — "
            "the paper's central result. Magnitude and costs go to the results section; next: Step 9 (shield-attacker) on STR14-SHIELD."
        )
    elif h["h1"] == "NO SHIFT":
        d = (
            "H1 NO SHIFT: the shield is free for the learned pursuer at this margin, contradicting Step 6's scripted shift and "
            "T3's prediction on a learned policy; report with the cost table."
        )
    else:
        d = (
            "H1 REVERSED: the shield helps the learned pursuer — a refutation of T3's direction on this policy class; "
            "report with the trajectories that explain it (visual gate)."
        )
    if h["h3"] == "RECOVERED":
        d += " H3-lite RECOVERED: shield-in-training closes the deployment gap; STRSH's best seed is the shielded artefact for Step 9."
    elif h["h3"] == "NOT RECOVERED":
        d += " H3-lite NOT RECOVERED: Step 10 needs the feasibility-constraint / margin-shaping components."
    return d


def load_cells() -> dict:
    out = {}
    for p in sorted(OUT.glob("s4_cell_*.json")):
        r = json.loads(p.read_text())
        out[r["cell"]] = r
    return out


def load_arms() -> dict:
    return {
        p.stem.removeprefix("s4_arm_"): json.loads(p.read_text())
        for p in sorted(OUT.glob("s4_arm_*.json"))
    }


def summarize(cells: dict, arms: dict) -> dict:
    h1 = h1_verdict(cells)
    fam = {}
    for opp in OPPONENTS:
        for arm in ("SHIELD", "ROBUST"):
            r = h1_verdict(cells, "STR14", opp, arm)
            fam[f"{opp}:{arm}"] = {
                k: r[k]
                for k in (
                    "verdict",
                    "delta",
                    "ci",
                    "p",
                    "b",
                    "c",
                    "rate_shielded",
                    "rate_unshielded",
                )
                if k in r
            }
    teacher = {
        arm: h1_verdict(cells, "TEACHER", "GREEDY12", arm)
        for arm in ("SHIELD", "ROBUST")
    }
    cost_keys = (
        "intervention_rate",
        "infeasibility_rate",
        "backup_rate",
        "stall_step_rate",
        "deadlock_episode_rate",
        "min_clearance_floor",
    )
    cost = {
        k: {ck: v["shield"].get(ck) for ck in cost_keys} | {"rate": v["rate"]}
        for k, v in cells.items()
    }
    hs = hsafe_verdict(cells, arms)
    h3 = h3_verdict(arms)
    h = {"hsafe": hs["verdict"], "h1": h1["verdict"], "h3": h3["verdict"]}
    return {
        "spec": "docs/pathway/shielded_learned_pursuer_prereg_2026-09-14.md",
        "hsafe": hs,
        "h1": h1,
        "h1_family": fam,
        "teacher_shift": teacher,
        "h3": h3,
        "cost": cost,
        "arms": {
            a: {"verdict": r.get("verdict"), "reason": r.get("reason")}
            for a, r in arms.items()
        },
        "complete": all(c in cells for c in CELLS) and all(a in arms for a in ARMS),
        "direction": direction(h),
    }


def write_figure(cells: dict, arms: dict, summary: dict, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    ax = axes[0]
    opps = list(OPPONENTS)
    for j, (arm, c) in enumerate((("SHIELD", "tab:blue"), ("ROBUST", "tab:orange"))):
        xs, ys, err = [], [], []
        for i, opp in enumerate(opps):
            r = summary["h1_family"].get(f"{opp}:{arm}", {})
            if "delta" in r:
                xs.append(i + (j - 0.5) * 0.25)
                ys.append(r["delta"])
                err.append(r["delta"] - r["ci"][0])
        if xs:
            ax.errorbar(
                xs,
                ys,
                yerr=err,
                fmt="o",
                color=c,
                capsize=3,
                label=f"STR14 {arm} − NONE",
            )
    t = summary["teacher_shift"].get("SHIELD", {})
    if "delta" in t:
        ax.errorbar(
            [-0.6],
            [t["delta"]],
            yerr=[t["delta"] - t["ci"][0]],
            fmt="s",
            color="k",
            capsize=3,
            label="TEACHER SHIELD − NONE (GREEDY12)",
        )
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xticks(range(len(opps)))
    ax.set_xticklabels(opps, fontsize=8)
    ax.set_ylabel("Δ capture rate (paired, 95 % CI)")
    ax.set_title(f"S4 — shift (H1: {summary['h1']['verdict']})")
    ax.legend(fontsize=7)
    ax = axes[1]
    keys = ("intervention_rate", "stall_step_rate", "deadlock_episode_rate")
    for j, cell in enumerate(
        (
            "STR14:GREEDY12:SHIELD",
            "TEACHER:GREEDY12:SHIELD",
            "STR14:GREEDY12:ROBUST",
            "TEACHER:GREEDY12:ROBUST",
        )
    ):
        c = summary["cost"].get(cell)
        if c:
            ax.bar(
                [i + j * 0.2 for i in range(len(keys))],
                [c[k] or 0 for k in keys],
                width=0.2,
                label=cell,
            )
    ax.set_xticks([i + 0.3 for i in range(len(keys))])
    ax.set_xticklabels(keys, fontsize=8)
    ax.set_title("cost (reported)")
    ax.legend(fontsize=6)
    ax = axes[2]
    for a, r in arms.items():
        rows = r.get("checkpoints") or []
        if rows:
            ax.plot(
                [x["steps"] for x in rows],
                [x["win_rate"] for x in rows],
                marker="o",
                label=f"{a} ({r.get('verdict')})",
            )
    for cell, c in (
        ("STR14:GREEDY12:SHIELD", "tab:blue"),
        ("STR14:GREEDY12:NONE", "tab:green"),
        ("TEACHER:GREEDY12:NONE", "k"),
    ):
        if cell in cells:
            ax.axhline(
                cells[cell]["rate"],
                ls=":",
                color=c,
                lw=1.2,
                label=f"{cell} = {cells[cell]['rate']:.3f}",
            )
    ax.set_xlabel("env steps")
    ax.set_title(f"shield-in-training (H3-lite: {summary['h3']['verdict']})")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=6)
    fig.suptitle(
        "S4 — Shielded Games on the learned pursuer"
        + ("" if summary["complete"] else " — INCOMPLETE")
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_gif(path, device: str, every: int = 4) -> None:
    """Visual gate (spec section 6): STR14 unshielded vs shielded on the same
    GREEDY12 episodes — one where the shield changes the outcome, one where it
    does not — with the FOV cone and the ShieldOverlay (nominal vs executed)."""
    import matplotlib.pyplot as plt
    import numpy as np
    from stable_baselines3 import PPO

    from pe2.render import FovOverlay, Renderer, ShieldOverlay, Trails, save_gif
    from pe2.shield import CBFShield, NoOpShield
    from pe2.structured import SearchMDP

    cfg = cfg_for("GREEDY12")
    predict = PPO.load(STR14_CKPT, device=device).predict

    class Rec(SearchMDP):
        def __init__(self, *a, label, **k):
            super().__init__(*a, **k)
            self.label = label
            self.frames = []
            self.r = Renderer(cfg, dpi=80, figsize=(5, 5))
            self.t = Trails()
            self.live = {"nominal": None, "executed": None, "info": None}
            self.ov = [FovOverlay("pursuer")]
            # ShieldOverlay reads shield.margin / shield._x_lim and draws only
            # once live["info"] is a filter info dict; it no-ops before the
            # first step and there is no overlay at all with no shield.
            if self.shield is not None:
                self.ov.append(ShieldOverlay("pursuer", self.shield, self.live))

        def _step_inner(self, action, branch="law"):
            env = self.inner.env
            self.t.append("pursuer", *env.pursuer_pose[:2])
            self.t.append("evader", *env.evader_pose[:2])
            a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
            nominal = ((a[0] + 1) / 2 * cfg.v_max, a[1] * cfg.omega_max)
            out = super()._step_inner(action, branch)
            v, w = self.trace[-1][2], self.trace[-1][3]
            self.live.update(
                nominal=nominal,
                executed=((v + 1) / 2 * cfg.v_max, w * cfg.omega_max),
                info=self._last_sinfo,
            )
            if env.current_step % every == 0:
                self.frames.append(
                    self.r.draw(
                        env,
                        trails=self.t,
                        overlays=self.ov,
                        title_prefix=f"{self.label} [{branch}]",
                    )
                )
            return out

        def finish(self, outcome, step):
            for _ in range(10):
                self.frames.append(
                    self.r.draw(
                        self.inner.env,
                        trails=self.t,
                        overlays=self.ov,
                        title_prefix=f"{self.label}  {outcome.upper()} @ {step}",
                    )
                )
            plt.close(self.r.fig)

    def play(seed, shield, label):
        m = Rec(cfg, GreedyEvader, hold=HOLD, shield=shield, label=label)
        m.trace = []
        log = m.play_episode(predict, seed)
        m.finish(log["outcome"], log["steps"])
        return m.frames, log

    def find(pred):
        for s in range(
            OPPONENTS["GREEDY12"]["seed0"], OPPONENTS["GREEDY12"]["seed0"] + 80
        ):
            u = SearchMDP(
                cfg, GreedyEvader, hold=HOLD, shield=NoOpShield(cfg)
            ).play_episode(predict, s)
            sh = SearchMDP(
                cfg, GreedyEvader, hold=HOLD, shield=CBFShield(cfg)
            ).play_episode(predict, s)
            if pred(u, sh):
                return s
        return None

    changed = find(lambda u, sh: u["outcome"] != sh["outcome"])
    same = find(lambda u, sh: u["outcome"] == sh["outcome"] == "capture")
    frames = []
    for s in (changed, same):
        if s is None:
            continue
        fu, _ = play(s, NoOpShield(cfg), "STR14 unshielded")
        fs, _ = play(s, CBFShield(cfg), "STR14 shielded (δ=0)")
        n = max(len(fu), len(fs))
        fu += [fu[-1]] * (n - len(fu))
        fs += [fs[-1]] * (n - len(fs))
        frames += [np.concatenate([a, b], axis=1) for a, b in zip(fu, fs, strict=True)]
    save_gif(frames, str(path), fps=15)
    print(
        f"wrote {path} ({len(frames)} frames; seeds changed={changed}, same={same})",
        flush=True,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cell", nargs="*", default=[])
    p.add_argument("--arms", nargs="*", choices=list(ARMS), default=[])
    p.add_argument("--summarize", action="store_true")
    p.add_argument("--gif", action="store_true")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for cell in args.cell:
        run_cell(cell, args.device)
    for arm in args.arms:
        run_arm(arm, args.device)
    if args.summarize:
        cells, arms = load_cells(), load_arms()
        summary = summarize(cells, arms)
        (OUT / "s4_round.json").write_text(
            json.dumps({"summary": summary, "arms": arms}, indent=2)
        )
        write_figure(cells, arms, summary, OUT / "s4_round.png")
        print(
            json.dumps(
                {k: summary[k] for k in ("hsafe", "h1", "h3")}, indent=2, default=str
            )
        )
        print(json.dumps(summary["h1_family"], indent=2))
        print(f"\n{summary['direction']}")
    if args.gif:
        write_gif(OUT / "step_08_shield.gif", args.device)


if __name__ == "__main__":
    main()
