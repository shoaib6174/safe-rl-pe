"""S5 — Step 10 remedy: a shield-aware structured learner.
Spec: docs/pathway/shield_remedy_prereg_2026-09-15.md. Nothing here may change a
threshold, a seed, a stream, a rule or a budget after a number is seen.

Arms (S4's STRSH configuration + the ingredient; 10M env steps, shield in the loop):
  RA{19,20,21}  shield_obs                    RB{19,20,21}  stall penalty 1e-3
  RC{19,20,21}  both                          RZ19          shield_obs, all rewards zeroed (control)
Gate: final checkpoint, SHIELDED, on S4's primary stream 1 090 000+, paired with the S4
cells (TEACHER_SHIELD = the verdict bar, STR14_SHIELD, TEACHER_NONE) and the three S4
STRSH finals re-scored on that stream.

Usage:
  ./venv/bin/python scripts/derisk_s5_remedy.py --measure-start
  ./venv/bin/python scripts/derisk_s5_remedy.py --arms RA19        (one per process)
  ./venv/bin/python scripts/derisk_s5_remedy.py --summarize
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib

from pe2.config import EnvConfig
from pe2.derisk import gate_pass
from pe2.eval import wilson_ci, win_count
from pe2.metrics import summarize_shield
from pe2.scripted import GreedyEvader
from pe2.structured import HOLD

OUT = pathlib.Path("results/derisk")
START_PATH = OUT / "s5_start.json"
S4_CELL = {
    "TEACHER_SHIELD": OUT / "s4_cell_TEACHER_GREEDY12_SHIELD.json",
    "STR14_SHIELD": OUT / "s4_cell_STR14_GREEDY12_SHIELD.json",
    "TEACHER_NONE": OUT / "s4_cell_TEACHER_GREEDY12_NONE.json",
}
STRSH_CKPTS = {
    f"STRSH{s}": OUT / f"ckpts_STRSH{s}" / f"STRSH{s}_10000000.zip"
    for s in (16, 17, 18)
}
STRSH_EXPECTED = {
    "STRSH16": 210,
    "STRSH17": 216,
    "STRSH18": 208,
}  # S4 finals, k of 500 — reproduction check

# --- pre-registered (spec sections 2-5). Do not tune. ---
STALL_W = 1e-3
TOTAL_LONG = 10_000_000
EVERY_LONG = 1_000_000
EVAL_N = 200
GATE_N = 500
PROBE_SEED0 = 1_110_000
GATE_SEED0 = 1_090_000
SEEDS = (19, 20, 21)
PASS_MIN = 2
P_GATE = 0.05
GAMMA_S = 0.99
FAMILIES = {
    "RA": {"shield_obs": True, "stall_penalty": 0.0, "zero_reward": False},
    "RB": {"shield_obs": False, "stall_penalty": STALL_W, "zero_reward": False},
    "RC": {"shield_obs": True, "stall_penalty": STALL_W, "zero_reward": False},
}
ARMS: dict = {
    f"{f}{s}": {"family": f, "seed": s, **cfg}
    for f, cfg in FAMILIES.items()
    for s in SEEDS
}
ARMS["RZ19"] = {
    "family": "RZ",
    "seed": 19,
    "shield_obs": True,
    "stall_penalty": 0.0,
    "zero_reward": True,
}
ARTEFACT_ORDER = ("RC", "RA", "RB")
REFS = (
    "TEACHER_SHIELD",
    "STR14_SHIELD",
    "TEACHER_NONE",
    "STRSH16",
    "STRSH17",
    "STRSH18",
)


def _ci(k: int, n: int) -> dict:
    lo, hi = wilson_ci(k, n)
    return {"k": k, "n": n, "rate": k / n, "wilson": [lo, hi]}


def _write_json(path: pathlib.Path, obj) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def provenance() -> dict:
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


# ----------------------------------------------------------------------------
# guards
# ----------------------------------------------------------------------------
def require_not_measured() -> None:
    if START_PATH.exists():
        raise SystemExit(
            f"{START_PATH} already exists: references already measured (stop rule, spec section 6)."
        )
    if any(OUT.glob("s5_arm_*.json")) or any(
        (OUT / f"ckpts_{a}").exists() for a in ARMS
    ):
        raise SystemExit(
            "S5 arms have already run; the stop rule forbids re-measuring."
        )


def require_arm_not_run(arm: str) -> None:
    if (OUT / f"s5_arm_{arm}.json").exists() or (OUT / f"ckpts_{arm}").exists():
        raise SystemExit(
            f"arm {arm} has already run; the stop rule (spec section 6) forbids re-running it."
        )


def load_start() -> dict:
    if not START_PATH.exists():
        raise SystemExit(
            f"references missing: {START_PATH}. Run --measure-start FIRST."
        )
    s = json.loads(START_PATH.read_text())
    for k in REFS:
        if k not in s or "logs" not in s[k]:
            raise SystemExit(
                f"{k} (with paired logs) missing from {START_PATH}; re-run --measure-start."
            )
    return s


# ----------------------------------------------------------------------------
# references (spec section 3), BEFORE any arm
# ----------------------------------------------------------------------------
def measure_start(device: str) -> dict:
    from stable_baselines3 import PPO

    from pe2.shield import CBFShield
    from pe2.structured import evaluate_structured

    require_not_measured()
    cfg = EnvConfig()
    out: dict = {"gate_seed0": GATE_SEED0, "gate_n": GATE_N, "provenance": provenance()}
    for name, path in S4_CELL.items():
        if not path.exists():
            raise SystemExit(f"S4 cell missing: {path}")
        cell = json.loads(path.read_text())
        if cell.get("seed0") != GATE_SEED0 or len(cell["logs"]) != GATE_N:
            raise SystemExit(
                f"{name}: S4 cell is not the {GATE_N}-episode stream {GATE_SEED0}"
            )
        out[name] = _ci(*win_count(cell["logs"], "pursuer")) | {
            "logs": cell["logs"],
            "source": str(path),
        }
        print(
            f"{name} = {out[name]['k']}/{GATE_N} = {out[name]['rate']:.4f} (S4 cell)",
            flush=True,
        )
    for name, ckpt in STRSH_CKPTS.items():
        if not ckpt.exists():
            raise SystemExit(f"STRSH checkpoint missing: {ckpt}")
        logs = evaluate_structured(
            PPO.load(ckpt, device=device),
            cfg,
            GreedyEvader,
            GATE_N,
            GATE_SEED0,
            hold=HOLD,
            shield_factory=lambda: CBFShield(cfg),
        )
        k, n = win_count(logs, "pursuer")
        out[name] = _ci(k, n) | {
            "logs": logs,
            "ckpt": str(ckpt),
            "expected_k": STRSH_EXPECTED[name],
            "reproduced": k == STRSH_EXPECTED[name],
        }
        print(
            f"{name} = {k}/{n} = {k / n:.4f} (S4 recorded {STRSH_EXPECTED[name]}/500: "
            f"{'reproduced' if k == STRSH_EXPECTED[name] else 'NOT reproduced'})",
            flush=True,
        )
    _write_json(START_PATH, out)
    return out


# ----------------------------------------------------------------------------
# arms
# ----------------------------------------------------------------------------
class ShieldedStructuredProbe:
    """StructuredEvalProbe whose evaluations run SHIELDED with the arm's
    shield_obs; rows carry the probe's violation count. Env-step grid; stops learn."""

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
        shield_obs,
        hold=HOLD,
    ):
        from stable_baselines3.common.callbacks import BaseCallback

        from pe2.shield import CBFShield
        from pe2.structured import evaluate_structured

        if total % every != 0:
            raise ValueError(f"total ({total}) must be a multiple of every ({every})")
        rows: list[dict] = []

        def evaluate_at(
            model, env_steps: int, mark: int, train_violations: int
        ) -> dict:
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
                shield_obs=shield_obs,
            )
            k, n = win_count(logs, "pursuer")
            sh = summarize_shield(logs)
            row = {
                "steps": env_steps,
                "mark": mark,
                "ckpt": str(path),
                "win_rate": k / n,
                "k": k,
                "n": n,
                "violations": int(sum(lg["violations"] for lg in logs)),
                "train_violations": int(train_violations),
                "intervention_rate": sh["intervention_rate"],
                "stall_step_rate": sh["stall_step_rate"],
                "deadlock_episode_rate": sh["deadlock_episode_rate"],
            }
            rows.append(row)
            _write_json(report_path, {"checkpoints": rows})
            print(
                f"  @{env_steps:>9} env steps (mark {mark}): shielded win rate {k}/{n} = {k / n:.3f}, "
                f"violations {row['violations']} (train {row['train_violations']}), stall {row['stall_step_rate']:.3f}, "
                f"deadlock eps {row['deadlock_episode_rate']:.3f}",
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
                viol = int(inner_self.training_env.get_attr("violations_total")[0])
                while (
                    inner_self.next_mark <= total and env_steps >= inner_self.next_mark
                ):
                    evaluate_at(inner_self.model, env_steps, inner_self.next_mark, viol)
                    inner_self.next_mark += every
                return env_steps < total

        self.rows = rows
        self.callback = _CB()


def run_train_arm(
    arm: str, starts: dict, device: str, holder: dict | None = None
) -> dict:
    from stable_baselines3 import PPO

    from pe2.shield import CBFShield
    from pe2.structured import SearchMDP, evaluate_structured

    spec = ARMS[arm]
    cfg = EnvConfig()
    d = OUT / f"ckpts_{arm}"
    d.mkdir(parents=True, exist_ok=True)
    probe = ShieldedStructuredProbe(
        cfg,
        GreedyEvader,
        every=EVERY_LONG,
        total=TOTAL_LONG,
        eval_n=EVAL_N,
        seed0=PROBE_SEED0,
        ckpt_dir=d,
        report_path=OUT / f"s5_probe_{arm}.json",
        label=arm,
        shield_obs=spec["shield_obs"],
    )
    if holder is not None:
        holder["rows"] = probe.rows
    env = SearchMDP(
        cfg,
        GreedyEvader,
        seed0=spec["seed"] * 1000,
        hold=HOLD,
        r_acq=0.0,
        zero_reward=spec["zero_reward"],
        shield=CBFShield(cfg),
        shield_obs=spec["shield_obs"],
        stall_penalty=spec["stall_penalty"],
    )
    if holder is not None:
        holder["env"] = env
    model = PPO(
        "MlpPolicy", env, seed=spec["seed"], gamma=GAMMA_S, verbose=0, device=device
    )
    model.save(OUT / f"s5_init_{arm}.zip")
    model.learn(total_timesteps=TOTAL_LONG, callback=probe.callback)
    result = {
        "arm": arm,
        "family": spec["family"],
        "seed": spec["seed"],
        "settings": spec,
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
    logs = evaluate_structured(
        PPO.load(ckpt, device=device),
        cfg,
        GreedyEvader,
        GATE_N,
        GATE_SEED0,
        hold=HOLD,
        shield_factory=lambda: CBFShield(cfg),
        shield_obs=spec["shield_obs"],
    )
    result["final"] = {
        "ckpt": str(ckpt),
        "score": _ci(*win_count(logs, "pursuer")),
        "violations": int(sum(lg["violations"] for lg in logs)),
        "shield": summarize_shield(logs),
        "gate": {
            ref: gate_pass(logs, starts[ref]["logs"], "pursuer", P_GATE) for ref in REFS
        },
    }
    g = result["final"]["gate"]["TEACHER_SHIELD"]
    result["verdict"] = f"final {'PASS' if g['pass'] else 'FAIL'} vs TEACHER_SHIELD"
    result["reason"] = (
        f"final {g['k']}/{g['n']} = {g['rate']:.3f}, McNemar p={g['p']:.2e} (b={g['b']}, c={g['c']}) vs shielded teacher "
        f"{g['ref_rate']:.3f}; "
        + "; ".join(
            f"vs {r} {'PASS' if result['final']['gate'][r]['pass'] else 'fail'}"
            for r in REFS[1:]
        )
        + f"; violations train {result['train_violations']} probe {result['probe_violations']} gate {result['final']['violations']}"
    )
    return result


def run_arm(arm: str, device: str) -> dict:
    require_arm_not_run(arm)  # refuse BEFORE reading or writing anything
    starts = load_start()
    holder: dict = {}
    try:
        result = run_train_arm(arm, starts, device, holder)
    except BaseException as e:
        _write_json(
            OUT / f"s5_arm_{arm}.json",
            {
                "arm": arm,
                "family": ARMS[arm]["family"],
                "seed": ARMS[arm]["seed"],
                "verdict": "UNDETERMINED",
                "reason": f"crashed: {type(e).__name__}: {e}",
                "checkpoints": holder.get("rows", []),
                "train_violations": int(
                    getattr(holder.get("env"), "violations_total", 0)
                ),
                "probe_violations": int(
                    sum(r.get("violations", 0) for r in holder.get("rows", []))
                ),
                "provenance": provenance(),
            },
        )
        raise
    _write_json(OUT / f"s5_arm_{arm}.json", result)
    print(f"[{arm}] {result['verdict']}: {result['reason']}", flush=True)
    return result


# ----------------------------------------------------------------------------
# verdicts (spec section 5)
# ----------------------------------------------------------------------------
def _pass(r: dict | None) -> bool | None:
    try:
        return bool(r["final"]["gate"]["TEACHER_SHIELD"]["pass"])
    except (KeyError, TypeError):
        return None


def hypothesis_verdicts(results: dict) -> dict:
    rz = _pass(results.get("RZ19"))
    h3 = {}
    for fam in FAMILIES:
        per = {s: _pass(results.get(f"{fam}{s}")) for s in SEEDS}
        if any(v is None for v in per.values()):
            h3[fam] = "INCOMPLETE"
        elif rz:
            h3[fam] = "UNDETERMINED (harness)"
        else:
            h3[fam] = "PASS" if sum(per.values()) >= PASS_MIN else "FAIL"
    if rz is None or any(v == "INCOMPLETE" for v in h3.values()):
        overall = "INCOMPLETE"
    elif rz:
        overall = "UNDETERMINED (harness)"
    else:
        overall = "PASS" if any(v == "PASS" for v in h3.values()) else "FAIL"
    return {
        "H3": h3,
        "H3_overall": overall,
        "RZ": "INCOMPLETE" if rz is None else ("PASS" if rz else "FAIL"),
        "artefact": choose_artefact(results, h3) if overall == "PASS" else None,
    }


def choose_artefact(results: dict, h3: dict) -> str | None:
    """Spec section 5: among PASSING arms, the passing seed with the highest
    gate rate vs TEACHER_SHIELD; ties -> arm order RC, RA, RB, then lowest seed."""
    cands = []
    for rank, fam in enumerate(ARTEFACT_ORDER):
        if h3.get(fam) != "PASS":
            continue
        for s in SEEDS:
            r = results.get(f"{fam}{s}")
            if r and _pass(r):
                cands.append(
                    (
                        -r["final"]["gate"]["TEACHER_SHIELD"]["rate"],
                        rank,
                        s,
                        f"{fam}{s}",
                    )
                )
    return min(cands)[3] if cands else None


def train_violations_of(r: dict) -> int:
    """The arm's cumulative training violations. A record written by the crash
    handler (or an orphan probe file) has no top-level count, so fall back to the
    last probe row that carried one."""
    if "train_violations" in r:
        return int(r["train_violations"])
    for row in reversed(r.get("checkpoints") or []):
        if "train_violations" in row:
            return int(row["train_violations"])
    return 0


def hsafe_verdict(results: dict) -> dict:
    steps = 0
    offenders = []
    missing = [a for a in ARMS if a not in results or "final" not in results[a]]
    for a, r in results.items():
        v = (
            train_violations_of(r)
            + int(r.get("probe_violations", 0))
            + int((r.get("final") or {}).get("violations", 0))
        )
        if "probe_violations" not in r:
            v += int(sum(x.get("violations", 0) for x in r.get("checkpoints", [])))
        steps += v
        if v:
            offenders.append(a)
    if steps:
        verdict = "FAILS"
    elif missing:
        verdict = "INCOMPLETE"
    else:
        verdict = "HOLDS"
    return {
        "verdict": verdict,
        "shielded_violation_steps": steps,
        "offenders": offenders,
        "missing": missing,
    }


def direction(h: dict) -> str:
    if h["hsafe"] == "FAILS":
        return "H-SAFE FAILS: a shielded arm recorded a contact step. Stop; census; no further round without a user ruling."
    if h["hsafe"] == "INCOMPLETE" or h["H3_overall"] == "INCOMPLETE":
        return "incomplete: all nine remedy arms and the RZ control must run"
    if h["RZ"] == "PASS":
        return "RZ PASSES: the augmented observation with no signal beats the shielded teacher — harness invalid; diagnose."
    if h["H3_overall"] == "PASS":
        return (
            "H3 PASSES: a safe learned pursuer that beats the shielded teacher exists — R3's first evidence in the "
            "adversarial setting, and the artefact for Step 9 (the attacker faces both STR14-SHIELD and this policy). "
            "Which ingredient did it is the reported reading."
        )
    return (
        "H3 FAILS (RZ valid): neither cheap ingredient recovers the loss at this budget; Step 10 needs the learned "
        "feasibility constraint or a search policy that plans around the barrier surfaces explicitly. Step 9 proceeds "
        "against STR14-SHIELD alone."
    )


def load_results() -> dict:
    return {
        p.stem.removeprefix("s5_arm_"): json.loads(p.read_text())
        for p in sorted(OUT.glob("s5_arm_*.json"))
    }


def orphan_records(results: dict) -> dict:
    """Probe files with no arm record: the process died before `run_arm` could
    write one. Their rows are evidence and must reach the H-SAFE census."""
    out = {}
    for path in sorted(OUT.glob("s5_probe_*.json")):
        arm = path.stem.removeprefix("s5_probe_")
        if arm in results:
            continue
        spec = ARMS.get(arm)
        out[arm] = {
            "arm": arm,
            "family": spec["family"] if spec else arm[:2],
            "seed": spec["seed"] if spec else None,
            "verdict": "UNDETERMINED",
            "reason": "orphan probe file (process died without a record)",
            "probe": str(path),
            "checkpoints": json.loads(path.read_text()).get("checkpoints", []),
        }
    return out


def summarize(results: dict, starts: dict) -> dict:
    h = hypothesis_verdicts(results)
    orphans = orphan_records(results)
    hs = hsafe_verdict(results | orphans)
    cost_keys = (
        "intervention_rate",
        "infeasibility_rate",
        "stall_step_rate",
        "deadlock_episode_rate",
    )
    cost = {
        a: {k: r["final"]["shield"].get(k) for k in cost_keys}
        | {"rate": r["final"]["score"]["rate"]}
        for a, r in results.items()
        if "final" in r
    }
    ingredient = {}
    for fam in FAMILIES:
        finals = [
            results[f"{fam}{s}"]["final"]["score"]["rate"]
            for s in SEEDS
            if f"{fam}{s}" in results and "final" in results[f"{fam}{s}"]
        ]
        ingredient[fam] = {
            "passes": sum(bool(_pass(results.get(f"{fam}{s}"))) for s in SEEDS),
            "mean_rate": (sum(finals) / len(finals)) if finals else None,
            "n_finished": len(finals),
        }
    d = direction(
        {"hsafe": hs["verdict"], "H3_overall": h["H3_overall"], "RZ": h["RZ"]}
    )
    return {
        "spec": "docs/pathway/shield_remedy_prereg_2026-09-15.md",
        "hsafe": hs,
        "hypotheses": h,
        "bars": {k: starts[k]["rate"] for k in REFS},
        "strsh_reproduced": {
            k: starts[k].get("reproduced") for k in ("STRSH16", "STRSH17", "STRSH18")
        },
        "arms": {
            a: {"verdict": r.get("verdict"), "reason": r.get("reason")}
            for a, r in results.items()
        },
        "orphans": {
            a: {
                "probe": r["probe"],
                "reason": r["reason"],
                "checkpoints": len(r["checkpoints"]),
                "probe_violations": int(
                    sum(x.get("violations", 0) for x in r["checkpoints"])
                ),
                "train_violations": train_violations_of(r),
            }
            for a, r in orphans.items()
        },
        "cost": cost,
        "ingredient": ingredient,
        "complete": all(a in results and "final" in results[a] for a in ARMS),
        "direction": d,
    }


def write_figure(results: dict, starts: dict, summary: dict, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    ax = axes[0]
    col = {"RA": "tab:blue", "RB": "tab:orange", "RC": "tab:green", "RZ": "tab:red"}
    for a, r in results.items():
        rows = r.get("checkpoints") or []
        if rows:
            ax.plot(
                [x["steps"] for x in rows],
                [x["win_rate"] for x in rows],
                marker=".",
                color=col[r["family"]],
                alpha=0.8,
                label=f"{a} ({summary['arms'][a]['verdict']})",
            )
    for ref, c in (
        ("TEACHER_SHIELD", "k"),
        ("STR14_SHIELD", "tab:purple"),
        ("TEACHER_NONE", "tab:gray"),
    ):
        ax.axhline(
            starts[ref]["rate"],
            ls=":",
            color=c,
            lw=1.2,
            label=f"{ref} = {starts[ref]['rate']:.3f}",
        )
    for s in ("STRSH16", "STRSH17", "STRSH18"):
        ax.axhline(
            starts[s]["rate"],
            ls="--",
            color="tab:brown",
            lw=0.6,
            label=f"{s} = {starts[s]['rate']:.3f}",
        )
    ax.set_xlabel("env steps")
    ax.set_ylabel(f"shielded probe capture rate (seeds {PROBE_SEED0}+)")
    ax.set_title(f"S5 — remedy arms (H3: {summary['hypotheses']['H3_overall']})")
    ax.legend(fontsize=5, ncol=2)
    ax = axes[1]
    names = [a for a in results if "final" in results[a]]
    if names:
        xs = range(len(names))
        ax.bar(
            [x - 0.2 for x in xs],
            [summary["cost"][a]["stall_step_rate"] for a in names],
            width=0.2,
            label="stall rate",
        )
        ax.bar(
            xs,
            [summary["cost"][a]["deadlock_episode_rate"] for a in names],
            width=0.2,
            label="deadlock episode rate",
        )
        ax.bar(
            [x + 0.2 for x in xs],
            [summary["cost"][a]["rate"] for a in names],
            width=0.2,
            label="gate capture rate",
        )
        ax.set_xticks(list(xs))
        ax.set_xticklabels(names, fontsize=7, rotation=45)
        ax.set_title("final checkpoints: cost vs capture (shielded, gate stream)")
        ax.legend(fontsize=7)
    fig.suptitle(
        "S5 — shield-aware remedy" + ("" if summary["complete"] else " — INCOMPLETE")
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--measure-start", action="store_true")
    p.add_argument("--arms", nargs="*", choices=list(ARMS), default=[])
    p.add_argument("--summarize", action="store_true")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.measure_start:
        measure_start(args.device)
        return
    for arm in args.arms:
        run_arm(arm, args.device)
    if args.summarize:
        results, starts = load_results(), load_start()
        summary = summarize(results, starts)
        _write_json(OUT / "s5_round.json", {"summary": summary, "arms": results})
        write_figure(results, starts, summary, OUT / "s5_round.png")
        print(
            json.dumps(
                {
                    k: summary[k]
                    for k in ("hsafe", "hypotheses", "ingredient", "strsh_reproduced")
                },
                indent=2,
                default=str,
            )
        )
        print(json.dumps(summary["arms"], indent=2))
        print(f"\n{summary['direction']}")


if __name__ == "__main__":
    main()
