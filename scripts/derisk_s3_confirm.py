"""S3 — confirmatory round for the structured learner (spec:
docs/pathway/structured_confirmatory_prereg_2026-09-14.md). Nothing here may
change a threshold, a seed, a stream, a rule or a budget after a number is seen.

Arms:
  STR11..STR15  SBLONG's configuration, five fresh seeds, 10M env steps  <- the test
  SZLONG11      same, every reward zeroed                                  <- negative control
  SBLONG        the existing final checkpoint, gated on the new stream     <- reported, not counted

Usage:
  ./venv/bin/python scripts/derisk_s3_confirm.py --measure-start
  ./venv/bin/python scripts/derisk_s3_confirm.py --arms STR11        (one process per arm)
  ./venv/bin/python scripts/derisk_s3_confirm.py --summarize
"""

from __future__ import annotations

import argparse
import json
import pathlib

from pe2.config import EnvConfig
from pe2.derisk import gate_pass
from pe2.eval import mcnemar_pvalue, wilson_ci, win_count
from pe2.scripted import (
    FovLimitedEvader,
    GreedyEvader,
    GreedyPursuer,
    ObstacleHuggingEvader,
    OpenSpaceEvader,
    SearchingPursuer,
)
from pe2.structured import HOLD

OUT = pathlib.Path("results/derisk")
START_PATH = OUT / "s3_start.json"
PPO_CLONE = OUT / "ppo_bc_searching_pursuer.zip"
SAC_CLONE = OUT / "sac_bc_searching_pursuer.zip"
SEL7_100K = OUT / "ckpts_SEL7" / "SEL7_100000.zip"
SBLONG_CKPT = OUT / "ckpts_SBLONG" / "SBLONG_10000000.zip"

# --- pre-registered (spec sections 2-5). Do not tune. ---
TOTAL_LONG = 10_000_000
EVERY_LONG = 1_000_000
EVAL_N = 200
GATE_N = 500
PROBE_SEED0 = 1_060_000
GATE_SEED0 = 1_070_000
FAMILY_SEED0 = 1_080_000
FAMILY_N = 200
SEEDS = (11, 12, 13, 14, 15)
PASS_MIN = 4
P_GATE = 0.05
GAMMA_S = 0.99
REFS = ("TEACHER", "PPO_CLONE", "SAC_CLONE", "SEL7_100K", "GREEDY_FULLSTATE")
FAMILY = {
    "GreedyEvader": GreedyEvader,
    "FovLimitedEvader": FovLimitedEvader,
    "ObstacleHuggingEvader": ObstacleHuggingEvader,
    "OpenSpaceEvader": OpenSpaceEvader,
}

ARMS: dict = {
    f"STR{s}": {"kind": "train", "seed": s, "zero_reward": False} for s in SEEDS
}
ARMS["SZLONG11"] = {"kind": "control", "seed": 11, "zero_reward": True}
ARMS["SBLONG"] = {"kind": "existing", "seed": 7, "ckpt": str(SBLONG_CKPT)}


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


# ----------------------------------------------------------------------------
# guards
# ----------------------------------------------------------------------------
def require_not_measured() -> None:
    if START_PATH.exists():
        raise SystemExit(
            f"{START_PATH} already exists: the bars are already measured (stop rule, spec section 6)."
        )
    if any(OUT.glob("s3_arm_*.json")) or any(
        (OUT / f"ckpts_{a}").exists() for a in ARMS if ARMS[a]["kind"] != "existing"
    ):
        raise SystemExit(
            "S3 arms have already run; the stop rule forbids re-measuring."
        )


def require_arm_not_run(arm: str) -> None:
    if (OUT / f"s3_arm_{arm}.json").exists() or (
        ARMS[arm]["kind"] != "existing" and (OUT / f"ckpts_{arm}").exists()
    ):
        raise SystemExit(
            f"arm {arm} has already run; the stop rule (spec section 6) forbids re-running an arm."
        )


def load_start() -> dict:
    if not START_PATH.exists():
        raise SystemExit(f"gate bars missing: {START_PATH}. Run --measure-start FIRST.")
    s = json.loads(START_PATH.read_text())
    for k in REFS:
        if k not in s or "logs" not in s[k]:
            raise SystemExit(
                f"{k} (with paired logs) missing from {START_PATH}; re-run --measure-start."
            )
    return s


# ----------------------------------------------------------------------------
# bars (spec section 2), BEFORE any arm
# ----------------------------------------------------------------------------
def measure_start(device: str) -> dict:
    from stable_baselines3 import PPO, SAC

    from pe2.exploitability import evaluate_sb3_policy
    from pe2.structured import evaluate_scripted

    require_not_measured()
    cfg = EnvConfig()
    refs = {
        "TEACHER": lambda: evaluate_scripted(
            cfg, SearchingPursuer, GreedyEvader, GATE_N, GATE_SEED0
        ),
        "PPO_CLONE": lambda: evaluate_sb3_policy(
            PPO.load(PPO_CLONE, device=device),
            cfg,
            "pursuer",
            GreedyEvader,
            GATE_N,
            GATE_SEED0,
        ),
        "SAC_CLONE": lambda: evaluate_sb3_policy(
            SAC.load(SAC_CLONE, device=device),
            cfg,
            "pursuer",
            GreedyEvader,
            GATE_N,
            GATE_SEED0,
        ),
        "SEL7_100K": lambda: evaluate_sb3_policy(
            PPO.load(SEL7_100K, device=device),
            cfg,
            "pursuer",
            GreedyEvader,
            GATE_N,
            GATE_SEED0,
        ),
        "GREEDY_FULLSTATE": lambda: evaluate_scripted(
            cfg, GreedyPursuer, GreedyEvader, GATE_N, GATE_SEED0
        ),
    }
    out: dict = {"gate_seed0": GATE_SEED0, "gate_n": GATE_N, "provenance": provenance()}
    for name, fn in refs.items():
        logs = fn()
        out[name] = _ci(*win_count(logs, "pursuer")) | {"logs": logs}
        lo, hi = out[name]["wilson"]
        print(
            f"{name} = {out[name]['k']}/{GATE_N} = {out[name]['rate']:.4f}, Wilson [{lo:.4f}, {hi:.4f}]",
            flush=True,
        )
    START_PATH.write_text(json.dumps(out, indent=2))
    return out


# ----------------------------------------------------------------------------
# arms
# ----------------------------------------------------------------------------
def gate_final(ckpt, starts: dict, cfg, device: str) -> dict:
    from stable_baselines3 import PPO

    from pe2.structured import evaluate_structured

    model = PPO.load(ckpt, device=device)
    logs = evaluate_structured(model, cfg, GreedyEvader, GATE_N, GATE_SEED0, hold=HOLD)
    return {
        "ckpt": str(ckpt),
        "score": _ci(*win_count(logs, "pursuer")),
        "gate": {
            ref: gate_pass(logs, starts[ref]["logs"], "pursuer", P_GATE) for ref in REFS
        },
    }


def run_train_arm(
    arm: str, starts: dict, device: str, holder: dict | None = None
) -> dict:
    from stable_baselines3 import PPO

    from pe2.structured import SearchMDP, StructuredEvalProbe

    spec = ARMS[arm]
    cfg = EnvConfig()
    d = OUT / f"ckpts_{arm}"
    d.mkdir(parents=True, exist_ok=True)
    probe = StructuredEvalProbe(
        cfg,
        GreedyEvader,
        every=EVERY_LONG,
        total=TOTAL_LONG,
        eval_n=EVAL_N,
        seed0=PROBE_SEED0,
        ckpt_dir=d,
        report_path=OUT / f"s3_probe_{arm}.json",
        label=arm,
        hold=HOLD,
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
    )
    model = PPO(
        "MlpPolicy", env, seed=spec["seed"], gamma=GAMMA_S, verbose=0, device=device
    )
    init_path = OUT / f"s3_init_{spec['seed']}.zip"
    if not init_path.exists():
        model.save(
            init_path
        )  # STR11 and SZLONG11 share seed 11: literally the same init
    model.learn(total_timesteps=TOTAL_LONG, callback=probe.callback)
    result = {
        "arm": arm,
        "kind": spec["kind"],
        "seed": spec["seed"],
        "settings": spec,
        "init": str(init_path),
        "env_steps": int(env.env_steps),
        "decisions": int(model.num_timesteps),
        "probe_seed0": PROBE_SEED0,
        "gate_seed0": GATE_SEED0,
        "checkpoints": probe.rows,
        "provenance": provenance(),
    }
    ckpt = d / f"{arm}_{TOTAL_LONG}.zip"
    if not any(r["mark"] == TOTAL_LONG for r in probe.rows) or not ckpt.exists():
        result["verdict"] = "UNDETERMINED"
        result["reason"] = (
            f"final probe at mark {TOTAL_LONG} never fired (env steps reached {env.env_steps})"
        )
        return result
    result["final"] = gate_final(ckpt, starts, cfg, device)
    t = result["final"]["gate"]["TEACHER"]
    result["verdict"] = f"final {'PASS' if t['pass'] else 'FAIL'} vs TEACHER"
    result["reason"] = (
        f"final {t['k']}/{t['n']} = {t['rate']:.3f}, McNemar p={t['p']:.2e} (b={t['b']}, c={t['c']}) vs teacher {t['ref_rate']:.3f}; "
        + "; ".join(
            f"vs {r} {'PASS' if result['final']['gate'][r]['pass'] else 'fail'}"
            for r in REFS[1:]
        )
    )
    return result


def run_existing_arm(arm: str, starts: dict, device: str) -> dict:
    spec = ARMS[arm]
    ckpt = pathlib.Path(spec["ckpt"])
    if not ckpt.exists():
        raise SystemExit(f"{arm}: checkpoint missing: {ckpt}")
    cfg = EnvConfig()
    result = {
        "arm": arm,
        "kind": spec["kind"],
        "seed": spec["seed"],
        "settings": spec,
        "gate_seed0": GATE_SEED0,
        "checkpoints": [],
        "provenance": provenance(),
        "final": gate_final(ckpt, starts, cfg, device),
    }
    t = result["final"]["gate"]["TEACHER"]
    result["verdict"] = (
        f"final {'PASS' if t['pass'] else 'FAIL'} vs TEACHER (reported, not counted)"
    )
    result["reason"] = (
        f"final {t['k']}/{t['n']} = {t['rate']:.3f}, McNemar p={t['p']:.2e} (b={t['b']}, c={t['c']}) vs teacher {t['ref_rate']:.3f}"
    )
    return result


def run_arm(arm: str, device: str) -> dict:
    starts = load_start()
    require_arm_not_run(arm)
    holder: dict = {}
    try:
        result = (
            run_existing_arm(arm, starts, device)
            if ARMS[arm]["kind"] == "existing"
            else run_train_arm(arm, starts, device, holder)
        )
    except Exception as e:
        (OUT / f"s3_arm_{arm}.json").write_text(
            json.dumps(
                {
                    "arm": arm,
                    "kind": ARMS[arm]["kind"],
                    "seed": ARMS[arm]["seed"],
                    "verdict": "UNDETERMINED",
                    "reason": f"crashed: {type(e).__name__}: {e}",
                    "checkpoints": holder.get("rows", []),
                    "provenance": provenance(),
                },
                indent=2,
            )
        )
        raise
    (OUT / f"s3_arm_{arm}.json").write_text(json.dumps(result, indent=2))
    print(f"[{arm}] {result['verdict']}: {result['reason']}", flush=True)
    return result


# ----------------------------------------------------------------------------
# verdicts (spec section 4), artefact rule, direction (section 6)
# ----------------------------------------------------------------------------
def _teacher_pass(r: dict) -> bool | None:
    try:
        return bool(r["final"]["gate"]["TEACHER"]["pass"])
    except (KeyError, TypeError):
        return None


def hypothesis_verdict(results: dict) -> dict:
    fresh = {f"STR{s}": results.get(f"STR{s}") for s in SEEDS}
    passes = {a: _teacher_pass(r) if r else None for a, r in fresh.items()}
    ctrl = results.get("SZLONG11")
    ctrl_pass = _teacher_pass(ctrl) if ctrl else None
    if any(v is None for v in passes.values()) or ctrl_pass is None:
        return {
            "H_STR": "INCOMPLETE",
            "passes": sum(bool(v) for v in passes.values()),
            "harness": "INCOMPLETE",
            "artefact": None,
            "per_seed": passes,
        }
    harness = "INVALID" if ctrl_pass else "VALID"
    n = sum(passes.values())
    if harness == "INVALID":
        h = "UNDETERMINED (harness)"
    else:
        h = "PASS" if n >= PASS_MIN else "FAIL"
    artefact = choose_artefact(results) if h == "PASS" else None
    return {
        "H_STR": h,
        "passes": n,
        "harness": harness,
        "artefact": artefact,
        "per_seed": passes,
    }


def choose_artefact(results: dict) -> str | None:
    """Spec section 4: the passing FRESH seed with the highest gate rate vs
    TEACHER; ties -> lowest seed. SBLONG is never eligible."""
    cands = []
    for s in SEEDS:
        r = results.get(f"STR{s}")
        if r and _teacher_pass(r):
            cands.append((-r["final"]["gate"]["TEACHER"]["rate"], s))
    if not cands:
        return None
    return f"STR{min(cands)[1]}"


def direction(h: dict) -> str:
    if h["H_STR"] == "INCOMPLETE":
        return "incomplete: all five fresh seeds and the SZLONG11 control must run"
    if h["harness"] == "INVALID":
        return (
            "SZLONG11 PASSES vs the teacher: the harness is invalid at this budget (unrewarded drift beats the "
            "teacher inside the fixed law); nothing is read; diagnose before any further round."
        )
    if h["H_STR"] == "PASS":
        return (
            f"H-STR PASSES ({h.get('passes')}/5): the structured learner is the paper's learned pursuer — improves on its "
            f"scripted teacher without selection, anchoring, or the possibility of destroying the capture law. Artefact: "
            f"{h.get('artefact')}. Next pre-registration: the shield on the artefact policy (Step 8)."
        )
    return (
        f"H-STR FAILS ({h.get('passes')}/5, control valid): SBLONG's 0.68 was seed-specific; report the seed distribution; "
        "the selection-based PPO artefact (D110 H-SEL) is the fallback pursuer for the shield round."
    )


# ----------------------------------------------------------------------------
# characterisation + family (spec section 5), reported only
# ----------------------------------------------------------------------------
def family_row(logs_policy: list[dict], logs_teacher: list[dict]) -> dict:
    k, n = win_count(logs_policy, "pursuer")
    kt, nt = win_count(logs_teacher, "pursuer")
    p, b, c = mcnemar_pvalue(logs_policy, logs_teacher, "pursuer")
    return _ci(k, n) | {"teacher_rate": kt / nt, "p": p, "b": b, "c": c}


def characterise_and_family(results: dict, h: dict, device: str) -> dict:
    from stable_baselines3 import PPO

    from pe2.structured import (
        OrbitSearch,
        characterise,
        evaluate_scripted,
        evaluate_structured,
    )

    cfg = EnvConfig()
    policies = {"TEACHER": (OrbitSearch(cfg).predict, 1)}
    if "SBLONG" in results:
        policies["SBLONG"] = (PPO.load(SBLONG_CKPT, device=device).predict, HOLD)
    if h.get("artefact"):
        policies[h["artefact"]] = (
            PPO.load(results[h["artefact"]]["final"]["ckpt"], device=device).predict,
            HOLD,
        )
    out = {"characterisation": {}, "family": {}}
    for name, (predict, hold) in policies.items():
        out["characterisation"][name] = characterise(
            predict, cfg, GreedyEvader, GATE_N, GATE_SEED0, hold=hold
        )
    for ev_name, ev in FAMILY.items():
        teacher_logs = evaluate_scripted(
            cfg, SearchingPursuer, ev, FAMILY_N, FAMILY_SEED0
        )
        out["family"][ev_name] = {"TEACHER": _ci(*win_count(teacher_logs, "pursuer"))}
        for name, (predict, hold) in policies.items():
            if name == "TEACHER":
                continue

            class _P:  # predict shim for evaluate_structured
                def __init__(self, f):
                    self.predict = f

            logs = evaluate_structured(
                _P(predict), cfg, ev, FAMILY_N, FAMILY_SEED0, hold=hold
            )
            out["family"][ev_name][name] = family_row(logs, teacher_logs)
    return out


def summarize(
    results: dict, starts: dict, device: str = "cpu", extras: bool = True
) -> dict:
    h = hypothesis_verdict(results)
    d = direction(h)
    summary = {
        "spec": "docs/pathway/structured_confirmatory_prereg_2026-09-14.md",
        "hypothesis": h,
        "bars": {k: starts[k]["rate"] for k in REFS},
        "arms": {
            a: {
                "verdict": r.get("verdict"),
                "reason": r.get("reason"),
                "final_rate": (r.get("final") or {}).get("score", {}).get("rate"),
            }
            for a, r in results.items()
        },
        "secondary": {
            a: {ref: bool(results[a]["final"]["gate"][ref]["pass"]) for ref in REFS}
            for a in results
            if "final" in results[a]
        },
        "complete": all(a in results for a in ARMS),
        "direction": d,
    }
    if extras:
        summary |= characterise_and_family(results, h, device)
    return summary


def load_results() -> dict:
    return {
        p.stem.removeprefix("s3_arm_"): json.loads(p.read_text())
        for p in sorted(OUT.glob("s3_arm_*.json"))
    }


def write_figure(results: dict, starts: dict, summary: dict, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    ax = axes[0]
    for arm, r in results.items():
        rows = r.get("checkpoints") or []
        if rows:
            ax.plot(
                [x["steps"] for x in rows],
                [x["win_rate"] for x in rows],
                marker="o",
                color="tab:red" if r.get("kind") == "control" else None,
                label=f"{arm} ({summary['arms'][arm]['verdict']})",
            )
    for ref, c in zip(
        REFS, ("k", "tab:red", "tab:purple", "tab:orange", "tab:gray"), strict=True
    ):
        ax.axhline(
            starts[ref]["rate"],
            ls=":",
            color=c,
            lw=1.2,
            label=f"{ref} (gate) = {starts[ref]['rate']:.3f}",
        )
    ax.set_xlabel("env steps")
    ax.set_ylabel(f"probe capture rate (seeds {PROBE_SEED0}+)")
    ax.set_title("S3 — structured learner, five fresh seeds + zero-reward control")
    ax.legend(fontsize=6, ncol=2)
    ax = axes[1]
    fam = summary.get("family", {})
    if fam:
        names = list(fam)
        pols = [p for p in next(iter(fam.values())) if p != "TEACHER"]
        width = 0.8 / (len(pols) + 1)
        for j, pol in enumerate(["TEACHER", *pols]):
            xs = [i + j * width for i in range(len(names))]
            ys = [fam[n][pol]["rate"] for n in names]
            ax.bar(xs, ys, width=width, label=pol)
        ax.set_xticks([i + 0.4 - width / 2 for i in range(len(names))])
        ax.set_xticklabels(names, fontsize=8)
        ax.set_ylabel(f"capture rate (N={FAMILY_N}, seeds {FAMILY_SEED0}+)")
        ax.set_title("opponent family (reported only)")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7)
    fig.suptitle(
        "S3 — confirmatory round" + ("" if summary["complete"] else " — INCOMPLETE")
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
        summary = summarize(results, starts, args.device)
        (OUT / "s3_round.json").write_text(
            json.dumps({"summary": summary, "arms": results}, indent=2)
        )
        (OUT / "s3_characterisation.json").write_text(
            json.dumps(
                {
                    "characterisation": summary.get("characterisation"),
                    "family": summary.get("family"),
                },
                indent=2,
            )
        )
        write_figure(results, starts, summary, OUT / "s3_round.png")
        print(json.dumps(summary["hypothesis"], indent=2))
        print(json.dumps(summary["arms"], indent=2))
        print(json.dumps(summary.get("characterisation"), indent=2))
        print(f"\n{summary['direction']}")


if __name__ == "__main__":
    main()
