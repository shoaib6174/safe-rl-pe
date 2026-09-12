"""S1 — structured search: RL learns the search sub-policy under a fixed
intercept law (spec: docs/pathway/structured_search_prereg_2026-09-12.md).
Nothing here may change a threshold, a seed or a budget after a number is
seen.

Four arms, concurrent, 1M env steps each:
  SA  PPO from scratch on SearchMDP, terminal +/-1 + 0.2 per acquisition  <- the test
  SB  same, terminal only                                                 <- density ablation
  SZ  same, every reward zeroed                                           <- negative control
  CP  PPO-GAE lambda=1.0 from a PPO clone of SearchingPursuer, per-step   <- option-1 control

Usage (in this order):
  ./venv/bin/python scripts/derisk_s1_structured.py --preflight       (V1: wrapper == teacher)
  ./venv/bin/python scripts/derisk_s1_structured.py --measure-start   (bars + PPO clone + init)
  ./venv/bin/python scripts/derisk_s1_structured.py --arms SA         (one process per arm)
  ./venv/bin/python scripts/derisk_s1_structured.py --summarize
"""

from __future__ import annotations

import argparse
import json
import pathlib

from pe2.config import EnvConfig
from pe2.derisk import (
    CLONE_CI_LO,
    ROUND_PASS_FROM,
    round_verdict,
    structured_verdict,
)
from pe2.eval import wilson_ci, win_count
from pe2.scripted import GreedyEvader, SearchingPursuer
from pe2.structured import HOLD, R_ACQ

OUT = pathlib.Path("results/derisk")
START_PATH = OUT / "s1_start.json"
PREFLIGHT_PATH = OUT / "s1_preflight.json"
INIT_PATH = OUT / "s1_ppo_init.zip"
CP_CLONE = OUT / "ppo_bc_searching_pursuer.zip"

# --- pre-registered (spec sections 2-5). Do not tune. ---
TOTAL = 1_000_000  # env steps per arm
EVERY = 100_000
EVAL_N = 200
FINAL_EVAL_N = 500
PASS_FROM = ROUND_PASS_FROM  # CP only: 500_000, bound to the library constant
EVAL_SEED0 = 1_010_000  # fresh: <= 1_000_999 consumed
DEMO_SEED0 = 1_020_000
DEMO_EPISODES = 300
BC_EPOCHS = 50
SEED = 7
GAMMA_S = 0.99  # structured arms, per decision
CP_LAMBDA = 1.0  # the control's GAE lambda (deviates from T0.4's 0.95 on purpose)
V2_MIN_LO = 0.35  # the PPO clone must be a competent prior
ARMS = {
    "SA": {"r_acq": R_ACQ, "zero_reward": False},
    "SB": {"r_acq": 0.0, "zero_reward": False},
    "SZ": {"r_acq": 0.0, "zero_reward": True},
    "CP": "control",
}
STRUCTURED = ("SA", "SB", "SZ")


def _ci(k: int, n: int) -> dict:
    lo, hi = wilson_ci(k, n)
    return {"k": k, "n": n, "rate": k / n, "wilson": [lo, hi]}


def _print(name: str, entry: dict) -> None:
    lo, hi = entry["wilson"]
    print(
        f"{name} = {entry['k']}/{entry['n']} = {entry['rate']:.4f}, "
        f"Wilson [{lo:.4f}, {hi:.4f}]",
        flush=True,
    )


# ----------------------------------------------------------------------------
# pre-flight (spec section 2, V1)
# ----------------------------------------------------------------------------
def preflight() -> dict:
    """V1: SearchMDP + orbit at hold=1 must reproduce SearchingPursuer
    episode-for-episode on all FINAL_EVAL_N bar seeds. A mismatch halts the
    round before any bar or arm exists."""
    from pe2.structured import OrbitSearch, evaluate_scripted, evaluate_structured

    cfg = EnvConfig()
    teacher = evaluate_scripted(
        cfg, SearchingPursuer, GreedyEvader, FINAL_EVAL_N, EVAL_SEED0
    )
    wrapped = evaluate_structured(
        OrbitSearch(cfg), cfg, GreedyEvader, FINAL_EVAL_N, EVAL_SEED0, hold=1
    )
    mism = [
        t["seed"]
        for t, w in zip(teacher, wrapped, strict=True)
        if (t["outcome"], t["steps"], t["los_breaks"])
        != (w["outcome"], w["steps"], w["los_breaks"])
    ]
    k, n = win_count(teacher, "pursuer")
    out = {
        "pass": not mism,
        "n": FINAL_EVAL_N,
        "eval_seed0": EVAL_SEED0,
        "mismatches": len(mism),
        "mismatch_seeds": mism[:20],
        "teacher_rate": k / n,
    }
    PREFLIGHT_PATH.write_text(json.dumps(out, indent=2))
    print(
        f"V1 {'PASS' if out['pass'] else 'FAIL'}: {len(mism)} mismatches in {n} "
        f"episodes (teacher rate {k / n:.3f})",
        flush=True,
    )
    if mism:
        raise SystemExit(
            "V1 FAILED: the wrapper does not reproduce the teacher; halting."
        )
    return out


def require_preflight() -> None:
    if not PREFLIGHT_PATH.exists():
        raise SystemExit(
            f"preflight missing: {PREFLIGHT_PATH}. Run --preflight FIRST (spec V1)."
        )
    if not json.loads(PREFLIGHT_PATH.read_text()).get("pass"):
        raise SystemExit("V1 preflight FAILED; the round may not proceed.")


# ----------------------------------------------------------------------------
# bars (spec section 3), measured BEFORE any arm trains
# ----------------------------------------------------------------------------
def make_structured_env(cfg, arm: str):
    from pe2.structured import SearchMDP

    return SearchMDP(cfg, GreedyEvader, seed0=SEED * 1000, hold=HOLD, **ARMS[arm])


def measure_start(device: str) -> dict:
    from stable_baselines3 import PPO

    from pe2.exploitability import (
        collect_demonstrations,
        evaluate_sb3_policy,
        train_bc_ppo,
    )
    from pe2.structured import OrbitSearch, RandomSearch, evaluate_structured

    require_preflight()
    cfg = EnvConfig()
    out: dict = {}

    logs = evaluate_structured(
        OrbitSearch(cfg), cfg, GreedyEvader, FINAL_EVAL_N, EVAL_SEED0, hold=1
    )
    out["BAR_ORBIT"] = _ci(*win_count(logs, "pursuer")) | {"hold": 1}
    _print("BAR_ORBIT", out["BAR_ORBIT"])

    logs = evaluate_structured(
        RandomSearch(0), cfg, GreedyEvader, FINAL_EVAL_N, EVAL_SEED0, hold=HOLD
    )
    out["FLOOR_RAND"] = _ci(*win_count(logs, "pursuer")) | {"hold": HOLD, "rng_seed": 0}
    _print("FLOOR_RAND", out["FLOOR_RAND"])

    # The untrained init of SA/SB/SZ, saved so each arm loads literally this policy.
    init = PPO(
        "MlpPolicy",
        make_structured_env(cfg, "SA"),
        seed=SEED,
        gamma=GAMMA_S,
        verbose=0,
        device=device,
    )
    init.save(INIT_PATH)
    logs = evaluate_structured(
        init, cfg, GreedyEvader, FINAL_EVAL_N, EVAL_SEED0, hold=HOLD
    )
    out["START_S"] = _ci(*win_count(logs, "pursuer")) | {
        "init": str(INIT_PATH),
        "seed": SEED,
    }
    _print("START_S", out["START_S"])

    # The control's prior: PPO clone of the searching teacher (T2 recipe).
    demos = collect_demonstrations(
        cfg, "pursuer", SearchingPursuer, GreedyEvader, DEMO_EPISODES, DEMO_SEED0
    )
    bc, losses = train_bc_ppo(
        cfg, "pursuer", GreedyEvader, demos, seed=SEED, device=device, epochs=BC_EPOCHS
    )
    bc.save(CP_CLONE)
    logs = evaluate_sb3_policy(
        bc, cfg, "pursuer", GreedyEvader, n_episodes=FINAL_EVAL_N, seed0=EVAL_SEED0
    )
    entry = _ci(*win_count(logs, "pursuer"))
    entry |= {
        "clone": str(CP_CLONE),
        "final_bc_loss": losses[-1],
        "demo_episodes": DEMO_EPISODES,
        "demo_seed0": DEMO_SEED0,
        "v2_pass": entry["wilson"][0] >= V2_MIN_LO,
    }
    out["START_CP"] = entry
    _print("START_CP", entry)
    print(
        f"V2 {'PASS' if entry['v2_pass'] else 'FAIL'}: START_CP Wilson lo "
        f"{entry['wilson'][0]:.4f} vs {V2_MIN_LO}",
        flush=True,
    )
    out["eval_seed0"] = EVAL_SEED0
    START_PATH.write_text(json.dumps(out, indent=2))
    return out


def load_start() -> dict:
    if not START_PATH.exists():
        raise SystemExit(
            f"start bars missing: {START_PATH}. Run --measure-start FIRST; the "
            "pre-registration requires it before any arm trains."
        )
    s = json.loads(START_PATH.read_text())
    for k in ("BAR_ORBIT", "FLOOR_RAND", "START_S", "START_CP"):
        if k not in s:
            raise SystemExit(f"{k} missing from {START_PATH}; re-run --measure-start.")
    return s


# ----------------------------------------------------------------------------
# arms
# ----------------------------------------------------------------------------
def run_structured_arm(arm: str, starts: dict, device: str) -> dict:
    from stable_baselines3 import PPO

    from pe2.structured import StructuredEvalProbe, evaluate_structured

    cfg = EnvConfig()
    d = OUT / f"ckpts_{arm}"
    d.mkdir(parents=True, exist_ok=True)
    probe = StructuredEvalProbe(
        cfg,
        GreedyEvader,
        every=EVERY,
        total=TOTAL,
        eval_n=EVAL_N,
        seed0=EVAL_SEED0,
        ckpt_dir=d,
        report_path=OUT / f"s1_probe_{arm}.json",
        label=arm,
        hold=HOLD,
    )
    env = make_structured_env(cfg, arm)
    model = PPO.load(INIT_PATH, env=env, device=device, seed=SEED)
    # The budget is enforced in ENV steps by the probe (it stops learn());
    # the decision budget here is only a ceiling that can never bind first.
    model.learn(total_timesteps=TOTAL, callback=probe.callback)
    result = {
        "arm": arm,
        "settings": ARMS[arm],
        "seed": SEED,
        "gamma": GAMMA_S,
        "hold": HOLD,
        "env_steps": int(env.env_steps),
        "decisions": int(model.num_timesteps),
        "eval_n": EVAL_N,
        "eval_seed0": EVAL_SEED0,
        "checkpoints": probe.rows,
    }
    ckpt = d / f"{arm}_{TOTAL}.zip"
    if not any(r["mark"] == TOTAL for r in probe.rows) or not ckpt.exists():
        verdict = "UNDETERMINED"
        reason = (
            f"final probe at mark {TOTAL} never fired (env steps reached "
            f"{env.env_steps}); judged with no final probe rather than losing the "
            "probe history gathered before the crash"
        )
    else:
        final = PPO.load(ckpt, device=device)
        logs = evaluate_structured(
            final, cfg, GreedyEvader, FINAL_EVAL_N, EVAL_SEED0, hold=HOLD
        )
        result["final"] = _ci(*win_count(logs, "pursuer")) | {"ckpt": str(ckpt)}
        verdict, reason = structured_verdict(
            result["final"]["k"],
            result["final"]["n"],
            starts["BAR_ORBIT"]["rate"],
            starts["FLOOR_RAND"]["wilson"][1],
            starts["START_S"]["wilson"][1],
        )
    result["verdict"], result["reason"] = verdict, reason
    return result


def run_control_arm(starts: dict, device: str) -> dict:
    from stable_baselines3 import PPO

    from pe2.exploitability import (
        CheckpointEvalProbe,
        evaluate_sb3_policy,
        finetune_ppo,
    )

    arm = "CP"
    cfg = EnvConfig()
    d = OUT / f"ckpts_{arm}"
    d.mkdir(parents=True, exist_ok=True)
    probe = CheckpointEvalProbe(
        cfg,
        "pursuer",
        GreedyEvader,
        every=EVERY,
        eval_n=EVAL_N,
        seed0=EVAL_SEED0,
        ckpt_dir=d,
        report_path=OUT / f"s1_probe_{arm}.json",
        label=arm,
    )
    finetune_ppo(
        cfg,
        "pursuer",
        GreedyEvader,
        CP_CLONE,
        TOTAL,
        seed=SEED,
        device=device,
        callback=probe.callback,
        custom_objects={"gae_lambda": CP_LAMBDA},
    )
    bar = starts["START_CP"]["rate"]
    result = {
        "arm": arm,
        "prior": str(CP_CLONE),
        "gae_lambda": CP_LAMBDA,
        "seed": SEED,
        "timesteps": TOTAL,
        "eval_n": EVAL_N,
        "eval_seed0": EVAL_SEED0,
        "bar": {"name": "START_CP", "rate": bar},
        "checkpoints": probe.rows,
    }
    ckpt = d / f"{arm}_{TOTAL}.zip"
    if not ckpt.exists():
        verdict = "UNDETERMINED"
        reason = f"final checkpoint missing: {ckpt} was never written"
    else:
        final = PPO.load(ckpt, device=device)
        logs = evaluate_sb3_policy(
            final,
            cfg,
            "pursuer",
            GreedyEvader,
            n_episodes=FINAL_EVAL_N,
            seed0=EVAL_SEED0,
        )
        result["final"] = _ci(*win_count(logs, "pursuer")) | {"ckpt": str(ckpt)}
        verdict, reason = round_verdict(
            probe.rows,
            result["final"]["k"],
            result["final"]["n"],
            bar,
            from_step=PASS_FROM,
        )
    if not starts["START_CP"].get("v2_pass", True):
        verdict = f"{verdict} (weak prior)"
        reason += (
            f"; V2 failed: START_CP Wilson lo below {V2_MIN_LO}, not the "
            "competent prior"
        )
    result["verdict"], result["reason"] = verdict, reason
    return result


def run_arm(arm: str, device: str) -> dict:
    starts = load_start()  # FIRST: refuse before training if the bars are missing
    if ARMS[arm] == "control":
        result = run_control_arm(starts, device)
    else:
        if not INIT_PATH.exists():
            raise SystemExit(f"init missing: {INIT_PATH}; re-run --measure-start.")
        result = run_structured_arm(arm, starts, device)
    (OUT / f"s1_arm_{arm}.json").write_text(json.dumps(result, indent=2))
    print(f"[{arm}] verdict={result['verdict']}: {result['reason']}", flush=True)
    return result


# ----------------------------------------------------------------------------
# summary (spec section 6), fixed before the run
# ----------------------------------------------------------------------------
_RANK = {"FAILED": 0, "LEARNED": 1, "IMPROVED": 2}


def direction(v: dict) -> str:
    sa, sb, sz, cp = (v.get(a) for a in ("SA", "SB", "SZ", "CP"))
    if any(x is None for x in (sa, sb, sz, cp)):
        return "incomplete: all four arms must run"
    if sz in ("LEARNED", "IMPROVED"):
        return (
            "SZ LEARNED: harness invalid — the zero-reward control moved, so "
            "something other than the reward is moving the policy; nothing else "
            "is read (spec section 6)."
        )
    cp_base = cp.split(" ")[0]
    if sa == "IMPROVED" and cp_base == "IMPROVED":
        return (
            "SA IMPROVED and CP IMPROVED: a non-bootstrapping learner suffices on "
            "the per-step problem. The unstructured learned pursuer is the simpler "
            "artefact; the structure is reported as an ablation, not the "
            "contribution. Route C withdrawn either way."
        )
    if sa == "IMPROVED":
        return (
            f"SA IMPROVED, CP {cp}: hypothesis S confirmed, C refuted — the "
            "structure is what makes learning work here. Route C withdrawn. The "
            "paper becomes learned search under a fixed intercept law with a hard "
            "CBF shield; next pre-registration adds the shield to the structured "
            "pursuer. SB says whether density was needed."
        )
    if sa == "LEARNED":
        return (
            f"SA LEARNED, not IMPROVED (CP {cp}): the search MDP is learnable but "
            "the orbit is hard to beat from an 11-dim memoryless observation. The "
            "next question is the search policy's observation (coverage / "
            "staleness memory), not the learner. Route C re-scoped to per-step "
            "fine-tuning; not withdrawn."
        )
    return (
        f"SA FAILED (SZ valid, CP {cp}): PPO cannot learn even a ~20-decision "
        "search MDP at this budget. The negative result is, for the first time, "
        "about the game and not about a critic — Route C stands on a stronger "
        "footing; D82/D100/D102/D103 are re-scoped as one-step-TD results."
    )


def summarize(results: dict, starts: dict) -> dict:
    v = {a: r["verdict"] for a, r in results.items()}
    arms = {
        a: {"verdict": r["verdict"], "reason": r["reason"]} for a, r in results.items()
    }
    if v.get("SZ") in ("LEARNED", "IMPROVED"):
        for a in STRUCTURED:
            if a in arms and a != "SZ":
                arms[a]["verdict"] = "UNDETERMINED (harness)"
    d = direction(v)
    sa, sb = results.get("SA"), results.get("SB")
    if (
        sa
        and sb
        and "final" in sa
        and "final" in sb
        and (
            _RANK.get(v["SB"], -1) > _RANK.get(v["SA"], -1)
            or (
                _RANK.get(v["SB"], -1) == _RANK.get(v["SA"], -1)
                and sb["final"]["rate"] >= sa["final"]["rate"]
            )
        )
    ):
        d += " SB >= SA: the event reward added nothing; prefer SB as the simpler arm."
    return {
        "spec": "docs/pathway/structured_search_prereg_2026-09-12.md",
        "arms": arms,
        "bars": {
            k: starts[k]["rate"]
            for k in ("BAR_ORBIT", "FLOOR_RAND", "START_S", "START_CP")
        },
        "v2_pass": bool(starts["START_CP"].get("v2_pass", True)),
        "complete": all(a in results for a in ARMS),
        "direction": d,
    }


def load_results() -> dict:
    return {
        p.stem.removeprefix("s1_arm_"): json.loads(p.read_text())
        for p in sorted(OUT.glob("s1_arm_*.json"))
    }


def write_figure(results: dict, starts: dict, summary: dict, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for arm, r in results.items():
        rows = r["checkpoints"]
        if rows:
            ax.plot(
                [x["steps"] for x in rows],
                [x["win_rate"] for x in rows],
                marker="o",
                label=f"{arm} ({summary['arms'][arm]['verdict']})",
            )
        if "final" in r:
            lo, hi = r["final"]["wilson"]
            rate = r["final"]["rate"]
            ax.errorbar(
                [TOTAL + 20_000],
                [rate],
                yerr=[[rate - lo], [hi - rate]],
                fmt="s",
                capsize=3,
            )
    refs = (
        ("BAR_ORBIT", "tab:green", "the bar"),
        ("FLOOR_RAND", "tab:orange", "random floor"),
        ("START_S", "tab:purple", "untrained init"),
        ("START_CP", "tab:red", "PPO clone"),
    )
    for key, color, what in refs:
        ax.axhline(
            starts[key]["rate"],
            color=color,
            ls=":",
            lw=1.3,
            label=f"{key} = {starts[key]['rate']:.3f} ({what})",
        )
    ax.axhline(
        CLONE_CI_LO,
        color="k",
        ls="--",
        lw=0.8,
        label=f"CP collapse floor {CLONE_CI_LO}",
    )
    ax.axvline(
        PASS_FROM,
        color="grey",
        ls="-.",
        lw=0.8,
        label=f"CP pass window from {PASS_FROM}",
    )
    ax.set_xlabel("env steps")
    ax.set_ylabel(f"capture rate vs GreedyEvader (paired seeds {EVAL_SEED0}+)")
    ax.set_title(
        "S1 — structured search under a fixed intercept law"
        + ("" if summary["complete"] else " — INCOMPLETE")
    )
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--measure-start", action="store_true")
    p.add_argument("--arms", nargs="*", choices=list(ARMS), default=[])
    p.add_argument("--summarize", action="store_true")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.preflight:
        preflight()
        return
    if args.measure_start:
        measure_start(args.device)
        return
    for arm in args.arms:
        run_arm(arm, args.device)
    if args.summarize:
        results = load_results()
        starts = load_start()
        summary = summarize(results, starts)
        (OUT / "s1_round.json").write_text(
            json.dumps({"summary": summary, "bars": starts, "arms": results}, indent=2)
        )
        write_figure(results, starts, summary, OUT / "s1_round.png")
        print(json.dumps(summary["arms"], indent=2))
        print(f"\n{summary['direction']}")


if __name__ == "__main__":
    main()
