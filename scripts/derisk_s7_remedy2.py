"""S7 — Step 10 iteration 2: a shield-aware pursuer trained against the structured attacker family
(spec: docs/pathway/shield_remedy_2_prereg_2026-09-16.md). Nothing here may change a threshold, a seed,
a stream, a rule or a budget after a number is seen.

Stage 1: RCM{28,29,30} = S5's RC configuration trained against the MIXTURE (Greedy, Hug7, structured
SplitHug, structured CTL-D125/126/127); gate vs the held-out attackers CTL-D225/226/227, Greedy, Hug7,
paired with RC19 on the same stream.
Stage 2 (only if H3-2 passes): ATK2-{31,32,33} fresh structured attackers trained against the S7 artefact
(S6 machinery via importlib), gate on a fresh stream; H-RACE reported vs RACE_BAR = 0.58.

Usage:
  ./venv/bin/python scripts/derisk_s7_remedy2.py --preflight        (V1-E on 200 seeds)
  ./venv/bin/python scripts/derisk_s7_remedy2.py --measure-start    (RC19 reference on the gate stream)
  ./venv/bin/python scripts/derisk_s7_remedy2.py --arms RCM28       (one per process)
  ./venv/bin/python scripts/derisk_s7_remedy2.py --summarize
  ./venv/bin/python scripts/derisk_s7_remedy2.py --race-start | --race-arms ATK2-31 | --race-summarize
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pathlib

from pe2.config import EnvConfig
from pe2.derisk import gate_pass, paired_indicator_test
from pe2.evade import D_POS
from pe2.eval import wilson_ci, win_count
from pe2.metrics import summarize_shield
from pe2.scripted import GreedyEvader, ObstacleHuggingEvader
from pe2.structured import HOLD

OUT = pathlib.Path("results/derisk")
START_PATH = OUT / "s7_start.json"
PREFLIGHT_PATH = OUT / "s7_preflight.json"
RACE_START_PATH = OUT / "s7_race_start.json"
RC19_CKPT = OUT / "ckpts_RC19" / "RC19_10000000.zip"
STR14_CKPT = OUT / "ckpts_STR14" / "STR14_10000000.zip"
ATT_CKPT = {
    f"CTL-D1{s}": OUT / f"ckpts_CTL-D1{s}" / f"CTL-D1{s}_10000000.zip"
    for s in (25, 26, 27)
}
HELD_OUT_CKPT = {
    f"ATT{s}": OUT / f"ckpts_CTL-D2{s}" / f"CTL-D2{s}_10000000.zip"
    for s in (25, 26, 27)
}

# --- pre-registered (spec sections 3-5). Do not tune. ---
SEEDS = (28, 29, 30)
PROBE_SEED0 = 1_160_000
GATE_SEED0 = 1_170_000
GATE_N = 500
EVAL_N = 200
TOTAL_LONG = 10_000_000
EVERY_LONG = 1_000_000
PASS_MIN = 2
P_GATE = 0.05
GAMMA_S = 0.99
RC_SETTINGS = {"shield_obs": True, "stall_penalty": 1e-3, "r_acq": 0.0}
MIXTURE = ("GREEDY", "HUG7", "SPLITHUG", "CTL-D125", "CTL-D126", "CTL-D127")
HELD_OUT = ("ATT25", "ATT26", "ATT27")
GATE_OPPONENTS = HELD_OUT + ("GREEDY", "HUG7")
ARMS = {f"RCM{s}": {"seed": s} for s in SEEDS}
SEEDS2 = (31, 32, 33)
PROBE2_SEED0 = 1_180_000
GATE2_SEED0 = 1_190_000
RACE_BAR = 0.58
RACE_COMPARATORS = ("HUG7", "GREEDY")
PREFLIGHT_N = 200

_MODELS: dict = {}


def _model(path, device):
    from stable_baselines3 import PPO

    key = str(path)
    if key not in _MODELS:
        _MODELS[key] = PPO.load(path, device=device)
    return _MODELS[key]


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


def _ci(k, n):
    lo, hi = wilson_ci(k, n)
    return {"k": k, "n": n, "rate": k / n, "wilson": [lo, hi]}


# ----------------------------------------------------------------------------
# opponents
# ----------------------------------------------------------------------------
def structured_evader_factory(ckpt, device):
    from pe2.evade import StructuredEvaderPolicy

    cfg = EnvConfig()
    model = _model(ckpt, device)
    return lambda: StructuredEvaderPolicy(model, cfg, hold=10, d_pos=D_POS)


def splithug_factory(device):
    from pe2.evade import SplitHugEvader, StructuredEvaderPolicy

    cfg = EnvConfig()
    return lambda: StructuredEvaderPolicy(
        SplitHugEvader(cfg), cfg, hold=10, d_pos=D_POS
    )


def mixture_factory(seed: int, device: str):
    from pe2.evade import MixtureOpponentFactory

    members = [GreedyEvader, ObstacleHuggingEvader, splithug_factory(device)] + [
        structured_evader_factory(ATT_CKPT[k], device)
        for k in ("CTL-D125", "CTL-D126", "CTL-D127")
    ]
    return MixtureOpponentFactory(members, seed=seed)


def gate_opponent_factory(name: str, device: str):
    if name == "GREEDY":
        return GreedyEvader
    if name == "HUG7":
        return ObstacleHuggingEvader
    return structured_evader_factory(HELD_OUT_CKPT[name], device)


# ----------------------------------------------------------------------------
# guards
# ----------------------------------------------------------------------------
def require_preflight() -> None:
    if not PREFLIGHT_PATH.exists():
        raise SystemExit(
            f"preflight missing: {PREFLIGHT_PATH}. Run --preflight FIRST (spec V1-E)."
        )
    if not json.loads(PREFLIGHT_PATH.read_text()).get("pass"):
        raise SystemExit("V1-E preflight FAILED; the round may not proceed.")


def require_not_measured() -> None:
    if START_PATH.exists():
        raise SystemExit(f"{START_PATH} already exists (stop rule).")
    if any(OUT.glob("s7_arm_*.json")) or any(
        (OUT / f"ckpts_{a}").exists() for a in ARMS
    ):
        raise SystemExit(
            "S7 arms have already run; the stop rule forbids re-measuring."
        )


def require_arm_not_run(arm: str) -> None:
    if (OUT / f"s7_arm_{arm}.json").exists() or (OUT / f"ckpts_{arm}").exists():
        raise SystemExit(
            f"arm {arm} has already run; the stop rule forbids re-running it."
        )


def load_start() -> dict:
    if not START_PATH.exists():
        raise SystemExit(f"reference missing: {START_PATH}. Run --measure-start FIRST.")
    s = json.loads(START_PATH.read_text())
    for o in GATE_OPPONENTS:
        if o not in s.get("RC19", {}) or "deadlock_ind" not in s["RC19"][o]:
            raise SystemExit(
                f"RC19/{o} missing from {START_PATH}; re-run --measure-start."
            )
    return s


# ----------------------------------------------------------------------------
# preflight (V1-E) and reference
# ----------------------------------------------------------------------------
def preflight(device: str) -> dict:
    import numpy as np

    from pe2.evade import EvadeMDP, StructuredEvaderPolicy
    from pe2.exploitability import SingleAgentWrapper
    from pe2.shield import CBFShield
    from pe2.structured import StructuredPursuerPolicy

    cfg = EnvConfig()
    d1 = _model(STR14_CKPT, device)
    ev = _model(ATT_CKPT["CTL-D125"], device)
    mism = []
    for i in range(PREFLIGHT_N):
        seed = GATE_SEED0 + i
        ref = EvadeMDP(
            cfg,
            lambda: StructuredPursuerPolicy(d1, cfg, CBFShield(cfg), HOLD, False),
            hold=10,
        ).play_episode(ev.predict, seed)
        w = SingleAgentWrapper(
            cfg, "pursuer", lambda: StructuredEvaderPolicy(ev, cfg, hold=10)
        )
        w.reset(seed=seed)
        pol = StructuredPursuerPolicy(d1, cfg, CBFShield(cfg), HOLD, False)
        pol.reset()
        while True:
            v, om = pol.act(w._last_joint_obs["pursuer"], w.env)
            _, _, term, trunc, info = w.step(
                np.array([2 * v / cfg.v_max - 1, om / cfg.omega_max], dtype=np.float32)
            )
            pol.record(
                bool(info["pursuer_wall_contact"] or info["pursuer_obstacle_contact"])
            )
            if term or trunc:
                break
        got = (
            info["outcome"],
            w.env.current_step,
            pol.metrics.n_stalled,
            pol.metrics.longest_stall_run,
        )
        exp = (
            ref["outcome"],
            ref["steps"],
            ref["pursuer_shield"]["n_stalled"],
            ref["pursuer_shield"]["longest_stall_run"],
        )
        if got != exp:
            mism.append({"seed": seed, "got": got, "expected": exp})
    out = {
        "pass": not mism,
        "n": PREFLIGHT_N,
        "seed0": GATE_SEED0,
        "mismatches": len(mism),
        "examples": mism[:10],
        "provenance": provenance(),
    }
    _write_json(PREFLIGHT_PATH, out)
    print(
        f"V1-E {'PASS' if out['pass'] else 'FAIL'}: {len(mism)} mismatches in {PREFLIGHT_N} episodes",
        flush=True,
    )
    if mism:
        raise SystemExit(
            "V1-E FAILED: the evader object does not reproduce EvadeMDP; halting."
        )
    return out


def gate_pursuer(
    model, opponents: dict, device: str, holder: dict | None = None
) -> dict:
    """Shielded, shield_obs=True gate of a pursuer against each gate opponent on GATE_SEED0.

    Each opponent's entry is stashed into `holder["gate"]` as soon as it completes, so a
    crash part-way through the gate still hands `run_arm` the violations already observed."""
    from pe2.shield import CBFShield
    from pe2.structured import evaluate_structured

    cfg = EnvConfig()
    out = {}
    for name, factory in opponents.items():
        logs = evaluate_structured(
            model,
            cfg,
            factory,
            GATE_N,
            GATE_SEED0,
            hold=HOLD,
            shield_factory=lambda: CBFShield(cfg),
            shield_obs=True,
        )
        k, n = win_count(logs, "pursuer")
        sh = summarize_shield(logs)
        out[name] = {
            "deadlock_ind": [bool(lg["shield"]["deadlock"]) for lg in logs],
            "capture_ind": [lg["outcome"] == "capture" for lg in logs],
            "capture_rate": k / n,
            "capture": _ci(k, n),
            "pursuer": {
                kk: sh[kk]
                for kk in (
                    "deadlock_episode_rate",
                    "stall_step_rate",
                    "intervention_rate",
                    "infeasibility_rate",
                    "backup_rate",
                    "min_clearance_floor",
                )
            },
            "pursuer_violations": int(sh["violation_steps"]),
        }
        if holder is not None:
            holder.setdefault("gate", {})[name] = out[name]
        print(
            f"    vs {name:7s}: capture {k / n:.3f}, deadlock eps {sh['deadlock_episode_rate']:.3f}, "
            f"stall {sh['stall_step_rate']:.3f}, violations {sh['violation_steps']}",
            flush=True,
        )
    return out


def measure_start(device: str) -> dict:
    require_preflight()
    require_not_measured()
    opps = {name: gate_opponent_factory(name, device) for name in GATE_OPPONENTS}
    print("RC19 reference on the gate stream:", flush=True)
    out = {
        "gate_seed0": GATE_SEED0,
        "gate_n": GATE_N,
        "provenance": provenance(),
        "RC19": gate_pursuer(_model(RC19_CKPT, device), opps, device),
    }
    _write_json(START_PATH, out)
    return out


# ----------------------------------------------------------------------------
# stage-1 arms
# ----------------------------------------------------------------------------
class ShieldedStructuredProbe:
    """S5's shielded env-step-grid probe with shield_obs fixed True (every S7 arm
    is shield-aware); rows carry the probe's and the training env's violations."""

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

        def evaluate_at(model, env_steps, mark, train_violations):
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
                shield_obs=True,
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
                "violations": int(sh["violation_steps"]),
                "stall_step_rate": sh["stall_step_rate"],
                "deadlock_episode_rate": sh["deadlock_episode_rate"],
                "train_violations": int(train_violations),
            }
            rows.append(row)
            _write_json(report_path, {"checkpoints": rows})
            print(
                f"  @{env_steps:>9} env steps (mark {mark}): shielded win rate vs Greedy {k}/{n} = {k / n:.3f}, "
                f"violations {row['violations']}, deadlock eps {row['deadlock_episode_rate']:.3f}",
                flush=True,
            )
            return row

        class _CB(BaseCallback):
            def __init__(inner_self) -> None:
                super().__init__()
                inner_self.next_mark = every

            def _on_step(inner_self) -> bool:
                counts = inner_self.training_env.get_attr("env_steps")
                assert len(counts) == 1, "single env assumed (budget in env steps)"
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
    from pe2.structured import SearchMDP

    seed = ARMS[arm]["seed"]
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
        report_path=OUT / f"s7_probe_{arm}.json",
        label=arm,
    )
    if holder is not None:
        holder["rows"] = probe.rows
    env = SearchMDP(
        cfg,
        mixture_factory(seed, device),
        seed0=seed * 1000,
        hold=HOLD,
        r_acq=RC_SETTINGS["r_acq"],
        zero_reward=False,
        shield=CBFShield(cfg),
        shield_obs=RC_SETTINGS["shield_obs"],
        stall_penalty=RC_SETTINGS["stall_penalty"],
    )
    if holder is not None:
        holder["env"] = env
    model = PPO("MlpPolicy", env, seed=seed, gamma=GAMMA_S, verbose=0, device=device)
    model.save(OUT / f"s7_init_{arm}.zip")
    model.learn(total_timesteps=TOTAL_LONG, callback=probe.callback)
    result = {
        "arm": arm,
        "seed": seed,
        "settings": RC_SETTINGS,
        "mixture": list(MIXTURE),
        "env_steps": int(env.env_steps),
        "decisions": int(model.num_timesteps),
        "train_violations": int(env.violations_total),
        "probe_violations": int(sum(r.get("violations", 0) for r in probe.rows)),
        "mixture_draws": env.inner.opponent_factory.draws[:2000],
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
    opps = {name: gate_opponent_factory(name, device) for name in GATE_OPPONENTS}
    print(f"[{arm}] gate:", flush=True)
    g = gate_pursuer(PPO.load(ckpt, device=device), opps, device, holder)
    result["final"] = {
        "ckpt": str(ckpt),
        "gate": g,
        "pursuer_violations": int(sum(v["pursuer_violations"] for v in g.values())),
    }
    per = {
        o: paired_indicator_test(
            starts["RC19"][o]["deadlock_ind"], g[o]["deadlock_ind"], P_GATE
        )
        for o in HELD_OUT
    }
    n_pass = sum(v["pass"] for v in per.values())
    result["verdict"] = f"deadlock lower than RC19 on {n_pass}/3 held-out attackers"
    result["reason"] = (
        "; ".join(
            f"{o}: RC19 {per[o]['rate_a']:.3f} vs S7 {per[o]['rate_b']:.3f} "
            f"p={per[o]['p']:.1e} {'PASS' if per[o]['pass'] else 'fail'}"
            for o in HELD_OUT
        )
        + f"; capture vs Greedy {g['GREEDY']['capture_rate']:.3f} "
        f"(RC19 {starts['RC19']['GREEDY']['capture_rate']:.3f}); violations train "
        f"{result['train_violations']} probe {result['probe_violations']} gate "
        f"{result['final']['pursuer_violations']}"
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
            OUT / f"s7_arm_{arm}.json",
            {
                "arm": arm,
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
                "gate_violations": int(
                    sum(
                        v["pursuer_violations"] for v in holder.get("gate", {}).values()
                    )
                ),
                "gate_partial": holder.get("gate", {}),
                "provenance": provenance(),
            },
        )
        raise
    _write_json(OUT / f"s7_arm_{arm}.json", result)
    print(f"[{arm}] {result['verdict']}: {result['reason']}", flush=True)
    return result


# ----------------------------------------------------------------------------
# stage-1 verdicts
# ----------------------------------------------------------------------------
def h32_verdicts(results: dict, starts: dict) -> dict:
    per_seed, passing = {}, []
    for s in SEEDS:
        r = results.get(f"RCM{s}")
        if r is None or "final" not in r:
            per_seed[s] = {"verdict": "INCOMPLETE"}
            continue
        g = r["final"]["gate"]
        tests = {
            o: paired_indicator_test(
                starts["RC19"][o]["deadlock_ind"], g[o]["deadlock_ind"], P_GATE
            )
            for o in HELD_OUT
        }
        n = sum(v["pass"] for v in tests.values())
        cap = gate_pass(  # is RC19 significantly better at capture vs Greedy than the S7 pursuer?
            [
                {"seed": i, "role": "pursuer", "outcome": "capture" if c else "timeout"}
                for i, c in enumerate(starts["RC19"]["GREEDY"]["capture_ind"])
            ],
            [
                {"seed": i, "role": "pursuer", "outcome": "capture" if c else "timeout"}
                for i, c in enumerate(g["GREEDY"]["capture_ind"])
            ],
            "pursuer",
            P_GATE,
        )
        weaker = bool(cap["pass"])
        mean_dl = sum(g[o]["pursuer"]["deadlock_episode_rate"] for o in HELD_OUT) / len(
            HELD_OUT
        )
        if n >= PASS_MIN:
            v = "PASS (robust but weaker)" if weaker else "PASS"
            passing.append((mean_dl, s))
        else:
            v = "FAIL"
        per_seed[s] = {
            "verdict": v,
            "passes": n,
            "tests": tests,
            "capture_guard": {
                "rc19_better_p": cap["p"],
                "weaker": weaker,
                "s7_capture": g["GREEDY"]["capture_rate"],
                "rc19_capture": starts["RC19"]["GREEDY"]["capture_rate"],
            },
            "mean_deadlock_heldout": mean_dl,
        }
    if any(v["verdict"] == "INCOMPLETE" for v in per_seed.values()):
        h = "INCOMPLETE"
    else:
        h = (
            "PASS"
            if sum(v["verdict"].startswith("PASS") for v in per_seed.values())
            >= PASS_MIN
            else "FAIL"
        )
    artefact = f"RCM{min(passing)[1]}" if (h == "PASS" and passing) else None
    return {"H32": h, "per_seed": per_seed, "artefact": artefact}


def hsafe_verdict(
    results: dict, starts: dict | None = None, race: dict | None = None
) -> dict:
    steps, offenders = 0, []
    missing = [a for a in ARMS if a not in results or "final" not in results[a]]
    for a, r in results.items():
        v = (
            int(r.get("train_violations", 0))
            + int(r.get("probe_violations", 0))
            # a crash record has no "final"; gate_violations is what the gate had
            # already observed when the process died (absent on a complete record)
            + int(r.get("gate_violations", 0))
            + int((r.get("final") or {}).get("pursuer_violations", 0))
        )
        if "probe_violations" not in r:
            v += int(sum(x.get("violations", 0) for x in r.get("checkpoints", [])))
        steps += v
        if v:
            offenders.append(a)
    if starts is not None:
        for o in GATE_OPPONENTS:
            v = int(
                ((starts.get("RC19") or {}).get(o) or {}).get("pursuer_violations") or 0
            )
            steps += v
            if v:
                offenders.append(f"reference:RC19/{o}")
    if race is not None:
        # Stage 2 is inside H-SAFE (spec section 1: "everywhere ... stage 2").
        # race_summary has already pooled each ATK2 arm's train + probe + gate
        # pursuer contact steps; fold the census in rather than recount it.
        steps += int(race.get("pursuer_violation_steps", 0))
        offenders += [f"race:{a}" for a in race.get("offenders", [])]
    return {
        "verdict": "FAILS" if steps else ("INCOMPLETE" if missing else "HOLDS"),
        "pursuer_violation_steps": steps,
        "offenders": offenders,
        "missing": missing,
    }


def direction(h: dict) -> str:
    if h["hsafe"] == "FAILS":
        return "H-SAFE FAILS: a shielded pursuer recorded a contact step. Stop; census; no further round without a user ruling."
    if h["hsafe"] == "INCOMPLETE" or h["H32"] == "INCOMPLETE":
        return "incomplete: all three RCM arms must run"
    if h["H32"] == "FAIL":
        return (
            "H3-2 FAILS: family training does not buy resistance at this budget — the structured attack is a "
            "property of the shield's geometry, not of the pursuer's policy."
        )
    if h.get("H_RACE") == "HOLDS":
        return (
            "H3-2 PASSES and H-RACE holds: training against the structured family closes most of the attacker's "
            "gain — the arms race converges toward the defender (R3's strongest form)."
        )
    if h.get("H_RACE") == "INCOMPLETE":
        return "H3-2 PASSES; stage 2 incomplete: all three ATK2 arms must run"
    if h.get("H_RACE") == "FAILS":
        return (
            "H3-2 PASSES but fresh attackers adapt (H-RACE fails): robust to unseen attackers of the old kind, "
            "an open arms race against new ones — reported as such."
        )
    return (
        "H3-2 PASSES: stage 2 (fresh attackers vs the artefact) decides whether the race converges — run "
        "--race-start / --race-arms / --race-summarize."
    )


def load_results() -> dict:
    return {
        p.stem.removeprefix("s7_arm_"): json.loads(p.read_text())
        for p in sorted(OUT.glob("s7_arm_RCM*.json"))
    }


def summarize(results: dict, starts: dict, race: dict | None = None) -> dict:
    h = h32_verdicts(results, starts)
    hs = hsafe_verdict(results, starts, race)
    table = {}
    for a, r in results.items():
        g = (r.get("final") or {}).get("gate") or {}
        table[a] = {
            o: {
                "deadlock": g[o]["pursuer"]["deadlock_episode_rate"],
                "capture": g[o]["capture_rate"],
            }
            for o in g
        }
    ref = {
        o: {
            "deadlock": starts["RC19"][o]["pursuer"]["deadlock_episode_rate"],
            "capture": starts["RC19"][o]["capture_rate"],
        }
        for o in GATE_OPPONENTS
    }
    hr = race["H_RACE"] if race else None
    return {
        "spec": "docs/pathway/shield_remedy_2_prereg_2026-09-16.md",
        "hsafe": hs,
        "hypotheses": h,
        "arms": table,
        "rc19": ref,
        "race": race,
        "complete": all(a in results and "final" in results[a] for a in ARMS),
        "direction": direction({"hsafe": hs["verdict"], "H32": h["H32"], "H_RACE": hr}),
    }


def write_figure(results: dict, starts: dict, summary: dict, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    ax = axes[0]
    opps = list(GATE_OPPONENTS)
    x = range(len(opps))
    ax.bar(
        [i - 0.3 for i in x],
        [summary["rc19"][o]["deadlock"] for o in opps],
        width=0.15,
        color="k",
        label="RC19 (S5)",
    )
    for j, a in enumerate(sorted(summary["arms"])):
        ax.bar(
            [i - 0.15 + 0.15 * j for i in x],
            [summary["arms"][a].get(o, {}).get("deadlock", 0) for o in opps],
            width=0.15,
            label=f"{a} ({summary['hypotheses']['per_seed'][int(a[3:])]['verdict']})",
        )
    ax.set_xticks(list(x))
    ax.set_xticklabels(opps)
    ax.set_ylabel("pursuer deadlock-episode rate (gate)")
    ax.set_title(f"S7 stage 1: H3-2 {summary['hypotheses']['H32']}")
    ax.legend(fontsize=7)
    ax = axes[1]
    ax.bar(
        [i - 0.3 for i in x],
        [summary["rc19"][o]["capture"] for o in opps],
        width=0.15,
        color="k",
        label="RC19",
    )
    for j, a in enumerate(sorted(summary["arms"])):
        ax.bar(
            [i - 0.15 + 0.15 * j for i in x],
            [summary["arms"][a].get(o, {}).get("capture", 0) for o in opps],
            width=0.15,
            label=a,
        )
    ax.set_xticks(list(x))
    ax.set_xticklabels(opps)
    ax.set_ylabel("capture rate (gate)")
    ax.set_title("capture")
    ax.legend(fontsize=7)
    fig.suptitle(
        "S7 — shield-aware pursuer trained against the structured family"
        + ("" if summary["complete"] else " — INCOMPLETE")
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ----------------------------------------------------------------------------
# stage 2: the race (S6 machinery)
# ----------------------------------------------------------------------------
def _s6():
    p = pathlib.Path(__file__).resolve().parent / "derisk_s6_attacker.py"
    spec = importlib.util.spec_from_file_location("derisk_s6_attacker", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def race_config(artefact_ckpt) -> dict:
    return {
        "D3": {"ckpt": pathlib.Path(artefact_ckpt), "shield_obs": True, "hold": HOLD}
    }


def race_setup(S6, artefact_ckpt):
    S6.OUT = OUT
    S6.START_PATH = RACE_START_PATH
    S6.ARTEFACT_PREFIX = "s7race"
    S6.PROBE_SEED0 = PROBE2_SEED0
    S6.GATE_SEED0 = GATE2_SEED0
    S6.GATE_N = GATE_N
    S6.EVAL_N = EVAL_N
    S6.TOTAL_LONG = TOTAL_LONG
    S6.EVERY_LONG = EVERY_LONG
    S6.SEEDS = SEEDS2
    S6.DEFENDERS = race_config(artefact_ckpt)
    S6.COMPARATORS = RACE_COMPARATORS
    S6.ARMS = {
        f"ATK2-{s}": {
            "family": "CTL",
            "defender": "D3",
            "seed": s,
            "beta": 0.0,
            "zero_reward": False,
        }
        for s in SEEDS2
    }

    def load_race_start():
        if not RACE_START_PATH.exists():
            raise SystemExit(
                f"race comparators missing: {RACE_START_PATH}. Run --race-start FIRST."
            )
        s = json.loads(RACE_START_PATH.read_text())
        for c in RACE_COMPARATORS:  # as S6's own load_start validates its comparators
            e = (s.get("D3") or {}).get(c)
            if not e or "pursuer_violations" not in e or "pursuer" not in e:
                raise SystemExit(
                    f"D3/{c} missing from {RACE_START_PATH}; re-run --race-start."
                )
        return s

    S6.load_start = load_race_start


def _artefact_ckpt() -> pathlib.Path:
    round_path = OUT / "s7_round.json"
    if not round_path.exists():
        raise SystemExit(
            "s7_round.json missing: run --summarize first (stage 2 needs the stage-1 artefact)"
        )
    s = json.loads(round_path.read_text())["summary"]
    art = s["hypotheses"]["artefact"]
    if not art:
        raise SystemExit(
            "no S7 artefact: H3-2 did not pass; stage 2 does not run (spec section 5)."
        )
    return pathlib.Path(
        json.loads((OUT / f"s7_arm_{art}.json").read_text())["final"]["ckpt"]
    )


def stage1_before() -> dict | None:
    """Spec section 5's "before" number: the S7 artefact's deadlock-episode rate against
    the held-out attackers, taken from the stage-1 gate in `s7_round.json`."""
    path = OUT / "s7_round.json"
    if not path.exists():
        return None
    doc = json.loads(path.read_text())
    art = ((doc.get("summary") or {}).get("hypotheses") or {}).get("artefact")
    g = (((doc.get("arms") or {}).get(art) or {}).get("final") or {}).get("gate")
    if not art or not g:
        return None
    return {
        "artefact": art,
        "heldout_deadlock": {
            o: g[o]["pursuer"]["deadlock_episode_rate"] for o in HELD_OUT if o in g
        },
    }


def race_start(device: str) -> dict:
    from pe2.evade import play_scripted_evader

    S6 = _s6()
    race_setup(S6, _artefact_ckpt())
    if RACE_START_PATH.exists():
        raise SystemExit(f"{RACE_START_PATH} already exists (stop rule).")
    cfg = EnvConfig()
    f = S6.defender_factory("D3", device)
    out = {
        "gate_seed0": GATE2_SEED0,
        "gate_n": GATE_N,
        "artefact": str(S6.DEFENDERS["D3"]["ckpt"]),
        "provenance": provenance(),
        "D3": {
            "HUG7": S6._entry(
                play_scripted_evader(cfg, ObstacleHuggingEvader, f, GATE_N, GATE2_SEED0)
            ),
            "GREEDY": S6._entry(
                play_scripted_evader(cfg, GreedyEvader, f, GATE_N, GATE2_SEED0)
            ),
        },
    }
    for c, e in out["D3"].items():
        print(
            f"D3 {c:6s}: pursuer deadlock eps {e['pursuer']['deadlock_episode_rate']:.3f} "
            f"violations {e['pursuer_violations']} | escape {e['pursuer']['escape_rate']:.3f}",
            flush=True,
        )
    _write_json(RACE_START_PATH, out)
    return out


def race_arm(arm: str, device: str) -> dict:
    S6 = _s6()
    race_setup(S6, _artefact_ckpt())
    if arm not in S6.ARMS:
        raise SystemExit(f"unknown race arm {arm!r}; choose from {sorted(S6.ARMS)}")
    return S6.run_arm(arm, device)


def race_summary(
    arms: dict, race_start: dict | None = None, before: dict | None = None
) -> dict:
    inds = [
        arms[a]["final"]["deadlock_ind"] for a in sorted(arms) if "final" in arms[a]
    ]
    pooled = [x for ind in inds for x in ind]
    k, n = sum(map(bool, pooled)), len(pooled)
    lo, hi = wilson_ci(k, n) if n else (None, None)
    # Spec section 5 asks for EACH attacker's rate with a Wilson CI; amendment A2 keeps
    # H-RACE pooled and reports the per-arm intervals beside it.
    per = {}
    for a in sorted(arms):
        if "final" not in arms[a]:
            continue
        ind = arms[a]["final"]["deadlock_ind"]
        ka, na = sum(map(bool, ind)), len(ind)
        alo, ahi = wilson_ci(ka, na) if na else (None, None)
        per[a] = {
            "deadlock_rate": arms[a]["final"]["pursuer"]["deadlock_episode_rate"],
            "escape_rate": arms[a]["final"]["pursuer"]["escape_rate"],
            "pursuer_violations": arms[a]["final"]["pursuer_violations"],
            "wilson": [alo, ahi],
        }
    complete = len(inds) == len(SEEDS2)
    # H-SAFE census over stage 2 (S6 record shape: train / probe / gate). Every
    # ATK2 record counts, including a crash record with no "final".
    viol, offenders = 0, []
    for a in sorted(arms):
        r = arms[a]
        v = (
            int(r.get("train_pursuer_violations", 0))
            + int(r.get("probe_pursuer_violations", 0))
            + int((r.get("final") or {}).get("pursuer_violations", 0) or 0)
        )
        if "probe_pursuer_violations" not in r:
            v += int(
                sum(x.get("pursuer_violations", 0) for x in r.get("checkpoints", []))
            )
        viol += v
        if v:
            offenders.append(a)
    if race_start is not None:
        # Spec section 1 puts the comparators inside H-SAFE too: race_start runs the
        # frozen artefact, shielded, for GATE_N episodes against each of Hug7 and Greedy.
        for name, e in sorted((race_start.get("D3") or {}).items()):
            v = int((e or {}).get("pursuer_violations") or 0)
            viol += v
            if v:
                offenders.append(f"race_start:D3/{name}")
    return {
        "per_arm": per,
        "pooled_rate": (k / n) if n else None,
        "pooled_wilson": [lo, hi],
        "race_bar": RACE_BAR,
        "all_arms_below_bar": bool(per)
        and all(
            v["wilson"][1] is not None and v["wilson"][1] < RACE_BAR
            for v in per.values()
        ),
        "before": before,
        "pursuer_violation_steps": viol,
        "offenders": offenders,
        "H_RACE": (
            "INCOMPLETE" if not complete else ("HOLDS" if hi < RACE_BAR else "FAILS")
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--measure-start", action="store_true")
    p.add_argument("--arms", nargs="*", choices=list(ARMS), default=[])
    p.add_argument("--summarize", action="store_true")
    p.add_argument("--race-start", action="store_true")
    p.add_argument("--race-arms", nargs="*", default=[])
    p.add_argument("--race-summarize", action="store_true")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.preflight:
        preflight(args.device)
        return
    if args.measure_start:
        measure_start(args.device)
        return
    for arm in args.arms:
        run_arm(arm, args.device)
    if args.summarize:
        results, starts = load_results(), load_start()
        race = None
        if (OUT / "s7_race.json").exists():
            race = json.loads((OUT / "s7_race.json").read_text())
        summary = summarize(results, starts, race)
        _write_json(OUT / "s7_round.json", {"summary": summary, "arms": results})
        write_figure(results, starts, summary, OUT / "s7_round.png")
        print(
            json.dumps(
                {k: summary[k] for k in ("hsafe", "hypotheses", "rc19", "arms")},
                indent=2,
                default=str,
            )
        )
        print(f"\n{summary['direction']}")
    if args.race_start:
        race_start(args.device)
    for arm in args.race_arms:
        race_arm(arm, args.device)
    if args.race_summarize:
        arms = {
            p_.stem.removeprefix("s7race_arm_"): json.loads(p_.read_text())
            for p_ in sorted(OUT.glob("s7race_arm_*.json"))
        }
        rs = (
            json.loads(RACE_START_PATH.read_text())
            if RACE_START_PATH.exists()
            else None
        )
        r = race_summary(arms, rs, stage1_before())
        _write_json(OUT / "s7_race.json", r)
        print(json.dumps(r, indent=2, default=str))
        results, starts = load_results(), load_start()
        summary = summarize(results, starts, r)
        _write_json(OUT / "s7_round.json", {"summary": summary, "arms": results})
        write_figure(results, starts, summary, OUT / "s7_round.png")
        print(f"\n{summary['direction']}")


if __name__ == "__main__":
    main()
