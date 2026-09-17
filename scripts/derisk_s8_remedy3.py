"""S8 — Step 10 iteration 3: a capture-preserving best response to the structured attack
(spec: docs/pathway/shield_remedy_3_prereg_2026-09-17.md). Nothing here may change a threshold,
a seed, a stream, a rule or a budget after a number is seen.

Stage 1: seven arms on the WEIGHTED family (Greedy .50, Hug7 .20, SplitHug .10, CTL-D125/126/127
1/15 each) — primary FT-A1-{34,35,36} (KL anchor to RC19, beta 0.1), secondary FT-A2-{34,35,36}
(beta 0.01), control SCR-W34 (S7's from-scratch learner on the same weighted family). Gate vs the
held-out attackers CTL-D225/226/227, Greedy and Hug7, paired with BOTH references RC19 and RCM28.
H4 is two-sided: deadlock lower than RC19 on >= 2/3 held-out attackers AND Greedy capture not
significantly below RC19's.
Stage 2 (only if H4 passes): ATK3-{37,38,39} fresh structured attackers trained against the S8
artefact (S6 machinery via S7), gate on a fresh stream; H-RACE-3 vs RACE_BAR = 0.58, with S7's
pooled 0.147 reported beside it.

Usage:
  ./venv/bin/python scripts/derisk_s8_remedy3.py --preflight       (V-A0: beta=0 == plain PPO)
  ./venv/bin/python scripts/derisk_s8_remedy3.py --measure-start   (RC19 AND RCM28 on the gate stream)
  ./venv/bin/python scripts/derisk_s8_remedy3.py --arms FT-A1-34   (one per process)
  ./venv/bin/python scripts/derisk_s8_remedy3.py --summarize
  ./venv/bin/python scripts/derisk_s8_remedy3.py --race-start | --race-arms ATK3-37 | --race-summarize
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib

from pe2.config import EnvConfig
from pe2.derisk import gate_pass, paired_indicator_test
from pe2.scripted import GreedyEvader, ObstacleHuggingEvader
from pe2.structured import HOLD

OUT = pathlib.Path("results/derisk")
# Artefact namespace. The S10 runner reuses this module's machinery with
# ARTEFACT_PREFIX = "s10", so every artefact path the round writes or looks for
# must come from the prefix: through the helpers below, or — for the three
# constants S10 overrides directly — derived from it here at import.
ARTEFACT_PREFIX = "s8"
START_PATH = OUT / f"{ARTEFACT_PREFIX}_start.json"
PREFLIGHT_PATH = OUT / f"{ARTEFACT_PREFIX}_preflight.json"
RACE_START_PATH = OUT / f"{ARTEFACT_PREFIX}_race_start.json"

# --- pre-registered (spec sections 2-5). Do not tune. ---
SEEDS = (34, 35, 36)
BETAS = {"FT-A1": 0.1, "FT-A2": 0.01}
PRIMARY, SECONDARY, CONTROL = "FT-A1", "FT-A2", "SCR"
ARMS = {
    **{
        f"{fam}-{s}": {"seed": s, "family": fam, "beta": b}
        for fam, b in BETAS.items()
        for s in SEEDS
    },
    "SCR-W34": {"seed": 34, "family": CONTROL, "beta": None},
}
# Sampling weights, in MIXTURE order (spec section 3): Greedy .50, Hug7 .20,
# SplitHug .10, and the three structured CTL attackers 1/15 each (.20 total).
WEIGHTS = (0.50, 0.20, 0.10, 1 / 15, 1 / 15, 1 / 15)
TOTAL = 5_000_000
EVERY = 500_000
PROBE_SEED0 = 1_210_000
GATE_SEED0 = 1_200_000
GATE_N = 500
EVAL_N = 200
PASS_MIN = 2
P_GATE = 0.05
GAMMA = 0.99
PREFLIGHT_SEED = 34
PREFLIGHT_STEPS = 2048
SEEDS2 = (37, 38, 39)
PROBE2_SEED0 = 1_220_000
GATE2_SEED0 = 1_230_000
RACE_BAR = 0.58
# D116: the pooled deadlock rate fresh attackers reached against the S7 artefact.
S7_RACE_RATE = 0.147
REFERENCES = ("RC19", "RCM28")


# ----------------------------------------------------------------------------
# artefact paths (every one of them under ARTEFACT_PREFIX)
# ----------------------------------------------------------------------------
def _arm_file(arm: str) -> pathlib.Path:
    return OUT / f"{ARTEFACT_PREFIX}_arm_{arm}.json"


def _probe_file(arm: str) -> pathlib.Path:
    return OUT / f"{ARTEFACT_PREFIX}_probe_{arm}.json"


def _round_file(ext: str) -> pathlib.Path:
    return OUT / f"{ARTEFACT_PREFIX}_round.{ext}"


def _race_file() -> pathlib.Path:
    return OUT / f"{ARTEFACT_PREFIX}_race.json"


def _race_arm_glob() -> str:
    return f"{ARTEFACT_PREFIX}race_arm_*.json"


def _s7():
    p = pathlib.Path(__file__).resolve().parent / "derisk_s7_remedy2.py"
    spec = importlib.util.spec_from_file_location("derisk_s7_remedy2", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


S7 = _s7()
MIXTURE = S7.MIXTURE
HELD_OUT = S7.HELD_OUT
GATE_OPPONENTS = S7.GATE_OPPONENTS
RC19_CKPT = S7.RC19_CKPT
RCM28_CKPT = OUT / "ckpts_RCM28" / f"RCM28_{S7.TOTAL_LONG}.zip"

if len(WEIGHTS) != len(MIXTURE):
    raise AssertionError(f"{len(WEIGHTS)} weights for {len(MIXTURE)} mixture members")


def apply_overrides() -> None:
    """Point the reused S7 machinery at the S8 round.

    `S7.gate_pursuer` reads GATE_SEED0 and GATE_N as its OWN module globals, and
    `S7.race_setup` reads OUT / RACE_START_PATH / PROBE2_SEED0 / GATE2_SEED0 /
    SEEDS2 the same way, so an S7 module reused unaltered would gate the S8 arms
    on S7's consumed stream (1 170 000+) and write into S7's artefacts. Every
    such global is re-pointed here; `race_setup` below re-asserts the stage-2
    ones on the S6 module after S7's `race_setup` has run.

    The S7 module object is private to this process (importlib, not sys.modules):
    `scripts/derisk_s7_remedy2.py` on disk and its own round are untouched.
    """
    S7.OUT = OUT
    S7.GATE_SEED0 = GATE_SEED0
    S7.GATE_N = GATE_N
    S7.EVAL_N = EVAL_N
    S7.RACE_START_PATH = RACE_START_PATH
    S7.PROBE2_SEED0 = PROBE2_SEED0
    S7.GATE2_SEED0 = GATE2_SEED0
    S7.SEEDS2 = SEEDS2
    S7.RACE_BAR = RACE_BAR


apply_overrides()


# ----------------------------------------------------------------------------
# env, opponents, learner
# ----------------------------------------------------------------------------
def mixture_factory(seed: int, device: str):
    """S7's family in MIXTURE order, sampled with the spec's task weights."""
    from pe2.evade import MixtureOpponentFactory

    members = [GreedyEvader, ObstacleHuggingEvader, S7.splithug_factory(device)] + [
        S7.structured_evader_factory(S7.ATT_CKPT[k], device)
        for k in ("CTL-D125", "CTL-D126", "CTL-D127")
    ]
    return MixtureOpponentFactory(members, seed=seed, weights=WEIGHTS)


def make_env(seed: int, device: str):
    """S5's RC configuration (D113) on the weighted family — unchanged by spec
    section 3: shield on, shield_obs, stall penalty 1e-3, no acquisition bonus."""
    from pe2.shield import CBFShield
    from pe2.structured import SearchMDP

    cfg = EnvConfig()
    return SearchMDP(
        cfg,
        mixture_factory(seed, device),
        seed0=seed * 1000,
        hold=HOLD,
        r_acq=S7.RC_SETTINGS["r_acq"],
        zero_reward=False,
        shield=CBFShield(cfg),
        shield_obs=S7.RC_SETTINGS["shield_obs"],
        stall_penalty=S7.RC_SETTINGS["stall_penalty"],
    )


def make_learner(arm: str, env, device: str):
    """Control: S7's learner from scratch. Fine-tunes: RC19 reloaded as a
    `KLAnchoredPPO` whose anchor is RC19 itself, set BEFORE any step. RC19's
    saved gamma is already 0.99, so `load` restores the spec's discount."""
    from stable_baselines3 import PPO

    spec = ARMS[arm]
    seed = spec["seed"]
    if spec["family"] == CONTROL:
        return PPO("MlpPolicy", env, seed=seed, gamma=GAMMA, verbose=0, device=device)
    from pe2.anchored_ppo import KLAnchoredPPO

    model = KLAnchoredPPO.load(RC19_CKPT, env=env, seed=seed, device=device)
    model.beta = spec["beta"]
    model.set_anchor(model.policy)
    return model


# ----------------------------------------------------------------------------
# guards
# ----------------------------------------------------------------------------
def require_preflight() -> None:
    if not PREFLIGHT_PATH.exists():
        raise SystemExit(
            f"preflight missing: {PREFLIGHT_PATH}. Run --preflight FIRST (spec V-A0)."
        )
    if not json.loads(PREFLIGHT_PATH.read_text()).get("pass"):
        raise SystemExit("V-A0 preflight FAILED; the round may not proceed.")


def require_not_measured() -> None:
    if START_PATH.exists():
        raise SystemExit(f"{START_PATH} already exists (stop rule).")
    if any(OUT.glob(f"{ARTEFACT_PREFIX}_arm_*.json")) or any(
        (OUT / f"ckpts_{a}").exists() for a in ARMS
    ):
        raise SystemExit(
            "S8 arms have already run; the stop rule forbids re-measuring."
        )


def require_arm_not_run(arm: str) -> None:
    if _arm_file(arm).exists() or (OUT / f"ckpts_{arm}").exists():
        raise SystemExit(
            f"arm {arm} has already run; the stop rule forbids re-running it."
        )


def load_start() -> dict:
    """Spec section 3: an arm refuses to run without BOTH references."""
    if not START_PATH.exists():
        raise SystemExit(f"reference missing: {START_PATH}. Run --measure-start FIRST.")
    s = json.loads(START_PATH.read_text())
    for ref in REFERENCES:
        for o in GATE_OPPONENTS:
            if o not in s.get(ref, {}) or "deadlock_ind" not in s[ref][o]:
                raise SystemExit(
                    f"{ref}/{o} missing from {START_PATH}; re-run --measure-start."
                )
    return s


# ----------------------------------------------------------------------------
# preflight (V-A0) and references
# ----------------------------------------------------------------------------
def preflight(device: str) -> dict:
    """V-A0 (spec section 2): on THIS env, `KLAnchoredPPO.load(RC19)` with
    `beta = 0` must be bit-identical to `PPO.load(RC19)` at the same seed —
    `num_timesteps`, the rollout buffer's returns and every policy parameter.

    One `learn(2048)` is exactly one PPO rollout plus one `train()` call, which
    is where the anchor code lives; with hold = 10 those 2 048 learner decisions
    cover far more than the spec's 4 096 env steps (the report records how many).
    A mismatch halts the round.
    """
    import torch
    from stable_baselines3 import PPO

    from pe2.anchored_ppo import KLAnchoredPPO

    # Both envs are built up front, by the same call, so neither learner can see
    # a different opponent stream: MixtureOpponentFactory's RNG is per env.
    env_a = make_env(PREFLIGHT_SEED, device)
    env_b = make_env(PREFLIGHT_SEED, device)

    anchored = KLAnchoredPPO.load(
        RC19_CKPT, env=env_a, seed=PREFLIGHT_SEED, device=device
    )
    anchored.beta = 0.0
    anchored.set_anchor(anchored.policy)
    anchored.learn(total_timesteps=PREFLIGHT_STEPS)

    plain = PPO.load(RC19_CKPT, env=env_b, seed=PREFLIGHT_SEED, device=device)
    plain.learn(total_timesteps=PREFLIGHT_STEPS)

    mismatched: list[dict] = []
    if anchored.num_timesteps != plain.num_timesteps:
        mismatched.append(
            {
                "what": "num_timesteps",
                "anchored": int(anchored.num_timesteps),
                "plain": int(plain.num_timesteps),
            }
        )
    if env_a.env_steps != env_b.env_steps:
        mismatched.append(
            {
                "what": "env_steps",
                "anchored": int(env_a.env_steps),
                "plain": int(env_b.env_steps),
            }
        )
    ret_a = torch.as_tensor(anchored.rollout_buffer.returns)
    ret_b = torch.as_tensor(plain.rollout_buffer.returns)
    if not torch.equal(ret_a, ret_b):
        mismatched.append(
            {
                "what": "rollout_returns",
                "max_abs_diff": float((ret_a - ret_b).abs().max()),
            }
        )
    sd_a, sd_b = anchored.policy.state_dict(), plain.policy.state_dict()
    if sd_a.keys() != sd_b.keys():
        mismatched.append(
            {
                "what": "state_dict_keys",
                "anchored_only": sorted(set(sd_a) - set(sd_b)),
                "plain_only": sorted(set(sd_b) - set(sd_a)),
            }
        )
    for k in sd_a:
        if k in sd_b and not torch.equal(sd_a[k], sd_b[k]):
            mismatched.append(
                {
                    "what": f"policy.{k}",
                    "max_abs_diff": float((sd_a[k] - sd_b[k]).abs().max()),
                }
            )
    out = {
        "pass": not mismatched,
        "seed": PREFLIGHT_SEED,
        "steps": PREFLIGHT_STEPS,
        "env_steps": int(env_a.env_steps),
        "num_timesteps": int(anchored.num_timesteps),
        "n_params": len(sd_a),
        "anchor_kl": anchored.last_anchor_kl,
        "mismatched": mismatched,
        "provenance": S7.provenance(),
    }
    S7._write_json(PREFLIGHT_PATH, out)
    print(
        f"V-A0 {'PASS' if out['pass'] else 'FAIL'}: {len(mismatched)} mismatches over "
        f"{out['n_params']} policy tensors + num_timesteps + rollout returns "
        f"({out['num_timesteps']} decisions, {out['env_steps']} env steps, "
        f"measured anchor KL {out['anchor_kl']})",
        flush=True,
    )
    if mismatched:
        raise SystemExit(
            "V-A0 FAILED: the anchored learner at beta=0 is not plain PPO; halting."
        )
    return out


def measure_start(device: str) -> dict:
    require_preflight()
    require_not_measured()
    opps = {name: S7.gate_opponent_factory(name, device) for name in GATE_OPPONENTS}
    out: dict = {
        "gate_seed0": GATE_SEED0,
        "gate_n": GATE_N,
        "provenance": S7.provenance(),
    }
    for ref, ckpt in (("RC19", RC19_CKPT), ("RCM28", RCM28_CKPT)):
        print(f"{ref} reference on the gate stream:", flush=True)
        out[ref] = S7.gate_pursuer(S7._model(ckpt, device), opps, device)
    S7._write_json(START_PATH, out)
    return out


# ----------------------------------------------------------------------------
# stage-1 arms
# ----------------------------------------------------------------------------
def _anchor_kl_callback(rows: list, trace: list):
    """Stamps the learner's last measured `anchor_kl` onto the probe grid.

    The anchored arms report `train/anchor_kl` on their own schedule and SB3's
    logger clears it; `KLAnchoredPPO` therefore also exports `last_anchor_kl`,
    which this callback samples whenever the probe has just written a row. Runs
    AFTER the probe's callback in the list, so `rows` is already extended.
    `None` means "not measured" — the from-scratch control has no anchor.
    """
    from stable_baselines3.common.callbacks import BaseCallback

    class _KL(BaseCallback):
        def _on_step(self) -> bool:
            while len(trace) < len(rows):
                row = rows[len(trace)]
                trace.append(
                    {
                        "steps": row["steps"],
                        "mark": row["mark"],
                        "anchor_kl": getattr(self.model, "last_anchor_kl", None),
                    }
                )
            return True

    return _KL()


def _logs(ind: list) -> list[dict]:
    """A capture indicator as the episode-log records `gate_pass` consumes."""
    return [
        {"seed": i, "role": "pursuer", "outcome": "capture" if c else "timeout"}
        for i, c in enumerate(ind)
    ]


def _capture_guard(gate: dict, starts: dict) -> dict:
    """Spec section 4: McNemar on the Greedy capture indicator, S8 vs RC19. The
    guard FAILS iff RC19 is significantly better (p < 0.05, b > c its way)."""
    cap = gate_pass(
        _logs(starts["RC19"]["GREEDY"]["capture_ind"]),
        _logs(gate["GREEDY"]["capture_ind"]),
        "pursuer",
        P_GATE,
    )
    return {
        "failed": bool(cap["pass"]),
        "rc19_better_p": cap["p"],
        "b": cap["b"],
        "c": cap["c"],
        "s8_capture": gate["GREEDY"]["capture_rate"],
        "rc19_capture": starts["RC19"]["GREEDY"]["capture_rate"],
    }


def _vs_rcm28(gate: dict, starts: dict) -> dict:
    """Spec section 4, reported with no verdict: the arm against the disengaged
    reference on the same stream."""
    ref = starts.get("RCM28") or {}
    return {
        "greedy_capture": gate["GREEDY"]["capture_rate"],
        "rcm28_greedy_capture": (ref.get("GREEDY") or {}).get("capture_rate"),
        "heldout_deadlock": {
            o: gate[o]["pursuer"]["deadlock_episode_rate"] for o in HELD_OUT
        },
        "rcm28_heldout_deadlock": {
            o: ref[o]["pursuer"]["deadlock_episode_rate"] for o in HELD_OUT if o in ref
        },
        "deadlock_tests": {
            o: paired_indicator_test(
                ref[o]["deadlock_ind"], gate[o]["deadlock_ind"], P_GATE
            )
            for o in HELD_OUT
            if o in ref
        },
    }


def _arm_verdict(gate: dict, starts: dict) -> dict:
    """H4(seed) (spec section 4): both clauses, priced."""
    tests = {
        o: paired_indicator_test(
            starts["RC19"][o]["deadlock_ind"], gate[o]["deadlock_ind"], P_GATE
        )
        for o in HELD_OUT
    }
    n_pass = sum(v["pass"] for v in tests.values())
    guard = _capture_guard(gate, starts)
    deadlock_ok = n_pass >= PASS_MIN
    if deadlock_ok and not guard["failed"]:
        verdict = "PASS"
    elif deadlock_ok:
        verdict = "FAIL (capture lost)"
    elif not guard["failed"]:
        verdict = "FAIL (still pinned)"
    else:
        verdict = "FAIL (both)"
    return {
        "verdict": verdict,
        "passes": n_pass,
        "tests": tests,
        "capture_guard": guard,
        "mean_deadlock_heldout": sum(
            gate[o]["pursuer"]["deadlock_episode_rate"] for o in HELD_OUT
        )
        / len(HELD_OUT),
        "greedy_capture": gate["GREEDY"]["capture_rate"],
        "vs_rcm28": _vs_rcm28(gate, starts),
    }


def run_train_arm(
    arm: str, starts: dict, device: str, holder: dict | None = None
) -> dict:
    from stable_baselines3 import PPO

    spec = ARMS[arm]
    seed = spec["seed"]
    cfg = EnvConfig()
    d = OUT / f"ckpts_{arm}"
    d.mkdir(parents=True, exist_ok=True)
    probe = S7.ShieldedStructuredProbe(
        cfg,
        GreedyEvader,
        every=EVERY,
        total=TOTAL,
        eval_n=EVAL_N,
        seed0=PROBE_SEED0,
        ckpt_dir=d,
        report_path=_probe_file(arm),
        label=arm,
    )
    trace: list = []
    if holder is not None:
        holder["rows"] = probe.rows
        holder["anchor_kl_trace"] = trace
    env = make_env(seed, device)
    if holder is not None:
        holder["env"] = env
    model = make_learner(arm, env, device)
    model.learn(
        total_timesteps=TOTAL,
        callback=[probe.callback, _anchor_kl_callback(probe.rows, trace)],
    )
    result = {
        "arm": arm,
        **spec,
        "settings": S7.RC_SETTINGS,
        "mixture": list(MIXTURE),
        "weights": list(WEIGHTS),
        "env_steps": int(env.env_steps),
        "decisions": int(model.num_timesteps),
        "train_violations": int(env.violations_total),
        "probe_violations": int(sum(r.get("violations", 0) for r in probe.rows)),
        "mixture_draws": env.inner.opponent_factory.draws[:2000],
        "anchor_kl_trace": trace,
        "checkpoints": probe.rows,
        "provenance": S7.provenance(),
    }
    ckpt = d / f"{arm}_{TOTAL}.zip"
    if not any(r["mark"] == TOTAL for r in probe.rows) or not ckpt.exists():
        result["verdict"], result["reason"] = (
            "UNDETERMINED",
            f"final probe at mark {TOTAL} never fired (env steps {env.env_steps})",
        )
        return result
    opps = {name: S7.gate_opponent_factory(name, device) for name in GATE_OPPONENTS}
    print(f"[{arm}] gate:", flush=True)
    g = S7.gate_pursuer(PPO.load(ckpt, device=device), opps, device, holder)
    result["final"] = {
        "ckpt": str(ckpt),
        "gate": g,
        "pursuer_violations": int(sum(v["pursuer_violations"] for v in g.values())),
    }
    v = _arm_verdict(g, starts)
    result["verdict"] = v["verdict"]
    result["reason"] = (
        "; ".join(
            f"{o}: RC19 {v['tests'][o]['rate_a']:.3f} vs S8 {v['tests'][o]['rate_b']:.3f} "
            f"p={v['tests'][o]['p']:.1e} {'PASS' if v['tests'][o]['pass'] else 'fail'}"
            for o in HELD_OUT
        )
        + f"; deadlock {v['passes']}/3; capture vs Greedy {v['greedy_capture']:.3f} "
        f"(RC19 {v['capture_guard']['rc19_capture']:.3f}, guard "
        f"{'FAILED' if v['capture_guard']['failed'] else 'ok'} p="
        f"{v['capture_guard']['rc19_better_p']:.1e}); violations train "
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
        S7._write_json(
            _arm_file(arm),
            {
                "arm": arm,
                **ARMS[arm],
                "verdict": "UNDETERMINED",
                "reason": f"crashed: {type(e).__name__}: {e}",
                "checkpoints": holder.get("rows", []),
                "anchor_kl_trace": holder.get("anchor_kl_trace", []),
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
                "provenance": S7.provenance(),
            },
        )
        raise
    S7._write_json(_arm_file(arm), result)
    print(f"[{arm}] {result['verdict']}: {result['reason']}", flush=True)
    return result


# ----------------------------------------------------------------------------
# stage-1 verdicts (spec section 4)
# ----------------------------------------------------------------------------
def _fail_mode(per_arm: dict) -> str:
    """Which of spec section 6's two FAIL rows the PRIMARY family landed on.
    `FAIL (both)` counts towards both flavours; a tie is `mixed`."""
    flavours = [per_arm[a]["verdict"] for a in ARMS if ARMS[a]["family"] == PRIMARY]
    lost = sum(v in ("FAIL (capture lost)", "FAIL (both)") for v in flavours)
    pinned = sum(v in ("FAIL (still pinned)", "FAIL (both)") for v in flavours)
    if lost > pinned:
        return "capture lost"
    if pinned > lost:
        return "still pinned"
    return "mixed"


def h4_verdicts(results: dict, starts: dict) -> dict:
    per_arm: dict = {}
    for a in ARMS:
        r = results.get(a)
        if r is None or "final" not in r:
            per_arm[a] = {"verdict": "INCOMPLETE"}
            continue
        per_arm[a] = _arm_verdict(r["final"]["gate"], starts)
    if any(v["verdict"] == "INCOMPLETE" for v in per_arm.values()):
        return {
            "H4": "INCOMPLETE",
            "per_arm": per_arm,
            "artefact": None,
            "fail_mode": None,
            "passing": None,
        }
    passing = {
        fam: [
            a
            for a in ARMS
            if ARMS[a]["family"] == fam and per_arm[a]["verdict"] == "PASS"
        ]
        for fam in (PRIMARY, SECONDARY)
    }
    if len(passing[PRIMARY]) >= PASS_MIN:
        h4, family = "PASS", PRIMARY
    elif len(passing[SECONDARY]) >= PASS_MIN:
        h4, family = "PASS (β = 0.01)", SECONDARY
    else:
        h4, family = "FAIL", None
    artefact = (
        min(
            passing[family],
            # lowest mean held-out deadlock; ties -> highest Greedy capture; ties -> lowest seed
            key=lambda a: (
                per_arm[a]["mean_deadlock_heldout"],
                -per_arm[a]["greedy_capture"],
                ARMS[a]["seed"],
            ),
        )
        if family
        else None
    )
    return {
        "H4": h4,
        "per_arm": per_arm,
        "artefact": artefact,
        "fail_mode": _fail_mode(per_arm) if h4 == "FAIL" else None,
        "passing": passing,
    }


def hsafe_verdict(
    results: dict, starts: dict | None = None, race: dict | None = None
) -> dict:
    """H-SAFE (spec section 2): 0 pursuer contact steps anywhere in the round —
    every arm's training, probes and gate, BOTH references, and stage 2."""
    steps, offenders = 0, []
    missing = [a for a in ARMS if a not in results or "final" not in results[a]]
    for a in sorted(results):
        r = results[a]
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
        for ref in REFERENCES:
            for o in GATE_OPPONENTS:
                v = int(
                    ((starts.get(ref) or {}).get(o) or {}).get("pursuer_violations")
                    or 0
                )
                steps += v
                if v:
                    offenders.append(f"reference:{ref}/{o}")
    if race is not None:
        # race_summary has already pooled each ATK3 arm's train + probe + gate
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
    """Spec section 6, row by row."""
    if h["hsafe"] == "FAILS":
        return (
            "H-SAFE FAILS: a shielded pursuer recorded a contact step. Stop; census; no further "
            "round without a user ruling."
        )
    if h["hsafe"] == "INCOMPLETE" or h["H4"] == "INCOMPLETE":
        return "incomplete: all seven S8 arms must run"
    if h["H4"] == "FAIL":
        mode = h.get("fail_mode")
        if mode == "capture lost":
            return (
                "H4 FAILS (deadlock ok, capture lost): the anchor did not protect capture; the "
                "trade-off is structural at this budget — RCM28 (disengaged) and RC19 (huntable) "
                "are the two ends and no anchored path joins them."
            )
        if mode == "still pinned":
            return (
                "H4 FAILS (capture kept, still pinned): the anchor held too tightly; report with "
                "the KL trace; beta is not re-tuned (stop rule)."
            )
        return (
            "H4 FAILS on both clauses: the primary seeds split between losing capture and staying "
            "pinned — report both flavours with the KL trace; beta is not re-tuned (stop rule)."
        )
    if h.get("H_RACE") == "HOLDS":
        return (
            "H4 PASSES and H-RACE-3 holds: a safe learned pursuer that resists the structured attack "
            "AND keeps its capture exists; the arms race converges without disengagement. The paper's "
            "R3 becomes a method."
        )
    if h.get("H_RACE") == "INCOMPLETE":
        return "H4 PASSES; stage 2 incomplete: all three ATK3 arms must run"
    if h.get("H_RACE") == "FAILS":
        return (
            "H4 PASSES but fresh attackers adapt (H-RACE-3 fails): capture kept, old attackers "
            "resisted, fresh ones adapt — the price of hunting is exploitability, reported as the "
            "trade-off curve."
        )
    return (
        "H4 PASSES: stage 2 (fresh attackers vs the artefact) decides whether the race converges — "
        "run --race-start / --race-arms / --race-summarize."
    )


def load_results() -> dict:
    return {
        p.stem.removeprefix(f"{ARTEFACT_PREFIX}_arm_"): json.loads(p.read_text())
        for p in sorted(OUT.glob(f"{ARTEFACT_PREFIX}_arm_*.json"))
    }


def summarize(results: dict, starts: dict, race: dict | None = None) -> dict:
    h = h4_verdicts(results, starts)
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
    refs = {
        ref: {
            o: {
                "deadlock": starts[ref][o]["pursuer"]["deadlock_episode_rate"],
                "capture": starts[ref][o]["capture_rate"],
            }
            for o in GATE_OPPONENTS
        }
        for ref in REFERENCES
    }
    return {
        "spec": "docs/pathway/shield_remedy_3_prereg_2026-09-17.md",
        "hsafe": hs,
        "hypotheses": h,
        "arms": table,
        "references": refs,
        "anchor_kl": {
            a: r["anchor_kl_trace"]
            for a, r in results.items()
            if r.get("anchor_kl_trace")
        },
        "race": race,
        "complete": all(a in results and "final" in results[a] for a in ARMS),
        "direction": direction(
            {
                "hsafe": hs["verdict"],
                "H4": h["H4"],
                "fail_mode": h.get("fail_mode"),
                "H_RACE": race["H_RACE"] if race else None,
            }
        ),
    }


def write_figure(results: dict, starts: dict, summary: dict, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    opps = list(GATE_OPPONENTS)
    x = range(len(opps))
    series = [
        (r, summary["references"][r], c)
        for r, c in zip(REFERENCES, ("k", "0.55"), strict=True)
    ]
    series += [(a, summary["arms"][a], None) for a in ARMS if a in summary["arms"]]
    w = 0.84 / len(series)
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    for ax, key in ((axes[0], "deadlock"), (axes[1], "capture")):
        for j, (name, d, col) in enumerate(series):
            ax.bar(
                [i - 0.42 + w * (j + 0.5) for i in x],
                [d.get(o, {}).get(key, 0) for o in opps],
                width=w,
                color=col,
                label=name,
            )
        ax.set_xticks(list(x))
        ax.set_xticklabels(opps)
    axes[0].set_ylabel("pursuer deadlock-episode rate (gate)")
    axes[0].set_title(f"S8 stage 1: H4 {summary['hypotheses']['H4']}")
    axes[0].legend(fontsize=7, ncol=2)
    axes[1].set_ylabel("capture rate (gate)")
    axes[1].set_title("capture (the second, new clause of the gate)")
    fig.suptitle(
        "S8 — anchored fine-tune of RC19 against the weighted family"
        + ("" if summary["complete"] else " — INCOMPLETE")
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ----------------------------------------------------------------------------
# stage 2: the race (S6 machinery, via S7)
# ----------------------------------------------------------------------------
def race_setup(S6, artefact_ckpt) -> None:
    """S7's race wiring (D3 = the artefact, shielded, shield_obs) with the
    S8-specific knobs re-asserted on top: S7's `race_setup` reads its OWN
    globals, so the prefix, the two fresh streams, the seeds, the arm names and
    the race-start file are set here after it has run."""
    S7.race_setup(S6, artefact_ckpt)
    S6.OUT = OUT
    S6.START_PATH = RACE_START_PATH
    S6.ARTEFACT_PREFIX = f"{ARTEFACT_PREFIX}race"
    S6.PROBE_SEED0 = PROBE2_SEED0
    S6.GATE_SEED0 = GATE2_SEED0
    S6.SEEDS = SEEDS2
    S6.ARMS = {
        f"ATK3-{s}": {
            "family": "CTL",
            "defender": "D3",
            "seed": s,
            "beta": 0.0,
            "zero_reward": False,
        }
        for s in SEEDS2
    }

    def load_race_start() -> dict:
        if not RACE_START_PATH.exists():
            raise SystemExit(
                f"race comparators missing: {RACE_START_PATH}. Run --race-start FIRST."
            )
        s = json.loads(RACE_START_PATH.read_text())
        for (
            c
        ) in S7.RACE_COMPARATORS:  # as S6's own load_start validates its comparators
            e = (s.get("D3") or {}).get(c)
            if not e or "pursuer_violations" not in e or "pursuer" not in e:
                raise SystemExit(
                    f"D3/{c} missing from {RACE_START_PATH}; re-run --race-start."
                )
        return s

    S6.load_start = load_race_start


def _artefact_ckpt() -> pathlib.Path:
    round_path = _round_file("json")
    if not round_path.exists():
        raise SystemExit(
            f"{round_path.name} missing: run --summarize first "
            "(stage 2 needs the stage-1 artefact)"
        )
    art = json.loads(round_path.read_text())["summary"]["hypotheses"]["artefact"]
    if not art:
        raise SystemExit(
            "no S8 artefact: H4 did not pass; stage 2 does not run (spec section 5)."
        )
    return pathlib.Path(json.loads(_arm_file(art).read_text())["final"]["ckpt"])


def stage1_before() -> dict | None:
    """Spec section 5's "before" number: the S8 artefact's deadlock-episode rate
    against the held-out attackers, taken from the stage-1 gate."""
    path = _round_file("json")
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

    S6 = S7._s6()
    race_setup(S6, _artefact_ckpt())
    if RACE_START_PATH.exists():
        raise SystemExit(f"{RACE_START_PATH} already exists (stop rule).")
    cfg = EnvConfig()
    f = S6.defender_factory("D3", device)
    out = {
        "gate_seed0": GATE2_SEED0,
        "gate_n": GATE_N,
        "artefact": str(S6.DEFENDERS["D3"]["ckpt"]),
        "provenance": S7.provenance(),
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
    S7._write_json(RACE_START_PATH, out)
    return out


def race_arm(arm: str, device: str) -> dict:
    S6 = S7._s6()
    race_setup(S6, _artefact_ckpt())
    if arm not in S6.ARMS:
        raise SystemExit(f"unknown race arm {arm!r}; choose from {sorted(S6.ARMS)}")
    return S6.run_arm(arm, device)


def race_summary(
    arms: dict, race_start: dict | None = None, before: dict | None = None
) -> dict:
    """S7's pooled H-RACE with spec section 5's extra comparison point: S7's own
    fresh-attacker rate (D116), reported, never a pass condition."""
    r = S7.race_summary(arms, race_start, before)
    r["s7_race_rate"] = S7_RACE_RATE
    r["below_s7"] = bool(
        r["pooled_rate"] is not None and r["pooled_rate"] < S7_RACE_RATE
    )
    return r


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
        if _race_file().exists():
            race = json.loads(_race_file().read_text())
        summary = summarize(results, starts, race)
        S7._write_json(_round_file("json"), {"summary": summary, "arms": results})
        write_figure(results, starts, summary, _round_file("png"))
        print(
            json.dumps(
                {k: summary[k] for k in ("hsafe", "hypotheses", "references", "arms")},
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
            p_.stem.removeprefix(f"{ARTEFACT_PREFIX}race_arm_"): json.loads(
                p_.read_text()
            )
            for p_ in sorted(OUT.glob(_race_arm_glob()))
        }
        rs = (
            json.loads(RACE_START_PATH.read_text())
            if RACE_START_PATH.exists()
            else None
        )
        r = race_summary(arms, rs, stage1_before())
        S7._write_json(_race_file(), r)
        print(json.dumps(r, indent=2, default=str))
        results, starts = load_results(), load_start()
        summary = summarize(results, starts, r)
        S7._write_json(_round_file("json"), {"summary": summary, "arms": results})
        write_figure(results, starts, summary, _round_file("png"))
        print(f"\n{summary['direction']}")


if __name__ == "__main__":
    main()
