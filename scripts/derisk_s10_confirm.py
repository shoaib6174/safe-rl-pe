"""S10 — confirmatory round for S8's existence proof: the β = 0.01 anchored fine-tune on
five fresh seeds (spec: docs/pathway/shield_remedy_3b_prereg_2026-09-17.md). Nothing here
may change a threshold, a seed, a stream, a rule or a budget after a number is seen.

Stage 1: five arms FT-B-{40..44} — each is EXACTLY S8's FT-A2 learner (RC configuration,
init RC19, anchor RC19, β = 0.01, S8's WEIGHTS and MIXTURE, 5 M env steps, probes every
500 k). Only the seeds are new. Gate on the fresh stream 1 250 000+ against S8's held-out
attackers ATT25/26/27 plus Greedy and Hug7, paired with RC19. Three references are measured
before any arm: RC19, RCM28 and FT-A2-35 — the last is S8's passing seed, reported as a
fourth row; it is not an arm and casts no vote.
H5 passes iff >= 3 of the 5 arms pass S8's unchanged two-sided per-arm test.
Stage 2 (only if H5 passes): ATK4-{45,46,47} fresh structured attackers vs the S10 artefact
(S6 machinery via S7/S8) on stream 1 270 000+ / 1 280 000+; H-RACE-4 vs RACE_BAR = 0.58,
with S7's pooled 0.147 reported beside it.

The whole round is S8's machinery under S10's constants: this module loads
`scripts/derisk_s8_remedy3.py` privately (importlib, not sys.modules) and re-points its
globals — so S8's own script and artefacts on disk are untouched.

V-A0 (the preflight) is NOT re-run. S8 verified on this exact learner, env and checkpoint
that `KLAnchoredPPO` at β = 0 is bit-identical to plain PPO; S10 changes no part of that
code path, so `PREFLIGHT_PATH` is deliberately left at S8's `s8_preflight.json` and
`require_preflight` (S8's function, re-exported below) checks that S8's V-A0 exists and
passed before any reference is measured.

Usage:
  ./venv/bin/python scripts/derisk_s10_confirm.py --measure-start   (RC19, RCM28, FT-A2-35)
  ./venv/bin/python scripts/derisk_s10_confirm.py --arms FT-B-40    (one per process)
  ./venv/bin/python scripts/derisk_s10_confirm.py --summarize
  ./venv/bin/python scripts/derisk_s10_confirm.py --race-start | --race-arms ATK4-45 |
                                                  --race-summarize
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib


def _s8():
    p = pathlib.Path(__file__).resolve().parent / "derisk_s8_remedy3.py"
    spec = importlib.util.spec_from_file_location("derisk_s8_remedy3", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


S8 = _s8()
OUT = S8.OUT
ARTEFACT_PREFIX = "s10"
START_PATH = OUT / f"{ARTEFACT_PREFIX}_start.json"
RACE_START_PATH = OUT / f"{ARTEFACT_PREFIX}_race_start.json"
# Deliberately S8's: see the module docstring (V-A0 is not re-run).
PREFLIGHT_PATH = S8.PREFLIGHT_PATH
require_preflight = S8.require_preflight

# --- pre-registered (spec sections 2-5). Do not tune. ---
SEEDS = (40, 41, 42, 43, 44)
FAMILY = "FT-B"
BETA = S8.BETAS["FT-A2"]  # 0.01 — spec section 2: the learner does not change
ARMS = {f"{FAMILY}-{s}": {"seed": s, "family": FAMILY, "beta": BETA} for s in SEEDS}
TOTAL = S8.TOTAL  # 5 M env steps, probes every 500 k, both exactly S8's
EVERY = S8.EVERY
PROBE_SEED0 = 1_260_000
GATE_SEED0 = 1_250_000
GATE_N = S8.GATE_N  # 500
EVAL_N = S8.EVAL_N  # 200
# Arms out of five that must PASS for H5 (spec section 3). NOT S8's own PASS_MIN,
# which is the per-arm count of held-out attackers and stays at 2 of 3.
PASS_MIN = 3
SEEDS2 = (45, 46, 47)
PROBE2_SEED0 = 1_270_000
GATE2_SEED0 = 1_280_000
RACE_BAR = S8.RACE_BAR  # 0.58
S7_RACE_RATE = S8.S7_RACE_RATE  # 0.147, D116
REPORT_ONLY = "FT-A2-35"  # S8's passing seed: a reference row, never an arm
REFERENCES = ("RC19", "RCM28", REPORT_ONLY)
REFERENCE_CKPT = dict(S8.REFERENCE_CKPT) | {
    REPORT_ONLY: OUT / f"ckpts_{REPORT_ONLY}" / f"{REPORT_ONLY}_{S8.TOTAL}.zip"
}


_S8_RACE_SETUP = S8.race_setup  # captured before `apply_overrides` replaces it


def race_setup(S6, artefact_ckpt) -> None:
    """S8's race wiring (which is S7's, with S8's knobs on top) plus the only
    stage-2 thing S10 changes: the arm names. Everything else S8's `race_setup`
    sets — OUT, the race-start file, the `s10race` prefix, the two fresh streams
    and SEEDS2 — is already this round's, because it reads the S8 globals that
    `apply_overrides` re-pointed."""
    _S8_RACE_SETUP(S6, artefact_ckpt)
    S6.ARMS = {
        f"ATK4-{s}": {
            "family": "CTL",
            "defender": "D3",
            "seed": s,
            "beta": 0.0,
            "zero_reward": False,
        }
        for s in SEEDS2
    }


def apply_overrides() -> None:
    """Point the reused S8 machinery at the S10 round.

    S8's functions read every constant as their OWN module globals at call time,
    and three of its paths (START_PATH, PREFLIGHT_PATH, RACE_START_PATH) were
    bound from its prefix at import — so setting the prefix alone would leave
    them on S8's files. START_PATH and RACE_START_PATH are therefore re-pointed
    explicitly here; PREFLIGHT_PATH deliberately is not (see the module
    docstring: V-A0 is S8's). `S8.apply_overrides()` is called
    LAST because it is what pushes OUT / GATE_SEED0 / GATE_N / EVAL_N and the
    stage-2 streams down onto the S7 module that actually gates and races; it
    must see S10's values, not S8's. `S8.race_setup` is replaced by this
    module's so that `S8.race_start` / `race_arm`, which call it by name, wire
    up the ATK4 arms.
    """
    S8.ARTEFACT_PREFIX = ARTEFACT_PREFIX
    S8.START_PATH = START_PATH
    S8.RACE_START_PATH = RACE_START_PATH
    S8.SEEDS = SEEDS
    S8.ARMS = ARMS
    S8.TOTAL = TOTAL
    S8.EVERY = EVERY
    S8.PROBE_SEED0 = PROBE_SEED0
    S8.GATE_SEED0 = GATE_SEED0
    S8.SEEDS2 = SEEDS2
    S8.PROBE2_SEED0 = PROBE2_SEED0
    S8.GATE2_SEED0 = GATE2_SEED0
    S8.REFERENCES = REFERENCES
    S8.REFERENCE_CKPT = REFERENCE_CKPT
    S8.race_setup = race_setup
    S8.apply_overrides()


apply_overrides()


# ----------------------------------------------------------------------------
# stage-1 verdicts (spec section 3)
# ----------------------------------------------------------------------------
def _fail_counts(per_arm: dict) -> dict:
    """Spec section 3's reported census of the two fail flavours. As in S8's
    `_fail_mode`, `FAIL (both)` counts towards both flavours (and is also
    reported on its own so the two counts can be read back)."""
    v = [per_arm[a]["verdict"] for a in ARMS]
    return {
        "still pinned": sum(x in ("FAIL (still pinned)", "FAIL (both)") for x in v),
        "capture lost": sum(x in ("FAIL (capture lost)", "FAIL (both)") for x in v),
        "both": sum(x == "FAIL (both)" for x in v),
    }


def h5_verdicts(results: dict, starts: dict) -> dict:
    """H5 (spec section 3): >= PASS_MIN of the five arms pass S8's per-arm test.

    The per-arm test is `S8._arm_verdict`, unchanged — paired deadlock tests vs
    RC19 on the held-out attackers plus the Greedy capture guard. Only the arms
    in ARMS are judged, so the FT-A2-35 reference row cannot vote or be chosen
    as the artefact even if a record for it is sitting in `results`. The
    artefact is named only when H5 passes: spec section 4 runs stage 2 only
    then, and `S8._artefact_ckpt` refuses without one.
    """
    per_arm: dict = {}
    for a in ARMS:
        r = results.get(a)
        if r is None or "final" not in r:
            per_arm[a] = {"verdict": "INCOMPLETE"}
            continue
        per_arm[a] = S8._arm_verdict(r["final"]["gate"], starts)
    if any(v["verdict"] == "INCOMPLETE" for v in per_arm.values()):
        return {
            "H5": "INCOMPLETE",
            "per_arm": per_arm,
            "artefact": None,
            "passing": None,
            "n_pass": None,
            "fail_counts": None,
        }
    passing = [a for a in ARMS if per_arm[a]["verdict"] == "PASS"]
    h5 = "PASS" if len(passing) >= PASS_MIN else "FAIL"
    artefact = (
        min(
            passing,
            # lowest mean held-out deadlock; ties -> highest Greedy capture; ties -> lowest seed
            key=lambda a: (
                per_arm[a]["mean_deadlock_heldout"],
                -per_arm[a]["greedy_capture"],
                ARMS[a]["seed"],
            ),
        )
        if h5 == "PASS"
        else None
    )
    return {
        "H5": h5,
        "per_arm": per_arm,
        "artefact": artefact,
        "passing": passing,
        "n_pass": len(passing),
        "fail_counts": _fail_counts(per_arm),
    }


def reference_row(starts: dict) -> dict | None:
    """Spec section 3, reported with no verdict: FT-A2-35 on THIS stream — does
    S8's passing seed reproduce out of sample?"""
    ref = starts.get(REPORT_ONLY)
    if not ref:
        return None
    held = {o: ref[o]["pursuer"]["deadlock_episode_rate"] for o in S8.HELD_OUT}
    return {
        "reference": REPORT_ONLY,
        "heldout_deadlock": held,
        "mean_deadlock_heldout": sum(held.values()) / len(held),
        "greedy_capture": ref["GREEDY"]["capture_rate"],
        "note": "S8's passing seed on the S10 gate stream; reported, casts no vote",
    }


def direction(h: dict) -> str:
    """Spec section 5, row by row."""
    if h["hsafe"] == "FAILS":
        return (
            "H-SAFE FAILS: a shielded pursuer recorded a contact step. Stop; census; no further "
            "round without a user ruling."
        )
    if h["hsafe"] == "INCOMPLETE" or h["H5"] == "INCOMPLETE":
        return "incomplete: all five S10 arms must run"
    if h["H5"] == "FAIL":
        if h["n_pass"]:
            return (
                f"H5 FAILS ({h['n_pass']}/5 pass): the configuration finds the policy sometimes; "
                "report the rate with the KL traces — selection across seeds (as S2's H-SEL) is "
                "the honest next question, not a re-tune."
            )
        return (
            "H5 FAILS (0/5): FT-A2-35 was a fluke; D119's reading is withdrawn to one "
            "observation."
        )
    passed = (
        f"H5 PASSES ({h['n_pass']}/5): the trade-off is not structural — an anchored fine-tune "
        "reliably yields a safe pursuer that resists the structured attack AND hunts. R3 becomes "
        "a method"
    )
    if h["H_RACE"] == "HOLDS":
        return (
            f"{passed}, and H-RACE-4 holds: fresh attackers do not pin the artefact, so the arms "
            "race converges without disengagement (S7's 0.147 beside it)."
        )
    if h["H_RACE"] == "FAILS":
        return (
            f"{passed}, but fresh attackers adapt (H-RACE-4 fails): the price of hunting is "
            "exploitability, reported as the trade-off curve."
        )
    if h["H_RACE"] == "INCOMPLETE":
        return f"{passed}; stage 2 incomplete: all three ATK4 arms must run"
    return (
        f"{passed}; stage 2 measures the race — run --race-start / --race-arms / "
        "--race-summarize."
    )


def summarize(results: dict, starts: dict, race: dict | None = None) -> dict:
    h = h5_verdicts(results, starts)
    hs = S8.hsafe_verdict(results, starts, race)
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
            for o in S8.GATE_OPPONENTS
        }
        for ref in REFERENCES
    }
    return {
        "spec": "docs/pathway/shield_remedy_3b_prereg_2026-09-17.md",
        "hsafe": hs,
        "hypotheses": h,
        "arms": table,
        "references": refs,
        "reference_ft_a2_35": reference_row(starts),
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
                "H5": h["H5"],
                "n_pass": h["n_pass"],
                "H_RACE": race["H_RACE"] if race else None,
            }
        ),
    }


def write_figure(results: dict, starts: dict, summary: dict, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    opps = list(S8.GATE_OPPONENTS)
    x = range(len(opps))
    series = [
        (r, summary["references"][r], c)
        for r, c in zip(REFERENCES, ("k", "0.55", "tab:orange"), strict=True)
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
    axes[0].set_title(f"S10 stage 1: H5 {summary['hypotheses']['H5']}")
    axes[0].legend(fontsize=7, ncol=2)
    axes[1].set_ylabel("capture rate (gate)")
    axes[1].set_title("capture (the second clause of the gate)")
    fig.suptitle(
        "S10 — β = 0.01 anchored fine-tune of RC19 on five fresh seeds"
        + ("" if summary["complete"] else " — INCOMPLETE")
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _write_round(results: dict, starts: dict, race: dict | None = None) -> dict:
    summary = summarize(results, starts, race)
    S8.S7._write_json(S8._round_file("json"), {"summary": summary, "arms": results})
    write_figure(results, starts, summary, S8._round_file("png"))
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--measure-start", action="store_true")
    p.add_argument("--arms", nargs="*", choices=list(ARMS), default=[])
    p.add_argument("--summarize", action="store_true")
    p.add_argument("--race-start", action="store_true")
    p.add_argument("--race-arms", nargs="*", default=[])
    p.add_argument("--race-summarize", action="store_true")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.measure_start:
        S8.measure_start(args.device)
        return
    for arm in args.arms:
        S8.run_arm(arm, args.device)
    if args.summarize:
        results, starts = S8.load_results(), S8.load_start()
        race = None
        if S8._race_file().exists():
            race = json.loads(S8._race_file().read_text())
        summary = _write_round(results, starts, race)
        print(
            json.dumps(
                {
                    k: summary[k]
                    for k in (
                        "hsafe",
                        "hypotheses",
                        "references",
                        "reference_ft_a2_35",
                        "arms",
                    )
                },
                indent=2,
                default=str,
            )
        )
        print(f"\n{summary['direction']}")
    if args.race_start:
        S8.race_start(args.device)
    for arm in args.race_arms:
        S8.race_arm(arm, args.device)
    if args.race_summarize:
        arms = {
            p_.stem.removeprefix(f"{ARTEFACT_PREFIX}race_arm_"): json.loads(
                p_.read_text()
            )
            for p_ in sorted(OUT.glob(S8._race_arm_glob()))
        }
        rs = (
            json.loads(RACE_START_PATH.read_text())
            if RACE_START_PATH.exists()
            else None
        )
        r = S8.race_summary(arms, rs, S8.stage1_before())
        S8.S7._write_json(S8._race_file(), r)
        print(json.dumps(r, indent=2, default=str))
        summary = _write_round(S8.load_results(), S8.load_start(), r)
        print(f"\n{summary['direction']}")


if __name__ == "__main__":
    main()
