"""S12 — the selection round: "train six, select one, confirm" (spec:
docs/pathway/selection_round_prereg_2026-09-18.md). Nothing here may change a
threshold, a seed, a stream, a rule or a budget after a number is seen.

Twelve arms in two independent batches — `SEL-A-{48..53}` and `SEL-B-{54..59}` —
each EXACTLY S10's arm (RC configuration, init RC19, anchor RC19, β = 0.01, S8's
WEIGHTS and MIXTURE, 5 M env steps, probes every 500 k on 1 320 000+). Only the
seeds and the streams are new.

Two streams, and they never mix:
  * SELECTION, 1 290 000+ — every arm's gate, paired with RC19 measured on the
    same stream before any arm. This is what `select` reads.
  * CONFIRMATION, 1 300 000+ — RC19 and FT-B-41 (S10's artefact, reference only)
    plus the two selected policies. This is what `h6_verdicts` reads. Nothing
    selected on the selection stream is ever judged on it.

H6 passes iff BOTH selected policies pass S8's unchanged two-sided per-arm test
on the confirmation stream. Fresh attackers are not trained (spec section 2):
this round has no stage 2.

The whole round is S8's machinery under S12's constants: this module loads
`scripts/derisk_s8_remedy3.py` privately (importlib, not sys.modules) and
re-points its globals — so S8's own script and artefacts on disk are untouched.
S10 does the same; the only structural difference is the second stream, which
`_confirmation_stream` lends to S7's `gate_pursuer` and always takes back.

V-A0 (the preflight) is NOT re-run, for S10's reason: S8 verified on this exact
learner, env and checkpoint that `KLAnchoredPPO` at β = 0 is bit-identical to
plain PPO, and S12 changes no part of that code path. `PREFLIGHT_PATH` is
therefore deliberately left at S8's `s8_preflight.json`.

Usage:
  ./venv/bin/python scripts/derisk_s12_select.py --measure-start   (RC19, selection stream)
  ./venv/bin/python scripts/derisk_s12_select.py --arms SEL-A-48   (one per process)
  ./venv/bin/python scripts/derisk_s12_select.py --select
  ./venv/bin/python scripts/derisk_s12_select.py --confirm-start   (RC19, FT-B-41)
  ./venv/bin/python scripts/derisk_s12_select.py --confirm
  ./venv/bin/python scripts/derisk_s12_select.py --summarize
"""

from __future__ import annotations

import argparse
import contextlib
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
ARTEFACT_PREFIX = "s12"
START_PATH = OUT / f"{ARTEFACT_PREFIX}_start.json"
SELECTION_PATH = OUT / f"{ARTEFACT_PREFIX}_selection.json"
CONFIRM_START_PATH = OUT / f"{ARTEFACT_PREFIX}_confirm_start.json"
# S12 has no stage 2 (spec section 2). The race-start path is still re-pointed
# into this round's namespace so that no reused S8/S7 code path can reach S8's.
RACE_START_PATH = OUT / f"{ARTEFACT_PREFIX}_race_start.json"
# Deliberately S8's: see the module docstring (V-A0 is not re-run).
PREFLIGHT_PATH = S8.PREFLIGHT_PATH
require_preflight = S8.require_preflight
# The two earlier β = 0.01 rounds, read-only, for the pooled frontier rate.
S8_ROUND_PATH = OUT / "s8_round.json"
S10_ROUND_PATH = OUT / "s10_round.json"

# --- pre-registered (spec sections 2-5). Do not tune. ---
BATCHES = {"A": (48, 49, 50, 51, 52, 53), "B": (54, 55, 56, 57, 58, 59)}
FAMILY = "SEL"
BETA = S8.BETAS["FT-A2"]  # 0.01 — spec section 2: the recipe does not change
ARMS = {
    f"{FAMILY}-{b}-{s}": {"seed": s, "family": FAMILY, "beta": BETA, "batch": b}
    for b, seeds in BATCHES.items()
    for s in seeds
}
SEEDS = tuple(s for seeds in BATCHES.values() for s in seeds)
TOTAL = S8.TOTAL  # 5 M env steps, probes every 500 k, both exactly S8's
EVERY = S8.EVERY
SEL_SEED0 = 1_290_000  # the selection gate
CONFIRM_SEED0 = 1_300_000  # the confirmation gate
PROBE_SEED0 = 1_320_000  # the in-training probes
GATE_N = S8.GATE_N  # 500
EVAL_N = S8.EVAL_N  # 200
REFERENCES = ("RC19",)  # on the selection stream
FT_B41 = "FT-B-41"  # S10's artefact (D120): a confirmation reference, never an arm
FT_B41_CKPT = OUT / f"ckpts_{FT_B41}" / f"{FT_B41}_{S8.TOTAL}.zip"
CONFIRM_REFERENCES = ("RC19", FT_B41)  # on the confirmation stream
REFERENCE_CKPT = dict(S8.REFERENCE_CKPT) | {FT_B41: FT_B41_CKPT}
# The β = 0.01 arms of the two earlier rounds, for the pooled frontier rate.
PRIOR_ARM_PREFIX = {"S8": "FT-A2-", "S10": "FT-B-"}


def apply_overrides() -> None:
    """Point the reused S8 machinery at the S12 round.

    S8's functions read every constant as their OWN module globals at call time,
    and three of its paths (START_PATH, PREFLIGHT_PATH, RACE_START_PATH) were
    bound from its prefix at import — so setting the prefix alone would leave
    them on S8's files. `S8.apply_overrides()` is called LAST because it is what
    pushes OUT / GATE_SEED0 / GATE_N / EVAL_N down onto the S7 module that
    actually gates; it must see S12's selection stream, not S8's.

    GATE_SEED0 is the SELECTION stream: it is the gate every arm runs. The
    confirmation stream is lent to S7 only inside `_confirmation_stream`.
    """
    S8.ARTEFACT_PREFIX = ARTEFACT_PREFIX
    S8.START_PATH = START_PATH
    S8.RACE_START_PATH = RACE_START_PATH
    S8.SEEDS = SEEDS
    S8.ARMS = ARMS
    S8.TOTAL = TOTAL
    S8.EVERY = EVERY
    S8.PROBE_SEED0 = PROBE_SEED0
    S8.GATE_SEED0 = SEL_SEED0
    S8.REFERENCES = REFERENCES
    S8.REFERENCE_CKPT = REFERENCE_CKPT
    S8.apply_overrides()


apply_overrides()


@contextlib.contextmanager
def _confirmation_stream():
    """Lend S7's gate the confirmation stream, and always take it back.

    `S7.gate_pursuer` reads GATE_SEED0 as its own module global, so this is the
    only way to gate on 1 300 000+ without leaving the selection stream
    re-pointed for whatever runs next in the same process."""
    prev = S8.S7.GATE_SEED0
    S8.S7.GATE_SEED0 = CONFIRM_SEED0
    try:
        yield
    finally:
        S8.S7.GATE_SEED0 = prev


# ----------------------------------------------------------------------------
# artefact paths and guards
# ----------------------------------------------------------------------------
def _confirm_file(arm: str) -> pathlib.Path:
    return OUT / f"{ARTEFACT_PREFIX}_confirm_{arm}.json"


def require_all_arms(results: dict) -> None:
    """Spec section 2: the selection rule reads a whole batch, so `--select`
    refuses until all twelve arms have a selection gate."""
    missing = [a for a in ARMS if a not in results or "final" not in (results[a] or {})]
    if missing:
        raise SystemExit(
            f"{len(missing)} of {len(ARMS)} arms have no selection gate: "
            f"{', '.join(missing)}. Run them FIRST."
        )


def load_selection() -> dict:
    if not SELECTION_PATH.exists():
        raise SystemExit(f"selection missing: {SELECTION_PATH}. Run --select FIRST.")
    return json.loads(SELECTION_PATH.read_text())


def load_confirm_start() -> dict:
    """Spec section 2: a selected policy refuses to be gated without BOTH
    confirmation references already measured on the same stream."""
    if not CONFIRM_START_PATH.exists():
        raise SystemExit(
            f"confirmation references missing: {CONFIRM_START_PATH}. "
            "Run --confirm-start FIRST."
        )
    s = json.loads(CONFIRM_START_PATH.read_text())
    for ref in CONFIRM_REFERENCES:
        for o in S8.GATE_OPPONENTS:
            if o not in s.get(ref, {}) or "deadlock_ind" not in s[ref][o]:
                raise SystemExit(
                    f"{ref}/{o} missing from {CONFIRM_START_PATH}; "
                    "re-run --confirm-start."
                )
    return s


def load_confirms() -> dict:
    """The confirmation records. `s12_confirm_start.json` is not one of them,
    which is why the glob carries the family."""
    return {
        p.stem.removeprefix(f"{ARTEFACT_PREFIX}_confirm_"): json.loads(p.read_text())
        for p in sorted(OUT.glob(f"{ARTEFACT_PREFIX}_confirm_{FAMILY}-*.json"))
    }


# ----------------------------------------------------------------------------
# selection (spec section 2)
# ----------------------------------------------------------------------------
def _rank(v: dict, arm: str) -> tuple:
    """The spec's order: lowest mean held-out deadlock; ties -> highest Greedy
    capture; ties -> lowest seed. S8's artefact rule, unchanged."""
    return (v["mean_deadlock_heldout"], -v["greedy_capture"], ARMS[arm]["seed"])


def _select_batch(batch: str, results: dict, starts: dict) -> dict:
    arms = [a for a in ARMS if ARMS[a]["batch"] == batch]
    per_arm: dict = {}
    for a in arms:
        r = results.get(a)
        if r is None or "final" not in r:
            per_arm[a] = {"verdict": "INCOMPLETE"}
            continue
        per_arm[a] = S8._arm_verdict(r["final"]["gate"], starts)
    complete = all(v["verdict"] != "INCOMPLETE" for v in per_arm.values())
    passing = [a for a in arms if per_arm[a]["verdict"] == "PASS"]
    # the capture guard alone: the precondition of the spec's fallback
    guard_ok = [
        a for a in arms if complete and not per_arm[a]["capture_guard"]["failed"]
    ]
    out = {
        "batch": batch,
        "per_arm": per_arm,
        "passing": passing,
        "capture_guard_ok": guard_ok,
        "complete": complete,
    }
    if not complete:
        missing = [a for a in arms if per_arm[a]["verdict"] == "INCOMPLETE"]
        return out | {
            "selected": None,
            "selected_passed_selection": False,
            "rule": "incomplete",
            "rule_trace": f"batch {batch} has no selection gate for {', '.join(missing)}",
        }
    if passing:
        selected = min(passing, key=lambda a: _rank(per_arm[a], a))
        return out | {
            "selected": selected,
            "selected_passed_selection": True,
            "rule": "passing",
            "rule_trace": (
                f"{len(passing)}/{len(arms)} arms passed the two-sided selection test; "
                f"{selected} has the lowest mean held-out deadlock among them "
                f"({per_arm[selected]['mean_deadlock_heldout']:.3f}, Greedy capture "
                f"{per_arm[selected]['greedy_capture']:.3f})"
            ),
        }
    # Spec section 2's fallback: no arm passed, so take the lowest mean held-out
    # deadlock among those whose capture guard holds. The batch's confirmation is
    # expected to FAIL, and is reported as such.
    if guard_ok:
        selected = min(guard_ok, key=lambda a: _rank(per_arm[a], a))
        rule = "fallback (capture guard holds)"
        trace = (
            f"no arm in batch {batch} passed; {selected} has the lowest mean held-out "
            f"deadlock ({per_arm[selected]['mean_deadlock_heldout']:.3f}) among the "
            f"{len(guard_ok)} whose capture guard holds. Its confirmation is expected "
            "to FAIL (spec section 2)"
        )
    else:
        # The spec does not name this case (it presumes at least one arm keeps
        # its capture). The fallback's intent is that a batch always hands the
        # confirmation stage a candidate, so the guard is dropped and the fact is
        # recorded here rather than left to a silent None.
        selected = min(arms, key=lambda a: _rank(per_arm[a], a))
        rule = "fallback (no capture guard holds)"
        trace = (
            f"no arm in batch {batch} passed and NONE kept its capture; {selected} has "
            f"the lowest mean held-out deadlock ({per_arm[selected]['mean_deadlock_heldout']:.3f}) "
            f"of the {len(arms)}. Its confirmation is expected to FAIL (spec section 2)"
        )
    return out | {
        "selected": selected,
        "selected_passed_selection": False,
        "rule": rule,
        "rule_trace": trace,
    }


def select(results: dict, starts: dict) -> dict:
    """Spec section 2's selection rule, run independently on each batch."""
    return {b: _select_batch(b, results, starts) for b in BATCHES}


# ----------------------------------------------------------------------------
# confirmation (spec section 2)
# ----------------------------------------------------------------------------
def confirm_start(device: str) -> dict:
    """RC19 and FT-B-41 on the confirmation stream, measured before either
    selected policy is gated on it."""
    require_preflight()
    if CONFIRM_START_PATH.exists():
        raise SystemExit(f"{CONFIRM_START_PATH} already exists (stop rule).")
    if any(OUT.glob(f"{ARTEFACT_PREFIX}_confirm_{FAMILY}-*.json")):
        raise SystemExit(
            "a selected policy has already been confirmed; the stop rule forbids "
            "re-measuring the confirmation references."
        )
    opps = {
        name: S8.S7.gate_opponent_factory(name, device) for name in S8.GATE_OPPONENTS
    }
    out: dict = {
        "gate_seed0": CONFIRM_SEED0,
        "gate_n": GATE_N,
        "provenance": S8.S7.provenance(),
    }
    with _confirmation_stream():
        for ref in CONFIRM_REFERENCES:
            print(f"{ref} reference on the confirmation stream:", flush=True)
            out[ref] = S8.S7.gate_pursuer(
                S8.S7._model(REFERENCE_CKPT[ref], device), opps, device
            )
    S8.S7._write_json(CONFIRM_START_PATH, out)
    return out


def confirm(device: str) -> dict:
    """Gate each selected policy on the confirmation stream. Each record has the
    same shape as an arm's `final`, so `S8._arm_verdict` reads it unchanged."""
    selection = load_selection()
    load_confirm_start()  # both references must already exist on this stream
    from stable_baselines3 import PPO

    out: dict = {}
    for b in BATCHES:
        arm = (selection.get(b) or {}).get("selected")
        if not arm:
            raise SystemExit(
                f"batch {b} selected nothing ({(selection.get(b) or {}).get('rule')}); "
                "the confirmation cannot run."
            )
        if _confirm_file(arm).exists():
            raise SystemExit(
                f"{arm} has already been confirmed; the stop rule forbids re-running it."
            )
        ckpt = pathlib.Path(json.loads(S8._arm_file(arm).read_text())["final"]["ckpt"])
        opps = {
            name: S8.S7.gate_opponent_factory(name, device)
            for name in S8.GATE_OPPONENTS
        }
        print(f"[{arm}] confirmation gate on {CONFIRM_SEED0}+:", flush=True)
        with _confirmation_stream():
            g = S8.S7.gate_pursuer(PPO.load(ckpt, device=device), opps, device)
        rec = {
            "arm": arm,
            "batch": b,
            "ckpt": str(ckpt),
            "gate": g,
            "pursuer_violations": int(sum(v["pursuer_violations"] for v in g.values())),
            "gate_seed0": CONFIRM_SEED0,
            "gate_n": GATE_N,
            "selected_passed_selection": selection[b]["selected_passed_selection"],
            "provenance": S8.S7.provenance(),
        }
        S8.S7._write_json(_confirm_file(arm), rec)
        out[arm] = rec
    return out


# ----------------------------------------------------------------------------
# verdicts (spec section 3)
# ----------------------------------------------------------------------------
def _reference_row(name: str, starts: dict | None) -> dict | None:
    ref = (starts or {}).get(name)
    if not ref:
        return None
    held = {o: ref[o]["pursuer"]["deadlock_episode_rate"] for o in S8.HELD_OUT}
    return {
        "reference": name,
        "heldout_deadlock": held,
        "mean_deadlock_heldout": sum(held.values()) / len(held),
        "greedy_capture": ref["GREEDY"]["capture_rate"],
    }


def _shrinkage(sel_v: dict, conf_v: dict) -> dict:
    """Spec section 3, reported with no verdict: what the selected policy loses
    (or keeps) when it moves from the stream it was chosen on to a fresh one."""
    a = {
        "mean_deadlock_heldout": sel_v["mean_deadlock_heldout"],
        "greedy_capture": sel_v["greedy_capture"],
    }
    b = {
        "mean_deadlock_heldout": conf_v["mean_deadlock_heldout"],
        "greedy_capture": conf_v["greedy_capture"],
    }
    return {
        "selection": a,
        "confirmation": b,
        "delta_mean_deadlock": b["mean_deadlock_heldout"] - a["mean_deadlock_heldout"],
        "delta_greedy_capture": b["greedy_capture"] - a["greedy_capture"],
    }


def h6_verdicts(
    selection: dict | None, confirms: dict | None, confirm_starts: dict | None
) -> dict:
    """H6 (spec section 3): both selected policies pass S8's unchanged two-sided
    per-arm test on the CONFIRMATION stream, against RC19 as measured there.

    INCOMPLETE if any of the twelve arms lacks a selection gate or either
    confirmation gate is missing."""
    selection, confirms = selection or {}, confirms or {}
    per_selected: dict = {}
    incomplete = False
    for b in BATCHES:
        bt = selection.get(b) or {}
        arm = bt.get("selected")
        rec = confirms.get(arm) if arm else None
        if not bt.get("complete") or rec is None or "gate" not in rec:
            per_selected[b] = {"selected": arm, "verdict": "INCOMPLETE"}
            incomplete = True
            continue
        v = S8._arm_verdict(rec["gate"], confirm_starts or {})
        sel_v = bt["per_arm"][arm]
        per_selected[b] = v | {
            "selected": arm,
            "selection_verdict": sel_v["verdict"],
            "selected_passed_selection": bt["selected_passed_selection"],
            "selection_rule": bt["rule"],
            "shrinkage": _shrinkage(sel_v, v),
        }
    passing = [b for b, v in per_selected.items() if v["verdict"] == "PASS"]
    if incomplete:
        h6 = "INCOMPLETE"
    else:
        h6 = {2: "PASS", 1: "PASS (1/2)", 0: "FAIL"}[len(passing)]
    artefact = (
        min(
            passing,
            key=lambda b: _rank(per_selected[b], per_selected[b]["selected"]),
        )
        if passing and not incomplete
        else None
    )
    artefact = per_selected[artefact]["selected"] if artefact else None
    ft_b41 = _reference_row(FT_B41, confirm_starts)
    return {
        "H6": h6,
        "per_selected": per_selected,
        "n_pass": None if incomplete else len(passing),
        "artefact": artefact,
        "ft_b41": ft_b41,
        "paper_artefact": _paper_artefact(artefact, per_selected, ft_b41),
    }


def _paper_artefact(
    artefact: str | None, per_selected: dict, ft_b41: dict | None
) -> dict | None:
    """Spec section 3: the paper's artefact is the better of {this round's
    artefact, FT-B-41} on the confirmation stream — reported, not re-selected."""
    if ft_b41 is None:
        return None
    note = "better of the S12 artefact and FT-B-41 on the confirmation stream; reported, not re-selected"
    if artefact is None:
        return {
            "policy": FT_B41,
            "mean_deadlock_heldout": ft_b41["mean_deadlock_heldout"],
            "note": "no S12 artefact (H6 did not deliver one); D120's artefact stands, "
            "reported, not re-selected",
        }
    mine = next(v for v in per_selected.values() if v.get("selected") == artefact)
    if mine["mean_deadlock_heldout"] <= ft_b41["mean_deadlock_heldout"]:
        return {
            "policy": artefact,
            "mean_deadlock_heldout": mine["mean_deadlock_heldout"],
            "note": note,
        }
    return {
        "policy": FT_B41,
        "mean_deadlock_heldout": ft_b41["mean_deadlock_heldout"],
        "note": note,
    }


def _prior_pass(round_doc: dict | None, prefix: str) -> tuple[int, int]:
    per = (((round_doc or {}).get("summary") or {}).get("hypotheses") or {}).get(
        "per_arm"
    ) or {}
    v = [r.get("verdict") for a, r in per.items() if a.startswith(prefix)]
    return sum(x == "PASS" for x in v), len(v)


def pooled_rate(s8_round: dict | None, s10_round: dict | None, selection: dict) -> dict:
    """H6-rate (spec section 1), reported with no verdict: the per-seed frontier
    rate over every β = 0.01 seed run to date — S8's FT-A2 arms, S10's FT-B arms
    and this round's twelve, on their own selection stream — with a Wilson
    interval. The two earlier rounds are read from their round files; nothing is
    re-judged here."""
    sources = {
        "S8": _prior_pass(s8_round, PRIOR_ARM_PREFIX["S8"]),
        "S10": _prior_pass(s10_round, PRIOR_ARM_PREFIX["S10"]),
    }
    v = [
        r["verdict"]
        for b in (selection or {})
        for r in ((selection[b] or {}).get("per_arm") or {}).values()
    ]
    measured = [x for x in v if x != "INCOMPLETE"]
    sources["S12"] = (sum(x == "PASS" for x in measured), len(measured))
    k = sum(a for a, _ in sources.values())
    n = sum(b for _, b in sources.values())
    out = {
        "sources": {s: {"k": a, "n": b} for s, (a, b) in sources.items()},
        "k": k,
        "n": n,
    }
    if n == 0:
        return out | {"rate": None, "wilson": None}
    return out | S8.S7._ci(k, n)


def hsafe(
    results: dict,
    starts: dict | None = None,
    confirms: dict | None = None,
    confirm_starts: dict | None = None,
) -> dict:
    """H-SAFE (spec section 2): 0 pursuer contact steps anywhere in the round.

    S8's census covers every arm's training, probes and selection gate plus the
    selection-stream reference; this adds the two confirmation gates and both
    confirmation references."""
    h = S8.hsafe_verdict(results, starts)
    steps = int(h["pursuer_violation_steps"])
    offenders = list(h["offenders"])
    for arm in sorted(confirms or {}):
        v = int((confirms[arm] or {}).get("pursuer_violations") or 0)
        steps += v
        if v:
            offenders.append(f"confirm:{arm}")
    for ref in CONFIRM_REFERENCES:
        for o in S8.GATE_OPPONENTS:
            v = int(
                (((confirm_starts or {}).get(ref) or {}).get(o) or {}).get(
                    "pursuer_violations"
                )
                or 0
            )
            steps += v
            if v:
                offenders.append(f"confirm-reference:{ref}/{o}")
    return {
        "verdict": "FAILS" if steps else ("INCOMPLETE" if h["missing"] else "HOLDS"),
        "pursuer_violation_steps": steps,
        "offenders": offenders,
        "missing": h["missing"],
    }


def direction(h: dict) -> str:
    """Spec section 4, row by row."""
    if h["hsafe"] == "FAILS":
        return (
            "H-SAFE FAILS: a shielded pursuer recorded a contact step. Stop; census; no further "
            "round without a user ruling."
        )
    if h["hsafe"] == "INCOMPLETE" or h["H6"] == "INCOMPLETE":
        return (
            "incomplete: all twelve S12 arms must run and both selected policies must be "
            "confirmed"
        )
    if h["H6"] == "FAIL":
        return (
            "H6 FAILS (0/2): selection on one stream does not transfer — the frontier policies "
            "of S8/S10 were stream-specific. D120's reading is withdrawn to 'frontier reachable, "
            "not selectable'."
        )
    if h["H6"] == "PASS (1/2)":
        return (
            "H6 PASSES (1/2): the procedure delivers about half the time at batch size six. The "
            "paper reports the rate and recommends a larger batch, without re-running."
        )
    return (
        "H6 PASSES (2/2): the method is 'train six, select, confirm', with a measured per-seed "
        "rate and a two-for-two confirmation — the paper's R3 procedure."
    )


# ----------------------------------------------------------------------------
# report
# ----------------------------------------------------------------------------
def _read_round(path: pathlib.Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _table(gate: dict) -> dict:
    return {
        o: {
            "deadlock": gate[o]["pursuer"]["deadlock_episode_rate"],
            "capture": gate[o]["capture_rate"],
        }
        for o in gate
    }


def summarize(
    results: dict,
    starts: dict,
    selection: dict | None = None,
    confirms: dict | None = None,
    confirm_starts: dict | None = None,
) -> dict:
    h = h6_verdicts(selection, confirms, confirm_starts)
    hs = hsafe(results, starts, confirms, confirm_starts)
    arms_done = all(a in results and "final" in results[a] for a in ARMS)
    return {
        "spec": "docs/pathway/selection_round_prereg_2026-09-18.md",
        "hsafe": hs,
        "hypotheses": h,
        "selection": selection,
        "arms": {
            a: _table((r.get("final") or {}).get("gate") or {})
            for a, r in results.items()
        },
        "references": {
            ref: _table(starts[ref]) for ref in REFERENCES if ref in (starts or {})
        },
        "confirm": {a: _table(r["gate"]) for a, r in (confirms or {}).items()},
        "confirm_references": {
            ref: _table((confirm_starts or {})[ref])
            for ref in CONFIRM_REFERENCES
            if ref in (confirm_starts or {})
        },
        "pooled_frontier_rate": pooled_rate(
            _read_round(S8_ROUND_PATH), _read_round(S10_ROUND_PATH), selection or {}
        ),
        "anchor_kl": {
            a: r["anchor_kl_trace"]
            for a, r in results.items()
            if r.get("anchor_kl_trace")
        },
        "streams": {"selection": SEL_SEED0, "confirmation": CONFIRM_SEED0},
        "complete": arms_done and len(confirms or {}) == len(BATCHES),
        "direction": direction(
            {"hsafe": hs["verdict"], "H6": h["H6"], "n_pass": h["n_pass"]}
        ),
    }


def write_figure(results: dict, starts: dict, summary: dict, path) -> None:
    """Two panels, as S10's: the gate's deadlock clause and its capture clause.
    Solid bars are the confirmation stream — both references and the two selected
    policies, the only things H6 is decided on. The twelve faded bars behind them
    are the selection stream, reported, never compared across streams."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    opps = list(S8.GATE_OPPONENTS)
    x = range(len(opps))
    faded = [(a, summary["arms"][a]) for a in ARMS if a in summary.get("arms", {})]
    cr = summary.get("confirm_references") or {}
    solid = [
        (f"{ref} (confirm)", cr[ref], col)
        for ref, col in zip(CONFIRM_REFERENCES, ("k", "tab:orange"), strict=True)
        if ref in cr
    ]
    solid += [
        (f"{a} (confirm)", d, None) for a, d in (summary.get("confirm") or {}).items()
    ]
    series = [(n, d, "0.7", 0.3) for n, d in faded] + [
        (n, d, c, 1.0) for n, d, c in solid
    ]
    w = 0.84 / max(1, len(series))
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    for ax, key in ((axes[0], "deadlock"), (axes[1], "capture")):
        for j, (name, d, col, alpha) in enumerate(series):
            ax.bar(
                [i - 0.42 + w * (j + 0.5) for i in x],
                [d.get(o, {}).get(key, 0) for o in opps],
                width=w,
                color=col,
                alpha=alpha,
                label=name,
            )
        ax.set_xticks(list(x))
        ax.set_xticklabels(opps)
    axes[0].set_ylabel("pursuer deadlock-episode rate (gate)")
    axes[0].set_title(f"S12: H6 {summary['hypotheses']['H6']}")
    axes[0].legend(fontsize=6, ncol=3)
    axes[1].set_ylabel("capture rate (gate)")
    axes[1].set_title("capture (the second clause of the gate)")
    fig.suptitle(
        "S12 — train six, select one, confirm: two independent batches of the "
        "β = 0.01 anchored fine-tune" + ("" if summary["complete"] else " — INCOMPLETE")
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _write_round(
    results: dict,
    starts: dict,
    selection: dict | None,
    confirms: dict | None,
    confirm_starts: dict | None,
) -> dict:
    summary = summarize(results, starts, selection, confirms, confirm_starts)
    S8.S7._write_json(
        S8._round_file("json"),
        {"summary": summary, "arms": results, "confirm": confirms or {}},
    )
    write_figure(results, starts, summary, S8._round_file("png"))
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--measure-start", action="store_true")
    p.add_argument("--arms", nargs="*", choices=list(ARMS), default=[])
    p.add_argument("--select", action="store_true")
    p.add_argument("--confirm-start", action="store_true")
    p.add_argument("--confirm", action="store_true")
    p.add_argument("--summarize", action="store_true")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.measure_start:
        S8.measure_start(args.device)
        return
    for arm in args.arms:
        S8.run_arm(arm, args.device)
    if args.select:
        results = S8.load_results()
        require_all_arms(results)
        if SELECTION_PATH.exists():
            raise SystemExit(f"{SELECTION_PATH} already exists (stop rule).")
        sel = select(results, S8.load_start())
        S8.S7._write_json(SELECTION_PATH, sel)
        for b, bt in sel.items():
            print(f"[batch {b}] {bt['selected']}: {bt['rule_trace']}", flush=True)
    if args.confirm_start:
        confirm_start(args.device)
    if args.confirm:
        confirm(args.device)
    if args.summarize:
        selection = load_selection() if SELECTION_PATH.exists() else None
        confirm_starts = (
            json.loads(CONFIRM_START_PATH.read_text())
            if CONFIRM_START_PATH.exists()
            else None
        )
        summary = _write_round(
            S8.load_results(),
            S8.load_start(),
            selection,
            load_confirms(),
            confirm_starts,
        )
        print(
            json.dumps(
                {
                    k: summary[k]
                    for k in (
                        "hsafe",
                        "hypotheses",
                        "references",
                        "confirm_references",
                        "confirm",
                        "pooled_frontier_rate",
                    )
                },
                indent=2,
                default=str,
            )
        )
        print(f"\n{summary['direction']}")


if __name__ == "__main__":
    main()
