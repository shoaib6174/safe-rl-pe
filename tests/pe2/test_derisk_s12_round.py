"""scripts/derisk_s12_select.py — the selection round's own decisions.

S12 is S8's machinery under S12's constants (spec:
docs/pathway/selection_round_prereg_2026-09-18.md), so these tests cover what
S12 itself decides: the overrides it pushes onto the reused S8/S7 modules, the
per-batch selection rule and its fallback, H6 on the confirmation stream, the
pooled frontier rate, H-SAFE over both stages, and the spec's direction rows.
"""

import importlib.util
import json
import pathlib

import pytest

_PATH = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "derisk_s12_select.py"


def _runner():
    spec = importlib.util.spec_from_file_location("derisk_s12_select", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _runner()


# ----------------------------------------------------------------------------
# fakes, in the shape S7's `gate_pursuer` writes
# ----------------------------------------------------------------------------
def _ind(rate, n=500):
    k = round(rate * n)
    return [i < k for i in range(n)]


def _gate(deadlock, capture):
    return {
        "deadlock_ind": _ind(deadlock),
        "capture_ind": _ind(capture),
        "pursuer_violations": 0,
        "pursuer": {
            "deadlock_episode_rate": deadlock,
            "stall_step_rate": 0.3,
            "intervention_rate": 0.4,
        },
        "capture_rate": capture,
    }


def _ref(att, greedy_capture, greedy_deadlock=0.33):
    return {
        "ATT25": _gate(att[0], 0.12),
        "ATT26": _gate(att[1], 0.12),
        "ATT27": _gate(att[2], 0.10),
        "GREEDY": _gate(greedy_deadlock, greedy_capture),
        "HUG7": _gate(0.33, 0.41),
    }


_RC19 = ((0.60, 0.56, 0.58), 0.568)
_WIN = (0.30, 0.32, 0.31)  # beats RC19 on all three held-out attackers
_BETTER = (0.20, 0.21, 0.22)  # ... by more
_PINNED = (0.60, 0.61, 0.62)  # matches RC19: no deadlock win
_LESS_PINNED = (0.55, 0.56, 0.57)  # still no win, but the lowest mean of the pinned
_KEEPS_CAPTURE = 0.55  # not significantly below RC19's 0.568
_LOSES_CAPTURE = 0.30


def _starts():
    """The single selection-stream reference of spec section 2."""
    return {"RC19": _ref(*_RC19)}


def _confirm_starts(ftb41=_WIN, ftb41_capture=0.60):
    """RC19 and FT-B-41 on the confirmation stream (spec section 2)."""
    return {
        "RC19": _ref(*_RC19),
        "FT-B-41": _ref(ftb41, ftb41_capture, greedy_deadlock=0.20),
    }


def _arm(name, att, greedy_capture):
    return {
        "arm": name,
        **R.ARMS[name],
        "train_violations": 0,
        "probe_violations": 0,
        "final": {
            "ckpt": f"results/derisk/ckpts_{name}/{name}_5000000.zip",
            "pursuer_violations": 0,
            "gate": {
                "ATT25": _gate(att[0], 0.1),
                "ATT26": _gate(att[1], 0.1),
                "ATT27": _gate(att[2], 0.1),
                "GREEDY": _gate(0.2, greedy_capture),
                "HUG7": _gate(0.2, 0.35),
            },
        },
        "checkpoints": [],
    }


def _round(att=None, cap=None):
    """All twelve arms; by default every one of them passes both clauses."""
    att, cap = att or {}, cap or {}
    return {a: _arm(a, att.get(a, _WIN), cap.get(a, _KEEPS_CAPTURE)) for a in R.ARMS}


def _confirm(arm, att, greedy_capture=_KEEPS_CAPTURE):
    """A confirmation record: the same shape as an arm's `final`."""
    return {
        "arm": arm,
        "batch": R.ARMS[arm]["batch"],
        "ckpt": f"results/derisk/ckpts_{arm}/{arm}_5000000.zip",
        "pursuer_violations": 0,
        "gate": {
            "ATT25": _gate(att[0], 0.1),
            "ATT26": _gate(att[1], 0.1),
            "ATT27": _gate(att[2], 0.1),
            "GREEDY": _gate(0.2, greedy_capture),
            "HUG7": _gate(0.2, 0.35),
        },
    }


def _gate_ok(holder=None):
    """What the stubbed `gate_pursuer` returns: a clean five-opponent gate."""
    g = {
        "ATT25": _gate(0.30, 0.1),
        "ATT26": _gate(0.32, 0.1),
        "ATT27": _gate(0.31, 0.1),
        "GREEDY": _gate(0.2, _KEEPS_CAPTURE),
        "HUG7": _gate(0.2, 0.35),
    }
    if holder is not None:
        holder["gate"] = g
    return g


def _confirms(a_att=_WIN, b_att=_WIN, a_cap=_KEEPS_CAPTURE, b_cap=_KEEPS_CAPTURE):
    sel = R.select(_round(), _starts())
    return {
        sel["A"]["selected"]: _confirm(sel["A"]["selected"], a_att, a_cap),
        sel["B"]["selected"]: _confirm(sel["B"]["selected"], b_att, b_cap),
    }


# fake prior rounds, in the shape `results/derisk/s{8,10}_round.json` carry
def _s8_round():
    """S8: FT-A2-35 is the one β = 0.01 seed that passed. FT-A1-* (β = 0.1) and
    the from-scratch control are not β = 0.01 seeds and must not be counted."""
    v = {
        "FT-A1-34": "FAIL (still pinned)",
        "FT-A1-35": "FAIL (still pinned)",
        "FT-A1-36": "FAIL (still pinned)",
        "FT-A2-34": "FAIL (still pinned)",
        "FT-A2-35": "PASS",
        "FT-A2-36": "FAIL (still pinned)",
        "SCR-W34": "PASS",
    }
    return {
        "summary": {
            "hypotheses": {"per_arm": {a: {"verdict": x} for a, x in v.items()}}
        }
    }


def _s10_round():
    v = {
        "FT-B-40": "PASS",
        "FT-B-41": "PASS",
        "FT-B-42": "PASS",
        "FT-B-43": "FAIL (still pinned)",
        "FT-B-44": "FAIL (still pinned)",
    }
    return {
        "summary": {
            "hypotheses": {"per_arm": {a: {"verdict": x} for a, x in v.items()}}
        }
    }


# ----------------------------------------------------------------------------
class TestProtocol:
    def test_constants(self):
        assert R.BATCHES == {
            "A": (48, 49, 50, 51, 52, 53),
            "B": (54, 55, 56, 57, 58, 59),
        }
        assert set(R.BATCHES["A"]).isdisjoint(R.BATCHES["B"])
        assert len(R.ARMS) == 12
        assert set(R.ARMS) == {f"SEL-A-{s}" for s in R.BATCHES["A"]} | {
            f"SEL-B-{s}" for s in R.BATCHES["B"]
        }
        assert R.ARMS["SEL-A-48"] == {
            "seed": 48,
            "family": "SEL",
            "beta": 0.01,
            "batch": "A",
        }
        assert R.SEEDS == tuple(range(48, 60))
        assert (R.TOTAL, R.EVERY) == (5_000_000, 500_000)
        assert R.SEL_SEED0 == 1_290_000
        assert R.CONFIRM_SEED0 == 1_300_000
        assert R.PROBE_SEED0 == 1_320_000
        assert (R.GATE_N, R.EVAL_N) == (500, 200)
        assert R.ARTEFACT_PREFIX == "s12"
        assert R.REFERENCES == ("RC19",)
        assert R.CONFIRM_REFERENCES == ("RC19", "FT-B-41")
        assert R.FT_B41_CKPT == R.OUT / "ckpts_FT-B-41" / "FT-B-41_5000000.zip"

    def test_the_learner_is_exactly_s10s_arm(self):
        """Spec section 2: the recipe does not change. SEL is an anchored
        fine-tune because it is not S8's CONTROL family."""
        assert (R.TOTAL, R.EVERY) == (R.S8.TOTAL, R.S8.EVERY)
        assert R.BETA == R.S8.BETAS["FT-A2"] == 0.01
        assert R.FAMILY == "SEL" and R.FAMILY != R.S8.CONTROL
        assert R.S8.WEIGHTS == (0.50, 0.20, 0.10, 1 / 15, 1 / 15, 1 / 15)
        assert R.S8.MIXTURE == R.S8.S7.MIXTURE
        # the per-arm two-sided test is S8's, unchanged: >= 2 of 3 held-out
        assert R.S8.PASS_MIN == 2 and R.S8.HELD_OUT == ("ATT25", "ATT26", "ATT27")

    def test_the_s8_module_is_pointed_at_the_s12_round(self):
        assert R.S8.ARTEFACT_PREFIX == "s12"
        assert R.S8.START_PATH == R.OUT / "s12_start.json"
        assert R.S8.ARMS == R.ARMS and R.S8.SEEDS == R.SEEDS
        assert R.S8.REFERENCES == ("RC19",)
        assert R.S8.REFERENCE_CKPT["RC19"] == R.S8.RC19_CKPT
        assert R.S8.REFERENCE_CKPT["FT-B-41"] == R.FT_B41_CKPT
        assert R.S8.GATE_SEED0 == R.SEL_SEED0  # the gate an arm runs is SELECTION
        assert R.S8.PROBE_SEED0 == 1_320_000
        assert R.S8._arm_file("SEL-A-48") == R.OUT / "s12_arm_SEL-A-48.json"
        assert R.S8._probe_file("SEL-A-48") == R.OUT / "s12_probe_SEL-A-48.json"
        assert R.S8._round_file("json") == R.OUT / "s12_round.json"
        assert R.S8.RACE_START_PATH == R.OUT / "s12_race_start.json"

    def test_the_s7_module_is_pointed_at_the_selection_stream(self):
        """S7's `gate_pursuer` reads GATE_SEED0/GATE_N as its OWN globals."""
        assert R.S8.S7.GATE_SEED0 == R.SEL_SEED0
        assert R.S8.S7.GATE_N == R.GATE_N and R.S8.S7.EVAL_N == R.EVAL_N
        assert R.S8.S7.OUT == R.OUT
        assert R.S8.S7.RACE_START_PATH == R.OUT / "s12_race_start.json"

    def test_the_confirmation_stream_is_borrowed_and_restored(self):
        """Nothing selected on 1 290 000+ is ever judged on it: the gate that
        judges a selected policy runs on 1 300 000+, and the selection stream is
        put back even if that gate raises."""
        assert R.S8.S7.GATE_SEED0 == R.SEL_SEED0
        with R._confirmation_stream():
            assert R.S8.S7.GATE_SEED0 == R.CONFIRM_SEED0
        assert R.S8.S7.GATE_SEED0 == R.SEL_SEED0
        with pytest.raises(RuntimeError), R._confirmation_stream():
            raise RuntimeError("boom")
        assert R.S8.S7.GATE_SEED0 == R.SEL_SEED0

    def test_preflight_is_s8s(self):
        """V-A0 is S8's and is not re-run: the recipe's code path is unchanged."""
        assert R.PREFLIGHT_PATH == R.OUT / "s8_preflight.json"
        assert R.S8.PREFLIGHT_PATH == R.PREFLIGHT_PATH
        assert R.require_preflight is R.S8.require_preflight

    def test_guards(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R.S8, "OUT", tmp_path)
        monkeypatch.setattr(R, "OUT", tmp_path)
        monkeypatch.setattr(R, "SELECTION_PATH", tmp_path / "s12_selection.json")
        monkeypatch.setattr(
            R, "CONFIRM_START_PATH", tmp_path / "s12_confirm_start.json"
        )
        # an arm may not be re-run, and S8's/S10's namespaces are not this round's
        R.S8.require_arm_not_run("SEL-A-48")
        (tmp_path / "s10_arm_SEL-A-48.json").write_text("{}")
        R.S8.require_arm_not_run("SEL-A-48")
        (tmp_path / "s12_arm_SEL-A-48.json").write_text("{}")
        with pytest.raises(SystemExit, match="stop rule"):
            R.S8.require_arm_not_run("SEL-A-48")
        # --confirm before --select
        with pytest.raises(SystemExit, match="--select FIRST"):
            R.confirm("cpu")
        # --confirm-start before --confirm
        R.SELECTION_PATH.write_text(json.dumps(R.select(_round(), _starts())))
        with pytest.raises(SystemExit, match="--confirm-start FIRST"):
            R.confirm("cpu")
        # a confirmation reference may not be measured twice
        R.CONFIRM_START_PATH.write_text("{}")
        with pytest.raises(SystemExit, match="already exists"):
            R.confirm_start("cpu")

    def test_select_refuses_before_all_twelve_arms(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R.S8, "OUT", tmp_path)
        done = _round()
        for a in list(R.ARMS)[:10]:
            (tmp_path / f"s12_arm_{a}.json").write_text(json.dumps(done[a]))
        with pytest.raises(SystemExit) as e:
            R.require_all_arms(R.S8.load_results())
        assert "SEL-B-58" in str(e.value) and "SEL-B-59" in str(e.value)
        # a crashed arm has a record but no gate: still missing
        (tmp_path / "s12_arm_SEL-B-58.json").write_text('{"verdict": "UNDETERMINED"}')
        with pytest.raises(SystemExit, match="SEL-B-58"):
            R.require_all_arms(R.S8.load_results())
        for a in list(R.ARMS)[10:]:
            (tmp_path / f"s12_arm_{a}.json").write_text(json.dumps(done[a]))
        R.require_all_arms(R.S8.load_results())  # satisfied: no raise

    def test_load_results_reads_the_s12_namespace(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R.S8, "OUT", tmp_path)
        (tmp_path / "s12_arm_SEL-A-48.json").write_text('{"arm": "SEL-A-48"}')
        (tmp_path / "s10_arm_FT-B-40.json").write_text('{"arm": "FT-B-40"}')
        assert set(R.S8.load_results()) == {"SEL-A-48"}

    def test_load_confirms_ignores_the_confirmation_start_file(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(R, "OUT", tmp_path)
        (tmp_path / "s12_confirm_start.json").write_text("{}")
        (tmp_path / "s12_confirm_SEL-A-48.json").write_text('{"arm": "SEL-A-48"}')
        assert set(R.load_confirms()) == {"SEL-A-48"}


# ----------------------------------------------------------------------------
class TestSelection:
    def test_lowest_mean_deadlock_among_the_passing(self):
        sel = R.select(
            _round(att={"SEL-A-50": _BETTER, "SEL-B-57": _BETTER}), _starts()
        )
        assert sel["A"]["selected"] == "SEL-A-50"
        assert sel["B"]["selected"] == "SEL-B-57"
        assert sel["A"]["selected_passed_selection"] is True
        assert sel["A"]["rule"] == "passing"
        assert len(sel["A"]["passing"]) == 6 and len(sel["A"]["per_arm"]) == 6
        assert set(sel["A"]["per_arm"]) == {f"SEL-A-{s}" for s in R.BATCHES["A"]}

    def test_ties_break_on_capture_then_seed(self):
        sel = R.select(_round(cap={"SEL-A-51": 0.60}), _starts())
        assert sel["A"]["selected"] == "SEL-A-51"  # tie on deadlock -> capture
        assert R.select(_round(), _starts())["A"]["selected"] == "SEL-A-48"  # -> seed

    def test_a_losing_arm_never_selects_even_with_the_lowest_deadlock(self):
        """The capture guard is a precondition of the fallback, not a tie-break."""
        sel = R.select(
            _round(
                att={a: _PINNED for a in R.ARMS} | {"SEL-A-49": (0.05, 0.05, 0.05)},
                cap={"SEL-A-49": _LOSES_CAPTURE},
            ),
            _starts(),
        )
        assert sel["A"]["per_arm"]["SEL-A-49"]["verdict"] == "FAIL (capture lost)"
        assert "SEL-A-49" not in sel["A"]["capture_guard_ok"]
        assert sel["A"]["selected"] != "SEL-A-49"

    def test_the_no_pass_fallback(self):
        att = {a: _PINNED for a in R.ARMS} | {"SEL-A-52": _LESS_PINNED}
        sel = R.select(_round(att=att), _starts())
        assert sel["A"]["passing"] == []
        assert sel["A"]["selected"] == "SEL-A-52"
        assert sel["A"]["selected_passed_selection"] is False
        assert "fallback" in sel["A"]["rule"]
        assert "expected to FAIL" in sel["A"]["rule_trace"]

    def test_the_degenerate_fallback_when_no_capture_guard_holds(self):
        sel = R.select(
            _round(
                att={a: _PINNED for a in R.ARMS} | {"SEL-A-52": _LESS_PINNED},
                cap={a: _LOSES_CAPTURE for a in R.ARMS},
            ),
            _starts(),
        )
        assert sel["A"]["capture_guard_ok"] == []
        assert sel["A"]["selected"] == "SEL-A-52"
        assert sel["A"]["selected_passed_selection"] is False
        assert "no capture guard" in sel["A"]["rule"]

    def test_a_missing_arm_makes_its_batch_incomplete(self):
        res = _round()
        del res["SEL-B-56"]
        sel = R.select(res, _starts())
        assert sel["A"]["complete"] is True and sel["A"]["selected"] is not None
        assert sel["B"]["complete"] is False and sel["B"]["selected"] is None
        assert sel["B"]["per_arm"]["SEL-B-56"]["verdict"] == "INCOMPLETE"
        assert sel["B"]["rule"] == "incomplete"
        no_final = _round()
        del no_final["SEL-B-56"]["final"]
        assert R.select(no_final, _starts())["B"]["complete"] is False

    def test_the_per_arm_test_is_s8s(self):
        v = R.select(_round(), _starts())["A"]["per_arm"]["SEL-A-48"]
        assert v["passes"] == 3 and set(v["tests"]) == set(R.S8.HELD_OUT)
        assert v["capture_guard"]["failed"] is False
        assert v["mean_deadlock_heldout"] == pytest.approx(sum(_WIN) / 3)


# ----------------------------------------------------------------------------
class TestH6:
    def test_both_confirmations_pass(self):
        sel = R.select(_round(), _starts())
        h = R.h6_verdicts(sel, _confirms(), _confirm_starts())
        assert h["H6"] == "PASS" and h["n_pass"] == 2
        assert set(h["per_selected"]) == {"A", "B"}
        assert h["per_selected"]["A"]["verdict"] == "PASS"

    def test_one_of_two(self):
        sel = R.select(_round(), _starts())
        h = R.h6_verdicts(sel, _confirms(b_att=_PINNED), _confirm_starts())
        assert h["H6"] == "PASS (1/2)" and h["n_pass"] == 1
        assert h["per_selected"]["B"]["verdict"] == "FAIL (still pinned)"
        assert h["artefact"] == sel["A"]["selected"]

    def test_neither(self):
        sel = R.select(_round(), _starts())
        h = R.h6_verdicts(
            sel, _confirms(a_att=_PINNED, b_att=_PINNED), _confirm_starts()
        )
        assert h["H6"] == "FAIL" and h["n_pass"] == 0 and h["artefact"] is None

    def test_incomplete(self):
        sel = R.select(_round(), _starts())
        assert R.h6_verdicts(sel, {}, _confirm_starts())["H6"] == "INCOMPLETE"
        one = _confirms()
        del one[sel["B"]["selected"]]
        assert R.h6_verdicts(sel, one, _confirm_starts())["H6"] == "INCOMPLETE"
        res = _round()
        del res["SEL-B-56"]
        short = R.select(res, _starts())
        assert (
            R.h6_verdicts(short, _confirms(), _confirm_starts())["H6"] == "INCOMPLETE"
        )

    def test_the_artefact_is_the_better_of_the_two(self):
        sel = R.select(_round(), _starts())
        h = R.h6_verdicts(sel, _confirms(b_att=_BETTER), _confirm_starts())
        assert h["artefact"] == sel["B"]["selected"]
        h2 = R.h6_verdicts(sel, _confirms(a_att=_BETTER), _confirm_starts())
        assert h2["artefact"] == sel["A"]["selected"]

    def test_the_paper_artefact_is_reported_against_ft_b_41(self):
        sel = R.select(_round(), _starts())
        # FT-B-41 is worse on the confirmation stream -> the S12 artefact stands
        h = R.h6_verdicts(sel, _confirms(a_att=_BETTER), _confirm_starts(ftb41=_PINNED))
        assert h["paper_artefact"]["policy"] == sel["A"]["selected"]
        assert "not re-selected" in h["paper_artefact"]["note"]
        # FT-B-41 is better -> D120's artefact stands, reported, not re-selected
        h2 = R.h6_verdicts(sel, _confirms(), _confirm_starts(ftb41=_BETTER))
        assert h2["paper_artefact"]["policy"] == "FT-B-41"
        # nothing passed -> D120's artefact stands
        h3 = R.h6_verdicts(
            sel, _confirms(a_att=_PINNED, b_att=_PINNED), _confirm_starts()
        )
        assert h3["paper_artefact"]["policy"] == "FT-B-41"

    def test_the_shrinkage_is_reported_on_both_streams(self):
        sel = R.select(_round(), _starts())
        h = R.h6_verdicts(sel, _confirms(a_att=_PINNED), _confirm_starts())
        sh = h["per_selected"]["A"]["shrinkage"]
        assert sh["selection"]["mean_deadlock_heldout"] == pytest.approx(sum(_WIN) / 3)
        assert sh["confirmation"]["mean_deadlock_heldout"] == pytest.approx(
            sum(_PINNED) / 3
        )
        assert sh["delta_mean_deadlock"] == pytest.approx(
            sum(_PINNED) / 3 - sum(_WIN) / 3
        )
        assert sh["delta_greedy_capture"] == pytest.approx(0.0)

    def test_the_confirmation_verdict_uses_the_confirmation_references(self):
        """A confirmation gate is judged against RC19 as measured on 1 300 000+,
        never against the selection-stream RC19."""
        sel = R.select(_round(), _starts())
        cs = _confirm_starts()
        cs["RC19"] = _ref(_WIN, 0.568)  # RC19 itself is no longer pinned here
        h = R.h6_verdicts(sel, _confirms(), cs)
        assert h["H6"] == "FAIL"
        assert h["per_selected"]["A"]["verdict"] == "FAIL (still pinned)"


# ----------------------------------------------------------------------------
class TestPooledRate:
    def test_counts_the_beta_001_seeds_of_s8_s10_and_s12(self):
        sel = R.select(
            _round(att={a: _PINNED for a in list(R.ARMS)[4:]}), _starts()
        )  # 4 of 12 pass
        p = R.pooled_rate(_s8_round(), _s10_round(), sel)
        assert p["sources"]["S8"] == {"k": 1, "n": 3}
        assert p["sources"]["S10"] == {"k": 3, "n": 5}
        assert p["sources"]["S12"] == {"k": 4, "n": 12}
        assert (p["k"], p["n"]) == (8, 20)
        assert p["rate"] == pytest.approx(0.4)
        lo, hi = p["wilson"]
        assert 0.0 < lo < 0.4 < hi < 1.0

    def test_incomplete_arms_are_not_counted(self):
        res = _round()
        del res["SEL-B-56"]
        sel = R.select(res, _starts())
        p = R.pooled_rate(_s8_round(), _s10_round(), sel)
        assert p["sources"]["S12"] == {"k": 11, "n": 11}
        assert (p["k"], p["n"]) == (15, 19)

    def test_missing_prior_rounds_do_not_crash(self):
        sel = R.select(_round(), _starts())
        p = R.pooled_rate({}, {}, sel)
        assert (p["k"], p["n"]) == (12, 12)
        assert R.pooled_rate({}, {}, {})["n"] == 0
        assert R.pooled_rate({}, {}, {})["rate"] is None


# ----------------------------------------------------------------------------
class TestSafety:
    def test_hsafe_counts_the_selection_reference_and_the_arms(self):
        assert R.hsafe(_round(), _starts())["verdict"] == "HOLDS"
        st = _starts()
        st["RC19"]["ATT26"]["pursuer_violations"] = 3
        h = R.hsafe(_round(), st)
        assert h["verdict"] == "FAILS" and h["pursuer_violation_steps"] == 3
        assert h["offenders"] == ["reference:RC19/ATT26"]
        res = _round()
        res["SEL-A-48"]["train_violations"] = 1
        assert R.hsafe(res, _starts())["offenders"] == ["SEL-A-48"]

    def test_hsafe_counts_the_confirmation_gates_and_both_references(self):
        sel = R.select(_round(), _starts())
        a = sel["A"]["selected"]
        c = _confirms()
        assert R.hsafe(_round(), _starts(), c, _confirm_starts())["verdict"] == "HOLDS"
        c[a]["pursuer_violations"] = 2
        h = R.hsafe(_round(), _starts(), c, _confirm_starts())
        assert h["verdict"] == "FAILS" and h["pursuer_violation_steps"] == 2
        assert h["offenders"] == [f"confirm:{a}"]
        cs = _confirm_starts()
        cs["FT-B-41"]["GREEDY"]["pursuer_violations"] = 5
        h2 = R.hsafe(_round(), _starts(), _confirms(), cs)
        assert h2["offenders"] == ["confirm-reference:FT-B-41/GREEDY"]
        cs2 = _confirm_starts()
        cs2["RC19"]["HUG7"]["pursuer_violations"] = 1
        h3 = R.hsafe(_round(), _starts(), _confirms(), cs2)
        assert h3["offenders"] == ["confirm-reference:RC19/HUG7"]

    def test_hsafe_is_incomplete_while_an_arm_is_missing(self):
        res = _round()
        del res["SEL-B-59"]
        h = R.hsafe(res, _starts())
        assert h["verdict"] == "INCOMPLETE" and h["missing"] == ["SEL-B-59"]


# ----------------------------------------------------------------------------
class TestDirection:
    def _d(self, **kw):
        return R.direction({"hsafe": "HOLDS", "H6": "PASS", "n_pass": 2} | kw)

    def test_hsafe_fails_stops_the_round(self):
        assert "Stop" in self._d(hsafe="FAILS")
        assert "Stop" in self._d(hsafe="FAILS", H6="FAIL", n_pass=0)

    def test_incomplete(self):
        assert "incomplete" in self._d(hsafe="INCOMPLETE")
        assert "incomplete" in self._d(H6="INCOMPLETE", n_pass=None)

    def test_pass_row(self):
        d = self._d()
        assert "train six, select" in d and "R3" in d

    def test_one_of_two_row(self):
        d = self._d(H6="PASS (1/2)", n_pass=1)
        assert (
            "half the time" in d and "larger batch" in d and "without re-running" in d
        )

    def test_fail_row(self):
        d = self._d(H6="FAIL", n_pass=0)
        assert "does not transfer" in d and "D120" in d
        assert "not selectable" in d


# ----------------------------------------------------------------------------
class TestSummary:
    def _summary(self, monkeypatch, tmp_path, **kw):
        monkeypatch.setattr(R, "S8_ROUND_PATH", tmp_path / "s8_round.json")
        monkeypatch.setattr(R, "S10_ROUND_PATH", tmp_path / "s10_round.json")
        (tmp_path / "s8_round.json").write_text(json.dumps(_s8_round()))
        (tmp_path / "s10_round.json").write_text(json.dumps(_s10_round()))
        res, st = _round(), _starts()
        return R.summarize(res, st, **kw)

    def test_complete_round(self, monkeypatch, tmp_path):
        sel = R.select(_round(), _starts())
        s = self._summary(
            monkeypatch,
            tmp_path,
            selection=sel,
            confirms=_confirms(),
            confirm_starts=_confirm_starts(),
        )
        assert s["spec"].endswith("selection_round_prereg_2026-09-18.md")
        assert s["hypotheses"]["H6"] == "PASS"
        assert s["hsafe"]["verdict"] == "HOLDS"
        assert set(s["arms"]) == set(R.ARMS)
        assert set(s["references"]) == {"RC19"}
        assert set(s["confirm_references"]) == {"RC19", "FT-B-41"}
        assert set(s["confirm"]) == set(_confirms())
        assert s["pooled_frontier_rate"]["n"] == 20
        assert s["complete"] is True
        assert "train six, select" in s["direction"]

    def test_stage_one_only(self, monkeypatch, tmp_path):
        s = self._summary(monkeypatch, tmp_path)
        assert s["hypotheses"]["H6"] == "INCOMPLETE"
        assert s["complete"] is False and "incomplete" in s["direction"]
        assert s["confirm"] == {} and s["confirm_references"] == {}
        assert s["pooled_frontier_rate"]["n"] == 8

    def test_write_figure(self, monkeypatch, tmp_path):
        pytest.importorskip("matplotlib")
        sel = R.select(_round(), _starts())
        s = self._summary(
            monkeypatch,
            tmp_path,
            selection=sel,
            confirms=_confirms(),
            confirm_starts=_confirm_starts(),
        )
        p = tmp_path / "s12_round.png"
        R.write_figure(_round(), _starts(), s, p)
        assert p.exists() and p.stat().st_size > 0

    def test_write_figure_before_the_confirmation(self, monkeypatch, tmp_path):
        pytest.importorskip("matplotlib")
        s = self._summary(monkeypatch, tmp_path)
        p = tmp_path / "s12_round_partial.png"
        R.write_figure(_round(), _starts(), s, p)
        assert p.exists() and p.stat().st_size > 0


# ----------------------------------------------------------------------------
class TestConfirmStage:
    """The confirmation stage's control flow: it must be resumable, and a gate
    that dies must leave its contact-step census on disk."""

    def _wire(self, tmp_path, monkeypatch, gate=None):
        monkeypatch.setattr(R, "OUT", tmp_path)
        monkeypatch.setattr(R.S8, "OUT", tmp_path)
        monkeypatch.setattr(R, "SELECTION_PATH", tmp_path / "s12_selection.json")
        monkeypatch.setattr(
            R, "CONFIRM_START_PATH", tmp_path / "s12_confirm_start.json"
        )
        res = _round()
        sel = R.select(res, _starts())
        R.SELECTION_PATH.write_text(json.dumps(sel))
        R.CONFIRM_START_PATH.write_text(json.dumps(_confirm_starts()))
        for a, r in res.items():
            (tmp_path / f"s12_arm_{a}.json").write_text(json.dumps(r))
        streams = []

        def fake_gate(model, opps, device, holder=None):
            streams.append(R.S8.S7.GATE_SEED0)
            return (gate or _gate_ok)(holder)

        monkeypatch.setattr(R.S8.S7, "gate_pursuer", fake_gate)
        monkeypatch.setattr(R.S8.S7, "gate_opponent_factory", lambda n, d: n)
        monkeypatch.setattr("stable_baselines3.PPO.load", lambda *a, **k: object())
        return sel, streams

    def test_an_already_confirmed_policy_is_skipped_not_refused(
        self, tmp_path, monkeypatch
    ):
        """A process that dies between the two batches must be re-runnable: the
        stop rule forbids re-gating a confirmed policy, not finishing the round."""
        sel, streams = self._wire(tmp_path, monkeypatch)
        a, b = sel["A"]["selected"], sel["B"]["selected"]
        done = _confirm(a, _WIN) | {"note": "from the first process"}
        R._confirm_file(a).write_text(json.dumps(done))
        out = R.confirm("cpu")
        assert streams == [R.CONFIRM_SEED0]  # exactly one gate ran, on 1 300 000+
        assert set(out) == {a, b}
        assert out[a]["note"] == "from the first process"  # A untouched
        assert R._confirm_file(b).exists()
        assert json.loads(R._confirm_file(a).read_text())["note"] == (
            "from the first process"
        )
        assert R.S8.S7.GATE_SEED0 == R.SEL_SEED0  # stream given back

    def test_refuses_only_when_every_selected_policy_is_confirmed(
        self, tmp_path, monkeypatch
    ):
        sel, _ = self._wire(tmp_path, monkeypatch)
        for bt in sel.values():
            R._confirm_file(bt["selected"]).write_text(
                json.dumps(_confirm(bt["selected"], _WIN))
            )
        with pytest.raises(SystemExit, match="every selected policy"):
            R.confirm("cpu")

    def test_a_crashed_gate_writes_its_census_and_re_raises(
        self, tmp_path, monkeypatch
    ):
        def dying(holder):
            if holder is not None:  # two opponents done before the crash
                holder["gate"] = {
                    "ATT25": _gate(0.3, 0.1) | {"pursuer_violations": 2},
                    "ATT26": _gate(0.3, 0.1) | {"pursuer_violations": 1},
                }
            raise RuntimeError("gate died")

        sel, _ = self._wire(tmp_path, monkeypatch, gate=dying)
        a = sel["A"]["selected"]
        with pytest.raises(RuntimeError, match="gate died"):
            R.confirm("cpu")
        rec = json.loads(R._confirm_file(a).read_text())
        assert rec["verdict"] == "UNDETERMINED" and "RuntimeError" in rec["reason"]
        assert rec["gate_violations"] == 3
        assert set(rec["gate_partial"]) == {"ATT25", "ATT26"}
        assert "gate" not in rec and rec["gate_seed0"] == R.CONFIRM_SEED0
        assert not R._confirm_file(sel["B"]["selected"]).exists()
        assert R.S8.S7.GATE_SEED0 == R.SEL_SEED0  # stream given back on the way out

    def test_hsafe_counts_a_crashed_confirmation(self):
        sel = R.select(_round(), _starts())
        a, b = sel["A"]["selected"], sel["B"]["selected"]
        crashed = {
            "arm": a,
            "verdict": "UNDETERMINED",
            "gate_violations": 3,
            "gate_partial": {},
        }
        h = R.hsafe(_round(), _starts(), {a: crashed}, _confirm_starts())
        assert h["verdict"] == "FAILS" and h["pursuer_violation_steps"] == 3
        assert h["offenders"] == [f"confirm:{a}"]
        assert f"confirm:{a}" in h["missing"]
        # a clean crash record (0 steps) is still INCOMPLETE, never HOLDS
        clean = dict(crashed, gate_violations=0)
        h2 = R.hsafe(_round(), _starts(), {a: clean}, _confirm_starts())
        assert h2["verdict"] == "INCOMPLETE"
        assert h2["missing"] == [
            f"confirm:{a}",
            "confirm:1/2 selected policies have a record",
        ]
        # one batch gated cleanly, the other not started
        h3 = R.hsafe(_round(), _starts(), {a: _confirm(a, _WIN)}, _confirm_starts())
        assert h3["verdict"] == "INCOMPLETE"
        assert h3["missing"] == ["confirm:1/2 selected policies have a record"]
        # both gated: nothing missing
        h4 = R.hsafe(
            _round(),
            _starts(),
            {a: _confirm(a, _WIN), b: _confirm(b, _WIN)},
            _confirm_starts(),
        )
        assert h4["verdict"] == "HOLDS" and h4["missing"] == []

    def test_hsafe_before_the_stage_starts_is_unchanged(self):
        """No confirmation record at all is stage 1, not a missing confirmation."""
        assert R.hsafe(_round(), _starts())["missing"] == []
        assert R.hsafe(_round(), _starts(), {}, None)["verdict"] == "HOLDS"

    def test_h6_is_incomplete_without_the_confirmation_references(self):
        """`s12_confirm_start.json` gone after a gate ran: degrade, never crash."""
        sel = R.select(_round(), _starts())
        for cs in (None, {}, {"FT-B-41": _ref(_WIN, 0.6)}):
            h = R.h6_verdicts(sel, _confirms(), cs)
            assert h["H6"] == "INCOMPLETE" and h["artefact"] is None
            assert h["per_selected"]["A"]["verdict"] == "INCOMPLETE"

    def test_the_paper_artefact_survives_a_missing_ft_b_41_row(self):
        sel = R.select(_round(), _starts())
        cs = _confirm_starts()
        del cs["FT-B-41"]
        h = R.h6_verdicts(sel, _confirms(), cs)
        assert h["H6"] == "PASS" and h["ft_b41"] is None
        assert h["paper_artefact"]["policy"] == h["artefact"]
        assert "reported alone" in h["paper_artefact"]["note"]
