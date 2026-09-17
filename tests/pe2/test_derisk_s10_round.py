"""scripts/derisk_s10_confirm.py — the confirmatory round's own decisions.

S10 is S8's machinery under S10's constants (spec:
docs/pathway/shield_remedy_3b_prereg_2026-09-17.md), so these tests cover what
S10 itself decides: the overrides it pushes onto the reused S8/S7/S6 modules,
H5 over five arms, and the spec's direction rows.
"""

import importlib.util
import json
import pathlib
import types

import pytest

_PATH = (
    pathlib.Path(__file__).resolve().parents[2] / "scripts" / "derisk_s10_confirm.py"
)


def _runner():
    spec = importlib.util.spec_from_file_location("derisk_s10_confirm", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _runner()


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


def _starts(ft_a2_35=(0.47, 0.47, 0.43), ft_a2_35_capture=0.672):
    """The three references of spec section 2 on the gate stream (D119's numbers
    for RC19 and FT-A2-35)."""
    return {
        "RC19": _ref((0.60, 0.56, 0.58), 0.568),
        "RCM28": _ref((0.10, 0.09, 0.10), 0.13, greedy_deadlock=0.04),
        "FT-A2-35": _ref(ft_a2_35, ft_a2_35_capture, greedy_deadlock=0.20),
    }


def _arm(name, att, greedy_capture):
    return {
        "arm": name,
        "seed": R.ARMS[name]["seed"],
        "family": R.ARMS[name]["family"],
        "train_violations": 0,
        "probe_violations": 0,
        "final": {
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


_WIN = (0.30, 0.32, 0.31)  # beats RC19 on all three held-out attackers
_PINNED = (0.60, 0.61, 0.62)  # matches RC19: no deadlock win
_KEEPS_CAPTURE = 0.55  # not significantly below RC19's 0.568
_LOSES_CAPTURE = 0.30


def _full(att=(_WIN,) * 5, cap=(_KEEPS_CAPTURE,) * 5):
    """Five FT-B arms; by default all five PASS both clauses."""
    return {
        f"FT-B-{s}": _arm(f"FT-B-{s}", a, c)
        for s, a, c in zip(R.SEEDS, att, cap, strict=True)
    }


def _n_of_5(n_pass):
    """n arms that pass both clauses; the rest stay pinned."""
    att = tuple(_WIN if i < n_pass else _PINNED for i in range(5))
    return _full(att=att)


class TestProtocol:
    def test_constants(self):
        assert R.SEEDS == (40, 41, 42, 43, 44)
        assert R.FAMILY == "FT-B" and R.BETA == 0.01
        assert set(R.ARMS) == {f"FT-B-{s}" for s in (40, 41, 42, 43, 44)}
        assert R.ARMS["FT-B-40"] == {"seed": 40, "family": "FT-B", "beta": 0.01}
        assert (R.TOTAL, R.EVERY) == (5_000_000, 500_000)
        assert (R.PROBE_SEED0, R.GATE_SEED0) == (1_260_000, 1_250_000)
        assert (R.SEEDS2, R.PROBE2_SEED0, R.GATE2_SEED0) == (
            (45, 46, 47),
            1_270_000,
            1_280_000,
        )
        assert (R.GATE_N, R.EVAL_N) == (500, 200)
        assert R.PASS_MIN == 3  # arms out of five (spec section 3)
        assert R.REFERENCES == ("RC19", "RCM28", "FT-A2-35")
        assert R.REPORT_ONLY == "FT-A2-35" and R.REPORT_ONLY not in R.ARMS
        assert R.ARTEFACT_PREFIX == "s10"
        assert (R.RACE_BAR, R.S7_RACE_RATE) == (0.58, 0.147)

    def test_the_learner_is_exactly_s8s_secondary_arm(self):
        """Spec section 2: nothing about the learner changes. The budget, the
        family and its weights come from S8; FT-B is an anchored fine-tune
        because it is not S8's CONTROL family."""
        assert (R.TOTAL, R.EVERY) == (R.S8.TOTAL, R.S8.EVERY)
        assert R.BETA == R.S8.BETAS["FT-A2"]
        assert R.FAMILY != R.S8.CONTROL
        assert R.S8.WEIGHTS == R.S8.WEIGHTS and R.S8.MIXTURE == R.S8.MIXTURE
        # the per-arm test is S8's, unchanged: >= 2 of 3 held-out attackers
        assert R.S8.PASS_MIN == 2 and R.S8.HELD_OUT == ("ATT25", "ATT26", "ATT27")

    def test_the_s8_module_is_pointed_at_the_s10_round(self):
        assert R.S8.ARTEFACT_PREFIX == "s10"
        assert R.S8.START_PATH == R.OUT / "s10_start.json"
        assert R.S8.RACE_START_PATH == R.OUT / "s10_race_start.json"
        assert R.S8.ARMS == R.ARMS and R.S8.SEEDS == R.SEEDS
        assert R.S8.REFERENCES == R.REFERENCES
        assert R.S8.REFERENCE_CKPT["FT-A2-35"] == (
            R.OUT / "ckpts_FT-A2-35" / "FT-A2-35_5000000.zip"
        )
        assert set(R.S8.REFERENCE_CKPT) >= set(R.REFERENCES)
        assert (R.S8.GATE_SEED0, R.S8.PROBE_SEED0) == (1_250_000, 1_260_000)
        assert (R.S8.SEEDS2, R.S8.PROBE2_SEED0, R.S8.GATE2_SEED0) == (
            (45, 46, 47),
            1_270_000,
            1_280_000,
        )
        # every artefact path follows the prefix
        assert R.S8._arm_file("FT-B-40") == R.OUT / "s10_arm_FT-B-40.json"
        assert R.S8._probe_file("FT-B-40") == R.OUT / "s10_probe_FT-B-40.json"
        assert R.S8._round_file("json") == R.OUT / "s10_round.json"
        assert R.S8._race_file() == R.OUT / "s10_race.json"
        assert R.S8._race_arm_glob() == "s10race_arm_*.json"

    def test_the_s7_module_is_pointed_at_the_s10_stream(self):
        """S7's `gate_pursuer` reads GATE_SEED0/GATE_N/OUT as its OWN globals, so
        S8's `apply_overrides` must run again after S10 changes them."""
        assert R.S8.S7.GATE_SEED0 == R.GATE_SEED0
        assert R.S8.S7.GATE_N == R.GATE_N and R.S8.S7.EVAL_N == R.EVAL_N
        assert R.S8.S7.OUT == R.OUT
        assert R.S8.S7.RACE_START_PATH == R.OUT / "s10_race_start.json"
        assert R.S8.S7.SEEDS2 == (45, 46, 47)
        assert (R.S8.S7.PROBE2_SEED0, R.S8.S7.GATE2_SEED0) == (1_270_000, 1_280_000)

    def test_preflight_is_s8s(self):
        """V-A0 is S8's and is not re-run: S10 changes nothing the preflight
        tested, so the guard reads S8's own artefact."""
        assert R.PREFLIGHT_PATH == R.OUT / "s8_preflight.json"
        assert R.S8.PREFLIGHT_PATH == R.PREFLIGHT_PATH
        assert R.require_preflight is R.S8.require_preflight

    def test_guards(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R.S8, "OUT", tmp_path)
        monkeypatch.setattr(R.S8, "START_PATH", tmp_path / "s10_start.json")
        monkeypatch.setattr(R.S8, "PREFLIGHT_PATH", tmp_path / "s8_preflight.json")
        with pytest.raises(SystemExit, match="preflight missing"):
            R.require_preflight()
        (tmp_path / "s8_preflight.json").write_text('{"pass": true}')
        R.require_preflight()  # satisfied: no raise

        entry = {"deadlock_ind": [], "capture_ind": []}
        two = {r: dict.fromkeys(R.S8.GATE_OPPONENTS, entry) for r in ("RC19", "RCM28")}
        (tmp_path / "s10_start.json").write_text(json.dumps(two))
        with pytest.raises(SystemExit, match="FT-A2-35"):
            R.S8.load_start()  # the third reference is mandatory
        three = {r: dict.fromkeys(R.S8.GATE_OPPONENTS, entry) for r in R.REFERENCES}
        (tmp_path / "s10_start.json").write_text(json.dumps(three))
        assert set(R.S8.load_start()) == set(R.REFERENCES)

        R.S8.require_arm_not_run("FT-B-40")  # nothing on disk yet
        (tmp_path / "s8_arm_FT-B-40.json").write_text("{}")
        R.S8.require_arm_not_run("FT-B-40")  # S8's namespace: not this round's
        (tmp_path / "s10_arm_FT-B-40.json").write_text("{}")
        with pytest.raises(SystemExit, match="stop rule"):
            R.S8.require_arm_not_run("FT-B-40")

    def test_load_results_reads_the_s10_namespace(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R.S8, "OUT", tmp_path)
        (tmp_path / "s10_arm_FT-B-40.json").write_text('{"arm": "FT-B-40"}')
        (tmp_path / "s8_arm_FT-A2-35.json").write_text('{"arm": "FT-A2-35"}')
        (tmp_path / "s10race_arm_ATK4-45.json").write_text('{"arm": "ATK4-45"}')
        assert set(R.S8.load_results()) == {"FT-B-40"}


class TestVerdicts:
    def test_three_of_five_passes(self):
        h = R.h5_verdicts(_n_of_5(3), _starts())
        assert h["H5"] == "PASS" and h["n_pass"] == 3
        assert h["passing"] == ["FT-B-40", "FT-B-41", "FT-B-42"]
        assert h["per_arm"]["FT-B-43"]["verdict"] == "FAIL (still pinned)"

    def test_two_of_five_fails_with_the_flavour_counts(self):
        h = R.h5_verdicts(_n_of_5(2), _starts())
        assert h["H5"] == "FAIL" and h["n_pass"] == 2 and h["artefact"] is None
        assert h["fail_counts"] == {
            "still pinned": 3,
            "capture lost": 0,
            "both": 0,
        }
        lost = R.h5_verdicts(
            _full(cap=(_LOSES_CAPTURE,) * 5),
            _starts(),
        )
        assert lost["H5"] == "FAIL" and lost["n_pass"] == 0
        assert lost["fail_counts"]["capture lost"] == 5
        both = R.h5_verdicts(
            _full(att=(_PINNED,) * 5, cap=(_LOSES_CAPTURE,) * 5), _starts()
        )
        assert both["per_arm"]["FT-B-40"]["verdict"] == "FAIL (both)"
        assert both["fail_counts"] == {
            "still pinned": 5,
            "capture lost": 5,
            "both": 5,
        }

    def test_a_missing_arm_is_incomplete(self):
        res = _full()
        del res["FT-B-42"]
        h = R.h5_verdicts(res, _starts())
        assert h["H5"] == "INCOMPLETE" and h["artefact"] is None
        assert h["per_arm"]["FT-B-42"]["verdict"] == "INCOMPLETE"
        assert h["fail_counts"] is None and h["n_pass"] is None
        no_final = _full()
        del no_final["FT-B-42"]["final"]
        assert R.h5_verdicts(no_final, _starts())["H5"] == "INCOMPLETE"

    def test_artefact_tiebreaks(self):
        # lowest mean held-out deadlock
        h = R.h5_verdicts(
            _full(att=(_WIN, (0.20, 0.21, 0.22), _WIN, _WIN, _WIN)), _starts()
        )
        assert h["artefact"] == "FT-B-41"
        # tie on deadlock -> highest Greedy capture
        h2 = R.h5_verdicts(
            _full(cap=(0.53, 0.55, 0.54, 0.53, 0.53)),
            _starts(),
        )
        assert h2["artefact"] == "FT-B-41"
        # tie on both -> lowest seed
        assert R.h5_verdicts(_full(), _starts())["artefact"] == "FT-B-40"

    def test_ft_a2_35_is_never_the_artefact_and_casts_no_vote(self):
        """Spec section 2: FT-A2-35 is a reported reference, not an arm."""
        res = _full(att=(_PINNED,) * 5)  # every real arm fails
        res["FT-A2-35"] = _arm("FT-B-40", _WIN, _KEEPS_CAPTURE) | {"arm": "FT-A2-35"}
        h = R.h5_verdicts(res, _starts())
        assert "FT-A2-35" not in h["per_arm"]
        assert h["H5"] == "FAIL" and h["artefact"] is None and h["n_pass"] == 0

    def test_the_per_arm_test_is_s8s(self):
        h = R.h5_verdicts(_full(), _starts())
        v = h["per_arm"]["FT-B-40"]
        assert v["passes"] == 3 and set(v["tests"]) == set(R.S8.HELD_OUT)
        assert v["capture_guard"]["failed"] is False
        assert v["vs_rcm28"]["rcm28_greedy_capture"] == pytest.approx(0.13)


class TestSafety:
    def test_hsafe_counts_all_three_references(self):
        assert R.S8.hsafe_verdict(_full(), _starts())["verdict"] == "HOLDS"
        st = _starts()
        st["FT-A2-35"]["ATT26"]["pursuer_violations"] = 3
        h = R.S8.hsafe_verdict(_full(), st)
        assert h["verdict"] == "FAILS" and h["pursuer_violation_steps"] == 3
        assert h["offenders"] == ["reference:FT-A2-35/ATT26"]

    def test_hsafe_covers_the_five_arms_and_stage_2(self):
        res = _full()
        del res["FT-B-44"]
        assert R.S8.hsafe_verdict(res, _starts())["missing"] == ["FT-B-44"]
        race = {"pursuer_violation_steps": 2, "offenders": ["ATK4-45"]}
        h = R.S8.hsafe_verdict(_full(), _starts(), race)
        assert h["verdict"] == "FAILS" and "race:ATK4-45" in h["offenders"]


class TestDirection:
    def _d(self, **kw):
        h = {"hsafe": "HOLDS", "H5": "PASS", "n_pass": 5, "H_RACE": None}
        return R.direction(h | kw)

    def test_hsafe_fails_stops_the_round(self):
        assert "Stop" in self._d(hsafe="FAILS")
        assert "Stop" in self._d(hsafe="FAILS", H5="FAIL", n_pass=0)

    def test_incomplete(self):
        assert "incomplete" in self._d(hsafe="INCOMPLETE")
        assert "incomplete" in self._d(H5="INCOMPLETE", n_pass=None)

    def test_pass_row(self):
        d = self._d()
        assert "not structural" in d and "R3 becomes a method" in d
        assert "--race-start" in d
        assert "H-RACE-4 holds" in self._d(H_RACE="HOLDS")
        assert "adapt" in self._d(H_RACE="FAILS")
        assert "stage 2 incomplete" in self._d(H_RACE="INCOMPLETE")

    def test_fail_rows(self):
        some = self._d(H5="FAIL", n_pass=2)
        assert "2/5" in some and "H-SEL" in some and "not a re-tune" in some
        assert "sometimes" in self._d(H5="FAIL", n_pass=1)
        none = self._d(H5="FAIL", n_pass=0)
        assert "fluke" in none and "one observation" in none and "D119" in none


class TestRace:
    def test_race_overrides(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R.S8, "OUT", tmp_path)
        monkeypatch.setattr(R.S8, "RACE_START_PATH", tmp_path / "s10_race_start.json")
        S6 = types.SimpleNamespace()
        R.race_setup(S6, tmp_path / "artefact.zip")
        assert S6.OUT == tmp_path
        assert S6.START_PATH == tmp_path / "s10_race_start.json"
        assert S6.ARTEFACT_PREFIX == "s10race"
        assert (S6.PROBE_SEED0, S6.GATE_SEED0) == (1_270_000, 1_280_000)
        assert (S6.GATE_N, S6.EVAL_N) == (R.GATE_N, R.EVAL_N)
        assert (S6.TOTAL_LONG, S6.EVERY_LONG) == (10_000_000, 1_000_000)
        assert S6.SEEDS == (45, 46, 47)
        assert set(S6.ARMS) == {"ATK4-45", "ATK4-46", "ATK4-47"}
        assert S6.ARMS["ATK4-45"] == {
            "family": "CTL",
            "defender": "D3",
            "seed": 45,
            "beta": 0.0,
            "zero_reward": False,
        }
        assert S6.DEFENDERS["D3"]["ckpt"] == tmp_path / "artefact.zip"
        assert S6.DEFENDERS["D3"]["shield_obs"] is True

    def test_s8s_race_entry_points_use_the_s10_wiring(self, tmp_path, monkeypatch):
        """`S8.race_start`/`race_arm` call `race_setup` by name on the S8 module,
        so S10's must be the one they find."""
        assert R.S8.race_setup is R.race_setup
        monkeypatch.setattr(R.S8, "OUT", tmp_path)
        monkeypatch.setattr(R.S8, "RACE_START_PATH", tmp_path / "s10_race_start.json")
        (tmp_path / "s10_round.json").write_text(
            json.dumps({"summary": {"hypotheses": {"artefact": "FT-B-41"}}})
        )
        (tmp_path / "s10_arm_FT-B-41.json").write_text(
            json.dumps({"final": {"ckpt": str(tmp_path / "ck.zip")}})
        )
        assert R.S8._artefact_ckpt() == tmp_path / "ck.zip"
        with pytest.raises(SystemExit, match="unknown race arm 'ATK3-37'"):
            R.S8.race_arm("ATK3-37", "cpu")

    def test_stage_2_needs_a_passing_stage_1(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R.S8, "OUT", tmp_path)
        with pytest.raises(SystemExit, match="run --summarize first"):
            R.S8._artefact_ckpt()
        (tmp_path / "s10_round.json").write_text(
            json.dumps({"summary": {"hypotheses": {"artefact": None}}})
        )
        with pytest.raises(SystemExit, match="artefact"):
            R.S8._artefact_ckpt()

    def test_race_summary_is_s8s_over_three_fresh_seeds(self):
        def arm(rate):
            return {
                "train_pursuer_violations": 0,
                "probe_pursuer_violations": 0,
                "final": {
                    "deadlock_ind": _ind(rate),
                    "pursuer": {
                        "deadlock_episode_rate": rate,
                        "escape_rate": 0.7,
                    },
                    "pursuer_violations": 0,
                },
            }

        arms = {f"ATK4-{s}": arm(0.12) for s in R.SEEDS2}
        r = R.S8.race_summary(arms)
        assert r["H_RACE"] == "HOLDS" and r["race_bar"] == 0.58
        assert r["s7_race_rate"] == 0.147 and r["below_s7"] is True
        assert set(r["per_arm"]) == set(arms)
        del arms["ATK4-47"]
        assert R.S8.race_summary(arms)["H_RACE"] == "INCOMPLETE"


class TestSummary:
    def test_summarize_reports_three_references_and_the_ft_a2_35_row(self):
        s = R.summarize(_full(), _starts())
        assert s["spec"].endswith("shield_remedy_3b_prereg_2026-09-17.md")
        assert s["hypotheses"]["H5"] == "PASS"
        assert s["hsafe"]["verdict"] == "HOLDS"
        assert set(s["references"]) == {"RC19", "RCM28", "FT-A2-35"}
        assert s["references"]["RC19"]["GREEDY"]["capture"] == pytest.approx(0.568)
        row = s["reference_ft_a2_35"]
        assert row["greedy_capture"] == pytest.approx(0.672)
        assert row["heldout_deadlock"]["ATT25"] == pytest.approx(0.47)
        assert row["mean_deadlock_heldout"] == pytest.approx((0.47 + 0.47 + 0.43) / 3)
        assert "no vote" in row["note"]
        assert set(s["arms"]) == set(R.ARMS)
        assert s["complete"] is True and "--race-start" in s["direction"]

    def test_summarize_is_incomplete_without_all_five(self):
        res = _full()
        del res["FT-B-44"]
        s = R.summarize(res, _starts())
        assert s["complete"] is False
        assert s["hypotheses"]["H5"] == "INCOMPLETE" and "incomplete" in s["direction"]

    def test_write_figure(self, tmp_path):
        pytest.importorskip("matplotlib")
        results, starts = _full(), _starts()
        p = tmp_path / "s10_round.png"
        R.write_figure(results, starts, R.summarize(results, starts), p)
        assert p.exists() and p.stat().st_size > 0
