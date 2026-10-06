"""scripts/derisk_s19_planner.py — the S19 round's cell table and verdicts.

Spec: docs/pathway/planner_intercept_prereg_2026-10-06.md INCLUDING amendments
A1 and A2. Everything here is either a constant the prereg fixes, a verdict
rule it states, or one of the properties the round's only claim rests on:

- the cell table really is 2 x 2 x 3, and the ONLY differences between the rows
  and the columns are the law object and the shield. That is asserted on the
  BUILT env, not on the table's own bookkeeping, because the claim "a capture
  difference is attributable to the law" is false the moment any other
  ingredient differs;
- the ATK column REFUSES to run without its frozen checkpoint;
- A1.1's zero guard: a LINE law-pin fraction of 0 is VACUOUS, never PASS;
- A2.2's null is the OBSTACLE-FREE bit-identity, and it FAILS when the two laws
  differ there;
- CG-C's seeds are the three construct seeds and nothing else;
- the provenance of every record carries the pre-registration's sha256.

No real episodes beyond one smoke `play_episode` per row.
"""

import hashlib
import importlib.util
import json
import pathlib

import numpy as np
import pytest

from pe2.config import EnvConfig
from pe2.planner import PlannerInterceptLaw
from pe2.structured import HOLD, InterceptLaw

_PATH = (
    pathlib.Path(__file__).resolve().parents[2] / "scripts" / "derisk_s19_planner.py"
)


def _runner():
    spec = importlib.util.spec_from_file_location("derisk_s19_planner", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _runner()
N = 100  # episodes in the synthetic records below


# ----------------------------------------------------------------------------
# synthetic records
# ----------------------------------------------------------------------------
def _ind(rate, n=N):
    k = round(rate * n)
    return [i < k for i in range(n)]


def _cell(name, capture, pin_frac=0.0, viol=0, n=N):
    """One cell record in the shape `run_cell` writes, with only the fields the
    verdicts read. The per-episode arrays are consistent with the aggregate so
    a verdict that recomputed one from the other would still agree."""
    spec = R.CELLS[name]
    pin = _ind(pin_frac, n)
    runs = [60 if p else 0 for p in pin]
    return {
        "name": name,
        "row": spec["row"],
        "col": spec["col"],
        "panel": spec["panel"],
        "opponent": spec["opponent"],
        "shielded": spec["shielded"],
        "n_episodes": n,
        "capture_ind": _ind(capture, n),
        "capture_rate": round(capture * n) / n,
        "violations": viol,
        "longest_law_run": runs,
        "law_pin_ge_40": pin,
        "n_unreachable": 0 if spec["row"] == "PLAN" else None,
        "aggregate": {
            "capture_rate": round(capture * n) / n,
            "episode_frac_law_pin_ge_40": sum(pin) / n,
            "law_pin_steps": R.LAW_PIN_STEPS,
            "longest_law_run_median": float(np.median(runs)),
            "longest_law_run_p90": float(np.percentile(runs, 90)),
        },
        "provenance": {"spec_sha256": "x"},
    }


def _round(capture, pins=None, viol=None):
    """A whole round. `capture` maps (row, col) -> {opponent: rate}; `pins`
    maps (row, 'SHIELD') -> {opponent: fraction}."""
    pins = pins or {}
    viol = viol or {}
    out = {}
    for (row, col), rates in capture.items():
        for o, rate in rates.items():
            name = R.cell_name(row, col, o)
            out[name] = _cell(
                name,
                rate,
                pin_frac=pins.get((row, col), {}).get(o, 0.0),
                viol=viol.get(name, 0),
            )
    return out


ALL = ("GREEDY", "HUG", "ATK")


def _flat(v):
    return dict.fromkeys(ALL, v)


def _pass_round():
    """PLAN captures more under the shield on all three opponents, carries the
    smaller shield cost on all three, and removes the law pins on all three."""
    return _round(
        {
            ("LINE", "OPEN"): _flat(0.60),
            ("PLAN", "OPEN"): _flat(0.65),
            ("LINE", "SHIELD"): _flat(0.30),
            ("PLAN", "SHIELD"): _flat(0.60),
        },
        pins={("LINE", "SHIELD"): _flat(0.40), ("PLAN", "SHIELD"): _flat(0.10)},
    )


# ----------------------------------------------------------------------------
# the cell table
# ----------------------------------------------------------------------------
class TestCellTable:
    def test_twelve_cells_two_rows_two_columns_three_opponents(self):
        assert len(R.CELLS) == 12
        assert list(R.ROWS) == ["LINE", "PLAN"]
        assert list(R.COLS) == ["OPEN", "SHIELD"]
        assert list(R.OPPONENTS) == ["GREEDY", "HUG", "ATK"]
        assert set(R.CELLS) == {
            f"{r}-{c}_{o}" for r in R.ROWS for c in R.COLS for o in R.OPPONENTS
        }

    def test_pre_registered_constants(self):
        assert R.N_EPISODES == 500
        assert R.GATE_SEED0 == 2_000_000
        assert R.CONSTRUCT_SEED0 == 1_990_000
        assert R.P_GATE == 0.05
        assert R.PASS_MIN == 2
        assert R.NULL_N == 50
        assert R.LAW_PIN_STEPS == 40  # A1.1, and `pe2.metrics.DEADLOCK_STEPS`
        assert R.H3_RELATIVE_DROP == 0.50
        assert R.OMP_THREADS == "4"
        assert R.ATK_CKPT == pathlib.Path(
            "results/derisk/ckpts_ATK4-45/ATK4-45_10000000.zip"
        )

    def test_every_cell_runs_the_same_hold_and_observation_flags(self):
        assert {c["hold"] for c in R.CELLS.values()} == {HOLD} == {10}
        assert (R.SHIELD_OBS, R.ESCAPE_OBS) == (True, True)

    def test_row_law_is_the_only_law_difference(self):
        cfg = EnvConfig()
        # LINE passes the `SearchMDP` default through, so the LINE row is the
        # law the project has always measured and not a re-wiring of it.
        assert R.row_law("LINE", cfg) is None
        assert isinstance(R.row_law("PLAN", cfg), PlannerInterceptLaw)

    @pytest.mark.parametrize("opponent", ALL)
    def test_built_envs_differ_only_in_law_and_shield(self, opponent, monkeypatch):
        """The round's whole attribution rests on this: build all four cells of
        one opponent and assert that law and shield are the only differences."""
        monkeypatch.setattr(R, "opponent_factory", lambda *a, **k: _still_evader)
        cfg = EnvConfig()
        envs = {
            (r, c): R.build_cell_env(R.cell_name(r, c, opponent), cfg)
            for r in R.ROWS
            for c in R.COLS
        }
        for (row, col), env in envs.items():
            assert type(env).__name__ == "BranchTaggedSearchMDP"
            assert env.cfg == cfg
            assert env.hold == HOLD
            assert env.shield_obs is True
            assert env.escape_obs is True
            assert env.stall_penalty == 0.0
            assert env.intervention_penalty == 0.0
            assert env.zero_reward is False
            assert env._driver is env.inner  # reward_wrapper=None
            # the law: exactly the row's
            want = InterceptLaw if row == "LINE" else PlannerInterceptLaw
            assert type(env.law) is want
            # the shield: exactly the column's
            if col == "SHIELD":
                assert type(env.shield).__name__ == "CBFShield"
                assert (env.shield.margin, env.shield.gamma) == (0.0, 0.2)
            else:
                assert env.shield is None
        # and the observation width is the same everywhere, so the search
        # policy is literally the same function of the same input
        assert len({e.observation_space.shape for e in envs.values()}) == 1

    def test_unknown_cell_name_refuses(self):
        with pytest.raises(SystemExit):
            R.run_cells(["PLAN-SHIELD_NOPE"])

    def test_default_config_is_the_locked_envconfig(self):
        assert R.cell_config() == EnvConfig()
        assert R.null_config().n_physical_obstacles == 0


class _StillEvader:
    """A do-nothing opponent, so the table assertions never load a checkpoint."""

    def reset(self):
        return None

    def act(self, obs, env):
        return (0.0, 0.0)


def _still_evader():
    return _StillEvader()


# ----------------------------------------------------------------------------
# the ATK refusal
# ----------------------------------------------------------------------------
class TestAtkRefusal:
    def test_require_atk_ckpt_refuses_when_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "ATK_CKPT", tmp_path / "nope.zip")
        with pytest.raises(SystemExit, match="ATK4-45"):
            R.require_atk_ckpt()

    def test_opponent_factory_refuses_when_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "ATK_CKPT", tmp_path / "nope.zip")
        with pytest.raises(SystemExit):
            R.opponent_factory("ATK", EnvConfig(), "cpu")

    def test_run_cells_refuses_before_running_anything(self, tmp_path, monkeypatch):
        """The refusal lands BEFORE the first episode: a pass that ran two cells
        and then refused would leave a half-measured round."""
        monkeypatch.setattr(R, "ATK_CKPT", tmp_path / "nope.zip")
        monkeypatch.setattr(R, "OUT", tmp_path)

        def _boom(*a, **k):
            raise AssertionError("an episode ran before the refusal")

        monkeypatch.setattr(R, "run_cell", _boom)
        with pytest.raises(SystemExit):
            R.run_cells(["LINE-SHIELD_GREEDY", "PLAN-SHIELD_ATK"])

    def test_scripted_opponents_need_no_checkpoint(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "ATK_CKPT", tmp_path / "nope.zip")
        from pe2.scripted import GreedyEvader, ObstacleHuggingEvader

        assert R.opponent_factory("GREEDY", EnvConfig(), "cpu") is GreedyEvader
        assert R.opponent_factory("HUG", EnvConfig(), "cpu") is ObstacleHuggingEvader


# ----------------------------------------------------------------------------
# H-PLAN-1
# ----------------------------------------------------------------------------
class TestHPlan1:
    def test_passes_when_plan_captures_more_under_the_shield(self):
        v = R.h_plan_1(_pass_round())
        assert v["verdict"] == "PASS"
        assert v["n_pass"] == 3
        assert all(b["direction"] == "PLAN > LINE" for b in v["opponents"].values())

    def test_two_of_three_is_enough(self):
        recs = _round(
            {
                ("LINE", "OPEN"): _flat(0.60),
                ("PLAN", "OPEN"): _flat(0.60),
                ("LINE", "SHIELD"): {"GREEDY": 0.30, "HUG": 0.30, "ATK": 0.60},
                ("PLAN", "SHIELD"): {"GREEDY": 0.60, "HUG": 0.60, "ATK": 0.30},
            }
        )
        v = R.h_plan_1(recs)
        assert v["verdict"] == "PASS"
        assert v["n_pass"] == 2
        # the direction is reported on the opponent that went the WRONG way
        assert v["opponents"]["ATK"]["direction"] == "PLAN < LINE"
        assert v["opponents"]["ATK"]["verdict"] == "FAIL"
        assert v["opponents"]["ATK"]["delta"] == pytest.approx(-0.30)

    def test_fails_when_the_direction_inverts_and_still_reports_it(self):
        recs = _round(
            {
                ("LINE", "OPEN"): _flat(0.60),
                ("PLAN", "OPEN"): _flat(0.60),
                ("LINE", "SHIELD"): _flat(0.60),
                ("PLAN", "SHIELD"): _flat(0.30),
            }
        )
        v = R.h_plan_1(recs)
        assert v["verdict"] == "FAIL"
        assert v["n_pass"] == 0
        for b in v["opponents"].values():
            assert b["direction"] == "PLAN < LINE"
            assert b["delta"] < 0
            assert "p" in b

    def test_a_tie_is_not_a_pass(self):
        recs = _round(
            {
                ("LINE", "OPEN"): _flat(0.60),
                ("PLAN", "OPEN"): _flat(0.60),
                ("LINE", "SHIELD"): _flat(0.40),
                ("PLAN", "SHIELD"): _flat(0.40),
            }
        )
        v = R.h_plan_1(recs)
        assert v["verdict"] == "FAIL"
        assert all(b["direction"] == "PLAN == LINE" for b in v["opponents"].values())

    def test_incomplete_when_a_cell_is_missing(self):
        recs = _pass_round()
        del recs["PLAN-SHIELD_HUG"]
        v = R.h_plan_1(recs)
        assert v["verdict"] == "INCOMPLETE"
        assert v["opponents"]["HUG"]["missing"] == ["PLAN-SHIELD_HUG"]


# ----------------------------------------------------------------------------
# H-PLAN-2 (as amended by A2.2)
# ----------------------------------------------------------------------------
class TestHPlan2:
    def test_passes_when_the_shield_cost_is_smaller_under_plan(self):
        v = R.h_plan_2(_pass_round())
        assert v["verdict"] == "PASS"
        assert v["cost_part"] is True
        assert v["gain_part"] is True
        b = v["opponents"]["GREEDY"]
        assert b["cost_line"] == pytest.approx(0.30)
        assert b["cost_plan"] == pytest.approx(0.05)
        assert b["cost_reduction"] == pytest.approx(0.25)

    def test_fails_when_the_shield_gain_is_not_significant(self):
        """A2.2 kept the gain conjunct: a smaller cost with no significant
        SHIELD gain is not H-PLAN-2."""
        recs = _round(
            {
                ("LINE", "OPEN"): _flat(0.90),
                ("PLAN", "OPEN"): _flat(0.65),
                ("LINE", "SHIELD"): _flat(0.60),
                ("PLAN", "SHIELD"): _flat(0.60),
            }
        )
        v = R.h_plan_2(recs)
        assert v["cost_part"] is True  # 0.30 vs 0.05
        assert v["gain_part"] is False
        assert v["verdict"] == "FAIL"

    def test_fails_when_the_cost_is_larger_under_plan(self):
        recs = _round(
            {
                ("LINE", "OPEN"): _flat(0.60),
                ("PLAN", "OPEN"): _flat(0.95),
                ("LINE", "SHIELD"): _flat(0.30),
                ("PLAN", "SHIELD"): _flat(0.60),
            }
        )
        v = R.h_plan_2(recs)
        assert v["gain_part"] is True
        assert v["cost_part"] is False
        assert v["verdict"] == "FAIL"

    def test_the_open_difference_is_reported_and_never_gated(self):
        """A2.2: the OPEN column is unshielded, NOT obstacle-free. A large OPEN
        difference must not stop H-PLAN-2 from passing."""
        recs = _round(
            {
                ("LINE", "OPEN"): _flat(0.40),
                ("PLAN", "OPEN"): _flat(0.70),  # +0.30, far outside the old band
                ("LINE", "SHIELD"): _flat(0.10),
                ("PLAN", "SHIELD"): _flat(0.60),
            }
        )
        v = R.h_plan_2(recs)
        assert v["verdict"] == "PASS"
        assert v["open_band_conjunct"] == "dropped by amendment A2.2"
        for b in v["opponents"].values():
            assert b["open_plan_minus_line"] == pytest.approx(0.30)

    def test_incomplete_when_an_open_cell_is_missing(self):
        recs = _pass_round()
        del recs["PLAN-OPEN_ATK"]
        v = R.h_plan_2(recs)
        assert v["verdict"] == "INCOMPLETE"
        assert v["opponents"]["ATK"]["missing"] == ["PLAN-OPEN_ATK"]


# ----------------------------------------------------------------------------
# H-PLAN-3 (as restated by A1.1), including the zero guard
# ----------------------------------------------------------------------------
class TestHPlan3:
    def test_passes_on_a_fifty_percent_relative_fall(self):
        recs = _round(
            {
                ("LINE", "OPEN"): _flat(0.60),
                ("PLAN", "OPEN"): _flat(0.60),
                ("LINE", "SHIELD"): _flat(0.30),
                ("PLAN", "SHIELD"): _flat(0.60),
            },
            pins={("LINE", "SHIELD"): _flat(0.20), ("PLAN", "SHIELD"): _flat(0.10)},
        )
        v = R.h_plan_3(recs)
        assert v["verdict"] == "PASS"
        assert v["n_pass"] == 3
        assert v["opponents"]["GREEDY"]["relative_drop"] == pytest.approx(0.50)

    def test_just_under_the_bar_fails(self):
        recs = _round(
            {
                ("LINE", "OPEN"): _flat(0.60),
                ("PLAN", "OPEN"): _flat(0.60),
                ("LINE", "SHIELD"): _flat(0.30),
                ("PLAN", "SHIELD"): _flat(0.60),
            },
            pins={("LINE", "SHIELD"): _flat(0.20), ("PLAN", "SHIELD"): _flat(0.11)},
        )
        v = R.h_plan_3(recs)
        assert v["verdict"] == "FAIL"
        assert v["n_pass"] == 0
        assert v["opponents"]["HUG"]["relative_drop"] == pytest.approx(0.45)

    def test_a_rise_in_the_pin_fraction_fails(self):
        recs = _round(
            {
                ("LINE", "OPEN"): _flat(0.60),
                ("PLAN", "OPEN"): _flat(0.60),
                ("LINE", "SHIELD"): _flat(0.30),
                ("PLAN", "SHIELD"): _flat(0.60),
            },
            pins={("LINE", "SHIELD"): _flat(0.10), ("PLAN", "SHIELD"): _flat(0.30)},
        )
        v = R.h_plan_3(recs)
        assert v["verdict"] == "FAIL"
        assert v["opponents"]["GREEDY"]["relative_drop"] < 0

    def test_zero_line_fraction_is_vacuous_never_pass(self):
        """A1.1's guard. 0 -> 0 is not a 50 % reduction and must never read as
        one — that is the uninformative shape A1.1 restated H-PLAN-3 to avoid."""
        recs = _round(
            {
                ("LINE", "OPEN"): _flat(0.60),
                ("PLAN", "OPEN"): _flat(0.60),
                ("LINE", "SHIELD"): _flat(0.30),
                ("PLAN", "SHIELD"): _flat(0.60),
            },
            pins={("LINE", "SHIELD"): _flat(0.0), ("PLAN", "SHIELD"): _flat(0.0)},
        )
        v = R.h_plan_3(recs)
        assert v["verdict"] == "VACUOUS"
        assert v["n_vacuous"] == 3
        assert v["n_pass"] == 0
        for b in v["opponents"].values():
            assert b["verdict"] == "VACUOUS"
            assert b["relative_drop"] is None

    def test_one_vacuous_opponent_cannot_be_counted_toward_the_bar(self):
        """Two real passes still PASS; a single pass plus a vacuous opponent
        cannot reach 2/3 and is VACUOUS rather than FAIL, because the shortfall
        is an absent phenomenon and not evidence against the planner."""
        base = {
            ("LINE", "OPEN"): _flat(0.60),
            ("PLAN", "OPEN"): _flat(0.60),
            ("LINE", "SHIELD"): _flat(0.30),
            ("PLAN", "SHIELD"): _flat(0.60),
        }
        two = R.h_plan_3(
            _round(
                base,
                pins={
                    ("LINE", "SHIELD"): {"GREEDY": 0.20, "HUG": 0.20, "ATK": 0.0},
                    ("PLAN", "SHIELD"): {"GREEDY": 0.05, "HUG": 0.05, "ATK": 0.0},
                },
            )
        )
        assert two["verdict"] == "PASS"
        assert (two["n_pass"], two["n_vacuous"]) == (2, 1)
        one = R.h_plan_3(
            _round(
                base,
                pins={
                    ("LINE", "SHIELD"): {"GREEDY": 0.20, "HUG": 0.0, "ATK": 0.0},
                    ("PLAN", "SHIELD"): {"GREEDY": 0.05, "HUG": 0.0, "ATK": 0.0},
                },
            )
        )
        assert one["verdict"] == "VACUOUS"
        assert (one["n_pass"], one["n_vacuous"]) == (1, 2)

    def test_one_pass_and_two_real_failures_is_fail_not_vacuous(self):
        recs = _round(
            {
                ("LINE", "OPEN"): _flat(0.60),
                ("PLAN", "OPEN"): _flat(0.60),
                ("LINE", "SHIELD"): _flat(0.30),
                ("PLAN", "SHIELD"): _flat(0.60),
            },
            pins={
                ("LINE", "SHIELD"): {"GREEDY": 0.20, "HUG": 0.20, "ATK": 0.20},
                ("PLAN", "SHIELD"): {"GREEDY": 0.05, "HUG": 0.20, "ATK": 0.20},
            },
        )
        v = R.h_plan_3(recs)
        assert v["verdict"] == "FAIL"
        assert (v["n_pass"], v["n_vacuous"]) == (1, 0)

    def test_median_and_p90_are_reported_beside_it(self):
        v = R.h_plan_3(_pass_round())
        b = v["opponents"]["GREEDY"]
        assert set(b["longest_law_run_median"]) == {"LINE", "PLAN"}
        assert set(b["longest_law_run_p90"]) == {"LINE", "PLAN"}
        assert b["law_pin_steps"] == 40

    def test_incomplete_when_a_shield_cell_is_missing(self):
        recs = _pass_round()
        del recs["LINE-SHIELD_GREEDY"]
        assert R.h_plan_3(recs)["verdict"] == "INCOMPLETE"


# ----------------------------------------------------------------------------
# H-SAFE
# ----------------------------------------------------------------------------
class TestHSafe:
    def test_holds_with_no_contact_in_the_shield_cells(self):
        v = R.h_safe(_pass_round())
        assert v["verdict"] == "HOLDS"
        assert v["violation_steps"] == 0
        assert len(v["gated_cells"]) == 6

    def test_fails_on_a_single_contact_step_in_a_shield_cell(self):
        recs = _pass_round()
        recs["PLAN-SHIELD_HUG"]["violations"] = 1
        v = R.h_safe(recs)
        assert v["verdict"] == "FAILS"
        assert v["offenders"] == ["PLAN-SHIELD_HUG"]
        assert v["violation_steps"] == 1

    def test_open_contacts_are_reported_not_gated(self):
        """An unshielded pursuer is allowed to touch things; that is the point
        of the OPEN column and must not read as a safety failure."""
        recs = _pass_round()
        recs["LINE-OPEN_GREEDY"]["violations"] = 523
        v = R.h_safe(recs)
        assert v["verdict"] == "HOLDS"
        assert (
            v["unshielded_contact_steps_reported_not_gated"]["LINE-OPEN_GREEDY"] == 523
        )

    def test_incomplete_when_a_shield_cell_is_missing(self):
        recs = _pass_round()
        del recs["PLAN-SHIELD_ATK"]
        assert R.h_safe(recs)["verdict"] == "INCOMPLETE"


# ----------------------------------------------------------------------------
# the NULL (amendment A2.2)
# ----------------------------------------------------------------------------
class TestNullOpen:
    def test_it_is_the_obstacle_free_arena_and_the_construct_stream(self):
        assert R.NULL_N == 50
        assert R.null_config().n_physical_obstacles == 0
        assert R.null_config().n_obstacles == EnvConfig().n_obstacles  # layout kept
        # the prereg's section 4 name and the artefact's name are one function
        assert R.null_identity is R.null_open

    def test_holds_on_an_obstacle_free_arena(self):
        v = R.null_open(n=3)
        assert v["verdict"] == "HOLDS"
        assert v["n_differing"] == 0
        assert v["seed0"] == R.CONSTRUCT_SEED0
        assert v["pre_registered"] is False  # n != NULL_N, and it says so
        assert "obstacle-free" in v["scenario"]
        # A2.1's residual must never fire on an empty map
        assert v["n_unreachable"] == 0

    def test_defaults_are_the_pre_registered_fifty_from_the_construct_stream(
        self, monkeypatch
    ):
        seen = {}

        class _Spy:
            law = PlannerInterceptLaw(EnvConfig())

            def play_episode(self, predict, seed):
                seen.setdefault("seeds", []).append(seed)
                return {"log": 1}

        monkeypatch.setattr(R, "build_cell_env", lambda *a, **k: _Spy())
        v = R.null_open()
        assert v["n_episodes"] == 50
        assert v["seed0"] == 1_990_000
        assert v["pre_registered"] is True
        assert sorted(set(seen["seeds"])) == [1_990_000 + i for i in range(50)]

    def test_fails_when_the_plan_law_differs_in_open_space(self, monkeypatch):
        """The load-bearing negative: a PLAN law that is NOT bit-identical on an
        empty map must make the null FAIL, because then no capture number in the
        round is about obstruction."""

        class _Divergent(PlannerInterceptLaw):
            def act(self, obs):
                a = super().act(obs)
                return None if a is None else np.asarray(a) * np.float32(0.5)

        monkeypatch.setattr(R, "PlannerInterceptLaw", _Divergent)
        v = R.null_open(n=2)
        assert v["verdict"] == "FAILS"
        assert v["n_differing"] == 2
        assert v["differing_seeds"] == [1_990_000, 1_990_001]

    def test_direction_stops_the_round_when_the_null_fails(self):
        v = R.verdicts(
            _pass_round(),
            {
                "verdict": "FAILS",
                "n_differing": 7,
                "n_episodes": 50,
                "differing_seeds": [1],
            },
        )
        text = R.direction(v)
        assert "NULL FAILS" in text
        assert "obstruction" in text
        # the H-PLAN clauses must not be read at all
        assert "H-PLAN-1 PASS" not in text

    def test_direction_refuses_to_interpret_an_unevaluated_null(self):
        text = R.direction(R.verdicts(_pass_round(), None))
        assert "NULL has not been evaluated" in text


# ----------------------------------------------------------------------------
# direction() — prereg section 4's refutation paragraph
# ----------------------------------------------------------------------------
HOLDS = {
    "verdict": "HOLDS",
    "n_differing": 0,
    "n_episodes": 50,
    "differing_seeds": [],
}


class TestDirection:
    def test_hsafe_failure_precedes_everything(self):
        recs = _pass_round()
        recs["LINE-SHIELD_GREEDY"]["violations"] = 3
        text = R.direction(R.verdicts(recs, HOLDS))
        assert text.startswith("H-SAFE FAILS")
        assert "nothing else is read" in text

    def test_incompleteness_precedes_the_hypotheses(self):
        recs = _pass_round()
        del recs["PLAN-SHIELD_ATK"]
        text = R.direction(R.verdicts(recs, HOLDS))
        assert "incomplete" in text

    def test_the_pass_narrative(self):
        text = R.direction(R.verdicts(_pass_round(), HOLDS))
        assert "H-PLAN-1 PASS" in text
        assert "H-PLAN-2 PASS" in text
        assert "H-PLAN-3 PASS" in text
        assert "all-step" in text or "ALL-STEP" in text  # A1.2's scope note

    def test_h_plan_1_failure_sends_s20_back_to_cg_c(self):
        recs = _round(
            {
                ("LINE", "OPEN"): _flat(0.60),
                ("PLAN", "OPEN"): _flat(0.60),
                ("LINE", "SHIELD"): _flat(0.60),
                ("PLAN", "SHIELD"): _flat(0.30),
            },
            pins={("LINE", "SHIELD"): _flat(0.40), ("PLAN", "SHIELD"): _flat(0.40)},
        )
        text = R.direction(R.verdicts(recs, HOLDS))
        assert "H-PLAN-1 FAILS" in text
        assert "re-diagnose" in text
        assert "s19_gifs" in text

    def test_the_open_difference_is_named_as_reported_not_gated(self):
        text = R.direction(R.verdicts(_pass_round(), HOLDS))
        assert "Reported, not gated" in text
        assert "unshielded" in text.lower()

    def test_a_vacuous_h_plan_3_is_narrated_as_vacuous(self):
        recs = _round(
            {
                ("LINE", "OPEN"): _flat(0.60),
                ("PLAN", "OPEN"): _flat(0.60),
                ("LINE", "SHIELD"): _flat(0.30),
                ("PLAN", "SHIELD"): _flat(0.60),
            },
            pins={("LINE", "SHIELD"): _flat(0.0), ("PLAN", "SHIELD"): _flat(0.0)},
        )
        text = R.direction(R.verdicts(recs, HOLDS))
        assert "H-PLAN-3 VACUOUS" in text


# ----------------------------------------------------------------------------
# CG-C
# ----------------------------------------------------------------------------
class TestGifs:
    def test_the_seed_list_is_exactly_the_three_construct_seeds(self):
        assert R.construct_seeds() == (1_990_000, 1_990_001, 1_990_002)
        assert R.GIF_N == 3
        assert R.GIF_OPPONENT == "GREEDY"

    def test_twelve_gifs_four_panels_times_three_seeds(self):
        assert R.PANELS == ("LINE-OPEN", "LINE-SHIELD", "PLAN-OPEN", "PLAN-SHIELD")
        assert len(R.PANELS) * R.GIF_N == 12

    def test_gif_paths_are_panel_and_seed(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        assert R.gif_path("PLAN-SHIELD", 1_990_001) == (
            tmp_path / "s19_gifs" / "PLAN-SHIELD_1990001.gif"
        )

    def test_each_panel_is_captioned_from_its_greedy_cell(self):
        assert R.gif_cell("PLAN-SHIELD") == "PLAN-SHIELD_GREEDY"
        assert R.gif_cell("LINE-OPEN") == "LINE-OPEN_GREEDY"

    def test_a_panel_without_a_record_refuses_before_rendering(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(R, "OUT", tmp_path)

        def _boom(*a, **k):
            raise AssertionError("a frame was rendered before the refusal")

        monkeypatch.setattr(R, "render_panel", _boom)
        with pytest.raises(SystemExit, match="never fabricated"):
            R.write_gifs(["PLAN-SHIELD"])

    def test_unknown_panel_refuses(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        with pytest.raises(SystemExit, match="unknown panel"):
            R.write_gifs(["PLAN-SHIELD_GREEDY"])

    def test_the_title_quotes_measured_numbers_only(self):
        rec = _cell("PLAN-SHIELD_GREEDY", 0.6, pin_frac=0.1)
        title = R.gif_title("PLAN-SHIELD", rec, 1_990_000)
        assert "PLAN-SHIELD vs GREEDY" in title
        assert "episode 1990000" in title
        assert "0.600" in title and "0.100" in title
        assert "PASS" not in title and "FAIL" not in title  # no verdict

    def test_existing_gifs_are_skipped(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        (tmp_path / "s19_gifs").mkdir(parents=True)
        for s in R.construct_seeds():
            R.gif_path("LINE-OPEN", s).write_bytes(b"")
        monkeypatch.setattr(
            R, "render_panel", lambda *a, **k: pytest.fail("re-rendered")
        )
        assert R.write_gifs(["LINE-OPEN"]) == []

    def test_the_recorder_is_the_cells_own_env(self):
        from pe2.recorder import BranchTaggedSearchMDP

        assert issubclass(R.CellRecorder, BranchTaggedSearchMDP)


# ----------------------------------------------------------------------------
# provenance
# ----------------------------------------------------------------------------
class TestProvenance:
    def test_it_carries_the_prereg_sha256(self):
        p = R.provenance()
        want = hashlib.sha256(R.spec_path().read_bytes()).hexdigest()
        assert p["spec"] == "docs/pathway/planner_intercept_prereg_2026-10-06.md"
        assert p["spec_sha256"] == want
        assert len(p["spec_sha256"]) == 64

    def test_it_carries_the_pe2_package_and_the_thread_setting(self):
        p = R.provenance()
        assert p["pe2_file"].endswith("pe2/__init__.py")
        assert p["omp_num_threads"] == "4"
        assert "git_commit" in p

    def test_the_spec_file_exists_at_the_recorded_path(self):
        assert R.spec_path().exists()


# ----------------------------------------------------------------------------
# the smoke: one real episode per row, to prove the env builds
# ----------------------------------------------------------------------------
class TestSmoke:
    @pytest.mark.parametrize("row", ["LINE", "PLAN"])
    def test_one_shielded_episode_per_row(self, row):
        from pe2.structured import OrbitSearch

        cfg = R.cell_config()
        mdp = R.build_cell_env(R.cell_name(row, "SHIELD", "GREEDY"), cfg)
        log = mdp.play_episode(OrbitSearch(cfg).predict, R.GATE_SEED0)
        assert log["outcome"] in ("capture", "timeout")
        b = log["branch"]
        assert b["n_steps"] > 0
        assert set(b["steps_by_branch"]) == {"law", "search"}
        assert b["intervention_total"] >= 0.0
        assert isinstance(b["longest_pinned_run_by_branch"]["law"], int)

    def test_a_cell_record_carries_every_pre_registered_field(
        self, tmp_path, monkeypatch
    ):
        """One two-episode cell, so the RECORD's schema is checked against the
        prereg rather than against this test's imagination."""
        monkeypatch.setattr(R, "OUT", tmp_path)
        rec = R.run_cell("PLAN-SHIELD_GREEDY", n_episodes=2)
        for key in (
            "capture_ind",
            "capture_rate",
            "wilson_capture",
            "violations",
            "intervention_total",
            "longest_law_run",
            "law_pin_ge_40",
            "n_unreachable",
            "branch",
            "provenance",
        ):
            assert key in rec, key
        assert len(rec["capture_ind"]) == 2
        assert len(rec["branch"]) == 2
        assert rec["seed0"] == R.GATE_SEED0
        assert rec["n_unreachable"] is not None  # PLAN counts A2.1's residual
        assert rec["aggregate"]["law_pin_steps"] == 40
        assert rec["aggregate"]["intervention_scope"] == "all steps (amendment A1.2)"
        path = R.write_cell(rec)
        assert json.loads(path.read_text())["name"] == "PLAN-SHIELD_GREEDY"

    def test_the_line_row_has_no_unreachable_counter(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        rec = R.run_cell("LINE-OPEN_GREEDY", n_episodes=1)
        assert rec["law"] == "InterceptLaw"
        assert rec["n_unreachable"] is None
        # No shield: the recorder's filter fields are the "nothing acted"
        # record, so nothing in the OPEN column is pinned BY A FILTER.
        assert rec["aggregate"]["episode_frac_law_pin_ge_40"] == 0.0
        assert rec["intervention_total"] == [0.0]
