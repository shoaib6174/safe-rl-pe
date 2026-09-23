"""scripts/derisk_s18_boundary.py — the boundary-condition round's decisions.

Spec: docs/pathway/boundary_condition_prereg_2026-09-23.md (sections 2-5, 7)
plus amendment A1. Everything here is either a constant the spec fixes, a
verdict rule it states, or one of the three properties the round's only claim
rests on:

- the arm table really is 2 x 3 x 3, with the penalty on exactly the PEN row and
  the right reward variant on each column;
- `lambda`, `w_vis` and `w_search` are READ from the calibration artefact and an
  arm that cannot read it REFUSES to run — a hardcoded weight would confound the
  column comparison with penalty strength, which is the one thing the design
  exists to avoid;
- the zero-motion metric counts the LEARNER'S REQUEST, not the executed command
  (the D129 error), and the gate env is built unwrapped (amendment A1.3).

Nothing here trains.
"""

import importlib.util
import json
import pathlib

import numpy as np
import pytest

_PATH = (
    pathlib.Path(__file__).resolve().parents[2] / "scripts" / "derisk_s18_boundary.py"
)


def _runner():
    spec = importlib.util.spec_from_file_location("derisk_s18_boundary", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _runner()
N = 100  # episodes in the synthetic gate entries below
S = 1200 * N  # env steps in one synthetic gate cell
D = 120 * N  # learner decisions in one synthetic gate cell

CALIBRATION = {
    "lambda": 0.5,
    "w_vis": 0.02,
    "w_search": 3.0,
    "means": {"intervention": 2.0, "vis": 12.5, "search": 0.0833},
    "n_episodes": 200,
    "seed0": 1_860_000,
    "provenance": {"git_commit": "deadbee"},
}


def _calibrate(tmp_path, monkeypatch, **overrides):
    """Task A's artefact as a fixture. It does not exist in this repo yet and
    must never be a precondition of these tests."""
    monkeypatch.setattr(R, "OUT", tmp_path)
    (tmp_path / "s18_lambda.json").write_text(json.dumps(CALIBRATION | overrides))
    return tmp_path / "s18_lambda.json"


def _ind(rate, n=N):
    k = round(rate * n)
    return [i < k for i in range(n)]


def _entry(capture, deadlock, zero, intervention=1.0, viol=0):
    """One gate cell in the shape `gate_model` writes."""
    n_zero = round(zero * D)
    return {
        "capture_ind": _ind(capture),
        "deadlock_ind": _ind(deadlock),
        "capture_rate": capture,
        "capture": {"k": round(capture * N), "n": N, "rate": capture},
        "pursuer": {
            "immobile_step_rate": 0.3,
            "deadlock_episode_rate": deadlock,
            "stall_step_rate": 0.3,
            "intervention_rate": 0.4,
            "infeasibility_rate": 0.0,
            "backup_rate": 0.0,
            "min_clearance_floor": 0.2,
        },
        "n_steps": S,
        "n_immobile": round(0.3 * S),
        "n_episodes": N,
        "pursuer_violations": viol,
        "zero_motion": {
            "n_zero_motion": n_zero,
            "n_decisions": D,
            "fraction": n_zero / D,
        },
        "intervention": {
            "total": intervention * N,
            "n_episodes": N,
            "per_episode_mean": intervention,
        },
    }


def _arm(name, capture, deadlock, zero, intervention=1.0, viol=0):
    return {
        "arm": name,
        **R.ARMS[name],
        "verdict": "MEASURED",
        "train_violations": 0,
        "probe_violations": 0,
        "checkpoints": [],
        "final": {
            "ckpt": f"ckpts_S18-{name}/{name}_5000000.zip",
            "pursuer_violations": viol,
            "gate": {
                # the robustness opponent is harder; the primary metric is GREEDY
                "GREEDY": _entry(capture, deadlock, zero, intervention),
                "HUG": _entry(capture / 2, deadlock, zero, intervention),
            },
        },
    }


# cell: (GreedyEvader capture, deadlock rate, zero-motion request fraction)
_DEFAULT = {cell: (0.40, 0.30, 0.20) for cell in R.CELL_ORDER}


def _full(**overrides):
    """A complete set of records: six cells, three seeds.

    Each keyword is a CELL name with its hyphen as an underscore (`SPARSE_PEN`)
    whose value is `(capture, deadlock, zero_motion)` applied to all three of
    its seeds."""
    cells = dict(_DEFAULT)
    for key, value in overrides.items():
        cells[key.replace("_", "-")] = value
    return {
        f"{cell}-{s}": _arm(f"{cell}-{s}", cap, dead, zero)
        for cell, (cap, dead, zero) in cells.items()
        for s in R.SEEDS
    }


def _crossover(**overrides):
    """The pre-registered PASS pattern: the penalty lowers capture in SPARSE and
    in DENSE-FLAT, and raises it in DENSE-VARY, with the zero-motion request
    fraction rising exactly where capture falls."""
    cells = {
        "SPARSE_PEN": (0.20, 0.30, 0.50),
        "DFLAT_PEN": (0.20, 0.30, 0.50),
        "DVARY_PEN": (0.60, 0.30, 0.10),
    }
    return _full(**(cells | overrides))


class _Full:
    """A policy that always asks for full speed. Under the filter its executed
    speed is repeatedly clamped to ~0, which is exactly the case that separates
    the REQUEST from the EXECUTION."""

    def predict(self, obs, deterministic=True):
        return np.array([1.0, 0.0], dtype=np.float32), None


class _Stopped:
    """A policy that always asks for no motion at all."""

    def predict(self, obs, deterministic=True):
        return np.array([-1.0, 0.0], dtype=np.float32), None


# ----------------------------------------------------------------------------
# sections 2-4: the protocol and the arm table
# ----------------------------------------------------------------------------
class TestProtocol:
    def test_constants(self):
        assert R.SEEDS == (70, 71, 72)
        assert (R.TOTAL, R.EVERY) == (5_000_000, 500_000)
        assert R.GATE_SEED0 == 1_900_000
        assert R.PROBE_SEED0 == 1_950_000
        assert R.GATE_N == 500
        assert R.GAMMA == 0.99
        assert (R.R_ACQ, R.SHIELD_OBS, R.ESCAPE_OBS) == (0.0, True, True)
        assert R.HOLD == 10
        assert (R.P_GATE, R.PASS_MIN) == (0.05, 2)
        assert R.GATE_OPPONENTS == ("GREEDY", "HUG")
        assert R.CAPTURE_OPPONENT == "GREEDY"
        assert R.ARTEFACT_PREFIX == "s18"
        assert R.SPEC.endswith("boundary_condition_prereg_2026-09-23.md")

    def test_amendment_a1_makes_every_stream_disjoint(self):
        """A1.1: calibration 1 860 000, construct gate 1 880 000, the gate
        1 900 000, probes 1 950 000. The gate must not overlap the probes."""
        assert R.GATE_SEED0 + R.GATE_N <= R.PROBE_SEED0
        assert R.PROBE_SEED0 + R.EVAL_N < 2_000_000
        assert R.GATE_SEED0 > 1_880_000  # after the construct gate's stream

    def test_the_recovery_is_off_and_d129s_penalty_is_not_an_arm(self):
        """Section 3: `stall_penalty` stays untouched in the code and is NOT a
        cell of this round. Section 4: RecoveryShieldV2 OFF, because D131's
        remedy would mask the effect under test."""
        assert R.STALL_PENALTY == 0.0
        assert R.RECOVERY is False
        assert all("stall_penalty" not in spec for spec in R.CELLS.values())

    def test_the_zero_motion_threshold_is_the_filters_own(self):
        from pe2.shield import STALL_V_FRAC

        assert R.ZERO_MOTION_V_FRAC == STALL_V_FRAC

    def test_the_arm_table_is_six_cells_by_three_seeds(self):
        assert R.CELL_ORDER == (
            "SPARSE-BASE",
            "SPARSE-PEN",
            "DFLAT-BASE",
            "DFLAT-PEN",
            "DVARY-BASE",
            "DVARY-PEN",
        )
        assert len(R.ARMS) == 18
        assert set(R.ARMS) == {f"{c}-{s}" for c in R.CELL_ORDER for s in (70, 71, 72)}

    def test_the_cell_and_seed_are_stored_not_parsed_off_the_name(self):
        assert R.ARMS["SPARSE-PEN-70"]["cell"] == "SPARSE-PEN"
        assert R.ARMS["SPARSE-PEN-70"]["seed"] == 70
        assert R.ARMS["DVARY-BASE-72"]["column"] == "DVARY"
        assert R.ARMS["DVARY-BASE-72"]["row"] == "BASE"

    def test_the_penalty_is_on_exactly_the_pen_row(self):
        for arm, spec in R.ARMS.items():
            assert spec["penalty"] is (spec["row"] == "PEN"), arm
        assert sum(s["penalty"] for s in R.ARMS.values()) == 9

    def test_the_three_columns_are_the_specs_table(self):
        """Section 2: SPARSE is neither dense nor varying, DENSE-FLAT is dense
        but flat on `I`, DENSE-VARY is both. DENSE-FLAT is the discriminating
        cell and must not be labelled as varying."""
        assert R.COLUMNS["SPARSE"] == {
            "weight": None,
            "dense": False,
            "varies_on_I": False,
        }
        assert R.COLUMNS["DFLAT"] == {
            "weight": "w_vis",
            "dense": True,
            "varies_on_I": False,
        }
        assert R.COLUMNS["DVARY"] == {
            "weight": "w_search",
            "dense": True,
            "varies_on_I": True,
        }

    def test_every_arm_has_the_same_observation_width(self):
        assert R.obs_width() == 16  # 11 env + 3 shield_obs + 2 escape_obs

    def test_a_mismatched_checkpoint_is_refused_at_load(self):
        class _Model:
            def __init__(self, width):
                self.observation_space = type("S", (), {"shape": (width,)})()

        R._check_width("SPARSE-BASE-70", _Model(16))  # agrees: no complaint
        with pytest.raises(SystemExit, match="do NOT gate this arm"):
            R._check_width("SPARSE-BASE-70", _Model(14))


# ----------------------------------------------------------------------------
# section 3 + A1.2: the calibration is read, never chosen
# ----------------------------------------------------------------------------
class TestCalibration:
    def test_an_arm_refuses_to_run_without_the_calibration_file(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(R, "OUT", tmp_path)
        with pytest.raises(SystemExit, match="missing calibration"):
            R.load_calibration()
        with pytest.raises(SystemExit, match="missing calibration"):
            R.run_arm("SPARSE-PEN-70")
        with pytest.raises(SystemExit, match="missing calibration"):
            R.gate_arm("SPARSE-PEN-70")

    def test_the_refusal_comes_before_anything_is_written(self, tmp_path, monkeypatch):
        """An arm that cannot read the calibration must leave no trace: no
        record, no checkpoint directory, nothing for the stop rule to trip on
        the next time the round is driven."""
        monkeypatch.setattr(R, "OUT", tmp_path)
        with pytest.raises(SystemExit):
            R.run_arm("DVARY-PEN-71")
        assert list(tmp_path.iterdir()) == []

    def test_it_reads_the_three_weights(self, tmp_path, monkeypatch):
        _calibrate(tmp_path, monkeypatch)
        c = R.load_calibration()
        assert (c["lambda"], c["w_vis"], c["w_search"]) == (0.5, 0.02, 3.0)
        assert c["means"]["intervention"] == 2.0
        assert c["seed0"] == 1_860_000

    @pytest.mark.parametrize("key", ["lambda", "w_vis", "w_search"])
    def test_a_missing_weight_is_refused(self, key, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        data = {k: v for k, v in CALIBRATION.items() if k != key}
        (tmp_path / "s18_lambda.json").write_text(json.dumps(data))
        with pytest.raises(SystemExit, match="missing"):
            R.load_calibration()

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf"), "0.5"])
    def test_a_non_positive_or_non_finite_weight_is_refused(
        self, bad, tmp_path, monkeypatch
    ):
        """A zero weight means the calibrated term never fired, so the column it
        defines would be mislabelled — exactly what the construct gate exists to
        prevent."""
        monkeypatch.setattr(R, "OUT", tmp_path)
        (tmp_path / "s18_lambda.json").write_text(
            json.dumps(CALIBRATION | {"w_search": bad})
        )
        with pytest.raises(SystemExit, match="w_search"):
            R.load_calibration()

    def test_unreadable_json_is_refused(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        (tmp_path / "s18_lambda.json").write_text("{not json")
        with pytest.raises(SystemExit, match="not readable JSON"):
            R.load_calibration()

    def test_no_weight_is_hardcoded_anywhere_in_the_module(self):
        """The one confound the design cannot survive: a penalty strength or a
        dense weight fixed in the source rather than measured."""
        assert not hasattr(R, "LAMBDA")
        assert not hasattr(R, "W_VIS")
        assert not hasattr(R, "W_SEARCH")
        for spec in R.CELLS.values():
            assert "lambda" not in spec and "intervention_penalty" not in spec

    def test_the_penalty_tracks_the_file_rather_than_a_constant(
        self, tmp_path, monkeypatch
    ):
        _calibrate(tmp_path, monkeypatch)
        first = R.make_env("SPARSE-PEN-70", R.load_calibration())
        assert first.intervention_penalty == 0.5
        _calibrate(tmp_path, monkeypatch, **{"lambda": 0.125})
        second = R.make_env("SPARSE-PEN-70", R.load_calibration())
        assert second.intervention_penalty == 0.125

    def test_a_base_arm_carries_no_penalty_whatever_lambda_is(self):
        assert R.intervention_penalty_for("SPARSE-BASE-70", CALIBRATION) == 0.0
        assert R.intervention_penalty_for("DVARY-BASE-72", CALIBRATION) == 0.0
        assert R.intervention_penalty_for("DVARY-PEN-72", CALIBRATION) == 0.5

    def test_one_lambda_is_shared_by_all_three_penalty_cells(self):
        """Section 3: "One `lambda`, shared by all three penalty cells, so the
        column comparison is not confounded by penalty strength"."""
        pens = {
            R.intervention_penalty_for(a, CALIBRATION)
            for a, s in R.ARMS.items()
            if s["penalty"]
        }
        assert pens == {CALIBRATION["lambda"]}


class TestRewardVariant:
    def test_sparse_takes_no_wrapper_at_all(self):
        """Not a dense wrapper at weight zero: the SPARSE column must be the
        S16/S17 setting itself."""
        assert R.reward_wrapper_for("SPARSE", CALIBRATION) is None

    def test_dense_flat_pays_the_calibrated_visibility_weight(self):
        from pe2.config import EnvConfig
        from pe2.exploitability import SingleAgentWrapper

        inner = SingleAgentWrapper(EnvConfig(), "pursuer", R.GreedyEvader)
        wrapped = R.reward_wrapper_for("DFLAT", CALIBRATION)(inner)
        assert wrapped.w_vis == 0.02 and wrapped.w_search == 0.0
        assert wrapped.w_evis == 0.0 and wrapped.w_pbrs == 0.0

    def test_dense_vary_pays_the_calibrated_staleness_weight(self):
        from pe2.config import EnvConfig
        from pe2.exploitability import SingleAgentWrapper

        inner = SingleAgentWrapper(EnvConfig(), "pursuer", R.GreedyEvader)
        wrapped = R.reward_wrapper_for("DVARY", CALIBRATION)(inner)
        assert wrapped.w_search == 3.0 and wrapped.w_vis == 0.0
        assert wrapped.tracker is not None  # the staleness grid actually exists

    @pytest.mark.parametrize(
        ("arm", "driver", "penalty"),
        [
            ("SPARSE-BASE-70", None, 0.0),
            ("SPARSE-PEN-71", None, 0.5),
            ("DFLAT-BASE-72", "DenseRewardWrapper", 0.0),
            ("DFLAT-PEN-70", "DenseRewardWrapper", 0.5),
            ("DVARY-BASE-71", "DenseRewardWrapper", 0.0),
            ("DVARY-PEN-72", "DenseRewardWrapper", 0.5),
        ],
    )
    def test_the_training_env_of_every_cell(self, arm, driver, penalty):
        env = R.make_env(arm, CALIBRATION)
        assert env.intervention_penalty == penalty
        assert env.stall_penalty == 0.0  # D129's quantity, never an arm here
        if driver is None:
            assert env._driver is env.inner
        else:
            assert type(env._driver).__name__ == driver
        assert env.observation_space.shape == (R.obs_width(),)
        assert env.shield_obs is True and env.escape_obs is True
        assert env.hold == 10 and env.r_acq == 0.0 and env.zero_reward is False
        assert type(env.shield).__name__ == "CBFShield"

    def test_the_learner_is_s17s_ppo_at_gamma_099(self):
        from stable_baselines3 import PPO

        model = R.make_learner(
            "DVARY-PEN-72", R.make_env("DVARY-PEN-72", CALIBRATION), "cpu"
        )
        assert isinstance(model, PPO)
        assert model.gamma == 0.99
        assert model.seed == 72
        assert model.observation_space.shape == (16,)


# ----------------------------------------------------------------------------
# H-ZERO's mechanism metric: the REQUEST, not the execution (the D129 error)
# ----------------------------------------------------------------------------
class TestIsZeroMotion:
    """`(a0 + 1) / 2 * v_max <= 0.05 * v_max`, on the learner's nominal."""

    def _cfg(self):
        from pe2.config import EnvConfig

        return EnvConfig()

    def test_a_full_speed_request_is_not_zero_motion(self):
        assert R.is_zero_motion([1.0, 0.0], self._cfg()) is False
        assert R.is_zero_motion([0.0, 0.0], self._cfg()) is False

    def test_a_stopped_request_is_zero_motion(self):
        assert R.is_zero_motion([-1.0, 0.0], self._cfg()) is True

    def test_the_bound_is_inclusive_in_the_envs_own_float32_units(self):
        """The action is float32 by the time `_step_inner` sees it, so the test
        is evaluated there too — the metric and the treatment must read the same
        number."""
        cfg = self._cfg()
        just_over = np.float32(-0.9)  # (a0+1)/2 = 0.0500000119 > 0.05
        just_under = np.nextafter(just_over, np.float32(-1.0))
        assert R.is_zero_motion([just_over, 0.0], cfg) is False
        assert R.is_zero_motion([just_under, 0.0], cfg) is True

    def test_the_turn_rate_is_irrelevant(self):
        """A policy that asks to spin in place while asking for no speed is
        still asking for no motion; H-ZERO is about the speed clause alone."""
        cfg = self._cfg()
        assert R.is_zero_motion([-1.0, 1.0], cfg) is True
        assert R.is_zero_motion([-1.0, -1.0], cfg) is True

    def test_an_out_of_box_request_is_clipped_like_the_env_clips_it(self):
        cfg = self._cfg()
        assert R.is_zero_motion([-5.0, 0.0], cfg) is True
        assert R.is_zero_motion([5.0, 0.0], cfg) is False


class TestRecordingSearchMDP:
    """The whole claim is that the agent stops ASKING to move. S16 measured the
    execution and thereby counted a remedy's own mechanism as the disease
    (D129); this records the action handed to `step()`."""

    def _cfg(self):
        from pe2.config import EnvConfig

        return EnvConfig()

    def test_a_full_speed_policy_records_no_zero_motion_though_it_is_immobile(self):
        """The separation that matters: the filter clamps the executed speed to
        ~0 for many steps, and not one of them counts against the policy,
        because the policy never asked to stop."""
        from pe2.metrics import summarize_shield

        logs, episodes = R.evaluate_arm_policy(
            _Full(), self._cfg(), R.GreedyEvader, 3, R.GATE_SEED0
        )
        req = R.request_summary(episodes)
        assert req["zero_motion"]["n_decisions"] > 0
        assert req["zero_motion"]["fraction"] == 0.0
        # ... while the EXECUTION was immobile a great deal of the time
        assert summarize_shield(logs)["immobile_step_rate"] > 0.0

    def test_a_stopped_policy_records_every_decision_as_zero_motion(self):
        _, episodes = R.evaluate_arm_policy(
            _Stopped(), self._cfg(), R.GreedyEvader, 3, R.GATE_SEED0
        )
        assert R.request_summary(episodes)["zero_motion"]["fraction"] == 1.0

    def test_one_record_per_episode(self):
        logs, episodes = R.evaluate_arm_policy(
            _Full(), self._cfg(), R.GreedyEvader, 4, R.GATE_SEED0
        )
        assert len(logs) == len(episodes) == 4
        assert all(e["decisions"] > 0 for e in episodes)

    def test_the_intervention_magnitude_is_the_envs_own_number(self):
        """Not a second copy of `||a_nominal - a_safe||` computed here: the
        value recorded is `info["intervention_magnitude"]`, the quantity the
        penalty is built from, so the metric cannot drift from the treatment."""
        from pe2.config import EnvConfig
        from pe2.shield import CBFShield

        cfg = EnvConfig()
        mdp = R.RecordingSearchMDP(
            cfg,
            R.GreedyEvader,
            hold=R.HOLD,
            shield=CBFShield(cfg),
            shield_obs=True,
            escape_obs=True,
            reward_wrapper=None,
        )
        obs, _ = mdp.reset(seed=R.GATE_SEED0)
        total = 0.0
        for _ in range(5):
            a, _ = _Full().predict(obs)
            obs, _r, term, trunc, info = mdp.step(a)
            total += info["intervention_magnitude"]
            if term or trunc:
                break
        assert mdp._episode["intervention"] == pytest.approx(total)
        assert total > 0.0  # a wall-seeking policy is corrected

    def test_a_policy_that_commands_nothing_is_never_corrected(self):
        _, episodes = R.evaluate_arm_policy(
            _Stopped(), self._cfg(), R.GreedyEvader, 2, R.GATE_SEED0
        )
        assert R.request_summary(episodes)["intervention"]["per_episode_mean"] == 0.0

    def test_recording_does_not_change_the_episodes(self):
        """The recorder must be observation-free: the same logs the library
        route produces."""
        from pe2.config import EnvConfig
        from pe2.shield import CBFShield
        from pe2.structured import OrbitSearch, evaluate_structured

        cfg = EnvConfig()
        pol = OrbitSearch(cfg)
        ref = evaluate_structured(
            pol,
            cfg,
            R.GreedyEvader,
            3,
            R.GATE_SEED0,
            shield_factory=lambda: CBFShield(cfg),
            shield_obs=True,
            escape_obs=True,
        )
        got, _ = R.evaluate_arm_policy(pol, cfg, R.GreedyEvader, 3, R.GATE_SEED0)
        assert got == ref


class TestRequestSummary:
    def test_the_fraction_is_decision_weighted_not_a_mean_of_episodes(self):
        """A two-decision episode must not weigh as much as a 200-decision
        one."""
        episodes = [
            {"decisions": 2, "zero_motion": 2, "intervention": 0.0},
            {"decisions": 198, "zero_motion": 0, "intervention": 0.0},
        ]
        assert R.request_summary(episodes)["zero_motion"]["fraction"] == pytest.approx(
            0.01
        )

    def test_the_intervention_mean_is_per_episode(self):
        episodes = [
            {"decisions": 5, "zero_motion": 0, "intervention": 3.0},
            {"decisions": 5, "zero_motion": 0, "intervention": 1.0},
        ]
        got = R.request_summary(episodes)["intervention"]
        assert got["per_episode_mean"] == pytest.approx(2.0)
        assert got["total"] == pytest.approx(4.0) and got["n_episodes"] == 2

    def test_an_empty_set_reports_zero_rather_than_dividing(self):
        got = R.request_summary([])
        assert got["zero_motion"]["fraction"] == 0.0
        assert got["intervention"]["per_episode_mean"] == 0.0


class TestUnwrappedGate:
    """Amendment A1.3: the reward wrapper is training-only and the gate builds
    its `SearchMDP` with `reward_wrapper=None` regardless, so the property does
    not rest on `play_episode` discarding rewards alone."""

    def test_the_gate_env_is_built_unwrapped_and_unpenalised(self, monkeypatch):
        from pe2.config import EnvConfig

        built = []
        base = R.RecordingSearchMDP

        class _Spy(base):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                built.append((self, kwargs))

        monkeypatch.setattr(R, "RecordingSearchMDP", _Spy)
        R.evaluate_arm_policy(_Full(), EnvConfig(), R.GreedyEvader, 2, R.GATE_SEED0)
        ((mdp, kwargs),) = built
        assert kwargs["reward_wrapper"] is None
        assert mdp._driver is mdp.inner
        assert mdp.intervention_penalty == 0.0
        assert mdp.stall_penalty == 0.0
        assert mdp.shield_obs is True and mdp.escape_obs is True
        assert mdp.hold == R.HOLD

    def test_a_fresh_shield_per_episode(self, monkeypatch):
        from pe2.config import EnvConfig
        from pe2.shield import CBFShield

        cfg = EnvConfig()
        built = []

        def spy():
            built.append(CBFShield(cfg))
            return built[-1]

        R.evaluate_arm_policy(
            _Full(), cfg, R.GreedyEvader, 3, R.GATE_SEED0, shield_factory=spy
        )
        assert len(built) == 3 and len({id(s) for s in built}) == 3

    def test_the_gate_cell_carries_every_pre_registered_metric(self, monkeypatch):
        """Section 4: capture (primary), deadlock, the zero-motion request
        fraction, the mean intervention magnitude and the violation census."""
        monkeypatch.setattr(R, "GATE_OPPONENTS", ("GREEDY",))
        monkeypatch.setattr(R, "GATE_N", 4)
        entry = R.gate_model(_Full())["GREEDY"]
        assert len(entry["capture_ind"]) == 4
        assert 0.0 <= entry["capture_rate"] <= 1.0
        assert entry["pursuer"]["deadlock_episode_rate"] >= 0.0
        assert entry["zero_motion"]["n_decisions"] > 0
        assert entry["zero_motion"]["fraction"] == 0.0
        assert entry["intervention"]["n_episodes"] == 4
        assert entry["pursuer_violations"] == 0


# ----------------------------------------------------------------------------
# section 5: the hypotheses
# ----------------------------------------------------------------------------
class TestPenaltyEffect:
    def test_it_reports_the_direction_per_seed_and_pooled(self):
        eff = R.penalty_effect(_crossover(), "SPARSE")
        assert eff["direction"] == "LOWERS"
        assert eff["n_lower"] == 3 and eff["n_raise"] == 0
        assert eff["per_seed"][70]["base"] == pytest.approx(0.40)
        assert eff["per_seed"][70]["pen"] == pytest.approx(0.20)
        assert eff["per_seed"][70]["delta"] == pytest.approx(-0.20)
        assert eff["pooled"]["drop"]["pass"] is True
        assert eff["pooled"]["gain"]["pass"] is False

    def test_the_pooled_test_uses_all_three_seeds_of_paired_episodes(self):
        eff = R.penalty_effect(_crossover(), "SPARSE")
        assert eff["pooled"]["drop"]["n"] == 3 * N

    def test_a_raising_column_reports_raises(self):
        eff = R.penalty_effect(_crossover(), "DVARY")
        assert eff["direction"] == "RAISES" and eff["n_raise"] == 3
        assert eff["pooled"]["gain"]["pass"] is True

    def test_a_split_column_reports_mixed_rather_than_a_direction(self):
        res = _crossover()
        # one seed up, one level with its control, one down: no direction has
        # the two seeds the spec requires
        res["SPARSE-PEN-70"] = _arm("SPARSE-PEN-70", 0.90, 0.30, 0.50)
        res["SPARSE-PEN-71"] = _arm("SPARSE-PEN-71", 0.40, 0.30, 0.50)
        eff = R.penalty_effect(res, "SPARSE")
        assert eff["direction"] == "MIXED"
        assert (eff["n_lower"], eff["n_raise"]) == (1, 1)

    def test_a_missing_arm_is_incomplete_and_has_no_pooled_test(self):
        res = _crossover()
        del res["SPARSE-PEN-71"]
        eff = R.penalty_effect(res, "SPARSE")
        assert eff["complete"] is False
        assert eff["direction"] == "INCOMPLETE" and eff["pooled"] is None


class TestHCross:
    """H-CROSS (primary): the penalty LOWERS capture in SPARSE and RAISES it in
    DENSE-VARY, sign on >= 2/3 seeds in EACH direction, each pooled difference
    significant at p < 0.05."""

    def test_the_pre_registered_crossover_passes(self):
        h = R.h_cross(_crossover())
        assert h["verdict"] == "PASS"
        assert h["sign_holds"] is True and h["significant"] is True

    def test_no_interaction_fails_and_says_which_way_it_went(self):
        """The refutation. It must be legible, not just "FAIL"."""
        h = R.h_cross(_full())  # penalty changes nothing anywhere
        assert h["verdict"] == "FAIL"
        assert h["sign_holds"] is False
        assert "SPARSE: penalty MIXED capture" in h["observed"]
        assert "DVARY: penalty MIXED capture" in h["observed"]

    def test_the_opposite_interaction_fails_and_reports_its_direction(self):
        """Penalty helps SPARSE and hurts DENSE-VARY: the sign is backwards, so
        the hypothesis fails, and the reported direction must say so."""
        h = R.h_cross(
            _full(SPARSE_PEN=(0.60, 0.30, 0.20), DVARY_PEN=(0.20, 0.30, 0.20))
        )
        assert h["verdict"] == "FAIL" and h["sign_holds"] is False
        assert "SPARSE: penalty RAISES capture" in h["observed"]
        assert "DVARY: penalty LOWERS capture" in h["observed"]

    def test_the_penalty_hurting_everywhere_fails(self):
        """D129's result reproduced in all three columns: no interaction, so
        Hypothesis B is refuted even though the penalty clearly does something."""
        h = R.h_cross(_full(SPARSE_PEN=(0.20, 0.3, 0.5), DVARY_PEN=(0.20, 0.3, 0.5)))
        assert h["verdict"] == "FAIL"
        assert "SPARSE: penalty LOWERS" in h["observed"]
        assert "DVARY: penalty LOWERS" in h["observed"]

    def test_the_right_sign_without_significance_fails(self):
        """One episode of difference on every seed: the sign holds 3/3 and the
        pooled paired test does not reach p < 0.05."""
        h = R.h_cross(_full(SPARSE_PEN=(0.39, 0.3, 0.2), DVARY_PEN=(0.41, 0.3, 0.2)))
        assert h["sign_holds"] is True
        assert h["significant"] is False
        assert h["verdict"] == "FAIL"

    def test_two_of_three_seeds_is_enough_and_one_is_not(self):
        res = _crossover()
        res["SPARSE-PEN-72"] = _arm("SPARSE-PEN-72", 0.90, 0.30, 0.50)
        assert R.h_cross(res)["sign_holds"] is True
        res["SPARSE-PEN-71"] = _arm("SPARSE-PEN-71", 0.90, 0.30, 0.50)
        assert R.h_cross(res)["sign_holds"] is False

    def test_it_reads_the_primary_opponent_only(self):
        """Capture vs GreedyEvader is the primary metric; HUG is
        robustness and cannot decide the hypothesis."""
        res = _crossover()
        for s in R.SEEDS:
            res[f"SPARSE-PEN-{s}"]["final"]["gate"]["HUG"] = _entry(0.99, 0.3, 0.5)
        assert R.h_cross(res)["verdict"] == "PASS"

    def test_incomplete(self):
        res = _crossover()
        del res["DVARY-BASE-70"]
        assert R.h_cross(res)["verdict"] == "INCOMPLETE"


class TestHMech:
    """H-MECH (the discriminating test): DENSE-FLAT patterns with SPARSE — the
    penalty lowers capture there too, on >= 2/3 seeds."""

    def test_dense_flat_patterning_with_sparse_passes(self):
        h = R.h_mech(_crossover())
        assert h["verdict"] == "PASS" and h["patterns_with"] == "SPARSE"

    def test_dense_flat_patterning_with_dense_vary_fails_and_names_it(self):
        """The pre-registered alternative outcome: the variable is density after
        all, not gradient on the intervened set."""
        h = R.h_mech(_crossover(DFLAT_PEN=(0.60, 0.30, 0.10)))
        assert h["verdict"] == "FAIL" and h["patterns_with"] == "DVARY"
        assert "DFLAT: penalty RAISES capture" in h["observed"]

    def test_a_split_dense_flat_column_patterns_with_neither(self):
        res = _crossover()
        res["DFLAT-PEN-70"] = _arm("DFLAT-PEN-70", 0.90, 0.30, 0.50)
        res["DFLAT-PEN-71"] = _arm("DFLAT-PEN-71", 0.40, 0.30, 0.50)
        h = R.h_mech(res)
        assert h["verdict"] == "FAIL" and h["patterns_with"] == "NEITHER"

    def test_two_of_three_seeds_is_enough(self):
        res = _crossover()
        res["DFLAT-PEN-72"] = _arm("DFLAT-PEN-72", 0.90, 0.30, 0.50)
        assert R.h_mech(res)["verdict"] == "PASS"

    def test_incomplete(self):
        res = _crossover()
        del res["DFLAT-PEN-72"]
        h = R.h_mech(res)
        assert h["verdict"] == "INCOMPLETE" and h["patterns_with"] is None


class TestHZero:
    """H-ZERO (mechanism): in every cell where the penalty lowers capture, the
    zero-motion REQUEST fraction rises against its matched no-penalty control."""

    def test_it_passes_when_the_requests_rise_where_capture_falls(self):
        h = R.h_zero(_crossover())
        assert h["verdict"] == "PASS"
        assert h["affected_columns"] == ["SPARSE", "DFLAT"]
        assert h["per_column"]["SPARSE"]["n_rise"] == 3
        assert h["per_column"]["SPARSE"]["per_seed"][70]["base"] == pytest.approx(0.20)
        assert h["per_column"]["SPARSE"]["per_seed"][70]["pen"] == pytest.approx(0.50)

    def test_a_column_whose_capture_fell_without_the_requests_rising_fails(self):
        """Without this, H-CROSS is a correlation: capture fell for some other
        reason and the "wins the penalty, loses the task" story is unsupported."""
        h = R.h_zero(_crossover(SPARSE_PEN=(0.20, 0.30, 0.05)))
        assert h["per_column"]["SPARSE"]["verdict"] == "FAIL"
        assert h["per_column"]["SPARSE"]["n_rise"] == 0
        assert h["verdict"] == "FAIL"

    def test_a_column_where_the_penalty_did_not_lower_capture_is_not_read(self):
        """DENSE-VARY's requests may do anything; the hypothesis says nothing
        about cells where capture did not fall."""
        h = R.h_zero(_crossover(DVARY_PEN=(0.60, 0.30, 0.90)))
        assert h["per_column"]["DVARY"]["verdict"] == "N/A"
        assert h["per_column"]["DVARY"]["capture_lowered"] is False
        assert h["verdict"] == "PASS"

    def test_it_is_vacuous_when_no_cell_lowered_capture(self):
        h = R.h_zero(_full(SPARSE_PEN=(0.60, 0.3, 0.2), DVARY_PEN=(0.60, 0.3, 0.2)))
        assert h["verdict"] == "VACUOUS" and h["affected_columns"] == []

    def test_every_affected_column_must_pass_not_just_one(self):
        h = R.h_zero(_crossover(DFLAT_PEN=(0.20, 0.30, 0.05)))
        assert h["per_column"]["SPARSE"]["verdict"] == "PASS"
        assert h["per_column"]["DFLAT"]["verdict"] == "FAIL"
        assert h["verdict"] == "FAIL"

    def test_two_of_three_seeds_is_enough(self):
        res = _crossover()
        res["SPARSE-PEN-72"] = _arm("SPARSE-PEN-72", 0.20, 0.30, 0.01)
        h = R.h_zero(res)
        assert h["per_column"]["SPARSE"]["n_rise"] == 2
        assert h["verdict"] == "PASS"

    def test_the_comparison_is_seed_matched(self):
        res = _crossover()
        res["SPARSE-BASE-70"] = _arm("SPARSE-BASE-70", 0.40, 0.30, 0.90)
        h = R.h_zero(res)
        assert h["per_column"]["SPARSE"]["per_seed"][70]["rises"] is False
        assert h["per_column"]["SPARSE"]["per_seed"][71]["rises"] is True

    def test_incomplete(self):
        res = _crossover()
        del res["SPARSE-PEN-70"]
        assert R.h_zero(res)["verdict"] == "INCOMPLETE"


class TestSafety:
    """H-SAFE (standing): zero pursuer contact steps across all 18 arms."""

    def test_holds_on_a_clean_round(self):
        h = R.hsafe(_full())
        assert h["verdict"] == "HOLDS" and h["pursuer_violation_steps"] == 0

    def test_a_contact_step_anywhere_fails_it(self):
        res = _full()
        res["DVARY-PEN-70"]["train_violations"] = 1
        assert R.hsafe(res)["verdict"] == "FAILS"
        res = _full()
        res["DFLAT-BASE-72"]["final"]["pursuer_violations"] = 3
        h = R.hsafe(res)
        assert h["verdict"] == "FAILS" and h["offenders"] == ["DFLAT-BASE-72"]
        assert h["pursuer_violation_steps"] == 3

    def test_probe_contact_steps_are_in_the_census(self):
        res = _full()
        res["SPARSE-PEN-71"]["probe_violations"] = 2
        assert R.hsafe(res)["pursuer_violation_steps"] == 2

    def test_missing_arms_are_incomplete(self):
        res = _full()
        del res["DVARY-PEN-72"]
        h = R.hsafe(res)
        assert h["verdict"] == "INCOMPLETE" and h["missing"] == ["DVARY-PEN-72"]

    def test_a_crash_record_still_counts_its_partial_gate(self):
        res = _full()
        del res["SPARSE-BASE-70"]["final"]
        del res["SPARSE-BASE-70"]["probe_violations"]
        res["SPARSE-BASE-70"]["gate_violations"] = 1
        res["SPARSE-BASE-70"]["checkpoints"] = [{"violations": 2}]
        h = R.hsafe(res)
        assert h["verdict"] == "FAILS" and h["pursuer_violation_steps"] == 3


# ----------------------------------------------------------------------------
# section 5's refutation paragraph
# ----------------------------------------------------------------------------
class TestDirection:
    def _d(self, **kw):
        h = {
            "hsafe": "HOLDS",
            "incomplete": False,
            "cross": "PASS",
            "mech": "PASS",
            "zero": "PASS",
            "cross_observed": "<cross>",
            "mech_observed": "<mech>",
            "zero_observed": "<zero>",
        }
        return R.direction(h | kw)

    def test_hsafe_failure_stops_the_round(self):
        d = self._d(hsafe="FAILS", cross="FAIL")
        assert "Stop" in d and "contact step" in d

    def test_incomplete(self):
        assert "incomplete" in self._d(hsafe="INCOMPLETE")
        assert "incomplete" in self._d(incomplete=True)

    def test_a_failed_h_cross_is_the_refutation_and_quotes_the_direction(self):
        d = self._d(cross="FAIL")
        assert "REFUTED" in d and "does not exist in this form" in d
        assert "<cross>" in d

    def test_h_mech_failing_means_the_variable_was_density(self):
        d = self._d(mech="FAIL")
        assert "DENSITY after all" in d and "S20 proposition" in d
        assert "<mech>" in d

    def test_h_zero_failing_leaves_h_cross_a_correlation(self):
        d = self._d(zero="FAIL")
        assert "correlation" in d and "<zero>" in d

    def test_everything_holding_states_the_boundary_condition(self):
        d = self._d()
        assert "boundary condition HOLDS" in d
        assert "gradient-on-I from density" in d


# ----------------------------------------------------------------------------
# summary, figure and the readers
# ----------------------------------------------------------------------------
class TestReaders:
    def test_the_mechanism_metrics_pool_both_gate_opponents(self):
        res = _crossover()
        gate = res["SPARSE-PEN-70"]["final"]["gate"]
        gate["GREEDY"]["zero_motion"] = {
            "n_zero_motion": 10,
            "n_decisions": 100,
            "fraction": 0.1,
        }
        gate["HUG"]["zero_motion"] = {
            "n_zero_motion": 0,
            "n_decisions": 900,
            "fraction": 0.0,
        }
        assert R.zero_motion_fraction(res, "SPARSE-PEN-70") == pytest.approx(0.01)

    def test_the_deadlock_rate_is_episode_weighted(self):
        res = _crossover()
        gate = res["SPARSE-PEN-70"]["final"]["gate"]
        gate["GREEDY"]["pursuer"]["deadlock_episode_rate"] = 0.10
        gate["GREEDY"]["n_episodes"] = 100
        gate["HUG"]["pursuer"]["deadlock_episode_rate"] = 0.50
        gate["HUG"]["n_episodes"] = 300
        assert R.deadlock_rate(res, "SPARSE-PEN-70") == pytest.approx(0.40)

    def test_the_intervention_mean_is_episode_weighted(self):
        res = _crossover()
        gate = res["SPARSE-PEN-70"]["final"]["gate"]
        gate["GREEDY"]["intervention"] = {
            "total": 100.0,
            "n_episodes": 100,
            "per_episode_mean": 1.0,
        }
        gate["HUG"]["intervention"] = {
            "total": 900.0,
            "n_episodes": 300,
            "per_episode_mean": 3.0,
        }
        assert R.mean_intervention_magnitude(res, "SPARSE-PEN-70") == pytest.approx(2.5)

    def test_a_missing_opponent_makes_a_pooled_metric_none(self):
        res = _crossover()
        del res["SPARSE-PEN-70"]["final"]["gate"]["HUG"]
        assert R.zero_motion_fraction(res, "SPARSE-PEN-70") is None
        assert R.deadlock_rate(res, "SPARSE-PEN-70") is None

    def test_an_unrun_arm_reads_as_none(self):
        assert R.capture_rate({}, "SPARSE-PEN-70") is None
        assert R.capture_ind({}, "SPARSE-PEN-70") is None


class TestSummary:
    def test_summarize(self):
        s = R.summarize(_crossover())
        assert s["spec"].endswith("boundary_condition_prereg_2026-09-23.md")
        assert s["hsafe"]["verdict"] == "HOLDS"
        assert s["hypotheses"]["H-CROSS"]["verdict"] == "PASS"
        assert s["hypotheses"]["H-MECH"]["verdict"] == "PASS"
        assert s["hypotheses"]["H-ZERO"]["verdict"] == "PASS"
        assert set(s["arms"]) == set(R.ARMS)
        assert s["arms"]["SPARSE-PEN-70"]["capture"] == pytest.approx(0.20)
        assert s["arms"]["SPARSE-PEN-70"]["zero_motion"] == pytest.approx(0.50)
        assert s["complete"] is True
        assert "boundary condition HOLDS" in s["direction"]

    def test_the_gate_block_records_that_evaluation_was_unwrapped(self):
        s = R.summarize(_crossover())
        assert s["gate"]["reward_wrapper"] is None  # amendment A1.3
        assert s["gate"]["seed0"] == 1_900_000 and s["gate"]["n"] == 500
        assert s["gate"]["opponents"] == ["GREEDY", "HUG"]

    def test_a_refuted_round_summarises_as_a_refutation(self):
        s = R.summarize(_full())
        assert s["hypotheses"]["H-CROSS"]["verdict"] == "FAIL"
        assert "REFUTED" in s["direction"]

    def test_summarize_is_incomplete_when_an_arm_is_missing(self):
        res = _crossover()
        del res["DVARY-PEN-72"]
        s = R.summarize(res)
        assert s["complete"] is False and "incomplete" in s["direction"]

    def test_the_summary_round_trips_through_json(self):
        assert json.loads(json.dumps(R.summarize(_crossover()), default=str))

    def test_write_figure(self, tmp_path):
        pytest.importorskip("matplotlib")
        results = _crossover()
        results["SPARSE-PEN-70"]["checkpoints"] = [
            {"steps": 500_000, "mark": 500_000, "win_rate": 0.3},
            {"steps": 1_000_000, "mark": 1_000_000, "win_rate": 0.4},
        ]
        p = tmp_path / "s18_round.png"
        R.write_figure(results, R.summarize(results), p)
        assert p.exists() and p.stat().st_size > 0

    def test_write_figure_survives_an_incomplete_round(self, tmp_path):
        pytest.importorskip("matplotlib")
        res = _crossover()
        del res["DVARY-PEN-72"]
        p = tmp_path / "s18_round.png"
        R.write_figure(res, R.summarize(res), p)
        assert p.exists()


# ----------------------------------------------------------------------------
# section 7: artefacts, guards and the stop rule
# ----------------------------------------------------------------------------
class TestGuards:
    def test_artefact_paths_all_carry_the_prefix(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        assert R._arm_file("SPARSE-PEN-70") == tmp_path / "s18_arm_SPARSE-PEN-70.json"
        assert (
            R._probe_file("DFLAT-BASE-71") == tmp_path / "s18_probe_DFLAT-BASE-71.json"
        )
        assert R._round_file("png") == tmp_path / "s18_round.png"
        assert R._lambda_file() == tmp_path / "s18_lambda.json"
        assert R._ckpt_dir("DVARY-PEN-72") == tmp_path / "ckpts_S18-DVARY-PEN-72"

    def test_an_arm_that_already_ran_is_refused(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        R.require_arm_not_run("SPARSE-PEN-70")
        (tmp_path / "s18_arm_SPARSE-PEN-70.json").write_text("{}")
        with pytest.raises(SystemExit, match="stop rule"):
            R.require_arm_not_run("SPARSE-PEN-70")
        with pytest.raises(SystemExit, match="stop rule"):
            R.require_arm_not_recorded("SPARSE-PEN-70")

    def test_a_checkpoint_directory_alone_refuses_a_second_training_run(
        self, tmp_path, monkeypatch
    ):
        """Every arm of this round trains and owns its directory, so the
        directory is independent evidence the arm already started."""
        monkeypatch.setattr(R, "OUT", tmp_path)
        (tmp_path / "ckpts_S18-DFLAT-PEN-71").mkdir()
        with pytest.raises(SystemExit, match="already trained"):
            R.require_arm_not_run("DFLAT-PEN-71")
        # ... but --gate-only exists precisely to finish such an arm
        R.require_arm_not_recorded("DFLAT-PEN-71")

    def test_gate_only_refuses_a_missing_final_checkpoint(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        with pytest.raises(SystemExit, match="no final checkpoint"):
            R.require_final_ckpt("DVARY-BASE-70")
        d = tmp_path / "ckpts_S18-DVARY-BASE-70"
        d.mkdir()
        (d / f"DVARY-BASE-70_{R.TOTAL}.zip").write_text("x")
        assert R.require_final_ckpt("DVARY-BASE-70").name.endswith("_5000000.zip")

    def test_no_gate_opponent_loads_a_learned_model(self):
        """Amendment A2: both gate opponents are scripted. A learned attacker
        anywhere in this round would make the gate depend on a policy trained
        under a different objective than the one under test."""
        from pe2.scripted import GreedyEvader, ObstacleHuggingEvader

        assert R.GATE_OPPONENTS == ("GREEDY", "HUG")
        assert R.gate_opponent_factory("GREEDY", "cpu") is GreedyEvader
        assert R.gate_opponent_factory("HUG", "cpu") is ObstacleHuggingEvader
        assert not hasattr(R, "STRUCT_CKPT")
        # the class may be NAMED in the amendment comment; what must not exist
        # is an import or an instantiation of it, or any checkpoint path
        src = pathlib.Path("scripts/derisk_s18_boundary.py").read_text()
        assert "from pe2.evade import" not in src
        assert "StructuredEvaderPolicy(" not in src
        assert "ATK4-45" not in src

    def test_an_unknown_gate_opponent_is_an_error(self):
        with pytest.raises(ValueError, match="unknown gate opponent"):
            R.gate_opponent_factory("ATK4-45", "cpu")

    def test_skip_existing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        names = ["SPARSE-BASE-70", "SPARSE-PEN-70", "DVARY-PEN-72"]
        assert R.todo(names) == ["SPARSE-BASE-70", "SPARSE-PEN-70", "DVARY-PEN-72"]
        (tmp_path / "s18_arm_SPARSE-PEN-70.json").write_text("{}")
        assert R.todo(names) == ["SPARSE-BASE-70", "DVARY-PEN-72"]

    def test_todo_defaults_to_every_arm_in_the_tables_order(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(R, "OUT", tmp_path)
        assert R.todo(None) == list(R.ARMS)
        assert R.todo([]) == list(R.ARMS)
        assert list(R.ARMS)[:3] == [
            "SPARSE-BASE-70",
            "SPARSE-BASE-71",
            "SPARSE-BASE-72",
        ]

    def test_load_results_reads_this_rounds_records_only(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        (tmp_path / "s18_arm_SPARSE-PEN-70.json").write_text('{"arm": "SPARSE-PEN-70"}')
        (tmp_path / "s17_arm_BASE-60.json").write_text('{"arm": "BASE-60"}')
        (tmp_path / "s18_lambda.json").write_text("{}")
        assert set(R.load_results()) == {"SPARSE-PEN-70"}

    def test_atomic_writes_leave_no_partial_file(self, tmp_path):
        p = tmp_path / "s18_arm_X.json"
        R._write_json(p, {"a": 1})
        assert json.loads(p.read_text()) == {"a": 1}
        assert not (tmp_path / "s18_arm_X.tmp").exists()

    def test_a_crash_record_is_undetermined_and_keeps_the_probe_history(self):
        rows = [{"steps": 500_000, "mark": 500_000, "win_rate": 0.3, "violations": 0}]
        rec = R._crash_record(
            "DVARY-PEN-70",
            RuntimeError("boom"),
            {"rows": rows, "gate": {"GREEDY": {"pursuer_violations": 2}}},
        )
        assert rec["verdict"] == "UNDETERMINED"
        assert "boom" in rec["reason"]
        assert rec["checkpoints"] == rows
        assert rec["gate_violations"] == 2
        assert rec["cell"] == "DVARY-PEN" and rec["seed"] == 70

    def test_the_cli_routes_training_gating_and_summarising(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(R, "OUT", tmp_path)
        trained, gated = [], []
        monkeypatch.setattr(R, "run_arm", lambda a, d: trained.append(a))
        monkeypatch.setattr(R, "gate_arm", lambda a, d: gated.append(a))

        monkeypatch.setattr("sys.argv", ["x", "--arms", "SPARSE-PEN-70"])
        R.main()
        assert trained == ["SPARSE-PEN-70"] and gated == []

        trained.clear()
        monkeypatch.setattr("sys.argv", ["x", "--arms", "DVARY-PEN-71", "--gate-only"])
        R.main()
        assert gated == ["DVARY-PEN-71"] and trained == []

        gated.clear()
        monkeypatch.setattr("sys.argv", ["x", "--arms"])
        R.main()
        assert trained == list(R.ARMS) and gated == []

        monkeypatch.setattr("sys.argv", ["x", "--arms", "NOPE-70"])
        with pytest.raises(SystemExit):  # argparse: not an arm of this round
            R.main()

    def test_the_cli_does_nothing_without_a_flag(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        calls = []
        monkeypatch.setattr(R, "run_arm", lambda a, d: calls.append(a))
        monkeypatch.setattr(R, "gate_arm", lambda a, d: calls.append(a))
        monkeypatch.setattr("sys.argv", ["x"])
        R.main()
        assert calls == []


# ----------------------------------------------------------------------------
# section 6, CG4: the behavioural gate
# ----------------------------------------------------------------------------
class _Space:
    shape = (R.obs_width(),)


class _FakeModel:
    """A checkpoint stand-in. Nothing here loads a real policy or renders a real
    frame — CG4's LOGIC is what is under test, not matplotlib."""

    observation_space = _Space()

    def predict(self, obs, deterministic=True):
        return np.zeros(2, dtype=np.float32), None


def _stage(tmp_path, cells=None, *, records=True, ckpts=True):
    """Lay down the arm records and final checkpoints CG4 reads."""
    for cell in R.CELL_ORDER if cells is None else cells:
        arm = R.gif_arm(cell)
        if records:
            R._write_json(R._arm_file(arm), _arm(arm, 0.40, 0.30, 0.20))
        if ckpts:
            d = R._ckpt_dir(arm)
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{arm}_{R.TOTAL}.zip").write_text("x")


def _stub_rendering(monkeypatch):
    """Replace the recorder, the checkpoint loader and the GIF writer with spies.

    Returns `(built, written)`: one entry per CellRecorder constructed (its
    kwargs plus the episode seed it was asked to play) and one path per GIF
    written."""
    built: list[dict] = []
    written: list[str] = []

    class _RecorderSpy:
        def __init__(self, cfg, opponent_factory, *, label, every=4, **kw):
            self.record = {
                "cfg": cfg,
                "opponent_factory": opponent_factory,
                "label": label,
                "every": every,
                **kw,
            }
            built.append(self.record)
            self.frames = ["frame"]

        def play_episode(self, predict, seed):
            self.record["seed"] = seed
            self.record["predict"] = predict
            return {"outcome": "timeout", "steps": 400}

        def finish(self, outcome, step):
            self.record["finished"] = (outcome, step)

    monkeypatch.setattr(R, "CellRecorder", _RecorderSpy)
    monkeypatch.setattr(R, "_model", lambda path, device: _FakeModel())
    monkeypatch.setattr(
        "pe2.render.save_gif",
        lambda frames, path, fps=20.0: written.append(path),
    )
    return built, written


class TestCG4:
    """CG4 (spec section 6): "Render 3 unselected episodes per cell and watch
    them, before reading the table."

    The retrospective's rule (section 4) made mandatory. What is tested here is
    everything that makes the GIFs evidence: the episodes are unselected, they
    are the GATE's episodes, no caption invents a number, and a cell whose
    record does not exist stops the pass rather than being captioned with a
    guess."""

    def test_the_episode_seeds_are_the_first_three_of_the_construct_stream(self):
        """Unselected by construction: three CONSECUTIVE seeds off A1's
        construct stream, fixed before any outcome exists."""
        assert R.CONSTRUCT_SEED0 == 1_880_000  # amendment A1's stream table
        assert R.GIF_N == 3
        assert R.construct_seeds() == (1_880_000, 1_880_001, 1_880_002)

    def test_the_construct_stream_is_disjoint_from_the_gate_and_the_probes(self):
        seeds = set(R.construct_seeds())
        assert not seeds & set(range(R.GATE_SEED0, R.GATE_SEED0 + R.GATE_N))
        assert not seeds & set(range(R.PROBE_SEED0, R.PROBE_SEED0 + R.EVAL_N))

    def test_every_cell_is_rendered_from_its_seed_70_arm(self):
        assert R.GIF_SEED == 70 and R.GIF_SEED == R.SEEDS[0]
        for cell in R.CELL_ORDER:
            arm = R.gif_arm(cell)
            assert arm in R.ARMS
            assert R.ARMS[arm]["cell"] == cell and R.ARMS[arm]["seed"] == 70

    def test_six_cells_times_three_episodes_is_eighteen_distinct_gifs(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(R, "OUT", tmp_path)
        paths = [
            R.gif_path(cell, seed)
            for cell in R.CELL_ORDER
            for seed in R.construct_seeds()
        ]
        assert len(R.CELL_ORDER) == 6
        assert len(paths) == 18 and len(set(paths)) == 18
        assert all(p.parent == tmp_path / "s18_gifs" for p in paths)

    def test_an_unknown_cell_is_rejected(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        with pytest.raises(SystemExit, match="unknown cell"):
            R.write_gifs(["NOPE"])
        # an ARM name is not a cell name, and must not quietly pass for one
        with pytest.raises(SystemExit, match="unknown cell"):
            R.write_gifs(["SPARSE-BASE-70"])

    def test_a_missing_arm_record_stops_the_pass_before_anything_renders(
        self, tmp_path, monkeypatch
    ):
        """The caption quotes measured rates, so a cell without a completed
        record would have to invent them. One missing record stops the WHOLE
        pass: a half-watched gate is how a construct error survives."""
        monkeypatch.setattr(R, "OUT", tmp_path)
        built, written = _stub_rendering(monkeypatch)
        _stage(tmp_path, R.CELL_ORDER[:-1])  # every cell but the last
        _stage(tmp_path, R.CELL_ORDER[-1:], records=False)  # its ckpt, no record
        with pytest.raises(SystemExit, match="no completed arm record"):
            R.write_gifs()
        assert built == [] and written == []
        assert not R.gifs_dir().exists()

    def test_a_record_without_a_gate_entry_is_not_a_completed_record(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(R, "OUT", tmp_path)
        _stage(tmp_path)
        crashed = _arm("DFLAT-PEN-70", 0.4, 0.3, 0.2)
        crashed.pop("final")
        crashed["verdict"] = "UNDETERMINED"
        R._write_json(R._arm_file("DFLAT-PEN-70"), crashed)
        with pytest.raises(SystemExit, match="no completed arm record"):
            R.read_arm("DFLAT-PEN")

    def test_a_missing_final_checkpoint_stops_the_pass_before_anything_renders(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(R, "OUT", tmp_path)
        built, written = _stub_rendering(monkeypatch)
        _stage(tmp_path, ckpts=False)
        with pytest.raises(SystemExit, match="no final checkpoint"):
            R.write_gifs()
        assert built == [] and written == []

    def test_an_existing_gif_is_skipped(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        _built, written = _stub_rendering(monkeypatch)
        _stage(tmp_path)
        R.gifs_dir().mkdir(parents=True)
        done = R.gif_path("SPARSE-BASE", 1_880_000)
        done.write_text("already watched")
        todo = R.write_gifs()
        assert ("SPARSE-BASE", 1_880_000) not in todo
        assert len(todo) == 17 and len(written) == 17
        assert str(done) not in written
        assert done.read_text() == "already watched"

    def test_a_full_pass_renders_three_unselected_episodes_of_every_cell(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(R, "OUT", tmp_path)
        built, written = _stub_rendering(monkeypatch)
        _stage(tmp_path)
        todo = R.write_gifs()
        assert todo == [
            (cell, seed) for cell in R.CELL_ORDER for seed in R.construct_seeds()
        ]
        assert [b["seed"] for b in built] == [
            s for _ in R.CELL_ORDER for s in R.construct_seeds()
        ]
        assert len(written) == 18

    def test_the_title_carries_the_cell_the_seeds_and_the_measured_rates(self):
        rec = _arm("SPARSE-PEN-70", 0.42, 0.31, 0.2)
        title = R.gif_title("SPARSE-PEN", rec, 1_880_001)
        assert "SPARSE-PEN" in title
        assert "1880001" in title  # the episode seed
        assert "70" in title  # the arm's training seed
        assert "0.420" in title and "0.310" in title
        assert f"N={N}" in title
        assert R.GIF_OPPONENT in title
        # visuals carry no verdict (spec section 6)
        for word in ("PASS", "FAIL", "H-CROSS", "H-MECH", "H-ZERO"):
            assert word not in title

    def test_the_title_reads_the_record_and_never_invents_a_number(self):
        """The one thing a caption may not do. Change the record, and the two
        numbers in the caption change with it."""
        title = R.gif_title("DVARY-BASE", _arm("DVARY-BASE-70", 0.777, 0.123, 0.2), 0)
        assert "0.777" in title and "0.123" in title
        assert "0.400" not in title and "0.300" not in title

    def test_the_title_quotes_the_opponent_the_episode_is_played_against(self):
        """The caption's rates and the rendered episode must be about the same
        opponent, or the GIF summarises a different measurement than it names."""
        assert R.GIF_OPPONENT == R.CAPTURE_OPPONENT
        assert R.GIF_OPPONENT in R.GATE_OPPONENTS
        rec = _arm("DFLAT-BASE-70", 0.4, 0.3, 0.2)
        gate = rec["final"]["gate"][R.GIF_OPPONENT]
        title = R.gif_title("DFLAT-BASE", rec, 1_880_000)
        assert f"{gate['capture_rate']:.3f}" in title
        assert f"{gate['pursuer']['deadlock_episode_rate']:.3f}" in title

    def test_the_rendered_episode_is_the_gates_episode(self, tmp_path, monkeypatch):
        """Amendment A1.3 and section 4: same hold, shield, `shield_obs`,
        `escape_obs`, opponent and `reward_wrapper=None` as
        `evaluate_arm_policy`. A look-alike episode would validate nothing."""
        from pe2.shield import CBFShield

        monkeypatch.setattr(R, "OUT", tmp_path)
        built, _written = _stub_rendering(monkeypatch)
        _stage(tmp_path, ["DFLAT-BASE"])
        R.render_cell("DFLAT-BASE", 1_880_002, "title")
        (kw,) = built
        assert kw["reward_wrapper"] is None  # A1.3
        assert kw["shield_obs"] is R.SHIELD_OBS and kw["escape_obs"] is R.ESCAPE_OBS
        assert kw["hold"] == R.HOLD
        assert isinstance(kw["shield"], CBFShield)
        assert kw["opponent_factory"] is R.gate_opponent_factory(R.GIF_OPPONENT, "cpu")
        assert kw["seed"] == 1_880_002
        assert kw["label"] == "title"
        assert kw["finished"] == ("timeout", 400)

    def test_the_recorder_is_the_gates_env_class(self):
        """`CellRecorder` subclasses the env the gate measures in, so the frames
        are of the measured trajectory rather than of a re-implementation."""
        assert issubclass(R.CellRecorder, R.RecordingSearchMDP)

    def test_writing_gifs_writes_no_record_and_no_round_artefact(
        self, tmp_path, monkeypatch
    ):
        """Visuals carry NO verdict (spec section 6)."""
        monkeypatch.setattr(R, "OUT", tmp_path)
        _stub_rendering(monkeypatch)
        _stage(tmp_path)
        before = {p: p.read_bytes() for p in sorted(tmp_path.glob("s18_*.json"))}
        R.write_gifs()
        after = {p: p.read_bytes() for p in sorted(tmp_path.glob("s18_*.json"))}
        assert after == before
        assert not R._round_file("json").exists()
        assert not R._round_file("png").exists()

    def test_the_cli_exposes_the_gate_and_runs_nothing_else(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(R, "OUT", tmp_path)
        calls, trained = [], []
        monkeypatch.setattr(R, "write_gifs", lambda *a: calls.append(a))
        monkeypatch.setattr(R, "run_arm", lambda a, d: trained.append(a))

        monkeypatch.setattr("sys.argv", ["x", "--gifs"])
        R.main()
        assert calls == [(None, "cpu", 4)] and trained == []

        calls.clear()
        monkeypatch.setattr("sys.argv", ["x", "--gifs", "SPARSE-BASE", "DVARY-PEN"])
        R.main()
        assert calls == [(["SPARSE-BASE", "DVARY-PEN"], "cpu", 4)]

        calls.clear()
        monkeypatch.setattr("sys.argv", ["x"])
        R.main()
        assert calls == []


class TestHZeroReadsItsTriggersOpponent:
    """H-ZERO is triggered by a capture DROP, and capture is measured on GREEDY
    alone. Reading the mechanism metric pooled over an opponent the trigger
    never looked at would be a different measurement wearing the same name --
    exactly the construct failure this round's gate exists to prevent."""

    @staticmethod
    def _res_with_split_zero_motion():
        """PEN's zero-motion RISES on GREEDY but the HUG half is rigged so the
        pooled number FALLS. A pooled reading flips the verdict."""
        res = _full(SPARSE_PEN=(0.20, 0.30, 0.20), SPARSE_BASE=(0.40, 0.30, 0.10))
        for s in R.SEEDS:
            pen = res[f"SPARSE-PEN-{s}"]["final"]["gate"]
            base = res[f"SPARSE-BASE-{s}"]["final"]["gate"]
            pen["GREEDY"]["zero_motion"] = {"n_zero_motion": 200, "n_decisions": 1000}
            base["GREEDY"]["zero_motion"] = {"n_zero_motion": 100, "n_decisions": 1000}
            pen["HUG"]["zero_motion"] = {"n_zero_motion": 0, "n_decisions": 9000}
            base["HUG"]["zero_motion"] = {"n_zero_motion": 4000, "n_decisions": 9000}
        return res

    def test_greedy_is_read_and_pooled_is_only_recorded(self):
        res = self._res_with_split_zero_motion()
        for s in R.SEEDS:
            greedy = R.zero_motion_fraction(res, f"SPARSE-PEN-{s}", R.CAPTURE_OPPONENT)
            pooled = R.zero_motion_fraction(res, f"SPARSE-PEN-{s}")
            assert greedy == pytest.approx(0.20)
            assert pooled == pytest.approx(0.02)

        col = R.h_zero(res)["per_column"]["SPARSE"]
        assert col["capture_lowered"] is True
        # rises on GREEDY (0.10 -> 0.20) on every seed; pooled would FALL
        assert col["n_rise"] == 3
        assert col["verdict"] == "PASS"
        for s in R.SEEDS:
            row = col["per_seed"][s]
            assert row["base"] == pytest.approx(0.10)
            assert row["pen"] == pytest.approx(0.20)
            assert row["base_pooled"] > row["pen_pooled"]
