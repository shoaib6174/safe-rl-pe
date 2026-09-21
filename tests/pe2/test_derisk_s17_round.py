"""scripts/derisk_s17_fixes_b.py — the redesigned loop-fixes round's decisions.

Spec: docs/pathway/loop_fixes_b_prereg_2026-09-21.md (sections 2-6) plus
amendment S17-A1. Everything here is either a constant the spec fixes, a
verdict rule it states, or one of the evaluation-path requirements the round
needs:

- a FRESH `RecoveryShieldV2` per episode (the S16 requirement, section 3);
- per-step logging of the policy's nominal and the executed command as
  SEPARATE fields (section 2, forced by D130), which is also the only place the
  executed omega exists -- `CBFShield.filter`'s info reports `v_exec` but no
  omega, and the spin fraction of H3 needs both;
- each family's `escape_obs` flag pinned to the observation width its source
  checkpoint was trained at, because four of the seven families replay S16
  checkpoints and feeding one the wrong width is silent, not loud.
"""

import importlib.util
import json
import pathlib

import numpy as np
import pytest

_PATH = (
    pathlib.Path(__file__).resolve().parents[2] / "scripts" / "derisk_s17_fixes_b.py"
)


def _runner():
    spec = importlib.util.spec_from_file_location("derisk_s17_fixes_b", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _runner()
N = 100  # episodes in the synthetic gate entries below
S = 1200 * N  # steps in one synthetic gate cell


def _ind(rate, n=N):
    k = round(rate * n)
    return [i < k for i in range(n)]


def _entry(capture, deadlock, spin, viol=0, recoveries=0):
    """One gate cell in the shape `gate_model` writes."""
    n_spin = round(spin * S)
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
        "spin": {"n_spin": n_spin, "n_steps": S, "fraction": n_spin / S},
        "recovery_kinds": {"forward": recoveries, "policy": 0, "rotation": 0},
        "recoveries": recoveries,
        "pursuer_violations": viol,
        "traces": [],
    }


def _arm(name, capture, deadlock, spin, viol=0):
    spec = R.ARMS[name]
    return {
        "arm": name,
        "family": spec["family"],
        "seed": spec["seed"],
        "train_violations": 0,
        "probe_violations": 0,
        "checkpoints": [],
        "final": {
            "ckpt": str(R.source_ckpt(name)),
            "pursuer_violations": viol,
            "gate": {
                "ATK4-45": _entry(0.10, deadlock, spin),
                "HUG7": _entry(0.30, deadlock, spin),
                "GREEDY": _entry(capture, deadlock, spin),
            },
        },
    }


_DEFAULT = {
    # family: (GreedyEvader capture, ATK4-45 deadlock rate, spin fraction)
    "BASE": (0.40, 0.40, 0.30),
    "OBS": (0.40, 0.30, 0.25),
    "GAM": (0.40, 0.30, 0.25),
    "OBSGAM": (0.40, 0.30, 0.25),
    "RECF-BASE": (0.40, 0.30, 0.25),
    "RECF-OBS": (0.40, 0.30, 0.25),
    "RECF-OBSGAM": (0.40, 0.30, 0.25),
}


def _full(**overrides):
    """A complete set of records: seven families, three seeds.

    Each keyword is a family name (dots for hyphens: `RECF_BASE`) whose value
    is `(capture, deadlock, spin)` applied to all three of its seeds."""
    cells = dict(_DEFAULT)
    for key, value in overrides.items():
        cells[key.replace("_", "-")] = value
    out = {}
    for fam, (cap, dead, spin) in cells.items():
        for s in R.SEEDS:
            name = f"{fam}-{s}"
            out[name] = _arm(name, cap, dead, spin)
    return out


# ----------------------------------------------------------------------------
# section 3: the arms
# ----------------------------------------------------------------------------
class TestProtocol:
    def test_constants(self):
        assert R.SEEDS == (60, 61, 62)
        assert (R.TOTAL, R.EVERY) == (5_000_000, 500_000)
        assert R.GATE_SEED0 >= 1_740_000
        assert R.GATE_N == 500
        assert R.PROBE_SEED0 != R.GATE_SEED0
        assert (R.DEADLOCK_BAR, R.SPIN_BAR, R.P_GATE) == (0.05, 0.05, 0.05)
        assert R.PASS_MIN == 2
        assert (R.SPIN_V_FRAC, R.SPIN_OMEGA_FRAC) == (0.05, 0.5)
        assert R.GATE_OPPONENTS == ("ATK4-45", "HUG7", "GREEDY")
        assert R.DEADLOCK_OPPONENT == "ATK4-45"
        assert R.CAPTURE_OPPONENT == "GREEDY"
        assert R.REFERENCE == "BASE"
        assert R.ARTEFACT_PREFIX == "s17"
        assert R.RECOVERY_AFTER == 20
        assert (R.HOLD, R.R_ACQ, R.SHIELD_OBS) == (10, 0.0, True)
        assert R.TRACE_EPISODES == 5
        assert R.SPEC.endswith("loop_fixes_b_prereg_2026-09-21.md")

    def test_the_spin_thresholds_are_the_shields_own_immobility_threshold(self):
        """H3's speed clause must be the same 0.05.v_max the filter calls
        immobile, or 'spinning' and 'immobile' could disagree by construction."""
        from pe2.shield import STALL_V_FRAC

        assert R.SPIN_V_FRAC == STALL_V_FRAC

    def test_the_arm_table_is_seven_families_by_three_seeds(self):
        assert R.FAMILY_ORDER == (
            "BASE",
            "OBS",
            "GAM",
            "OBSGAM",
            "RECF-BASE",
            "RECF-OBS",
            "RECF-OBSGAM",
        )
        assert len(R.ARMS) == 21
        assert set(R.ARMS) == {f"{f}-{s}" for f in R.FAMILY_ORDER for s in (60, 61, 62)}

    def test_the_family_and_seed_are_stored_not_parsed_off_the_name(self):
        """`RECF-BASE-60`.split('-') does not recover the family; every reader
        must go through the table."""
        assert R.ARMS["RECF-BASE-60"]["family"] == "RECF-BASE"
        assert R.ARMS["RECF-BASE-60"]["seed"] == 60
        assert R.ARMS["RECF-OBSGAM-62"]["family"] == "RECF-OBSGAM"
        assert R.ARMS["BASE-61"]["family"] == "BASE"

    def test_only_obsgam_trains(self):
        """Spec section 3: three training runs, not fifteen."""
        assert sorted(R.TRAIN_ARMS) == ["OBSGAM-60", "OBSGAM-61", "OBSGAM-62"]
        for arm, spec in R.ARMS.items():
            assert spec["trains"] is (spec["family"] == "OBSGAM"), arm

    def test_each_family_names_its_policy_source(self):
        """Spec section 3's `policies` column."""
        assert R.FAMILIES["BASE"]["source"] == "BASE"
        assert R.FAMILIES["OBS"]["source"] == "OBS"
        assert R.FAMILIES["GAM"]["source"] == "GAM"
        assert R.FAMILIES["OBSGAM"]["source"] == "OBSGAM"
        assert R.FAMILIES["RECF-BASE"]["source"] == "BASE"
        assert R.FAMILIES["RECF-OBS"]["source"] == "OBS"
        assert R.FAMILIES["RECF-OBSGAM"]["source"] == "OBSGAM"

    def test_only_the_recf_families_carry_the_recovery(self):
        for fam, spec in R.FAMILIES.items():
            assert spec["recovery"] is fam.startswith("RECF-"), fam

    def test_source_arm_and_checkpoint_paths(self):
        assert R.source_arm("RECF-BASE-60") == "BASE-60"
        assert R.source_arm("RECF-OBSGAM-62") == "OBSGAM-62"
        assert R.source_arm("GAM-61") == "GAM-61"
        assert R.source_ckpt("RECF-OBS-61").name == "OBS-61_5000000.zip"
        assert R.source_ckpt("RECF-OBS-61").parent.name == "ckpts_OBS-61"
        # the four reused families point at S16's directories, unchanged
        assert R.source_ckpt("BASE-60") == R.source_ckpt("RECF-BASE-60")

    def test_obsgam_is_both_halves_of_d128_and_nothing_else(self):
        """Spec section 3: escape features AND gamma = 0.999, else BASE."""
        base, obsgam = R.FAMILIES["BASE"], R.FAMILIES["OBSGAM"]
        assert obsgam["escape_obs"] is True and obsgam["gamma"] == 0.999
        assert obsgam["stall_penalty"] == base["stall_penalty"] == 1e-3
        assert R.FAMILIES["OBS"]["escape_obs"] is True
        assert R.FAMILIES["OBS"]["gamma"] == base["gamma"] == 0.99
        assert R.FAMILIES["GAM"]["gamma"] == 0.999
        assert R.FAMILIES["GAM"]["escape_obs"] is False


class TestObservationWidth:
    """A family replaying an S16 checkpoint at the wrong observation width
    either crashes or, worse, silently mis-shapes the policy's input."""

    def test_escape_obs_is_on_for_the_sixteen_dim_families_only(self):
        for fam, on in (
            ("BASE", False),
            ("OBS", True),
            ("GAM", False),
            ("OBSGAM", True),
            ("RECF-BASE", False),
            ("RECF-OBS", True),
            ("RECF-OBSGAM", True),
        ):
            assert R.FAMILIES[fam]["escape_obs"] is on, fam

    def test_declared_widths(self):
        for fam, width in (
            ("BASE", 14),
            ("OBS", 16),
            ("GAM", 14),
            ("OBSGAM", 16),
            ("RECF-BASE", 14),
            ("RECF-OBS", 16),
            ("RECF-OBSGAM", 16),
        ):
            assert R.obs_width(fam) == width, fam

    def test_a_recf_family_inherits_its_sources_width(self):
        for fam in R.FAMILIES:
            src = R.FAMILIES[fam]["source"]
            assert R.obs_width(fam) == R.obs_width(src), fam

    def test_the_training_env_has_the_declared_width(self):
        env = R.make_env("OBSGAM-60")
        assert env.observation_space.shape == (R.obs_width("OBSGAM"),)
        assert env.escape_obs is True and env.shield_obs is True
        assert env.stall_penalty == 1e-3 and env.hold == 10
        assert env.r_acq == 0.0 and env.zero_reward is False
        assert type(env.shield).__name__ == "CBFShield"

    def test_the_discount_reaches_the_learner(self):
        from stable_baselines3 import PPO

        model = R.make_learner("OBSGAM-60", R.make_env("OBSGAM-60"), "cpu")
        assert isinstance(model, PPO) and model.gamma == 0.999
        assert model.observation_space.shape == (16,)

    def test_a_mismatched_checkpoint_is_refused_at_load(self):
        """The last line of defence: SB3 would either raise deep inside
        `predict` or -- worse -- accept a width that matches for the wrong
        reason and feed the policy features it never trained on."""

        class _Model:
            def __init__(self, width):
                self.observation_space = type("S", (), {"shape": (width,)})()

        R._check_width("BASE-60", _Model(14))  # agrees: no complaint
        R._check_width("OBS-60", _Model(16))
        with pytest.raises(SystemExit, match="do NOT gate this arm"):
            R._check_width("OBS-60", _Model(14))
        with pytest.raises(SystemExit, match="escape_obs=False"):
            R._check_width("RECF-BASE-60", _Model(16))

    @pytest.mark.parametrize(
        ("arm", "width"),
        [("BASE-60", 14), ("OBS-60", 16), ("GAM-60", 14)],
    )
    def test_the_declared_width_matches_the_real_s16_checkpoint(self, arm, width):
        """The pin that matters: what the saved policy actually expects."""
        from stable_baselines3 import PPO

        ckpt = R.source_ckpt(arm)
        if not ckpt.exists():
            pytest.skip(f"{ckpt} not present")
        model = PPO.load(ckpt, device="cpu")
        assert model.observation_space.shape == (width,)
        assert R.obs_width(R.ARMS[arm]["family"]) == width


# ----------------------------------------------------------------------------
# section 2: per-step logging and the spin fraction
# ----------------------------------------------------------------------------
class TestIsSpin:
    """H3: a step spins iff `v_exec <= 0.05.v_max` AND `|w_exec| >= 0.5.w_max`."""

    def test_the_brief_cases(self):
        v_max, w_max = 1.0, 2.84
        assert R.is_spin(0.04 * v_max, 0.6 * w_max, v_max, w_max) is True
        assert R.is_spin(0.06 * v_max, 0.6 * w_max, v_max, w_max) is False
        assert R.is_spin(0.04 * v_max, 0.4 * w_max, v_max, w_max) is False

    def test_the_sign_of_omega_is_irrelevant(self):
        assert R.is_spin(0.0, -2.0, 1.0, 2.84) is True
        assert R.is_spin(0.0, 2.0, 1.0, 2.84) is True

    def test_both_bounds_are_inclusive(self):
        v_max, w_max = 1.0, 2.84
        assert R.is_spin(0.05 * v_max, 0.5 * w_max, v_max, w_max) is True
        assert R.is_spin(0.05 * v_max + 1e-9, 0.5 * w_max, v_max, w_max) is False
        assert R.is_spin(0.05 * v_max, 0.5 * w_max - 1e-9, v_max, w_max) is False

    def test_it_scales_with_the_caps_it_is_given(self):
        assert R.is_spin(0.5, 3.0, 20.0, 4.0) is True  # 0.5 <= 1.0, 3.0 >= 2.0
        assert R.is_spin(0.5, 3.0, 1.0, 4.0) is False  # 0.5 > 0.05


class _FakeShield:
    """A shield whose filter returns a scripted sequence of executed commands.

    Carries the attributes `StepRecorder` and `escape_features` reach through
    for, so the recorder's delegation is exercised rather than mocked away."""

    def __init__(self, outputs, kinds=None, cfg=None):
        from pe2.config import EnvConfig

        self.cfg = cfg or EnvConfig()
        self.outputs = list(outputs)
        self.kinds = list(kinds) if kinds else None
        self.calls = []
        self.n = 0

    def filter(self, action, state, obstacles, v_max=None):
        self.calls.append((tuple(map(float, action)), v_max))
        v, w = self.outputs[self.n]
        info = {
            "method": "exact",
            "feasible": True,
            "intervention": True,
            "v_nom": float(np.clip(float(action[0]), 0.0, v_max or self.cfg.v_max)),
            "v_exec": float(v),
            "stalled": False,
            "immobile": float(v) <= 0.05 * (v_max or self.cfg.v_max),
        }
        if self.kinds is not None:
            info["recovery_kind"] = self.kinds[self.n]
            info["recovery_fired"] = self.kinds[self.n] != "none"
        self.n += 1
        return np.array([v, w], dtype=np.float64), info


class TestStepRecorder:
    """Spec section 2: the policy's nominal and the executed command are
    recorded as SEPARATE fields, so D130's attribution is a measurement."""

    def test_a_hand_built_trace_counts_only_the_spinning_steps(self):
        v_max, w_max = 1.0, 2.84
        outputs = [
            (0.04 * v_max, 0.6 * w_max),  # spin
            (0.06 * v_max, 0.6 * w_max),  # too fast
            (0.04 * v_max, 0.4 * w_max),  # too slow a turn
            (0.0, -w_max),  # spin (negative omega)
        ]
        rec = R.StepRecorder(_FakeShield(outputs), keep_trace=True)
        for _ in outputs:
            rec.filter((1.0, 0.0), np.zeros(3), [])
        assert rec.n_steps == 4
        assert rec.n_spin == 2
        assert rec.spin_fraction == pytest.approx(0.5)

    def test_the_nominal_and_the_executed_are_separate_fields(self):
        rec = R.StepRecorder(_FakeShield([(0.0, 2.0)]), keep_trace=True)
        rec.filter((0.9, -1.5), np.zeros(3), [])
        (row,) = rec.trace
        assert row["v_nom"] == pytest.approx(0.9)
        assert row["w_nom"] == pytest.approx(-1.5)
        assert row["v_exec"] == pytest.approx(0.0)
        assert row["w_exec"] == pytest.approx(2.0)
        assert row["spin"] is True

    def test_the_nominal_is_clipped_to_the_action_box(self):
        rec = R.StepRecorder(_FakeShield([(0.0, 0.0)]), keep_trace=True)
        rec.filter((5.0, -99.0), np.zeros(3), [])
        (row,) = rec.trace
        assert row["v_nom"] == pytest.approx(1.0)
        assert row["w_nom"] == pytest.approx(-2.84)

    def test_the_recovery_category_is_recorded_every_step(self):
        kinds = ["none", "forward", "rotation", "none"]
        rec = R.StepRecorder(
            _FakeShield([(0.5, 0.0)] * 4, kinds=kinds), keep_trace=True
        )
        for _ in kinds:
            rec.filter((1.0, 0.0), np.zeros(3), [])
        assert [row["recovery_kind"] for row in rec.trace] == kinds
        assert rec.kinds == {"forward": 1, "policy": 0, "rotation": 1, "none": 2}

    def test_a_plain_shield_reports_every_step_as_no_recovery(self):
        rec = R.StepRecorder(_FakeShield([(0.5, 0.0)] * 3), keep_trace=True)
        for _ in range(3):
            rec.filter((1.0, 0.0), np.zeros(3), [])
        assert rec.kinds == {"forward": 0, "policy": 0, "rotation": 0, "none": 3}

    def test_keep_trace_false_aggregates_without_storing_steps(self):
        rec = R.StepRecorder(_FakeShield([(0.0, 2.0)] * 3), keep_trace=False)
        for _ in range(3):
            rec.filter((1.0, 0.0), np.zeros(3), [])
        assert rec.trace == []
        assert (rec.n_steps, rec.n_spin) == (3, 3)

    def test_the_caps_come_from_the_call_not_the_config(self):
        """`run_shielded_duel`'s evader branch passes a different v_max; the
        spin test must use the cap the filter was actually given."""
        rec = R.StepRecorder(_FakeShield([(0.05, 2.0)]), keep_trace=True)
        rec.filter((1.0, 0.0), np.zeros(3), [], v_max=2.0)
        assert rec.n_spin == 1  # 0.05 <= 0.05 * 2.0
        rec2 = R.StepRecorder(_FakeShield([(0.05, 2.0)]), keep_trace=True)
        rec2.filter((1.0, 0.0), np.zeros(3), [], v_max=0.5)
        assert rec2.n_spin == 0  # 0.05 > 0.05 * 0.5

    def test_it_passes_the_filter_result_through_untouched(self):
        from pe2.config import EnvConfig
        from pe2.shield import CBFShield

        cfg = EnvConfig()
        state = np.array([0.0, 0.0, 0.3])
        ref_u, ref_info = CBFShield(cfg).filter((1.0, 0.4), state, [], v_max=1.0)
        rec = R.StepRecorder(CBFShield(cfg), keep_trace=True)
        got_u, got_info = rec.filter((1.0, 0.4), state, [], v_max=1.0)
        assert np.array_equal(got_u, ref_u)
        assert got_info == ref_info  # the recorder adds nothing to the info

    def test_it_delegates_the_attributes_escape_features_reaches_through(self):
        from pe2.config import EnvConfig
        from pe2.shield import CBFShield, escape_features

        cfg = EnvConfig()
        state = np.array([4.5, 0.0, 0.0])
        obstacles: list[dict] = []
        ref = escape_features(CBFShield(cfg), state, obstacles)
        rec = R.StepRecorder(CBFShield(cfg), keep_trace=False)
        assert rec.cfg is rec.shield.cfg
        assert np.array_equal(escape_features(rec, state, obstacles), ref)

    def test_it_exposes_the_recovery_counters_of_the_shield_it_wraps(self):
        from pe2.config import EnvConfig
        from pe2.shield import RecoveryShieldV2

        rec = R.StepRecorder(RecoveryShieldV2(EnvConfig()), keep_trace=False)
        assert rec.n_recoveries == 0
        assert rec.n_recoveries_forward == 0

    def test_an_unknown_attribute_is_an_attribute_error(self):
        rec = R.StepRecorder(_FakeShield([]), keep_trace=False)
        with pytest.raises(AttributeError):
            _ = rec.no_such_attribute


class TestStepTally:
    """The recorder is per episode (the shield is), so a gate cell's numbers
    have to be totalled across the episodes, and only the first
    `TRACE_EPISODES` may keep a full trace."""

    def _tally(self, outputs, n, trace_episodes=R.TRACE_EPISODES):
        return R._StepTally(lambda: _FakeShield(outputs), trace_episodes)

    def test_a_fresh_recorder_per_episode(self):
        tally = self._tally([(0.0, 2.0)], 3)
        made = [tally() for _ in range(3)]
        assert len({id(m) for m in made}) == 3
        assert len({id(m.shield) for m in made}) == 3

    def test_totals_across_the_episodes(self):
        tally = self._tally([(0.0, 2.0), (0.5, 0.0)], 2)
        for _ in range(3):
            rec = tally()
            rec.filter((1.0, 0.0), np.zeros(3), [])
            rec.filter((1.0, 0.0), np.zeros(3), [])
        assert tally.n_steps == 6
        assert tally.n_spin == 3
        assert tally.spin_fraction == pytest.approx(0.5)
        assert tally.kinds == {"forward": 0, "policy": 0, "rotation": 0, "none": 6}

    def test_only_the_first_five_episodes_keep_a_trace(self):
        tally = self._tally([(0.0, 2.0)], 1)
        for _ in range(8):
            tally().filter((1.0, 0.0), np.zeros(3), [])
        assert len(tally.traces) == R.TRACE_EPISODES
        assert all(len(t) == 1 for t in tally.traces)
        assert [r.keep_trace for r in tally.recorders] == [True] * 5 + [False] * 3

    def test_an_empty_tally_reports_a_zero_fraction_rather_than_dividing(self):
        tally = self._tally([], 0)
        assert tally.n_steps == 0 and tally.spin_fraction == 0.0

    def test_recoveries_are_totalled_from_the_wrapped_shields(self):
        class _Fake(_FakeShield):
            def __init__(self, fired):
                super().__init__([])
                self.n_recoveries = fired

        made = iter([_Fake(4), _Fake(0), _Fake(6)])
        tally = R._StepTally(lambda: next(made), R.TRACE_EPISODES)
        for _ in range(3):
            tally()
        assert tally.recoveries == 10

    def test_a_plain_shield_tallies_zero_recoveries(self):
        from pe2.config import EnvConfig
        from pe2.shield import CBFShield

        tally = R._StepTally(lambda: CBFShield(EnvConfig()), R.TRACE_EPISODES)
        tally()
        assert tally.recoveries == 0


class TestPerEpisodeShield:
    """`evaluate_structured` builds ONE shield for a whole call and
    `RecoveryShieldV2`'s immobile run and counters are shield state with no
    episode hook (Task 1, concern 4), so a RECF arm would carry an open run
    across the paired-seed boundary. Spec section 3 requires a fresh one."""

    class _Stall:  # drives at walls, so the filter stalls it constantly
        def predict(self, obs, deterministic=True):
            return np.array([1.0, 0.0], dtype=np.float32), None

    def test_one_fresh_shield_per_episode_with_a_zeroed_counter(self):
        from pe2.config import EnvConfig
        from pe2.scripted import GreedyEvader
        from pe2.shield import RecoveryShieldV2

        built, snapshot = [], []

        def spy():
            s = RecoveryShieldV2(EnvConfig())
            built.append(s)
            snapshot.append(s._immobile_run)
            return s

        logs = R.evaluate_arm_policy(
            self._Stall(),
            EnvConfig(),
            GreedyEvader,
            3,
            R.GATE_SEED0,
            shield_factory=spy,
            shield_obs=False,
            escape_obs=False,
        )
        assert len(logs) == 3
        assert len(built) == 3 and len({id(s) for s in built}) == 3
        assert snapshot == [0, 0, 0]
        assert all(lg["violations"] == 0 for lg in logs)

    def test_the_library_route_would_have_shared_one_shield(self):
        """Why the loop above exists, measured rather than asserted."""
        from pe2.config import EnvConfig
        from pe2.scripted import GreedyEvader
        from pe2.shield import RecoveryShieldV2
        from pe2.structured import evaluate_structured

        built = []

        def make():
            built.append(RecoveryShieldV2(EnvConfig()))
            return built[-1]

        evaluate_structured(
            self._Stall(),
            EnvConfig(),
            GreedyEvader,
            2,
            R.GATE_SEED0,
            shield_factory=make,
        )
        assert len(built) == 1

    def test_identical_to_evaluate_structured_for_a_stateless_shield(self):
        from pe2.config import EnvConfig
        from pe2.scripted import GreedyEvader
        from pe2.shield import CBFShield
        from pe2.structured import OrbitSearch, evaluate_structured

        cfg = EnvConfig()
        pol = OrbitSearch(cfg)
        ref = evaluate_structured(
            pol,
            cfg,
            GreedyEvader,
            3,
            R.GATE_SEED0,
            shield_factory=lambda: CBFShield(cfg),
            shield_obs=True,
        )
        got = R.evaluate_arm_policy(
            pol,
            cfg,
            GreedyEvader,
            3,
            R.GATE_SEED0,
            shield_factory=lambda: CBFShield(cfg),
            shield_obs=True,
            escape_obs=False,
        )
        assert got == ref

    def test_the_recording_wrapper_does_not_change_the_episodes(self):
        """Wrapping the shield must be observation-free: the same logs."""
        from pe2.config import EnvConfig
        from pe2.scripted import GreedyEvader
        from pe2.shield import CBFShield
        from pe2.structured import OrbitSearch

        cfg = EnvConfig()
        pol = OrbitSearch(cfg)
        plain = R.evaluate_arm_policy(
            pol,
            cfg,
            GreedyEvader,
            3,
            R.GATE_SEED0,
            shield_factory=lambda: CBFShield(cfg),
            shield_obs=True,
            escape_obs=False,
        )
        tally = R._StepTally(lambda: CBFShield(cfg), R.TRACE_EPISODES)
        wrapped = R.evaluate_arm_policy(
            pol,
            cfg,
            GreedyEvader,
            3,
            R.GATE_SEED0,
            shield_factory=tally,
            shield_obs=True,
            escape_obs=False,
        )
        assert wrapped == plain
        assert tally.n_steps == sum(lg["shield"]["n_steps"] for lg in plain)


class TestGateCell:
    def test_a_recovery_gate_reports_kinds_spin_and_traces(self, monkeypatch):
        monkeypatch.setattr(R, "GATE_OPPONENTS", ("GREEDY",))
        monkeypatch.setattr(R, "GATE_N", 6)
        entry = R.gate_model(
            TestPerEpisodeShield._Stall(), escape_obs=False, recovery=True
        )["GREEDY"]
        assert entry["recoveries"] > 0
        assert entry["recovery_kinds"]["forward"] >= 0
        assert sum(entry["recovery_kinds"].values()) == entry["recoveries"]
        assert entry["spin"]["n_steps"] == entry["n_steps"]
        assert 0.0 <= entry["spin"]["fraction"] <= 1.0
        assert entry["pursuer_violations"] == 0
        assert len(entry["traces"]) == R.TRACE_EPISODES
        row = entry["traces"][0][0]
        assert {"v_nom", "w_nom", "v_exec", "w_exec", "recovery_kind", "spin"} <= set(
            row
        )

    def test_a_plain_gate_records_no_recovery_kind_but_still_logs_spin(
        self, monkeypatch
    ):
        monkeypatch.setattr(R, "GATE_OPPONENTS", ("GREEDY",))
        monkeypatch.setattr(R, "GATE_N", 4)
        entry = R.gate_model(
            TestPerEpisodeShield._Stall(), escape_obs=False, recovery=False
        )["GREEDY"]
        assert entry["recoveries"] == 0
        assert entry["recovery_kinds"] == {"forward": 0, "policy": 0, "rotation": 0}
        assert entry["spin"]["n_steps"] > 0
        assert len(entry["capture_ind"]) == 4


# ----------------------------------------------------------------------------
# section 4: the three hypotheses
# ----------------------------------------------------------------------------
class TestH1:
    """H1 (task): capture significantly ABOVE BASE's on the same seed
    (McNemar, p < 0.05) on >= 2 of 3 seeds."""

    def test_an_arm_that_beats_base_on_all_three_seeds_passes(self):
        h = R.h1_verdicts(_full(OBSGAM=(0.90, 0.30, 0.25)))
        assert h["OBSGAM"]["verdict"] == "PASS" and h["OBSGAM"]["n_pass"] == 3

    def test_an_arm_level_with_base_fails(self):
        h = R.h1_verdicts(_full())
        assert h["OBSGAM"]["verdict"] == "FAIL" and h["OBSGAM"]["n_pass"] == 0

    def test_an_arm_worse_than_base_fails(self):
        h = R.h1_verdicts(_full(OBSGAM=(0.05, 0.30, 0.25)))
        assert h["OBSGAM"]["verdict"] == "FAIL"

    def test_two_of_three_is_enough_and_one_is_not(self):
        res = _full()
        for s, cap in zip(R.SEEDS, (0.90, 0.90, 0.40), strict=True):
            res[f"OBSGAM-{s}"]["final"]["gate"]["GREEDY"] = _entry(cap, 0.30, 0.25)
        assert R.h1_verdicts(res)["OBSGAM"]["n_pass"] == 2
        assert R.h1_verdicts(res)["OBSGAM"]["verdict"] == "PASS"
        res[f"OBSGAM-{R.SEEDS[1]}"]["final"]["gate"]["GREEDY"] = _entry(
            0.40, 0.30, 0.25
        )
        assert R.h1_verdicts(res)["OBSGAM"]["verdict"] == "FAIL"

    def test_base_is_the_reference_and_never_tested_against_itself(self):
        h = R.h1_verdicts(_full())
        assert h["BASE"]["verdict"] == "REFERENCE"

    def test_the_comparison_is_seed_matched(self):
        """Seed 60 beats BASE-60, not BASE's mean."""
        res = _full()
        res["OBSGAM-60"]["final"]["gate"]["GREEDY"] = _entry(0.90, 0.30, 0.25)
        h = R.h1_verdicts(res)
        assert h["OBSGAM"]["per_seed"][60]["pass"] is True
        assert h["OBSGAM"]["per_seed"][61]["pass"] is False

    def test_incomplete(self):
        res = _full()
        del res["OBSGAM-61"]
        assert R.h1_verdicts(res)["OBSGAM"]["verdict"] == "INCOMPLETE"
        res = _full()
        del res["OBSGAM-61"]["final"]
        assert R.h1_verdicts(res)["OBSGAM"]["verdict"] == "INCOMPLETE"


class TestH2:
    """H2 (deadlock): deadlock-episode rate against ATK4-45 < 0.05 on >= 2/3."""

    def test_below_the_bar_on_all_three(self):
        h = R.h2_verdicts(_full(RECF_BASE=(0.40, 0.01, 0.25)))
        assert h["RECF-BASE"]["verdict"] == "PASS" and h["RECF-BASE"]["n_pass"] == 3
        assert h["RECF-BASE"]["deadlock"] == {60: 0.01, 61: 0.01, 62: 0.01}

    def test_the_bar_is_strict(self):
        h = R.h2_verdicts(_full(RECF_BASE=(0.40, R.DEADLOCK_BAR, 0.25)))
        assert h["RECF-BASE"]["n_pass"] == 0

    def test_two_of_three(self):
        res = _full()
        for s, d in zip(R.SEEDS, (0.01, 0.02, 0.40), strict=True):
            res[f"RECF-BASE-{s}"]["final"]["gate"]["ATK4-45"] = _entry(0.10, d, 0.25)
        assert R.h2_verdicts(res)["RECF-BASE"]["verdict"] == "PASS"

    def test_it_reads_the_deadlock_opponent_only(self):
        res = _full(RECF_BASE=(0.40, 0.01, 0.25))
        for s in R.SEEDS:
            res[f"RECF-BASE-{s}"]["final"]["gate"]["GREEDY"] = _entry(0.40, 0.90, 0.25)
        assert R.h2_verdicts(res)["RECF-BASE"]["verdict"] == "PASS"

    def test_base_is_reported_too(self):
        h = R.h2_verdicts(_full())
        assert h["BASE"]["verdict"] == "FAIL"

    def test_incomplete(self):
        res = _full()
        del res["RECF-OBS-62"]
        assert R.h2_verdicts(res)["RECF-OBS"]["verdict"] == "INCOMPLETE"


class TestH3:
    """H3 (no spinning): spin fraction over ALL gate steps < 0.05 on >= 2/3.
    This is the bar the round exists to impose -- no arm may win by rotating
    on the spot (D129)."""

    def test_below_the_bar_on_all_three(self):
        h = R.h3_verdicts(_full(RECF_BASE=(0.40, 0.01, 0.01)))
        assert h["RECF-BASE"]["verdict"] == "PASS" and h["RECF-BASE"]["n_pass"] == 3

    def test_the_bar_is_strict(self):
        h = R.h3_verdicts(_full(RECF_BASE=(0.40, 0.01, R.SPIN_BAR)))
        assert h["RECF-BASE"]["n_pass"] == 0

    def test_the_fraction_pools_every_gate_opponent(self):
        """'over all gate steps' -- one opponent cannot hide a spinning arm."""
        res = _full(RECF_BASE=(0.40, 0.01, 0.0))
        for s in R.SEEDS:
            res[f"RECF-BASE-{s}"]["final"]["gate"]["HUG7"] = _entry(0.30, 0.01, 0.9)
        h = R.h3_verdicts(res)
        assert h["RECF-BASE"]["spin"][60] == pytest.approx(0.3, abs=1e-6)
        assert h["RECF-BASE"]["verdict"] == "FAIL"

    def test_spin_fraction_is_a_step_weighted_pool_not_a_mean_of_rates(self):
        res = _full(RECF_BASE=(0.40, 0.01, 0.0))
        gate = res["RECF-BASE-60"]["final"]["gate"]
        gate["ATK4-45"]["spin"] = {"n_spin": 10, "n_steps": 100, "fraction": 0.1}
        gate["HUG7"]["spin"] = {"n_spin": 0, "n_steps": 900, "fraction": 0.0}
        gate["GREEDY"]["spin"] = {"n_spin": 0, "n_steps": 0, "fraction": 0.0}
        assert R.spin_fraction(res, "RECF-BASE-60") == pytest.approx(0.01)

    def test_two_of_three(self):
        res = _full(RECF_BASE=(0.40, 0.01, 0.0))
        for o in R.GATE_OPPONENTS:
            res[f"RECF-BASE-{R.SEEDS[2]}"]["final"]["gate"][o] = _entry(0.3, 0.01, 0.9)
        assert R.h3_verdicts(res)["RECF-BASE"]["verdict"] == "PASS"

    def test_incomplete(self):
        res = _full()
        del res["RECF-OBSGAM-60"]
        assert R.h3_verdicts(res)["RECF-OBSGAM"]["verdict"] == "INCOMPLETE"


class TestSolves:
    """An arm SOLVES iff H1, H2 and H3 all hold. BASE is the reference and is
    not eligible."""

    def _solves(self, results):
        return R.solves(
            R.h1_verdicts(results), R.h2_verdicts(results), R.h3_verdicts(results)
        )

    def test_all_three_bars(self):
        assert self._solves(_full(RECF_BASE=(0.90, 0.01, 0.01))) == ["RECF-BASE"]

    def test_an_arm_that_passes_h1_and_h2_but_spins_does_not_solve(self):
        """The case this round exists to catch: S16's `REC` drove deadlock to
        zero by rotating on the spot, and the old metrics called that a fix."""
        res = _full(RECF_BASE=(0.90, 0.00, 0.60))
        h1, h2, h3 = (
            R.h1_verdicts(res),
            R.h2_verdicts(res),
            R.h3_verdicts(res),
        )
        assert h1["RECF-BASE"]["verdict"] == "PASS"
        assert h2["RECF-BASE"]["verdict"] == "PASS"
        assert h3["RECF-BASE"]["verdict"] == "FAIL"
        assert R.solves(h1, h2, h3) == []

    def test_failing_h1_alone_is_enough_to_not_solve(self):
        assert self._solves(_full(RECF_BASE=(0.40, 0.01, 0.01))) == []

    def test_failing_h2_alone_is_enough_to_not_solve(self):
        assert self._solves(_full(RECF_BASE=(0.90, 0.40, 0.01))) == []

    def test_base_cannot_solve_even_when_every_bar_would_pass(self):
        res = _full(BASE=(0.90, 0.00, 0.00))
        h1, h2, h3 = R.h1_verdicts(res), R.h2_verdicts(res), R.h3_verdicts(res)
        assert h2["BASE"]["verdict"] == "PASS" and h3["BASE"]["verdict"] == "PASS"
        assert "BASE" not in R.solves(h1, h2, h3)

    def test_the_solvers_come_back_in_the_arm_tables_order(self):
        res = _full(
            RECF_OBSGAM=(0.90, 0.01, 0.01),
            OBSGAM=(0.90, 0.01, 0.01),
            RECF_BASE=(0.90, 0.01, 0.01),
        )
        assert self._solves(res) == ["OBSGAM", "RECF-BASE", "RECF-OBSGAM"]

    def test_incomplete_families_do_not_solve(self):
        res = _full(RECF_BASE=(0.90, 0.01, 0.01))
        del res["RECF-BASE-62"]
        assert self._solves(res) == []


class TestSafety:
    """H-SAFE: 0 pursuer contact steps in every cell."""

    def test_holds_on_a_clean_round(self):
        h = R.hsafe(_full())
        assert h["verdict"] == "HOLDS" and h["pursuer_violation_steps"] == 0

    def test_a_contact_step_anywhere_fails_it(self):
        res = _full()
        res["OBSGAM-60"]["train_violations"] = 1
        assert R.hsafe(res)["verdict"] == "FAILS"
        res = _full()
        res["RECF-OBS-62"]["final"]["pursuer_violations"] = 3
        h = R.hsafe(res)
        assert h["verdict"] == "FAILS" and h["offenders"] == ["RECF-OBS-62"]
        assert h["pursuer_violation_steps"] == 3

    def test_the_reused_families_are_in_the_census(self):
        res = _full()
        res["BASE-61"]["final"]["pursuer_violations"] = 4
        assert R.hsafe(res)["offenders"] == ["BASE-61"]

    def test_missing_arms_are_incomplete(self):
        res = _full()
        del res["RECF-OBSGAM-62"]
        h = R.hsafe(res)
        assert h["verdict"] == "INCOMPLETE" and h["missing"] == ["RECF-OBSGAM-62"]

    def test_a_crash_record_still_counts_its_partial_gate(self):
        res = _full()
        del res["OBSGAM-60"]["final"]
        del res["OBSGAM-60"]["probe_violations"]
        res["OBSGAM-60"]["gate_violations"] = 1
        res["OBSGAM-60"]["checkpoints"] = [{"violations": 2}]
        h = R.hsafe(res)
        assert h["verdict"] == "FAILS" and h["pursuer_violation_steps"] == 3


# ----------------------------------------------------------------------------
# section 5: what each outcome fixes
# ----------------------------------------------------------------------------
class TestDirection:
    def _d(self, **kw):
        h = {
            "hsafe": "HOLDS",
            "incomplete": False,
            "solving": ["RECF-BASE", "RECF-OBS", "RECF-OBSGAM"],
            "recf_base_h3": "PASS",
        }
        return R.direction(h | kw)

    def test_hsafe_failure_stops_the_round(self):
        d = self._d(hsafe="FAILS")
        assert "Stop" in d
        assert "Stop" in self._d(hsafe="FAILS", solving=[], recf_base_h3="FAIL")

    def test_incomplete(self):
        assert "incomplete" in self._d(hsafe="INCOMPLETE")
        assert "incomplete" in self._d(incomplete=True)

    def test_recf_solving_where_the_bare_arms_do_not(self):
        d = self._d()
        assert "engineering" in d and "forward escape" in d
        assert "cannot be won by spinning" in d

    def test_recf_base_failing_h3_leads(self):
        d = self._d(solving=[], recf_base_h3="FAIL")
        assert "still spins" in d and "D130" in d and "re-diagnos" in d

    def test_obsgam_solving_alone_is_a_learning_fix(self):
        d = self._d(solving=["OBSGAM"])
        assert "learning one" in d and "admissible set" in d
        assert "scripted recovery is unnecessary" in d

    def test_no_arm_solves_is_reported_as_a_failure(self):
        d = self._d(solving=[])
        assert "failure" in d and "jointly unsatisfiable" in d

    def test_a_mixed_outcome_names_every_solver(self):
        d = self._d(solving=["OBSGAM", "RECF-BASE"])
        assert "OBSGAM" in d and "RECF-BASE" in d

    def test_recf_base_failing_h3_is_reported_even_when_another_arm_solves(self):
        d = self._d(solving=["RECF-OBSGAM"], recf_base_h3="FAIL")
        assert "still spins" in d and "RECF-OBSGAM" in d


class TestSummary:
    def test_summarize(self):
        s = R.summarize(_full(RECF_BASE=(0.90, 0.01, 0.01)))
        assert s["spec"].endswith("loop_fixes_b_prereg_2026-09-21.md")
        assert s["hsafe"]["verdict"] == "HOLDS"
        assert s["hypotheses"]["H1"]["RECF-BASE"]["verdict"] == "PASS"
        assert s["hypotheses"]["H2"]["RECF-BASE"]["verdict"] == "PASS"
        assert s["hypotheses"]["H3"]["RECF-BASE"]["verdict"] == "PASS"
        assert s["solving"] == ["RECF-BASE"]
        assert set(s["arms"]) == set(R.ARMS)
        assert s["arms"]["RECF-BASE-60"]["capture"] == pytest.approx(0.90)
        assert s["arms"]["RECF-BASE-60"]["spin"] == pytest.approx(0.01, abs=1e-6)
        assert s["complete"] is True
        assert isinstance(s["direction"], str) and s["direction"]

    def test_summarize_is_incomplete_when_an_arm_is_missing(self):
        res = _full()
        del res["RECF-OBS-62"]
        s = R.summarize(res)
        assert s["complete"] is False and "incomplete" in s["direction"]

    def test_the_amendment_a1_note_travels_with_the_summary(self):
        """S17-A1: a zero `policy` count is expected, not a bug."""
        s = R.summarize(_full())
        assert "policy" in s["recovery_kinds_note"]
        assert "S17-A1" in s["recovery_kinds_note"]

    def test_write_figure(self, tmp_path):
        pytest.importorskip("matplotlib")
        results = _full(RECF_BASE=(0.90, 0.01, 0.01))
        results["OBSGAM-60"]["checkpoints"] = [
            {"steps": 500_000, "mark": 500_000, "win_rate": 0.3, "spin_fraction": 0.4},
            {
                "steps": 1_000_000,
                "mark": 1_000_000,
                "win_rate": 0.4,
                "spin_fraction": 0.3,
            },
        ]
        p = tmp_path / "s17_round.png"
        R.write_figure(results, R.summarize(results), p)
        assert p.exists() and p.stat().st_size > 0


# ----------------------------------------------------------------------------
# section 6: artefacts, guards and the stop rule
# ----------------------------------------------------------------------------
class TestGuards:
    def test_artefact_paths_all_carry_the_prefix(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        assert R._arm_file("RECF-BASE-60") == tmp_path / "s17_arm_RECF-BASE-60.json"
        assert R._probe_file("OBSGAM-61") == tmp_path / "s17_probe_OBSGAM-61.json"
        assert R._round_file("png") == tmp_path / "s17_round.png"
        assert R._ckpt_dir("OBSGAM-62") == tmp_path / "ckpts_OBSGAM-62"

    def test_an_arm_that_already_ran_is_refused(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        R.require_arm_not_run("RECF-BASE-60")
        (tmp_path / "s17_arm_RECF-BASE-60.json").write_text("{}")
        with pytest.raises(SystemExit, match="stop rule"):
            R.require_arm_not_run("RECF-BASE-60")
        R.require_arm_not_run("OBSGAM-60")
        (tmp_path / "ckpts_OBSGAM-60").mkdir()
        with pytest.raises(SystemExit, match="stop rule"):
            R.require_arm_not_run("OBSGAM-60")

    def test_a_re_gated_arm_is_not_refused_for_its_own_source_checkpoints(
        self, tmp_path, monkeypatch
    ):
        """`BASE-60` gates `ckpts_BASE-60`, which exists BEFORE the arm runs --
        it is S16's. Only a TRAINING arm's checkpoint directory is evidence
        that this round already ran it; for the six re-gated families the arm
        record is the only marker."""
        monkeypatch.setattr(R, "OUT", tmp_path)
        for arm in ("BASE-60", "OBS-60", "GAM-60", "RECF-BASE-60"):
            (tmp_path / f"ckpts_{R.source_arm(arm)}").mkdir(exist_ok=True)
            R.require_arm_not_run(arm)  # must not raise
        (tmp_path / "s17_arm_BASE-60.json").write_text("{}")
        with pytest.raises(SystemExit, match="stop rule"):
            R.require_arm_not_run("BASE-60")

    def test_only_obsgam_may_be_trained(self):
        with pytest.raises(SystemExit, match="does not train"):
            R.require_trainable("RECF-BASE-60")
        with pytest.raises(SystemExit, match="does not train"):
            R.require_trainable("OBS-60")
        R.require_trainable("OBSGAM-60")

    def test_gating_refuses_a_missing_source_checkpoint(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        with pytest.raises(SystemExit, match="RECF-OBSGAM-60 needs"):
            R.require_source_ckpt("RECF-OBSGAM-60")
        d = tmp_path / "ckpts_OBSGAM-60"
        d.mkdir()
        (d / f"OBSGAM-60_{R.TOTAL}.zip").write_text("x")
        assert R.require_source_ckpt("RECF-OBSGAM-60") == d / f"OBSGAM-60_{R.TOTAL}.zip"

    def test_skip_existing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        names = ["BASE-60", "BASE-61", "RECF-BASE-60"]
        assert R.todo(names) == names
        (tmp_path / "s17_arm_BASE-61.json").write_text("{}")
        assert R.todo(names) == ["BASE-60", "RECF-BASE-60"]
        for n in names:
            (tmp_path / f"s17_arm_{n}.json").write_text("{}")
        assert R.todo(names) == []

    def test_todo_defaults_to_every_arm_in_the_tables_order(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(R, "OUT", tmp_path)
        assert R.todo(None) == list(R.ARMS)
        assert list(R.ARMS)[:3] == ["BASE-60", "BASE-61", "BASE-62"]

    def test_the_cli_routes_training_and_gating_to_the_right_entry_points(
        self, tmp_path, monkeypatch
    ):
        """`--train` may only name a trainable arm; a bare `--gate` means every
        arm this round has not yet recorded."""
        monkeypatch.setattr(R, "OUT", tmp_path)
        trained, gated = [], []
        monkeypatch.setattr(R, "run_arm", lambda a, d: trained.append(a))
        monkeypatch.setattr(R, "gate_arm", lambda a, d: gated.append(a))
        monkeypatch.setattr(
            "sys.argv", ["x", "--train", "OBSGAM-61", "--gate", "RECF-OBS-60"]
        )
        R.main()
        assert trained == ["OBSGAM-61"] and gated == ["RECF-OBS-60"]

        trained.clear()
        gated.clear()
        monkeypatch.setattr("sys.argv", ["x", "--gate"])
        R.main()
        assert gated == list(R.ARMS) and trained == []

        monkeypatch.setattr("sys.argv", ["x", "--train", "BASE-60"])
        with pytest.raises(SystemExit):  # argparse: not a trainable choice
            R.main()

    def test_load_results_reads_this_rounds_records_only(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        (tmp_path / "s17_arm_BASE-60.json").write_text('{"arm": "BASE-60"}')
        (tmp_path / "s17_arm_RECF-OBS-62.json").write_text('{"arm": "RECF-OBS-62"}')
        (tmp_path / "s16_arm_REC-60.json").write_text('{"arm": "REC-60"}')
        assert set(R.load_results()) == {"BASE-60", "RECF-OBS-62"}

    def test_the_arm_record_round_trips_through_json(self):
        """Everything the runner writes has to be serialisable, traces included."""
        res = _full()
        res["RECF-BASE-60"]["final"]["gate"]["GREEDY"]["traces"] = [
            [
                {
                    "v_nom": 1.0,
                    "w_nom": 0.0,
                    "v_exec": 0.0,
                    "w_exec": 2.84,
                    "recovery_kind": "rotation",
                    "spin": True,
                }
            ]
        ]
        assert json.loads(json.dumps(R.summarize(res), default=str))
