"""scripts/derisk_s16_fixes.py — the loop-fixes round's own decisions.

Spec: docs/pathway/loop_fixes_prereg_2026-09-20.md (sections 2-5). Everything
here is either a constant the spec fixes, a verdict rule it states, or one of
the two evaluation-path corrections the round needs (a fresh `RecoveryShield`
per episode for `REC`, and the LSTM state threaded through `MEM`'s episodes).
"""

import importlib.util
import json
import pathlib

import numpy as np
import pytest

_PATH = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "derisk_s16_fixes.py"


def _runner():
    spec = importlib.util.spec_from_file_location("derisk_s16_fixes", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _runner()
N = 100  # episodes in the synthetic gate entries below


def _ind(rate, n=N):
    k = round(rate * n)
    return [i < k for i in range(n)]


def _entry(immobile, capture, viol=0):
    """One gate cell in the shape `gate_arm` writes."""
    return {
        "capture_ind": _ind(capture),
        "deadlock_ind": _ind(min(1.0, immobile + 0.1)),
        "capture_rate": capture,
        "pursuer": {
            "immobile_step_rate": immobile,
            "deadlock_episode_rate": min(1.0, immobile + 0.1),
            "stall_step_rate": immobile,
            "intervention_rate": 0.4,
            "infeasibility_rate": 0.0,
            "backup_rate": 0.0,
            "min_clearance_floor": 0.2,
        },
        "pursuer_violations": viol,
        "n_steps": 1200 * N,
        "n_immobile": round(immobile * 1200 * N),
    }


def _arm(name, immobile, capture=0.40, viol=0):
    fam, seed = name.split("-")[0], int(name.split("-")[1])
    return {
        "arm": name,
        "family": fam,
        "seed": seed,
        "train_violations": 0,
        "probe_violations": 0,
        "checkpoints": [],
        "final": {
            "ckpt": f"results/derisk/ckpts_{name}/{name}_5000000.zip",
            "pursuer_violations": viol,
            "gate": {
                "ATK4-45": _entry(immobile, 0.10),
                "HUG7": _entry(immobile, 0.30),
                "GREEDY": _entry(immobile, capture),
            },
        },
    }


def _full(
    base=(0.40, 0.42, 0.41),
    gam=(0.38, 0.39, 0.40),
    obs=(0.30, 0.31, 0.32),
    mem=(0.35, 0.36, 0.37),
    pen=(0.39, 0.40, 0.41),
    rec=(0.02, 0.03, 0.02),
    captures=None,
):
    """A complete set of records: five learned families plus REC, three seeds."""
    rates = {"BASE": base, "GAM": gam, "OBS": obs, "MEM": mem, "PEN": pen, "REC": rec}
    caps = captures or {}
    out = {}
    for fam, rs in rates.items():
        for s, r in zip(R.SEEDS, rs, strict=True):
            name = f"{fam}-{s}"
            out[name] = _arm(name, r, capture=caps.get(name, 0.40))
    return out


# ----------------------------------------------------------------------------
# section 2: the arms
# ----------------------------------------------------------------------------
class TestProtocol:
    def test_constants(self):
        assert R.SEEDS == (60, 61, 62)
        assert (R.TOTAL, R.EVERY) == (5_000_000, 500_000)
        assert (R.PROBE_SEED0, R.GATE_SEED0) == (1_730_000, 1_720_000)
        assert (R.GATE_N, R.EVAL_N) == (500, 200)
        assert (R.SOLVE_BAR, R.MATCH_BAR, R.GAM_BAR) == (0.10, 0.05, 0.05)
        assert (R.PASS_MIN, R.P_GATE) == (2, 0.05)
        assert R.GATE_OPPONENTS == ("ATK4-45", "HUG7", "GREEDY")
        assert R.PRIMARY_OPPONENT == "ATK4-45" and R.CAPTURE_OPPONENT == "GREEDY"
        assert R.ARTEFACT_PREFIX == "s16"
        assert R.RECOVERY_AFTER == 20
        assert (R.HOLD, R.R_ACQ, R.SHIELD_OBS) == (10, 0.0, True)
        assert R.SPEC.endswith("loop_fixes_prereg_2026-09-20.md")

    def test_the_arm_table_is_five_families_by_three_seeds(self):
        assert R.LEARNED == ("BASE", "GAM", "OBS", "MEM", "PEN")
        assert len(R.ARMS) == 15
        assert set(R.ARMS) == {f"{f}-{s}" for f in R.LEARNED for s in (60, 61, 62)}
        assert set(R.REC_ARMS) == {"REC-60", "REC-61", "REC-62"}
        assert R.REC_ARMS["REC-61"]["source"] == "BASE-61"
        assert len(R.ALL_ARMS) == 18
        assert R.FAMILY_ORDER == ("BASE", "GAM", "OBS", "MEM", "PEN", "REC")

    def test_each_arm_carries_the_spec_configuration(self):
        """Spec section 2's table, cell by cell."""
        assert R.ARMS["BASE-60"] == {
            "family": "BASE",
            "seed": 60,
            "gamma": 0.99,
            "stall_penalty": 1e-3,
            "escape_obs": False,
            "algo": "PPO",
        }
        assert R.ARMS["GAM-61"]["gamma"] == 0.999
        assert R.ARMS["OBS-62"]["escape_obs"] is True
        assert R.ARMS["MEM-60"]["algo"] == "RecurrentPPO"
        assert R.ARMS["PEN-61"]["stall_penalty"] == 1e-2

    def test_every_remedy_differs_from_base_in_exactly_one_setting(self):
        base = R.FAMILIES["BASE"]
        for fam in ("GAM", "OBS", "MEM", "PEN"):
            diff = {k for k, v in R.FAMILIES[fam].items() if base[k] != v}
            assert len(diff) == 1, f"{fam} changes {diff}"

    def test_only_mem_is_recurrent(self):
        assert R.is_recurrent("MEM-60") is True
        for arm in R.ARMS:
            if not arm.startswith("MEM"):
                assert R.is_recurrent(arm) is False
        assert R.is_recurrent("REC-60") is False  # REC replays BASE's PPO policy
        assert R.POLICY_STRING["PPO"] == "MlpPolicy"
        assert R.POLICY_STRING["RecurrentPPO"] == "MlpLstmPolicy"
        from sb3_contrib import RecurrentPPO
        from stable_baselines3 import PPO

        assert R.algo_class("PPO") is PPO
        assert R.algo_class("RecurrentPPO") is RecurrentPPO


class TestEnvAndLearner:
    """The settings must reach the objects, not just the table."""

    def test_env_width_and_penalty_per_arm(self):
        for arm, width, penalty in (
            ("BASE-60", 14, 1e-3),
            ("OBS-60", 16, 1e-3),
            ("PEN-60", 14, 1e-2),
            ("MEM-60", 14, 1e-3),
        ):
            env = R.make_env(arm)
            assert env.observation_space.shape == (width,), arm
            assert env.stall_penalty == penalty
            assert env.shield_obs is True and env.hold == 10
            assert env.r_acq == 0.0 and env.zero_reward is False
            assert type(env.shield).__name__ == "CBFShield"
        assert R.make_env("OBS-60").escape_obs is True
        assert R.make_env("BASE-60").escape_obs is False

    def test_the_discount_reaches_the_learner_and_mem_is_an_lstm(self):
        from sb3_contrib import RecurrentPPO
        from stable_baselines3 import PPO

        base = R.make_learner("BASE-60", R.make_env("BASE-60"), "cpu")
        assert isinstance(base, PPO) and base.gamma == 0.99
        gam = R.make_learner("GAM-60", R.make_env("GAM-60"), "cpu")
        assert isinstance(gam, PPO) and gam.gamma == 0.999
        mem = R.make_learner("MEM-60", R.make_env("MEM-60"), "cpu")
        assert isinstance(mem, RecurrentPPO) and mem.gamma == 0.99
        assert mem.policy.lstm_actor.input_size == 14  # the arm's observation


# ----------------------------------------------------------------------------
# the two evaluation-path corrections
# ----------------------------------------------------------------------------
class TestRecurrentPredictor:
    """`RecurrentPPO.predict` takes and returns the LSTM state; the
    `SearchMDP.play_episode` loop calls `predict(obs, deterministic=True)` and
    throws the second return value away, so an unthreaded MEM would be a
    memoryless policy wearing an LSTM."""

    class _Fake:
        def __init__(self):
            self.calls = []
            self.n = 0

        def predict(self, obs, state=None, episode_start=None, deterministic=False):
            self.calls.append((state, episode_start, deterministic))
            self.n += 1
            return np.zeros(2, dtype=np.float32), (f"state{self.n}",)

    def test_state_is_threaded_and_the_episode_start_flag_falls(self):
        fake = self._Fake()
        pred = R.RecurrentPredictor(fake)
        a, st = pred.predict(np.zeros(14, dtype=np.float32), deterministic=True)
        assert a.shape == (2,) and st == ("state1",)
        state, start, det = fake.calls[0]
        assert state is None and det is True
        assert np.array_equal(start, np.array([True]))
        pred.predict(np.zeros(14, dtype=np.float32), deterministic=True)
        state, start, _ = fake.calls[1]
        assert state == ("state1",)  # the previous call's state, threaded
        assert np.array_equal(start, np.array([False]))

    def test_reset_starts_a_fresh_episode(self):
        fake = self._Fake()
        pred = R.RecurrentPredictor(fake)
        pred.predict(np.zeros(14, dtype=np.float32))
        pred.reset()
        pred.predict(np.zeros(14, dtype=np.float32))
        state, start, _ = fake.calls[-1]
        assert state is None and np.array_equal(start, np.array([True]))

    def test_a_memoryless_model_is_used_unwrapped(self):
        model = object()
        assert R.predictor_for(model, recurrent=False) is model
        assert isinstance(R.predictor_for(model, recurrent=True), R.RecurrentPredictor)


class TestPerEpisodeShield:
    """`evaluate_structured` builds ONE shield for a whole call, and
    `RecoveryShield`'s immobile run is shield state with no episode hook (Task
    1, concern 1), so `REC` would carry an open run across the paired-seed
    boundary. The round's own evaluation loop rebuilds the shield per episode."""

    class _Stall:  # drives at walls, so the filter stalls it constantly
        def predict(self, obs, deterministic=True):
            return np.array([1.0, 0.0], dtype=np.float32), None

    def _factory(self, cls, **kw):
        from pe2.config import EnvConfig

        built = []

        def make():
            built.append(cls(EnvConfig(), **kw))
            return built[-1]

        return make, built

    def test_one_fresh_shield_per_episode_with_a_zeroed_counter(self):
        from pe2.config import EnvConfig
        from pe2.scripted import GreedyEvader
        from pe2.shield import RecoveryShield

        make, built = self._factory(RecoveryShield)
        seen, snapshot = [], []

        def spy():
            s = make()
            seen.append(s)
            snapshot.append(s._immobile_run)  # the counter AT handover
            return s

        # +27: the first stretch of the gate stream on which this policy times
        # out still immobile, so an open run really does reach the boundary.
        logs = R.evaluate_arm_policy(
            self._Stall(),
            EnvConfig(),
            GreedyEvader,
            3,
            R.GATE_SEED0 + 27,
            shield_factory=spy,
            shield_obs=False,
            escape_obs=False,
            recurrent=False,
        )
        assert len(logs) == 3
        assert len(seen) == 3 and len({id(s) for s in seen}) == 3
        assert snapshot == [0, 0, 0]  # every episode starts on a zeroed counter
        # the state that would have leaked exists: two of these three episodes
        # end with an immobile run of hundreds of steps still open
        assert sum(s._immobile_run > 0 for s in built) >= 2
        assert all(lg["violations"] == 0 for lg in logs)

    def test_the_library_route_would_have_shared_one_shield(self):
        """Why the loop above exists, measured rather than asserted."""
        from pe2.config import EnvConfig
        from pe2.scripted import GreedyEvader
        from pe2.shield import RecoveryShield
        from pe2.structured import evaluate_structured

        make, built = self._factory(RecoveryShield)
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
        """The only differences are the per-episode shield and the optional
        LSTM state; on a stateless shield and a memoryless policy the round's
        loop must reproduce the library's paired-seed evaluation exactly."""
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
            recurrent=False,
        )
        assert got == ref


class TestRecoveryTally:
    """Prereg amendment S16-A2: a firing executes what the filter admits, which
    is occasionally forward motion, and `RecoveryShield` now counts firings and
    the translating subset so the round can report that fraction. The counters
    live on the shield and this round builds one shield per episode, so they
    have to be totalled across an evaluation."""

    class _Fake:
        def __init__(self, fired, translating):
            self.n_recoveries = fired
            self.n_recoveries_translating = translating

    def test_totals_across_the_per_episode_shields(self):
        made = iter([self._Fake(4, 1), self._Fake(0, 0), self._Fake(6, 2)])
        tally = R._RecoveryTally(lambda: next(made))
        assert (tally.fired, tally.translating) == (0, 0)
        for _ in range(3):
            tally()
        assert (tally.fired, tally.translating) == (10, 3)
        assert len(tally.shields) == 3

    def test_a_plain_shield_tallies_zero(self):
        from pe2.config import EnvConfig
        from pe2.shield import CBFShield

        tally = R._RecoveryTally(lambda: CBFShield(EnvConfig()))
        tally()
        assert (tally.fired, tally.translating) == (0, 0)

    def test_a_recovery_gate_reports_the_firings(self, monkeypatch):
        monkeypatch.setattr(R, "GATE_OPPONENTS", ("GREEDY",))
        monkeypatch.setattr(R, "GATE_N", 3)
        monkeypatch.setattr(R, "GATE_SEED0", R.GATE_SEED0 + 27)
        entry = R.gate_arm(TestPerEpisodeShield._Stall(), recovery=True)["GREEDY"]
        assert entry["recoveries"] > 0  # the recovery really fired
        assert 0 <= entry["recoveries_translating"] <= entry["recoveries"]
        assert entry["pursuer_violations"] == 0
        plain = R.gate_arm(TestPerEpisodeShield._Stall(), recovery=False)["GREEDY"]
        assert "recoveries" not in plain  # only the REC arm reports them


# ----------------------------------------------------------------------------
# section 4: the three hypotheses
# ----------------------------------------------------------------------------
class TestFix1:
    def test_solves_needs_two_of_three_seeds_below_the_bar(self):
        h = R.fix1_verdicts(_full(obs=(0.05, 0.06, 0.40)))
        assert h["OBS"]["verdict"] == "SOLVES" and h["OBS"]["n_below_bar"] == 2
        assert h["BASE"]["verdict"] == "FAILS"
        assert h["REC"]["verdict"] == "SOLVES" and h["REC"]["n_below_bar"] == 3
        assert h["OBS"]["immobile"] == {60: 0.05, 61: 0.06, 62: 0.40}

    def test_one_seed_below_the_bar_is_not_enough(self):
        h = R.fix1_verdicts(_full(obs=(0.05, 0.40, 0.40)))
        assert h["OBS"]["verdict"] == "FAILS" and h["OBS"]["n_below_bar"] == 1

    def test_the_bar_is_strict(self):
        h = R.fix1_verdicts(_full(obs=(R.SOLVE_BAR, R.SOLVE_BAR, 0.05)))
        assert h["OBS"]["n_below_bar"] == 1  # 0.10 is not < 0.10

    def test_capture_guard_downgrades_to_solves_weaker(self):
        """BASE significantly better at capture on >= 2/3 seeds (paired
        McNemar, p < 0.05): the arm still solves, but weaker."""
        caps = {f"BASE-{s}": 0.60 for s in R.SEEDS}
        caps |= {f"OBS-{s}": 0.10 for s in R.SEEDS}
        h = R.fix1_verdicts(_full(obs=(0.05, 0.05, 0.05), captures=caps))
        assert h["OBS"]["verdict"] == "SOLVES (weaker)"
        assert sum(g["pass"] for g in h["OBS"]["capture_guard"].values()) >= 2

    def test_one_seed_losing_capture_does_not_downgrade(self):
        """Seed 60 only: BASE 0.60 vs OBS 0.10; the other two seeds are level."""
        caps = {"BASE-60": 0.60, "OBS-60": 0.10}
        h = R.fix1_verdicts(_full(obs=(0.05, 0.05, 0.05), captures=caps))
        assert sum(g["pass"] for g in h["OBS"]["capture_guard"].values()) == 1
        assert h["OBS"]["verdict"] == "SOLVES"

    def test_base_is_never_its_own_capture_guard_failure(self):
        h = R.fix1_verdicts(_full(base=(0.05, 0.05, 0.05)))
        assert h["BASE"]["verdict"] == "SOLVES"

    def test_incomplete(self):
        res = _full()
        del res["OBS-61"]
        h = R.fix1_verdicts(res)
        assert h["OBS"]["verdict"] == "INCOMPLETE"
        assert h["BASE"]["verdict"] == "FAILS"
        res = _full()
        del res["OBS-61"]["final"]  # crashed before the gate
        assert R.fix1_verdicts(res)["OBS"]["verdict"] == "INCOMPLETE"

    def test_solving_lists_every_family_that_solves(self):
        h = R.fix1_verdicts(_full(obs=(0.05, 0.05, 0.05)))
        assert R.solving(h) == ["OBS", "REC"]


class TestFix2:
    def test_matches_within_the_bar_of_recs_mean(self):
        h = R.fix2_verdicts(_full(rec=(0.02, 0.02, 0.02), obs=(0.06, 0.07, 0.06)))
        assert h["rec_mean"] == pytest.approx(0.02)
        assert h["per_family"]["OBS"]["mean"] == pytest.approx(0.0633333, abs=1e-6)
        assert h["per_family"]["OBS"]["verdict"] == "MATCHES"
        assert h["per_family"]["BASE"]["verdict"] == "DOES NOT MATCH"
        assert h["matching"] == ["OBS"]
        assert "REFUTED" in h["H_FIX_2"]

    def test_no_learned_arm_matching_confirms_the_expectation(self):
        h = R.fix2_verdicts(_full())
        assert h["matching"] == [] and h["base_matches"] is False
        assert "CONFIRMED" in h["H_FIX_2"]
        assert all(v["verdict"] == "DOES NOT MATCH" for v in h["per_family"].values())

    def test_a_remedy_that_matches_refutes_the_expectation(self):
        h = R.fix2_verdicts(_full(rec=(0.02, 0.02, 0.02), gam=(0.05, 0.05, 0.05)))
        assert h["per_family"]["GAM"]["verdict"] == "MATCHES"
        assert h["matching"] == ["GAM"] and "REFUTED" in h["H_FIX_2"]

    def test_the_bar_is_inclusive_and_one_sided(self):
        """Exactly 0.05 above REC matches; an arm BELOW REC matches a fortiori
        (it beats the baseline) and is flagged as beating it."""
        h = R.fix2_verdicts(_full(rec=(0.10, 0.10, 0.10), gam=(0.15, 0.15, 0.15)))
        assert h["per_family"]["GAM"]["verdict"] == "MATCHES"
        h = R.fix2_verdicts(_full(rec=(0.10, 0.10, 0.10), gam=(0.1501,) * 3))
        assert h["per_family"]["GAM"]["verdict"] == "DOES NOT MATCH"
        h = R.fix2_verdicts(_full(rec=(0.30, 0.30, 0.30), gam=(0.02, 0.02, 0.02)))
        g = h["per_family"]["GAM"]
        assert g["beats"] is True and g["two_sided_within"] is False
        # amendment S16-A3: beating REC must not hide inside the match label
        assert g["verdict"] == "MATCHES (BEATS REC)"
        assert h["beating"] == ["GAM"] and "GAM" in h["matching"]
        assert "BEATS REC" in h["H_FIX_2"]

    def test_beating_is_only_strictly_below_the_bar(self):
        h = R.fix2_verdicts(_full(rec=(0.30, 0.30, 0.30), gam=(0.25, 0.25, 0.25)))
        assert h["per_family"]["GAM"]["verdict"] == "MATCHES"
        assert h["beating"] == []

    def test_base_matching_rec_is_reported_separately(self):
        h = R.fix2_verdicts(_full(rec=(0.40, 0.40, 0.40)))
        assert h["base_matches"] is True
        assert h["matching"] == ["GAM", "OBS", "MEM", "PEN"]

    def test_incomplete_without_rec(self):
        res = _full()
        del res["REC-62"]
        h = R.fix2_verdicts(res)
        assert h["rec_mean"] is None and h["H_FIX_2"] == "INCOMPLETE"
        assert h["per_family"]["OBS"]["verdict"] == "INCOMPLETE"


class TestFix3:
    def test_gam_below_base_by_the_bar_on_two_of_three(self):
        h = R.fix3_verdicts(_full(base=(0.40, 0.40, 0.40), gam=(0.35, 0.34, 0.40)))
        assert h["verdict"] == "PASS" and h["n_pass"] == 2
        assert h["per_seed"][60]["delta"] == pytest.approx(0.05)

    def test_the_bar_is_inclusive_and_one_of_three_fails(self):
        h = R.fix3_verdicts(_full(base=(0.40, 0.40, 0.40), gam=(0.35, 0.36, 0.40)))
        assert h["n_pass"] == 1 and h["verdict"] == "FAIL"

    def test_incomplete(self):
        res = _full()
        del res["GAM-60"]
        assert R.fix3_verdicts(res)["verdict"] == "INCOMPLETE"


class TestSafety:
    def test_hsafe_holds_on_a_clean_round(self):
        h = R.hsafe(_full())
        assert h["verdict"] == "HOLDS" and h["pursuer_violation_steps"] == 0

    def test_a_contact_step_anywhere_fails_it(self):
        res = _full()
        res["BASE-60"]["train_violations"] = 1
        assert R.hsafe(res)["verdict"] == "FAILS"
        res = _full()
        res["MEM-62"]["probe_violations"] = 2
        h = R.hsafe(res)
        assert h["verdict"] == "FAILS" and h["offenders"] == ["MEM-62"]
        res = _full()
        res["OBS-61"]["final"]["pursuer_violations"] = 3
        assert R.hsafe(res)["pursuer_violation_steps"] == 3

    def test_rec_is_in_the_census(self):
        res = _full()
        res["REC-61"]["final"]["pursuer_violations"] = 4
        h = R.hsafe(res)
        assert h["verdict"] == "FAILS" and h["offenders"] == ["REC-61"]
        assert h["pursuer_violation_steps"] == 4

    def test_missing_arms_are_incomplete(self):
        res = _full()
        del res["REC-62"]
        h = R.hsafe(res)
        assert h["verdict"] == "INCOMPLETE" and h["missing"] == ["REC-62"]

    def test_a_crash_record_still_counts_its_partial_gate(self):
        res = _full()
        del res["PEN-60"]["final"]
        res["PEN-60"]["gate_violations"] = 1
        res["PEN-60"]["checkpoints"] = [{"violations": 2}]
        del res["PEN-60"]["probe_violations"]
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
            "solving": ["REC"],
            "matching": [],
            "beating": [],
            "base_matches": False,
        }
        return R.direction(h | kw)

    def test_hsafe_failure_stops_the_round(self):
        assert "Stop" in self._d(hsafe="FAILS")
        assert "Stop" in self._d(hsafe="FAILS", solving=[], matching=["OBS"])

    def test_incomplete(self):
        assert "incomplete" in self._d(hsafe="INCOMPLETE")
        assert "incomplete" in self._d(incomplete=True)

    def test_the_expected_row(self):
        d = self._d()
        assert "engineering one" in d and "not a learning one" in d

    def test_no_arm_solves(self):
        d = self._d(solving=[])
        assert "specific to the policies" in d and "redesign" in d.lower()

    def test_each_matching_row(self):
        obs = self._d(matching=["OBS"])
        # amendment S16-A1: the feature is the LINEARISED admissible set
        assert "observation-design rule" in obs and "LINEARISED" in obs
        assert "horizon rule" in self._d(matching=["GAM"])
        assert "memoryless architectures" in self._d(matching=["MEM"])
        assert "D113" in self._d(matching=["PEN"])

    def test_an_arm_that_beats_rec_leads_the_direction(self):
        d = self._d(matching=["OBS"], beating=["OBS"])
        assert "BEATS REC" in d and "observation-design rule" in d

    def test_several_matches_name_them_all(self):
        d = self._d(matching=["GAM", "OBS"])
        assert "horizon rule" in d and "OBS" in d

    def test_a_learned_arm_solving_where_rec_does_not(self):
        d = self._d(solving=["OBS"], matching=["OBS"])
        assert "REC" in d and "OBS" in d

    def test_base_matching_rec_is_a_harness_warning(self):
        d = self._d(base_matches=True, matching=["OBS"])
        assert "BASE" in d and "cannot discriminate" in d


class TestSummary:
    def test_summarize(self):
        s = R.summarize(_full(obs=(0.05, 0.05, 0.05)))
        assert s["spec"].endswith("loop_fixes_prereg_2026-09-20.md")
        assert s["hsafe"]["verdict"] == "HOLDS"
        assert s["hypotheses"]["H_FIX_1"]["OBS"]["verdict"] == "SOLVES"
        assert "H_FIX_2" in s["hypotheses"] and "H_FIX_3" in s["hypotheses"]
        assert set(s["arms"]) == set(R.ALL_ARMS)
        assert s["arms"]["OBS-60"]["immobile"] == pytest.approx(0.05)
        assert s["complete"] is True
        assert isinstance(s["direction"], str) and s["direction"]

    def test_summarize_is_incomplete_when_an_arm_is_missing(self):
        res = _full()
        del res["MEM-62"]
        s = R.summarize(res)
        assert s["complete"] is False and "incomplete" in s["direction"]

    def test_write_figure(self, tmp_path):
        pytest.importorskip("matplotlib")
        results = _full()
        results["BASE-60"]["checkpoints"] = [
            {
                "steps": 500_000,
                "mark": 500_000,
                "win_rate": 0.3,
                "immobile_step_rate": 0.4,
            },
            {
                "steps": 1_000_000,
                "mark": 1_000_000,
                "win_rate": 0.4,
                "immobile_step_rate": 0.3,
            },
        ]
        p = tmp_path / "s16_round.png"
        R.write_figure(results, R.summarize(results), p)
        assert p.exists() and p.stat().st_size > 0


# ----------------------------------------------------------------------------
# section 6: artefacts and the stop rule
# ----------------------------------------------------------------------------
class TestGuards:
    def test_an_arm_that_already_ran_is_refused(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        R.require_arm_not_run("BASE-60")  # nothing on disk yet
        (tmp_path / "s16_arm_BASE-60.json").write_text("{}")
        with pytest.raises(SystemExit, match="stop rule"):
            R.require_arm_not_run("BASE-60")
        R.require_arm_not_run("REC-60")
        (tmp_path / "ckpts_REC-60").mkdir()
        with pytest.raises(SystemExit, match="stop rule"):
            R.require_arm_not_run("REC-60")

    def test_an_arm_refuses_to_run_before_the_round_is_started(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(R, "OUT", tmp_path)
        monkeypatch.setattr(R, "START_PATH", tmp_path / "s16_start.json")
        with pytest.raises(SystemExit, match="measure-start"):
            R.load_start()
        (tmp_path / "s16_start.json").write_text(json.dumps({"gate_seed0": 1}))
        with pytest.raises(SystemExit, match="re-run --measure-start"):
            R.load_start()

    def test_the_start_file_pins_the_gate_opponent(self, tmp_path, monkeypatch):
        ckpt = tmp_path / "atk.zip"
        ckpt.write_bytes(b"attacker")
        monkeypatch.setattr(R, "OUT", tmp_path)
        monkeypatch.setattr(R, "START_PATH", tmp_path / "s16_start.json")
        monkeypatch.setattr(R, "ATK_CKPT", ckpt)
        start = R.measure_start()
        assert start["attacker"]["sha256"] == R._sha256(ckpt)
        assert R.load_start()["attacker"]["ckpt"] == str(ckpt)
        ckpt.write_bytes(b"a different attacker")
        with pytest.raises(SystemExit, match="changed since"):
            R.load_start()

    def test_the_round_is_measured_once(self, tmp_path, monkeypatch):
        ckpt = tmp_path / "atk.zip"
        ckpt.write_bytes(b"attacker")
        monkeypatch.setattr(R, "OUT", tmp_path)
        monkeypatch.setattr(R, "START_PATH", tmp_path / "s16_start.json")
        monkeypatch.setattr(R, "ATK_CKPT", ckpt)
        R.measure_start()
        with pytest.raises(SystemExit, match="stop rule"):
            R.measure_start()

    def test_measure_start_refuses_a_missing_attacker(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        monkeypatch.setattr(R, "START_PATH", tmp_path / "s16_start.json")
        monkeypatch.setattr(R, "ATK_CKPT", tmp_path / "nope.zip")
        with pytest.raises(SystemExit, match="gate opponent"):
            R.measure_start()

    def test_rec_refuses_to_run_before_base(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        with pytest.raises(SystemExit, match="BASE"):
            R.rec_seeds()
        d = tmp_path / "ckpts_BASE-61"
        d.mkdir()
        (d / f"BASE-61_{R.TOTAL}.zip").write_text("x")
        assert R.rec_seeds() == [61]
        (tmp_path / "s16_arm_REC-61.json").write_text("{}")  # already measured
        assert R.rec_seeds() == []

    def test_load_results_reads_every_arm_record(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        (tmp_path / "s16_arm_BASE-60.json").write_text('{"arm": "BASE-60"}')
        (tmp_path / "s16_arm_REC-60.json").write_text('{"arm": "REC-60"}')
        (tmp_path / "s15_arm_RC19.json").write_text('{"arm": "RC19"}')
        assert set(R.load_results()) == {"BASE-60", "REC-60"}

    def test_artefact_paths_all_carry_the_prefix(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        assert R._arm_file("BASE-60") == tmp_path / "s16_arm_BASE-60.json"
        assert R._probe_file("MEM-61") == tmp_path / "s16_probe_MEM-61.json"
        assert R._round_file("png") == tmp_path / "s16_round.png"
        assert R._ckpt_dir("OBS-62") == tmp_path / "ckpts_OBS-62"
