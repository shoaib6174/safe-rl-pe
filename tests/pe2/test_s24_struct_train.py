"""scripts/derisk_s24_struct_train.py — S24-FLAT Step F.2 (the `STRUCT` arm).

Spec: `docs/pathway/tactics_prereg_s24_flat_2026-10-14.md` sections 2, 3
(`STRUCT-s`) and 5 (CG-F1, STRUCT half).

Unlike Step F.1 (`FlatMDP`, a new `pe2` env class), this arm reuses
`pe2.structured.SearchMDP` unchanged -- the only new code is the trainer
script -- so everything below exercises the SCRIPT: the env/learner it
builds, CG-F1 on that construction, the artefact paths and the resumability
reading, a crash becoming UNDETERMINED and a SystemExit aborting instead,
that it prints no rate, and one 500-step PPO smoke that learns and writes
the probe JSON schema.

No real 3 M run: the smoke is 500 decisions with a tiny rollout.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import time

import pytest

from pe2.config import EnvConfig
from pe2.obs import obs_dim
from pe2.selectors import EvaderRulesV0
from pe2.shield import RECOVERY_AFTER, CBFShield
from pe2.structured import HOLD, InterceptLaw, SearchMDP
from pe2.tactics import TacticOpponent

_PATH = (
    pathlib.Path(__file__).resolve().parents[2]
    / "scripts"
    / "derisk_s24_struct_train.py"
)


def _runner():
    spec = importlib.util.spec_from_file_location("derisk_s24_struct_train", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _runner()


# ----------------------------------------------------------------------------
# section 2/3: the pre-registered constants and the arm table
# ----------------------------------------------------------------------------
class TestProtocol:
    def test_constants(self):
        assert R.SEEDS == (80, 81, 82)
        assert (R.TOTAL, R.EVERY) == (3_000_000, 250_000)
        assert R.PROBE_SEED0 == 2_600_000
        assert R.PROBE_N == 200
        assert R.GAMMA == 0.99
        assert R.RATIO == 1.2
        assert R.OPPONENT == "TACTIC-V0"
        assert R.PREFIX == "s24_struct"
        assert R.PREREG.endswith("tactics_prereg_s24_flat_2026-10-14.md")

    def test_the_arm_table_is_three_seeds_named_literally(self):
        assert set(R.ARMS) == {"STRUCT-80", "STRUCT-81", "STRUCT-82"}
        for seed in R.SEEDS:
            assert R.ARMS[f"STRUCT-{seed}"] == {"family": "STRUCT", "seed": seed}

    def test_the_checkpoint_label_is_the_seed_no_doubling(self):
        """F.1's `_label` function, reused: `STRUCT-80` carries the round and
        the arm exactly once, unlike F.1's own `FLAT` doubling note."""
        assert R._label("STRUCT-80") == "S24-STRUCT-80"
        assert R._ckpt_dir("STRUCT-81") == R.OUT / "ckpts_S24-STRUCT-81"

    def test_obs_len_is_the_base_width_plus_shield_and_escape_features(self):
        assert R.OBS_LEN == obs_dim(EnvConfig()) + 3 + 2
        assert R.OBS_LEN == 16

    def test_the_total_budget_is_a_multiple_of_the_probe_interval(self):
        assert R.TOTAL % R.EVERY == 0


# ----------------------------------------------------------------------------
# reading 1: the evader factory, without the pursuer side
# ----------------------------------------------------------------------------
class TestEvaderFactory:
    def test_it_builds_a_tacticopponent_over_evaderrulesv0(self):
        cfg = EnvConfig(evader_speed_ratio=1.2)
        evader = R.evader_factory_for(cfg)()
        assert isinstance(evader, TacticOpponent)
        assert isinstance(evader.select, EvaderRulesV0)

    def test_the_factory_shares_one_instance_across_calls(self):
        """`SingleAgentWrapper.reset` calls the factory then `.reset()` on
        what it returns every episode, so one shared, resettable instance
        is the correct shape -- `build_game`'s own `lambda: evader`."""
        factory = R.evader_factory_for(EnvConfig(evader_speed_ratio=1.2))
        assert factory() is factory()

    def test_it_never_builds_a_pursuerrulesv0_or_a_tacticlaw(self):
        """The whole point of reading 1: the pursuer side's classes are not
        even imported into this module, so nothing in it can construct one."""
        assert not hasattr(R, "PursuerRulesV0")
        assert not hasattr(R, "TacticLaw")


# ----------------------------------------------------------------------------
# the env and learner the deliverable fixes
# ----------------------------------------------------------------------------
class TestEnvConstruction:
    def test_make_env_matches_the_deliverable(self):
        env = R.make_env("STRUCT-80")
        assert isinstance(env, SearchMDP)
        assert env.hold == HOLD == 10
        assert (env.shield_obs, env.escape_obs) == (True, True)
        assert env.r_acq == 0.0
        assert env.zero_reward is False
        assert (env.stall_penalty, env.intervention_penalty) == (0.0, 0.0)
        assert env._driver is env.inner  # reward_wrapper=None
        assert isinstance(env.law, InterceptLaw)
        assert type(env.shield).__name__ == "RecoveryShieldV2"
        assert env.observation_space.shape == (R.OBS_LEN,)
        assert env.cfg.evader_speed_ratio == 1.2

    def test_the_training_stream_is_seed_times_1000(self):
        assert R.make_env("STRUCT-80").inner.seed0 == 80_000
        assert R.make_env("STRUCT-81").inner.seed0 == 81_000

    def test_make_gate_env_has_no_shield_baked_in(self):
        """Reading 4/9: paired evaluation needs a FRESH shield per episode,
        assigned by the caller -- not one shared across the whole call."""
        env = R.make_gate_env()
        assert env.shield is None
        assert (env.shield_obs, env.escape_obs) == (True, True)
        assert (env.hold, env.r_acq) == (HOLD, 0.0)
        assert isinstance(env.law, InterceptLaw)

    def test_make_learner_is_ppo_at_sb3_defaults_apart_from_gamma(self):
        from stable_baselines3 import PPO

        env = R.make_env("STRUCT-80")
        model = R.make_learner("STRUCT-80", env, "cpu")
        assert isinstance(model, PPO)
        assert model.gamma == 0.99
        assert model.observation_space.shape == (R.OBS_LEN,)
        assert str(model.action_space) == str(env.action_space)


# ----------------------------------------------------------------------------
# section 5: CG-F1 (STRUCT half)
# ----------------------------------------------------------------------------
class TestCGF1:
    def test_it_passes_on_the_real_construction(self):
        cg = R.assert_cg_f1(R.make_env("STRUCT-80"))
        assert cg["ok"] is True
        assert cg["config_diff"] == []
        assert cg["shield"] == {
            "class": "RecoveryShieldV2",
            "margin": 0.0,
            "gamma": 0.2,
            "recovery_after": RECOVERY_AFTER,
        }
        assert cg["opponent"] == {
            "class": "TacticOpponent",
            "rules_class": "EvaderRulesV0",
        }

    def test_a_config_mismatch_aborts_the_launch(self):
        env = R.make_env("STRUCT-80")
        env.cfg = EnvConfig(evader_speed_ratio=0.8)
        with pytest.raises(SystemExit, match="CG-F1 FAILED"):
            R.assert_cg_f1(env)

    def test_a_mismatched_shield_aborts_the_launch(self):
        env = R.make_env("STRUCT-80")
        env.shield = CBFShield(env.cfg)  # no recovery -- the wrong shield class
        with pytest.raises(SystemExit, match="CG-F1 FAILED"):
            R.assert_cg_f1(env)

    def test_the_check_is_recorded_not_just_asserted(self):
        """The brief: 'record the check in the arm JSON' -- `assert_cg_f1`
        returns the dict `train` stores under `cg_f1`, it does not just
        raise-or-pass silently."""
        cg = R.assert_cg_f1(R.make_env("STRUCT-81"))
        assert set(cg) >= {"gate", "prereg", "config_diff", "shield", "opponent", "ok"}
        assert cg["gate"] == "CG-F1"


# ----------------------------------------------------------------------------
# section 6 (by analogy with F.1/S17): artefacts and resumability
# ----------------------------------------------------------------------------
class TestGuards:
    def test_artefact_paths_all_carry_the_prefix(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        assert R._arm_file("STRUCT-80") == tmp_path / "s24_struct_arm_STRUCT-80.json"
        assert R._probe_file("STRUCT-81") == tmp_path / "s24_struct_probe_STRUCT-81.json"
        assert R._ckpt_dir("STRUCT-82") == tmp_path / "ckpts_S24-STRUCT-82"

    def test_an_arm_with_a_verdict_is_skipped(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(R, "OUT", tmp_path)
        (tmp_path / "s24_struct_arm_STRUCT-80.json").write_text(
            json.dumps({"verdict": "MEASURED"})
        )
        R.run(["STRUCT-80"], "cpu")
        assert "skipping" in capsys.readouterr().out

    def test_an_arm_record_with_no_verdict_is_not_skipped(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        (tmp_path / "s24_struct_arm_STRUCT-80.json").write_text(json.dumps({"partial": True}))
        called = []
        monkeypatch.setattr(R, "train", lambda arm, device: called.append(arm) or {"verdict": "MEASURED"})
        R.run(["STRUCT-80"], "cpu")
        assert called == ["STRUCT-80"]


class TestResume:
    def test_resume_from_picks_the_newest_checkpoint_and_matching_rows(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(R, "OUT", tmp_path)
        ckpt_dir = R._ckpt_dir("STRUCT-80")
        ckpt_dir.mkdir(parents=True)
        for mark in (250_000, 500_000):
            (ckpt_dir / f"S24-STRUCT-80_{mark}.zip").write_text("x")
        rows = [{"mark": 250_000}, {"mark": 500_000}, {"mark": 750_000}]
        R._probe_file("STRUCT-80").write_text(
            json.dumps({"arm": "STRUCT-80", "checkpoints": rows})
        )
        ckpt, offset, kept = R._resume_from("STRUCT-80")
        assert offset == 500_000
        assert ckpt == ckpt_dir / "S24-STRUCT-80_500000.zip"
        assert kept == rows[:2]

    def test_no_resume_once_the_final_mark_exists(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        ckpt_dir = R._ckpt_dir("STRUCT-80")
        ckpt_dir.mkdir(parents=True)
        (ckpt_dir / f"S24-STRUCT-80_{R.TOTAL}.zip").write_text("x")
        ckpt, offset, rows = R._resume_from("STRUCT-80")
        assert ckpt is None and offset == 0 and rows == []

    def test_no_checkpoints_means_no_resume(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "OUT", tmp_path)
        ckpt, offset, rows = R._resume_from("STRUCT-80")
        assert ckpt is None and offset == 0 and rows == []


# ----------------------------------------------------------------------------
# crash vs. launch-blocking abort
# ----------------------------------------------------------------------------
class TestCrashHandling:
    def test_a_crash_during_training_becomes_undetermined_with_history(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(R, "OUT", tmp_path)
        R._probe_file("STRUCT-80").parent.mkdir(parents=True, exist_ok=True)
        R._write_json(
            R._probe_file("STRUCT-80"),
            {"arm": "STRUCT-80", "checkpoints": [{"mark": 250_000}]},
        )

        def _boom(arm, device):
            raise RuntimeError("synthetic crash")

        monkeypatch.setattr(R, "train", _boom)
        R.run(["STRUCT-80"], "cpu")
        record = json.loads(R._arm_file("STRUCT-80").read_text())
        assert record["verdict"] == "UNDETERMINED"
        assert "synthetic crash" in record["reason"]
        assert record["checkpoints"] == [{"mark": 250_000}]  # history preserved

    def test_a_systemexit_aborts_rather_than_becoming_undetermined(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(R, "OUT", tmp_path)

        def _abort(arm, device):
            raise SystemExit("CG-F1 FAILED: synthetic")

        monkeypatch.setattr(R, "train", _abort)
        with pytest.raises(SystemExit):
            R.run(["STRUCT-80"], "cpu")
        assert not R._arm_file("STRUCT-80").exists()


# ----------------------------------------------------------------------------
# the stop rule only holds if training cannot show a number
# ----------------------------------------------------------------------------
def test_the_script_prints_no_capture_rate():
    source = _PATH.read_text()
    for line in source.splitlines():
        if "print(" not in line and 'f"[' not in line:
            continue
        assert "capture_rate" not in line, line
        assert "win_rate" not in line, line


# ----------------------------------------------------------------------------
# provenance
# ----------------------------------------------------------------------------
def test_provenance_fields():
    p = R.provenance()
    assert p["prereg"] == R.PREREG
    assert p["prereg_sha256"] and len(p["prereg_sha256"]) == 64
    assert p["baseline_sha256"] and len(p["baseline_sha256"]) == 64
    assert p["git_commit"] and len(p["git_commit"]) == 40
    assert p["pe2_file"].endswith("pe2/__init__.py")
    assert p["obs_len"] == R.OBS_LEN
    assert p["ratio"] == 1.2
    assert p["omp_num_threads"] is not None
    assert p["label_branch"].startswith("labels: none")
    assert p["learner"] == "STRUCT (SearchMDP + PPO, S17 BASE)"


# ----------------------------------------------------------------------------
# a short PPO smoke -- no real 3 M run
# ----------------------------------------------------------------------------
def test_ppo_smoke_learns_and_writes_the_probe_schema(tmp_path, monkeypatch, capsys):
    from stable_baselines3 import PPO

    monkeypatch.setattr(R, "OUT", tmp_path)
    env = R.make_env("STRUCT-80")
    model = PPO(
        "MlpPolicy", env, seed=80, gamma=R.GAMMA, n_steps=128, batch_size=64, verbose=0
    )
    t0 = time.time()
    model.learn(total_timesteps=500)
    elapsed = time.time() - t0
    assert model.num_timesteps >= 500
    # one decision is at least one env step (the hold loop breaks early at most);
    # no tighter bound holds here, unlike FLAT's, because the law can take over
    # before `hold` env steps elapse.
    assert env.env_steps >= model.num_timesteps

    monkeypatch.setattr(R, "PROBE_N", 3)
    probes = R._Probe("STRUCT-80", [])
    probes.at(model, env.env_steps, R.EVERY, R.train_evidence(model.get_env()))
    row = probes.rows[0]
    for key in (
        "train_violations",
        "opponent",
        "n",
        "k",
        "capture_rate",
        "mean_steps",
        "violations",
        "episodes_with_violations",
    ):
        assert key in row, key
    assert row["n"] == 3
    written = json.loads(R._probe_file("STRUCT-80").read_text())
    assert written["checkpoints"][0]["mark"] == R.EVERY
    assert written["checkpoints"][0]["ckpt"].endswith(f"S24-STRUCT-80_{R.EVERY}.zip")
    with capsys.disabled():
        print(
            f"\n[smoke] 500 PPO decisions = {env.env_steps} env steps in "
            f"{elapsed:.1f}s: {env.env_steps / elapsed:.0f} env steps/s "
            f"({500 / elapsed:.0f} decisions/s)"
        )
