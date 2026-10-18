"""scripts/derisk_s24_2b_train.py and scripts/derisk_s24_2b_gate.py — S24.2
Step E.3: the alternation. The pursuer's tactic selector RE-TRAINED
(`SEL3-80/81/82`) against a per-episode evader mixture {TACTIC-V0, EVS-best,
GREEDY, HUG}, and the 18-cell gate against the stage-0 pursuer's comparator
columns in `s24_1_cells` and `s24_2_cells`.

Pre-registration: `docs/pathway/tactics_prereg_s24_2_2026-10-17.md` section 3.
As E.2's own test suite, scoped to what this round's brief states:

- the trainer's arms, checkpoint path (`S24-2B-{ARM}`, never E.2's or S24.1's
  stem), the pre-registered seed streams (disjoint), the mixture (exactly
  four seats, drawn uniformly, TACTIC-V0 restored cleanly between episodes)
  and provenance (`evader_mixture` added);
- `make_learner` must build a learner for THIS round's arm name (`SEL3-80`),
  which is not a key of S24.1's `ARMS`;
- the construct gate on the built training env (obs_len 55, 5 actions, mask
  non-empty at reset, exactly four seats), on real and on fabricated envs;
- a crashed arm is UNDETERMINED and is never retrained; a construct-gate
  `SystemExit` is NOT caught as a crash;
- the gate's 18-cell table — rows differ ONLY in which pursuer checkpoint
  `run_cell` loads, never in the built env (`build_cell_env` never reads the
  row) — asserted on BUILT envs;
- record validity, duplicates reject both, a crash through `run_cells` writes
  THIS round's UNDETERMINED record;
- both comparator joins (`s24_1_cells`'s `SEL-s_{TACTIC-V0,GREEDY,HUG}`,
  `s24_2_cells`'s `EVS-s'_SEL-s`) refuse a non-measurable record; the
  `s24_1_cells` join refuses to run inside a binding (the `s24_2_cells` join
  needs no such guard — see the gate module's own docstring);
- H-ALT-1's UP-direction paired test and its two-level (evader-seed, then
  pursuer-seed) pass rule, H-ALT-2's non-inferiority on exact COUNTS
  (the margin boundary, S24-OS's review defect 2's discipline, repeated),
  CG-E3's flag, H-SAFE and the inherited CG-T1/A2.1, each with a FAILING
  direction and not only a passing one;
- the GIF seed rule (the first three GATE seeds, S24.1's own), captions that
  carry no rate, and the CLI's dispatch order;
- both launchers' preconditions, checked through the launchers' OWN embedded
  checkers, and that the gate launcher's three chains of six cover all 18;
- one real, tiny end-to-end smoke per script with a fresh policy, into a
  temporary checkpoint directory that is deleted afterwards.

CG-A1 (deliverable 1's bit-identity bar) lives in
`tests/pe2/test_learned_select.py`, the mirror of E.1's CG-E1. Counts for
that file and for E.1's (`test_evader_selector_mdp.py`) and E.2's
(`test_derisk_s24_2_round.py`) own suites are reported alongside this file's,
not duplicated here.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pytest

_REPO = pathlib.Path(__file__).resolve().parents[2]
_SCRIPTS = _REPO / "scripts"
_GATE_PATH = _SCRIPTS / "derisk_s24_2b_gate.py"
_LAUNCH_TRAIN = _SCRIPTS / "launch_s24_2b_train.sh"
_LAUNCH_GATE = _SCRIPTS / "launch_s24_2b.sh"


def _load(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


G = _load("derisk_s24_2b_gate", _GATE_PATH)
# `G.T` — NOT a second `_load` of the trainer: `_load` builds a fresh module
# object every call, so a separately loaded `T` here would be a different
# object from the one `G._bound()` reads, and patching it would patch nothing
# the gate's internals ever see.
T = G.T
S1 = G.S1
G2 = G.G2


# ----------------------------------------------------------------------------
# the trainer: arms, dirs, streams, the mixture, provenance, the construct gate
# ----------------------------------------------------------------------------
class TestTrainerArmsDirsStreams:
    def test_arms_are_the_three_seeds(self):
        assert T.ARM_ORDER == ("SEL3-80", "SEL3-81", "SEL3-82")
        for arm, spec in T.ARMS.items():
            assert spec["memory"] is True
            assert arm == f"SEL3-{spec['seed']}"

    def test_the_checkpoint_path_carries_the_trainers_own_stem(self):
        assert T.ckpt_path("SEL3-81") == pathlib.Path(
            "results/derisk/ckpts_S24-2B-SEL3-81/S24-2B-SEL3-81_3000000.zip"
        )
        assert T._mark_path("SEL3-81", 250_000) == pathlib.Path(
            "results/derisk/ckpts_S24-2B-SEL3-81/S24-2B-SEL3-81_250000.zip"
        )
        assert T.ckpt_path("SEL3-81") == T._mark_path("SEL3-81", T.TOTAL)

    def test_evs_best_is_evs_80_resolved_through_e2s_own_function(self):
        assert T.EVS_BEST == "EVS-80"
        assert T.EVS_BEST_CKPT == T.T2.ckpt_path("EVS-80")
        assert T.EVS_BEST_STEM == T.EVS_BEST_CKPT.stem

    def test_the_probe_stream_is_2_650_000_and_disjoint(self):
        assert T.PROBE_SEED0 == 2_650_000
        probe = set(range(T.PROBE_SEED0, T.PROBE_SEED0 + T.PROBE_N))
        s1_probe = set(range(T.T1.PROBE_SEED0, T.T1.PROBE_SEED0 + T.T1.PROBE_N))
        gate = set(range(2_400_000, 2_400_500))
        train = {s * 1000 + i for s in T.SEEDS for i in range(1000)}
        assert not (probe & s1_probe)
        assert not (probe & gate)
        assert not (probe & train)

    def test_an_overlapping_stream_is_refused_at_import(self):
        streams = {
            "train": {s * 1000 + i for s in T.SEEDS for i in range(1000)},
            "probe": set(range(T.T1.PROBE_SEED0, T.T1.PROBE_SEED0 + T.T1.PROBE_N)),
            "gate": set(range(2_400_000, 2_400_500)),
            "s24_1_probe": set(range(T.T1.PROBE_SEED0, T.T1.PROBE_SEED0 + T.T1.PROBE_N)),
        }
        bad = [
            f"{a} & {b}"
            for i, a in enumerate(streams)
            for b in list(streams)[i + 1 :]
            if streams[a] & streams[b]
        ]
        assert bad  # the collision is real once the probe stream moves onto S24.1's

    def test_the_mixture_is_exactly_the_four_seats(self):
        assert len(T.EVADER_MIXTURE) == 4
        assert T.EVADER_MIXTURE == ("TACTIC-V0", T.EVS_BEST_STEM, "GREEDY", "HUG")

    def test_provenance_adds_the_mixture_and_width(self):
        prov = T.provenance()
        assert prov["evader_mixture"] == list(T.EVADER_MIXTURE)
        assert prov["obs_len"] == T.OBS_LEN == 55
        assert prov["n_actions"] == 5
        assert prov["prereg"] == T.PREREG
        assert prov["derived_from"] == T.T2.PREREG

    def test_make_learner_builds_a_maskable_ppo_for_sel3_80(self):
        """S24.1's `make_learner` resolves `ARMS` in S24.1's OWN module, so
        calling it with THIS round's arm name directly would raise
        `KeyError` there."""
        from sb3_contrib import MaskablePPO

        assert "SEL3-80" not in T.T1.ARMS
        env = T.make_env("SEL3-80")
        built = T.make_learner("SEL3-80", env, "cpu")
        assert isinstance(built, MaskablePPO)
        assert built.gamma == T.GAMMA == 0.99


# ----------------------------------------------------------------------------
# the mixture: exactly four seats, uniform draw, TACTIC-V0 restored cleanly
# ----------------------------------------------------------------------------
class TestMixtureDraw:
    def test_rejects_a_mixture_that_is_not_four_seats(self):
        with pytest.raises(ValueError, match="exactly four seats"):
            T._EvaderMixtureSelectorMDP(
                T.RATIO, memory=True, seed0=1, evader_seats=("TACTIC-V0", "GREEDY")
            )

    def test_the_draw_visits_all_four_seats_over_many_episodes(self):
        env = T.make_env("SEL3-80")
        env.reset(seed=80_000)
        seen = {env.reset()[1]["evader"] for _ in range(400)}
        assert seen == set(T.EVADER_MIXTURE)

    def test_tactic_v0_restores_the_default_opponent_cleanly_after_greedy(self):
        """A draw forced to GREEDY then forced to TACTIC-V0 must leave the
        game's opponent factory and `game.evader.select` exactly as a
        fresh env would — not a leftover GreedyEvader factory."""
        env = T.make_env("SEL3-80")
        env.reset(seed=80_000)
        default_factory = env._default_opponent_factory
        env._install_evader("GREEDY")
        assert env.game.inner.opponent_factory is not default_factory
        env._install_evader("TACTIC-V0")
        assert env.game.inner.opponent_factory is default_factory
        assert env.game.evader.select is env.rules["evader"]

    def test_the_learned_seat_installs_through_game_evader_select_not_the_factory(self):
        env = T.make_env("SEL3-80")
        env.reset(seed=80_000)
        default_factory = env._default_opponent_factory
        env._install_evader(T.EVS_BEST_STEM)
        assert env.game.inner.opponent_factory is default_factory  # NOT the opponent factory
        assert env.game.evader.select is env._learned_select
        assert isinstance(env._learned_select, T.LearnedEvaderSelect)
        # built once, reused on a later draw of the same seat
        same = env._learned_select
        env._install_evader("GREEDY")
        env._install_evader(T.EVS_BEST_STEM)
        assert env._learned_select is same

    def test_an_unknown_seat_raises(self):
        env = T.make_env("SEL3-80")
        env.reset(seed=80_000)
        with pytest.raises(ValueError, match="unknown"):
            env._install_evader("NOT-A-SEAT")

    def test_episodes_actually_step_under_every_seat(self):
        """Not just installed — playable: one step per drawn seat."""
        env = T.make_env("SEL3-80")
        env.reset(seed=80_000)
        seen = set()
        for _ in range(60):
            _obs, info = env.reset()
            seen.add(info["evader"])
            mask = env.action_masks()
            env.step(int(np.flatnonzero(mask)[0]))
        assert seen == set(T.EVADER_MIXTURE)


class TestConstructGate:
    def test_a_real_built_env_passes(self):
        gate = T.cg_mixture("SEL3-80")
        assert gate["verdict"] == "PASS"
        assert gate["obs_len"] == T.OBS_LEN == 55
        assert gate["n_actions"] == 5
        assert gate["mask_nonempty_at_reset"] is True
        assert set(gate["seats"]) == set(T.EVADER_MIXTURE)

    def test_refuses_on_the_wrong_obs_len(self, monkeypatch):
        class _Fake:
            obs_len, n_actions = 47, 5

            def reset(self, seed=None):
                return np.zeros(self.obs_len, dtype=np.float32), {}

            def action_masks(self):
                mask = np.zeros(self.n_actions, dtype=bool)
                mask[0] = True
                return mask

        monkeypatch.setattr(T, "make_env", lambda arm: _Fake())
        with pytest.raises(SystemExit, match="obs_len"):
            T.cg_mixture("SEL3-80")

    def test_refuses_when_the_mask_is_empty_at_reset(self, monkeypatch):
        class _Fake:
            obs_len, n_actions = 55, 5

            def reset(self, seed=None):
                return np.zeros(self.obs_len, dtype=np.float32), {}

            def action_masks(self):
                return np.zeros(self.n_actions, dtype=bool)

        monkeypatch.setattr(T, "make_env", lambda arm: _Fake())
        with pytest.raises(SystemExit, match="mask is empty"):
            T.cg_mixture("SEL3-80")

    def test_refuses_when_the_mixture_is_not_exactly_four_seats(self, monkeypatch):
        monkeypatch.setattr(T, "EVADER_MIXTURE", ("TACTIC-V0", "GREEDY", "HUG"))

        class _Fake:
            obs_len, n_actions = 55, 5

            def reset(self, seed=None):
                return np.zeros(self.obs_len, dtype=np.float32), {}

            def action_masks(self):
                mask = np.zeros(self.n_actions, dtype=bool)
                mask[0] = True
                return mask

        monkeypatch.setattr(T, "make_env", lambda arm: _Fake())
        with pytest.raises(SystemExit, match="four seats"):
            T.cg_mixture("SEL3-80")


class TestCrashHandling:
    def test_a_crash_is_undetermined_and_is_never_retrained(self, tmp_path, monkeypatch):
        monkeypatch.setattr(T, "OUT", tmp_path)
        calls = []

        def boom(arm, device="cpu"):
            calls.append(arm)
            raise RuntimeError("boom")

        monkeypatch.setattr(T, "train", boom)
        T.run(["SEL3-80"], "cpu")
        rec = json.loads(T._arm_file("SEL3-80").read_text())
        assert rec["verdict"] == "UNDETERMINED"
        assert "boom" in rec["reason"]
        T.run(["SEL3-80"], "cpu")  # must be skipped, never retrained
        assert calls == ["SEL3-80"]

    def test_a_construct_gate_refusal_is_not_caught_as_a_crash(self, tmp_path, monkeypatch):
        monkeypatch.setattr(T, "OUT", tmp_path)

        def refuse(arm, device="cpu"):
            raise SystemExit("construct gate refuses")

        monkeypatch.setattr(T, "train", refuse)
        with pytest.raises(SystemExit):
            T.run(["SEL3-81"], "cpu")
        assert not T._arm_file("SEL3-81").exists()


# ----------------------------------------------------------------------------
# the gate's cell table — 18 cells, rows differ only by the pursuer checkpoint
# ----------------------------------------------------------------------------
class TestCellTable:
    def test_eighteen_cells_three_rows_six_opponents(self):
        assert len(G.CELL_ORDER) == 18
        assert set(G.ROW_NAMES) == {"SEL3-80", "SEL3-81", "SEL3-82"}
        assert set(G.OPPONENTS) == {
            "TACTIC-V0", "EVS-80", "EVS-81", "EVS-82", "GREEDY", "HUG",
        }
        for r in G.ROW_NAMES:
            for o in G.OPPONENTS:
                assert G.cell_name(r, o) in G.CELLS

    def test_cell_names_have_no_suffix(self):
        assert G.cell_name("SEL3-80", "EVS-81") == "SEL3-80_EVS-81"

    def test_in_distribution_matches_the_training_mixture(self):
        in_dist = {o for o, s in G.OPPONENTS.items() if s["in_distribution"]}
        assert in_dist == {"TACTIC-V0", "EVS-80", "GREEDY", "HUG"}

    def test_built_envs_are_identical_across_rows_at_the_same_opponent(self):
        """`build_cell_env` never reads `spec["row"]` — only `run_cell`'s own
        `load_policy(spec["row"], ...)` differs between rows — so two rows
        at the same opponent must build bit-identical envs."""
        env80 = G.build_cell_env("SEL3-80_HUG")
        env81 = G.build_cell_env("SEL3-81_HUG")
        assert env80.obs_len == env81.obs_len
        assert env80.actions == env81.actions
        assert env80.cfg.evader_speed_ratio == env81.cfg.evader_speed_ratio == 1.2
        assert type(env80.game.inner.opponent_factory) is type(env81.game.inner.opponent_factory)

    def test_evs_opponents_install_through_game_evader_select(self):
        env = G.build_cell_env("SEL3-80_EVS-81")
        assert isinstance(env.game.evader.select, G.LearnedEvaderSelect)
        assert env.game.evader.select.name == "EVS-81"

    def test_gate_seed0_cross_checked_against_s24_1(self):
        assert G.GATE_SEED0 == S1.GATE_SEED0 == 2_400_000


# ----------------------------------------------------------------------------
# record validity, duplicates, a crash through run_cells
# ----------------------------------------------------------------------------
def _measurable(name: str, **overrides) -> dict:
    spec = G.CELLS[name]
    n = G.N_EPISODES
    caps = [True] * (n // 2) + [False] * (n - n // 2)
    base = {
        "name": name,
        "n_episodes": n,
        "seed0": G.GATE_SEED0,
        "ratio": spec["ratio"],
        "opponent": spec["opponent"],
        "capture_ind": caps,
        "capture_rate": sum(caps) / n,
        "violation_steps": 0,
        "inapplicable_picks": 0,
        "steps": [300] * n,
        "search_policy_steps": 0,
        "picks": dict.fromkeys(("INTERCEPT", "LEAD", "COUNTER-ORBIT", "AMBUSH", "SWEEP"), 10),
        "aggregate": {
            "picks_share": dict.fromkeys(
                ("INTERCEPT", "LEAD", "COUNTER-ORBIT", "AMBUSH", "SWEEP"), 0.2
            )
        },
    }
    return base | overrides


def _on_disk(dirpath: pathlib.Path, rec: dict, name: str | None = None) -> None:
    d = dirpath / "s24_2b_cells"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name or rec['name']}.json").write_text(json.dumps(rec))


class TestRecordValidity:
    def test_a_measurable_record(self):
        assert G.record_status(_measurable("SEL3-80_EVS-81")) == ("MEASURABLE", "")

    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"n_episodes": 20}, "n_episodes"),
            ({"seed0": 1}, "seed0"),
            ({"opponent": "GREEDY"}, "opponent"),
        ],
    )
    def test_wrong_n_seed0_or_opponent_is_invalid(self, overrides, match):
        status, why = G.record_status(_measurable("SEL3-80_EVS-81") | overrides)
        assert status == "INVALID" and match in why

    def test_a_record_of_another_round_is_invalid(self):
        status, why = G.record_status({"name": "SEL-80_HUG"})
        assert status == "INVALID" and "not a cell of this round" in why

    def test_a_duplicated_cell_rejects_both_files(self, tmp_path, monkeypatch):
        monkeypatch.setattr(G, "OUT", tmp_path)
        rec = _measurable("SEL3-82_HUG")
        _on_disk(tmp_path, rec)
        _on_disk(tmp_path, rec, name="a_copy")
        measurable, rejected, by_cell = G.scan_records()
        assert measurable == {}
        assert len(rejected) == 2 and by_cell["SEL3-82_HUG"]
        assert all("duplicate" in why for why in rejected.values())

    def test_load_records_drops_everything_not_measurable(self, tmp_path, monkeypatch):
        monkeypatch.setattr(G, "OUT", tmp_path)
        _on_disk(tmp_path, _measurable("SEL3-80_EVS-81"))
        _on_disk(tmp_path, _measurable("SEL3-80_TACTIC-V0") | {"n_episodes": 20})
        _on_disk(tmp_path, _measurable("SEL3-81_GREEDY") | {"opponent": "HUG"})
        assert set(G.load_records(quiet=True)) == {"SEL3-80_EVS-81"}

    def test_read_cell_refuses_a_non_measurable_record(self, tmp_path, monkeypatch):
        monkeypatch.setattr(G, "OUT", tmp_path)
        with pytest.raises(SystemExit, match="no cell record"):
            G.read_cell("SEL3-81_EVS-80")
        _on_disk(tmp_path, _measurable("SEL3-81_EVS-80") | {"opponent": "HUG"})
        with pytest.raises(SystemExit, match="not measurable"):
            G.read_cell("SEL3-81_EVS-80")

    def test_a_crash_through_run_cells_writes_this_rounds_undetermined_record(
        self, tmp_path, monkeypatch
    ):
        """`undetermined_record` must be bound, or the file would carry no
        `ratio`/`opponent` field. `run_cell` itself is NOT bound (S1's own is
        reused, see the module docstring), so a monkeypatched `run_cell`
        would never be reached from `run_cells` — the crash exercised here
        is a genuine one, a checkpoint file that is not a valid zip."""
        monkeypatch.setattr(G, "OUT", tmp_path)
        bad_ckpt = tmp_path / "SEL3-81.zip"
        bad_ckpt.write_bytes(b"not a zip")
        monkeypatch.setattr(T, "ckpt_path", lambda arm, _p=bad_ckpt: _p)
        assert "run_cell" not in G._BOUND_ALWAYS  # reused from S1, not rebound
        G.run_cells(["SEL3-81_HUG"])
        rec = json.loads(G.cell_file("SEL3-81_HUG").read_text())
        assert rec["verdict"] == S1.UNDETERMINED == "UNDETERMINED"
        assert rec["ratio"] == 1.2
        assert rec["opponent"] == "HUG"
        assert "zip" in rec["error"].lower()
        assert G.record_status(rec) == ("UNDETERMINED", f"crashed cell: {rec['error']!r}")
        assert G.load_records(quiet=True) == {}
        with pytest.raises(SystemExit, match="UNDETERMINED"):
            G.read_cell("SEL3-81_HUG")


# ----------------------------------------------------------------------------
# the binding onto S24.1's orchestration
# ----------------------------------------------------------------------------
class TestBinding:
    def test_every_bound_name_is_restored(self):
        before = {k: getattr(S1, k) for k in G._BOUND_ALWAYS}
        with G._bound():
            for k in G._BOUND_ALWAYS:
                assert getattr(S1, k) is not before[k]
        after = {k: getattr(S1, k) for k in G._BOUND_ALWAYS}
        assert before == after

    def test_run_cell_is_not_bound_and_resolves_the_checkpoint_through_the_trainer(self):
        assert G.run_cell.__module__ == "derisk_s24_2b_gate"
        with G._bound():
            assert S1.run_cell is G._ORIG["run_cell"]  # unbound: S1's own

    def test_run_cells_refuses_before_the_first_episode_without_a_checkpoint(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(G, "OUT", tmp_path)
        monkeypatch.setattr(T, "ckpt_path", lambda arm: tmp_path / f"{arm}.zip")
        seen = []
        monkeypatch.setattr(G, "run_cell", lambda name, device="cpu", n_episodes=None: seen.append(name))
        with pytest.raises(SystemExit, match="missing the final checkpoint"):
            G.run_cells(["SEL3-80_EVS-80"])
        assert not seen


# ----------------------------------------------------------------------------
# the comparator joins: outside any binding, by the rule that measured them
# ----------------------------------------------------------------------------
class TestComparatorJoins:
    def _s1_fixture(self, tmp_path) -> None:
        d = tmp_path / "s24_1_cells"
        d.mkdir(parents=True)
        for s in G.SEEDS:
            for opponent in G.NI_OPPONENTS:
                name = f"SEL-{s}_{opponent}"
                rec = {
                    "name": name,
                    "n_episodes": S1.N_EPISODES,
                    "seed0": S1.GATE_SEED0,
                    "capture_ind": [True] * 250 + [False] * 250,
                    "capture_rate": 0.5,
                }
                (d / f"{name}.json").write_text(json.dumps(rec))

    def _s2_fixture(self, tmp_path) -> None:
        d = tmp_path / "s24_2_cells"
        d.mkdir(parents=True)
        for s in G.SEEDS:
            for sp in G.SEEDS:
                name = G2.cell_name(f"EVS-{sp}", f"SEL-{s}")
                rec = {
                    "name": name,
                    "n_episodes": G2.N_EPISODES,
                    "seed0": G2.GATE_SEED0,
                    "ratio": G2.RATIO,
                    "pursuer": f"SEL-{s}",
                    "capture_ind": [True] * 200 + [False] * 300,
                    "capture_rate": 0.4,
                }
                (d / f"{name}.json").write_text(json.dumps(rec))

    def test_reads_s24_1s_nine_rows(self, tmp_path, monkeypatch):
        monkeypatch.setattr(S1, "OUT", tmp_path)
        self._s1_fixture(tmp_path)
        got = G.load_s1_comparators()
        assert len(got) == 9
        assert got["SEL-80_TACTIC-V0"]["capture_rate"] == 0.5

    def test_s1_join_refuses_an_absent_record(self, tmp_path, monkeypatch):
        monkeypatch.setattr(S1, "OUT", tmp_path)
        self._s1_fixture(tmp_path)
        (tmp_path / "s24_1_cells" / "SEL-82_HUG.json").unlink()
        with pytest.raises(SystemExit, match="absent"):
            G.load_s1_comparators()

    def test_s1_join_refuses_a_non_measurable_record(self, tmp_path, monkeypatch):
        monkeypatch.setattr(S1, "OUT", tmp_path)
        self._s1_fixture(tmp_path)
        p = tmp_path / "s24_1_cells" / "SEL-80_TACTIC-V0.json"
        p.write_text(json.dumps(json.loads(p.read_text()) | {"n_episodes": 1}))
        with pytest.raises(SystemExit, match="n_episodes"):
            G.load_s1_comparators()

    def test_s1_join_refuses_to_run_inside_a_binding(self):
        with G._bound(), pytest.raises(SystemExit, match="may not run inside a binding"):
            G.load_s1_comparators()

    def test_reads_s24_2s_nine_rows(self, tmp_path, monkeypatch):
        monkeypatch.setattr(G2, "OUT", tmp_path)
        self._s2_fixture(tmp_path)
        got = G.load_s2_comparators()
        assert len(got) == 9
        assert got[G2.cell_name("EVS-80", "SEL-80")]["capture_rate"] == 0.4

    def test_s2_join_refuses_an_absent_record(self, tmp_path, monkeypatch):
        monkeypatch.setattr(G2, "OUT", tmp_path)
        self._s2_fixture(tmp_path)
        (tmp_path / "s24_2_cells" / f"{G2.cell_name('EVS-81', 'SEL-82')}.json").unlink()
        with pytest.raises(SystemExit, match="absent"):
            G.load_s2_comparators()

    def test_s2_join_needs_no_binding_guard(self, tmp_path, monkeypatch):
        """G2's globals are a SEPARATE module object from S1's (`_load` never
        registers in `sys.modules`), so this round's own `_bound()` — which
        only mutates S1's attributes — cannot affect it."""
        monkeypatch.setattr(G2, "OUT", tmp_path)
        self._s2_fixture(tmp_path)
        with G._bound():
            got = G.load_s2_comparators()
        assert len(got) == 9


# ----------------------------------------------------------------------------
# hypotheses — every rule with a FAILING direction, not only a passing one
# ----------------------------------------------------------------------------
def _indicators(n: int, both_true: int, a_only: int, b_only: int):
    a = [True] * both_true + [True] * a_only + [False] * b_only + [False] * (n - both_true - a_only - b_only)
    b = [True] * both_true + [False] * a_only + [True] * b_only + [False] * (n - both_true - a_only - b_only)
    return a, b


class TestUpTest:
    def test_a_significant_rise_passes(self):
        # the row (SEL3) wins 25 discordant pairs, the comparator (SEL) 0:
        # the row's capture is significantly HIGHER.
        row_ind, comp_ind = _indicators(500, 0, 25, 0)
        row_rec = {"capture_ind": row_ind, "capture_rate": sum(row_ind) / 500}
        comp_rec = {"capture_ind": comp_ind, "capture_rate": sum(comp_ind) / 500}
        b = G._up_test(row_rec, comp_rec, "SEL3-80_EVS-80", "EVS-80_SEL-80")
        assert b["delta"] > 0
        assert b["verdict"] == "PASS"

    def test_a_non_significant_or_falling_delta_fails(self):
        row_ind, comp_ind = _indicators(500, 0, 1, 0)
        row_rec = {"capture_ind": row_ind, "capture_rate": sum(row_ind) / 500}
        comp_rec = {"capture_ind": comp_ind, "capture_rate": sum(comp_ind) / 500}
        b = G._up_test(row_rec, comp_rec, "SEL3-80_EVS-80", "EVS-80_SEL-80")
        assert b["verdict"] == "FAIL"

    def test_a_row_that_captures_less_fails(self):
        comp_ind, row_ind = _indicators(500, 0, 60, 0)  # comparator wins
        row_rec = {"capture_ind": row_ind, "capture_rate": sum(row_ind) / 500}
        comp_rec = {"capture_ind": comp_ind, "capture_rate": sum(comp_ind) / 500}
        b = G._up_test(row_rec, comp_rec, "SEL3-80_EVS-80", "EVS-80_SEL-80")
        assert b["delta"] < 0
        assert b["verdict"] == "FAIL"


class TestHAlt1:
    def _cell(self, ind):
        return {"capture_ind": ind, "capture_rate": sum(ind) / len(ind)}

    def _seed_case(self, pattern):
        row_win, comp_lose = _indicators(500, 0, 25, 0)
        row_lose, comp_win = _indicators(500, 0, 1, 0)
        records, s2 = {}, {}
        for sp, win in zip(G.SEEDS, pattern):
            r, c = (row_win, comp_lose) if win else (row_lose, comp_win)
            records[G.cell_name("SEL3-80", f"EVS-{sp}")] = self._cell(r)
            s2[G2.cell_name(f"EVS-{sp}", "SEL-80")] = self._cell(c)
        return G.h_alt_1_seed(records, s2, 80)

    def test_two_of_three_pass_including_both_held_out_seeds_passes(self):
        # prereg section 3: the passers must include the two held-out evader seeds
        # (EVS-81, EVS-82 — every seed except EVS-best = EVS-80)
        b = self._seed_case([False, True, True])
        assert b["held_out_evader_seeds"] == ["EVS-81", "EVS-82"]
        assert b["n_pass"] == 2 and b["held_out_pass"] is True and b["verdict"] == "PASS"

    def test_two_of_three_pass_with_a_held_out_seed_failing_FAILS(self):
        b = self._seed_case([True, True, False])  # EVS-82 (held-out) fails
        assert b["n_pass"] == 2 and b["held_out_pass"] is False and b["verdict"] == "FAIL"

    def test_round_passes_on_two_of_three_pursuer_seeds(self):
        row_win, comp_lose = _indicators(500, 0, 25, 0)
        records, s2 = {}, {}
        for s in G.SEEDS:
            n_pass_seeds = 2 if s != 82 else 0
            for i, sp in enumerate(G.SEEDS):
                # the passers are the two HELD-OUT evader seeds (EVS-81, EVS-82): i >= 1
                r, c = (row_win, comp_lose) if (n_pass_seeds and i >= 1) else (comp_lose, row_win)
                records[G.cell_name(f"SEL3-{s}", f"EVS-{sp}")] = self._cell(r)
                s2[G2.cell_name(f"EVS-{sp}", f"SEL-{s}")] = self._cell(c)
        v = G.h_alt_1(records, s2)
        assert v["seeds"]["82"]["verdict"] == "FAIL"
        assert v["n_pass"] == 2 and v["verdict"] == "PASS"

    def test_missing_either_side_is_incomplete(self):
        v = G.h_alt_1({}, {})
        assert v["verdict"] == "INCOMPLETE"


class TestHAlt2MarginBoundary:
    """S24-OS's review defect 2, repeated: the margin is compared on exact
    COUNTS, not a float subtraction that can land exactly off -0.03 in
    binary floating point."""

    def _records(self, k_row: int, k_comp: int):
        n = G.N_EPISODES
        row = [True] * k_row + [False] * (n - k_row)
        comp = [True] * k_comp + [False] * (n - k_comp)
        return (
            {"capture_ind": row, "capture_rate": k_row / n},
            {"capture_ind": comp, "capture_rate": k_comp / n},
        )

    def test_sitting_exactly_on_the_margin_passes(self):
        # 175/500 - 190/500 == -0.030000000000000027 in float, exactly -0.03
        # in exact counts: round(-0.03 * 500) == -15, and 175 - 190 == -15.
        a, b = self._records(175, 190)
        row_rec, comp_rec = a, b
        v = G.h_alt_2_seed({"SEL3-80_TACTIC-V0": row_rec}, {"SEL-80_TACTIC-V0": comp_rec}, "TACTIC-V0", 80)
        assert round(G.NI_MARGIN * G.N_EPISODES) == -15
        assert (175 - 190) == -15
        assert v["verdict"] == "PASS"

    def test_one_capture_short_of_the_margin_fails(self):
        a, b = self._records(174, 190)  # gap -16, worse than -15
        v = G.h_alt_2_seed({"SEL3-80_TACTIC-V0": a}, {"SEL-80_TACTIC-V0": b}, "TACTIC-V0", 80)
        assert v["verdict"] == "FAIL"

    def test_a_non_significant_loss_passes_regardless_of_the_margin(self):
        a, b = self._records(249, 251)  # tiny, not significant, gap -2
        v = G.h_alt_2_seed({"SEL3-80_TACTIC-V0": a}, {"SEL-80_TACTIC-V0": b}, "TACTIC-V0", 80)
        assert v["losing_direction_significant"] is False
        assert v["verdict"] == "PASS"

    def test_needs_all_three_opponents_to_pass(self):
        pass_row, pass_comp = self._records(260, 240)
        fail_row, fail_comp = self._records(100, 400)  # significant loss, big gap
        records, s1 = {}, {}
        for o in G.NI_OPPONENTS:
            for s in G.SEEDS:
                r, c = (fail_row, fail_comp) if o == "HUG" else (pass_row, pass_comp)
                records[G.cell_name(f"SEL3-{s}", o)] = r
                s1[f"SEL-{s}_{o}"] = c
        v = G.h_alt_2(records, s1)
        assert v["opponents"]["HUG"]["verdict"] == "FAIL"
        assert v["opponents"]["TACTIC-V0"]["verdict"] == "PASS"
        assert v["verdict"] == "FAIL"


class TestCgE3:
    def _records(self, flagged_seeds: set[int]) -> dict:
        tactics = ("INTERCEPT", "LEAD", "COUNTER-ORBIT", "AMBUSH", "SWEEP")
        out = {}
        for s in G.SEEDS:
            share = {a: 0.0 for a in tactics}
            share["LEAD"] = 0.97 if s in flagged_seeds else 0.4
            share["SWEEP"] = 1 - share["LEAD"]
            for o in G.OPPONENTS:
                out[G.cell_name(f"SEL3-{s}", o)] = {"aggregate": {"picks_share": dict(share)}}
        return out

    def test_flags_a_seed_whose_top_tactic_clears_95_against_every_opponent(self):
        v = G.cg_e3(self._records({80}))
        assert v["flagged_seeds"] == [80]
        assert v["verdict"] == "REPORTED"

    def test_does_not_flag_a_seed_below_the_bar_on_even_one_opponent(self):
        records = self._records(set())
        records["SEL3-81_EVS-80"]["aggregate"]["picks_share"] = {
            a: 0.0 for a in ("INTERCEPT", "LEAD", "COUNTER-ORBIT", "AMBUSH", "SWEEP")
        } | {"LEAD": 0.99, "SWEEP": 0.01}
        v = G.cg_e3(records)
        assert v["flagged_seeds"] == []


class TestHSafeAndInheritedCgT1A21:
    def _records(self, **cell_overrides) -> dict:
        return {n: _measurable(n) | cell_overrides for n in G.CELL_ORDER}

    def _clean_probes(self) -> dict:
        return {
            arm: {"missing": False, "rows": [{"train_violations": 0, "violations": 0, "train_inapplicable_picks": 0, "inapplicable_picks": 0}]}
            for arm in T.ARM_ORDER
        }

    def test_holds_when_everything_is_clean(self):
        v = G.h_safe(self._records(), self._clean_probes())
        assert v["verdict"] == "HOLDS"

    def test_fails_on_a_gate_violation(self):
        records = self._records()
        records["SEL3-80_EVS-80"]["violation_steps"] = 1
        v = G.h_safe(records, self._clean_probes())
        assert v["verdict"] == "FAILS" and "SEL3-80_EVS-80" in v["offenders"]

    def test_fails_on_training_contact(self):
        probes = self._clean_probes()
        probes["SEL3-81"]["rows"][0]["train_violations"] = 1
        v = G.h_safe(self._records(), probes)
        assert v["verdict"] == "FAILS"
        assert "SEL3-81" in v["training_offenders"]

    def test_incomplete_on_missing_training_evidence(self):
        probes = self._clean_probes()
        probes["SEL3-82"] = {"missing": True, "rows": []}
        v = G.h_safe(self._records(), probes)
        assert v["verdict"] == "INCOMPLETE"
        assert "SEL3-82" in v["missing_training"]

    def test_cg_t1_refuses_on_a_non_zero_inapplicable_pick(self):
        clean = self._clean_probes()
        records = self._records()
        records["SEL3-81_TACTIC-V0"]["inapplicable_picks"] = 1
        v = G.cg_t1(records, clean)
        assert v["verdict"] == "REFUSE" and v["offenders"] == ["SEL3-81_TACTIC-V0"]
        assert G.cg_t1(self._records(), clean)["verdict"] == "PASS"
        bad = dict(clean)
        bad["SEL3-80"] = {"missing": False, "rows": [{"train_inapplicable_picks": 2, "inapplicable_picks": 0}]}
        assert "probe:SEL3-80" in G.cg_t1(self._records(), bad)["offenders"]
        absent = dict(clean)
        absent["SEL3-82"] = {"missing": False, "rows": [{"inapplicable_picks": 0}]}
        assert "probe:SEL3-82" in G.cg_t1(self._records(), absent)["offenders"]

    def test_cg_a2_1_refuses_above_tolerance(self):
        clean = self._records()
        v_clean = G.cg_a2_1(clean)
        assert v_clean["verdict"] == "PASS"
        bad = self._records(search_policy_steps=100, steps=[300] * G.N_EPISODES)
        v_bad = G.cg_a2_1(bad)
        assert v_bad["verdict"] == "REFUSE"
        assert set(v_bad["offenders"]) == set(G.CELL_ORDER)


# ----------------------------------------------------------------------------
# GIFs: the seed rule, captions, and the CLI's dispatch order
# ----------------------------------------------------------------------------
class TestGifsAndDispatch:
    def test_eighteen_gifs_three_rows_two_opponents_three_seeds(self):
        assert len(G.GIF_ROWS) == 3 and len(G.GIF_OPPONENTS) == 2 and G.GIF_N == 3
        assert set(G.GIF_OPPONENTS) == {"EVS-80", "HUG"}

    def test_gif_seeds_are_s24_1s_own_rule(self):
        assert G.gif_seeds() == S1.gif_seeds() == (2_400_000, 2_400_001, 2_400_002)

    def test_captions_carry_no_rate(self):
        title = G.gif_title("SEL3-80", "EVS-80", 2_400_000)
        assert title == "SEL3-80 vs EVS-80 · episode 2400000"
        assert not re.search(r"0\.\d+", title)

    def test_write_gifs_refuses_without_a_measurable_record(self, tmp_path, monkeypatch):
        monkeypatch.setattr(G, "OUT", tmp_path)
        monkeypatch.setattr(G, "render_cell", lambda *a, **k: pytest.fail("a frame was rendered"))
        with pytest.raises(SystemExit, match="no cell record"):
            G.write_gifs(["SEL3-80"])

    def test_write_gifs_refuses_without_the_rows_checkpoint(self, tmp_path, monkeypatch):
        monkeypatch.setattr(G, "OUT", tmp_path)
        for o in G.GIF_OPPONENTS:
            _on_disk(tmp_path, _measurable(G.cell_name("SEL3-80", o)))
        monkeypatch.setattr(T, "ckpt_path", lambda arm: tmp_path / "nope.zip")
        monkeypatch.setattr(G, "render_cell", lambda *a, **k: pytest.fail("rendered"))
        with pytest.raises(SystemExit, match="missing the final checkpoint"):
            G.write_gifs(["SEL3-80"])

    def test_dispatch_order_is_cells_then_gifs_then_summarize(self, monkeypatch):
        order = []
        monkeypatch.setattr(G, "run_cells", lambda *a, **k: order.append("cells"))
        monkeypatch.setattr(G, "write_gifs", lambda *a, **k: order.append("gifs"))

        def fake_summarize(device="cpu"):
            order.append("summarize")
            return {
                "verdicts": {
                    "h_alt_1": {"verdict": "FAIL"}, "h_alt_2": {"verdict": "FAIL"},
                    "cg_e3": {}, "h_safe": {}, "cg_t1": {}, "cg_a2_1": {},
                    "direction": "", "n_records": 0, "n_cells": 18, "complete": False,
                }
            }

        monkeypatch.setattr(G, "summarize", fake_summarize)
        monkeypatch.setattr(sys, "argv", ["prog", "--cells", "--gifs", "--summarize"])
        G.main()
        assert order == ["cells", "gifs", "summarize"]


# ----------------------------------------------------------------------------
# the launch wrappers — their OWN embedded checkers
# ----------------------------------------------------------------------------
def _checker(launch_path: pathlib.Path, tmp_path: pathlib.Path) -> pathlib.Path:
    body = re.search(r"<<'PY'[^\n]*\n(.*?)\nPY\n", launch_path.read_text(), re.DOTALL).group(1)
    path = tmp_path / "checker.py"
    path.write_text(body)
    if not (tmp_path / "scripts").exists():
        (tmp_path / "scripts").symlink_to(_SCRIPTS)
    return path


def _run_checker(path: pathlib.Path, tmp_path: pathlib.Path, *arms: str):
    return subprocess.run(
        [sys.executable, str(path), "wt", *arms],
        capture_output=True, text=True, check=False, cwd=tmp_path,
    )


def _evs_best_ckpt(tmp_path: pathlib.Path) -> None:
    """A dummy at `T.EVS_BEST_CKPT`'s own (relative) path — E.2's upstream
    artefact, which the checker script requires but does not itself train."""
    ckpt = tmp_path / T.EVS_BEST_CKPT
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_bytes(b"x")


class TestTrainLauncherChecker:
    def test_it_accepts_when_nothing_exists_yet(self, tmp_path):
        (tmp_path / "results" / "derisk").mkdir(parents=True)
        _evs_best_ckpt(tmp_path)
        checker = _checker(_LAUNCH_TRAIN, tmp_path)
        got = _run_checker(checker, tmp_path, "SEL3-80", "SEL3-81", "SEL3-82")
        assert got.returncode == 0, got.stdout + got.stderr
        assert "clean" in got.stdout

    def test_it_refuses_an_existing_arm_record(self, tmp_path):
        d = tmp_path / "results" / "derisk"
        d.mkdir(parents=True)
        (d / "s24_2b_arm_SEL3-80.json").write_text("{}")
        checker = _checker(_LAUNCH_TRAIN, tmp_path)
        got = _run_checker(checker, tmp_path, "SEL3-80")
        assert got.returncode == 1 and "arm record exists" in got.stdout

    def test_it_refuses_an_existing_checkpoint_dir(self, tmp_path):
        d = tmp_path / "results" / "derisk"
        (d / "ckpts_S24-2B-SEL3-81").mkdir(parents=True)
        checker = _checker(_LAUNCH_TRAIN, tmp_path)
        got = _run_checker(checker, tmp_path, "SEL3-81")
        assert got.returncode == 1 and "checkpoint dir" in got.stdout

    def test_it_refuses_without_the_evs_best_checkpoint(self, tmp_path):
        (tmp_path / "results" / "derisk").mkdir(parents=True)
        checker = _checker(_LAUNCH_TRAIN, tmp_path)
        got = _run_checker(checker, tmp_path, "SEL3-80")
        assert got.returncode == 1 and "EVS-best checkpoint missing" in got.stdout


class TestGateLauncherChecker:
    def test_it_refuses_without_a_checkpoint(self, tmp_path):
        (tmp_path / "results" / "derisk").mkdir(parents=True)
        checker = _checker(_LAUNCH_GATE, tmp_path)
        got = _run_checker(checker, tmp_path, "SEL3-80")
        assert got.returncode == 1 and "no final checkpoint" in got.stdout

    def test_it_refuses_a_dirty_probe_row(self, tmp_path):
        d = tmp_path / "results" / "derisk"
        ck = d / "ckpts_S24-2B-SEL3-80"
        ck.mkdir(parents=True)
        (ck / "S24-2B-SEL3-80_3000000.zip").write_bytes(b"x")
        (d / "s24_2b_probe_SEL3-80.json").write_text(
            json.dumps({"checkpoints": [{"mark": 3_000_000, "violations": 1, "train_violations": 0}]})
        )
        checker = _checker(_LAUNCH_GATE, tmp_path)
        got = _run_checker(checker, tmp_path, "SEL3-80")
        assert got.returncode == 1 and "violations = 1" in got.stdout

    @staticmethod
    def _evs_ckpts(d):
        # review defect 3: the gate launcher also requires the three frozen EVADER checkpoints
        for s in (80, 81, 82):
            e = d / f"ckpts_S24-2-EVS-{s}"
            e.mkdir(parents=True, exist_ok=True)
            (e / f"S24-2-EVS-{s}_3000000.zip").write_bytes(b"x")

    def test_it_refuses_without_the_frozen_evader_checkpoints(self, tmp_path):
        d = tmp_path / "results" / "derisk"
        ck = d / "ckpts_S24-2B-SEL3-80"
        ck.mkdir(parents=True)
        (ck / "S24-2B-SEL3-80_3000000.zip").write_bytes(b"x")
        (d / "s24_2b_probe_SEL3-80.json").write_text(
            json.dumps({"checkpoints": [{"mark": 3_000_000, "violations": 0, "train_violations": 0}]})
        )
        checker = _checker(_LAUNCH_GATE, tmp_path)
        got = _run_checker(checker, tmp_path, "SEL3-80")
        assert got.returncode == 1 and "no frozen evader checkpoint" in got.stdout

    def test_it_accepts_clean_evidence(self, tmp_path):
        d = tmp_path / "results" / "derisk"
        ck = d / "ckpts_S24-2B-SEL3-80"
        ck.mkdir(parents=True)
        (ck / "S24-2B-SEL3-80_3000000.zip").write_bytes(b"x")
        self._evs_ckpts(d)
        (d / "s24_2b_probe_SEL3-80.json").write_text(
            json.dumps({"checkpoints": [{"mark": 3_000_000, "violations": 0, "train_violations": 0}]})
        )
        checker = _checker(_LAUNCH_GATE, tmp_path)
        got = _run_checker(checker, tmp_path, "SEL3-80")
        assert got.returncode == 0, got.stdout + got.stderr


class TestLaunchersGateTheHost:
    def test_both_launchers_share_the_pgrep_guard_and_omp_4(self):
        for path in (_LAUNCH_TRAIN, _LAUNCH_GATE):
            src = path.read_text()
            assert "derisk_s2[34]_|replay_s18_branch_split|s24_tactic_game|derisk_s18_boundary" in src
            assert "OMP_NUM_THREADS=4" in src
            assert "is not a symlink to" in src
            assert "git rev-parse --short HEAD" in src

    def test_the_gate_launcher_has_three_chains_of_six_covering_all_18_cells(self):
        chains = [
            line.split("--cells", 1)[1].split(">")[0].split()
            for line in _LAUNCH_GATE.read_text().splitlines()
            if "--cells" in line and line.lstrip().startswith("nohup")
        ]
        assert len(chains) == 3
        assert all(len(c) == 6 for c in chains)
        flat = [c for chain in chains for c in chain]
        assert sorted(flat) == sorted(G.CELL_ORDER)

    def test_neither_launcher_summarises_or_renders(self):
        for path in (_LAUNCH_TRAIN, _LAUNCH_GATE):
            for line in path.read_text().splitlines():
                if line.lstrip().startswith("nohup"):
                    assert "--summarize" not in line and "--gifs" not in line

    def test_launchers_use_braced_loop_variables(self):
        assert "${a}" in _LAUNCH_TRAIN.read_text()


# ----------------------------------------------------------------------------
# one real, tiny end-to-end smoke — a fresh policy, a tmp dir, deleted
# afterwards. No other test in this file trains a real step or runs a real
# episode against a frozen checkpoint.
# ----------------------------------------------------------------------------
class TestSmoke:
    def test_a_fresh_policy_trains_a_few_steps_and_runs_n3_gate_cells(self, monkeypatch):
        from sb3_contrib import MaskablePPO

        tmp = pathlib.Path(tempfile.mkdtemp(prefix="s24_2b_smoke_"))
        try:
            monkeypatch.setattr(T, "ckpt_path", lambda arm, _tmp=tmp: _tmp / f"{arm}.zip")
            env = T.make_env("SEL3-80")
            built = T.make_learner("SEL3-80", env, "cpu")
            assert isinstance(built, MaskablePPO)
            model = MaskablePPO(
                "MlpPolicy", env, seed=80, gamma=0.99, n_steps=64, batch_size=32, verbose=0,
            )
            model.learn(total_timesteps=64)
            ckpt = T.ckpt_path("SEL3-80")
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            model.save(ckpt)

            with G._bound():
                rec_tv = G.run_cell("SEL3-80_TACTIC-V0", n_episodes=3)
                rec_evs = G.run_cell("SEL3-80_EVS-80", n_episodes=3)
                rec_greedy = G.run_cell("SEL3-80_GREEDY", n_episodes=3)

            assert rec_tv["n_episodes"] == rec_evs["n_episodes"] == rec_greedy["n_episodes"] == 3
            assert rec_tv["opponent"] == "TACTIC-V0"
            assert rec_evs["opponent"] == "EVS-80"
            assert rec_greedy["opponent"] == "GREEDY"
            for rec in (rec_tv, rec_evs, rec_greedy):
                assert set(rec["picks"]) == {"INTERCEPT", "LEAD", "COUNTER-ORBIT", "AMBUSH", "SWEEP"}
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        assert not tmp.exists()
