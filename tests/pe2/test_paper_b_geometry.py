"""Paper B round B1 runner — cell table, state cutting, the geometry probes on
a constructed pose, the record rules and the verdict functions. One short
real episode as a smoke."""
from __future__ import annotations

import importlib.util
import json
import pathlib
import tempfile

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("pbg", REPO / "scripts/paper_b_geometry.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)

from pe2.config import EnvConfig  # noqa: E402
from pe2.shield import CBFShield  # noqa: E402


def _row(t, immobile, stalled):
    return {"immobile": immobile, "stalled": stalled, "t": t, "pose": [0.0, 0.0, 0.0]}


class TestCells:
    def test_eight_cells_fresh_stream(self):
        assert len(M.CELLS) == 8 and M.SEED0 == 3_000_000 and M.N_EPISODES == 100 and M.K == 40

    def test_cut_states_at_the_40th_and_140th_step_and_self_stops(self):
        rows = [_row(t, False, False) for t in range(10)] + [_row(t, True, True) for t in range(10, 170)] + [_row(t, False, False) for t in range(170, 180)] \
            + [_row(t, True, False) for t in range(180, 230)]
        s = M.cut_states(rows)
        kinds = [(x["kind"], x["t"], x["run_at_sample"], x["run_length"], x["censored"]) for x in s]
        assert kinds == [("stalled", 49, 40, 160, False), ("stalled", 149, 140, 160, False), ("self_stop", 219, 40, 50, True)]
        assert all("dwell_beyond_sample" in x for x in s)


class TestProbes:
    def test_blocked_heading_is_a_pair_trap_with_an_admissible_rotation(self):
        cfg = EnvConfig()
        sh = CBFShield(cfg)
        obs = [{"x": 0.0, "y": 0.0, "radius": 0.5}]
        row = {"pose": [-0.655, 0.0, 0.0], "v_nom": 1.0, "w_nom": 0.0}  # 5 mm from the surface, heading into it
        p = M.probe_state(None, sh, cfg, row, obs, None, seed=0, with_leave=False)
        assert p["trap_heading"] is True and p["trap_position"] is False and p["headings_with_forward"] > 0
        assert p["rot_admissible"] is True and p["tier"] in ("exact", "relaxed", "backup")  # oracle: the filter returns a turn unchanged
        assert p["rot_admissible_exact"] is False  # the exact tier does not (the VCP soft row), which is why both are reported (A3)
        assert p["trap_heading_exact"] is True and p["n_oracle_grid"] >= 1
        assert p["u_shield"][0] <= 0.05  # the filter stops it
        # the linearised rows can be jointly infeasible this close to the surface (the shield is then on a lower
        # tier); either way the exact projection must not yield forward motion
        assert p["qp"]["forward"] is False and (p["qp"]["found"] or p["tier"] != "exact")

    def test_open_space_is_no_trap_and_qp_agrees_with_the_filter(self):
        cfg = EnvConfig()
        sh = CBFShield(cfg)
        row = {"pose": [0.0, 0.0, 0.3], "v_nom": 0.8, "w_nom": 0.5}
        p = M.probe_state(None, sh, cfg, row, [], None, seed=0, with_leave=False)
        assert p["trap_heading"] is False and p["trap_position"] is False and p["tier"] == "exact"
        assert p["qp"]["agree"] is True and p["qp"]["forward"] is True
        assert p["u_shield"] == pytest.approx([0.8, 0.5])

    def test_leave_probability_is_seeded_and_in_range(self):
        cfg = EnvConfig()
        sh = CBFShield(cfg)
        S = M.sweep()
        from stable_baselines3 import PPO

        model = PPO.load(S.POLICIES["BASE-61"]["ckpt"], device="cpu")
        obs = np.zeros(model.observation_space.shape[0], dtype=np.float32)
        a = M.leave_probability(model, sh, np.array([0.0, 0.0, 0.0]), [], obs, seed=7, n=50)
        b = M.leave_probability(model, sh, np.array([0.0, 0.0, 0.0]), [], obs, seed=7, n=50)
        assert a["p_leave"] == b["p_leave"] and 0.0 <= a["p_leave"] <= 1.0 and len(a["sigma"]) == 2
        if a["p_leave"] > 0:
            assert a["predicted_dwell_steps"] == pytest.approx(10 / a["p_leave"])


class TestVerdicts:
    def _records(self, exact=1.0, trap_h=1.0, trap_p=0.0, agree=1.0, dl_det=0.3, dl_stoch=0.25, obs_len=200, pred=200.0):
        recs = {}
        for p in M.POLICIES:
            states = [{"kind": "stalled", "tier": "exact" if i < exact * 10 else "relaxed", "trap_heading": i < trap_h * 10, "trap_position": i < trap_p * 10,
                       "rot_admissible": True, "qp": {"agree": i < agree * 10, "forward": False}, "leave": {"predicted_dwell_steps": pred}, "branch": "search"} for i in range(10)]
            # A2: a law-branch stall with an absurd prediction must not enter clause 2
            states.append({"kind": "stalled", "tier": "exact", "trap_heading": True, "trap_position": False, "rot_admissible": True,
                           "qp": {"agree": True, "forward": False}, "leave": {"predicted_dwell_steps": None}, "branch": "law"})
            recs[M.cell_name(p, "det")] = {"states": states, "deadlock_episode_rate": dl_det, "episodes": [], "violation_steps": 0}
            base = {"tier": "exact", "trap_heading": True, "trap_position": False, "rot_admissible": True, "qp": {"agree": True, "forward": False}}
            stoch_states = [{"kind": "stalled", "run_at_sample": 40, "run_length": 40 + obs_len, "branch": "search", "dwell_beyond_sample": obs_len,
                             "leave": {"predicted_dwell_steps": pred}, **base} for _ in range(5)]
            stoch_states.append({"kind": "stalled", "run_at_sample": 40, "run_length": 1100, "branch": "law", "dwell_beyond_sample": None, "leave": {"predicted_dwell_steps": None}, **base})
            stoch_states.append({"kind": "stalled", "run_at_sample": 140, "run_length": 400, "branch": "search", "dwell_beyond_sample": 1, "leave": {"predicted_dwell_steps": 1.0}, **base})
            recs[M.cell_name(p, "stoch")] = {"states": stoch_states, "deadlock_episode_rate": dl_stoch, "violation_steps": 0,
                                             "episodes": [{"stall_runs": [{"length": 40 + obs_len, "censored": False, "branch": "search"}]}]}
        return recs

    def test_h_b1_1_to_3_pass_and_fail(self):
        cls = M.classification(self._records())
        assert M.h_b1_1(cls)["verdict"] == "PASS" and M.h_b1_2(cls)["verdict"] == "PASS" and M.h_b1_3(cls)["verdict"] == "PASS"
        assert cls["BASE-61"]["law_branch_share"] == pytest.approx(1 / 11)
        cls = M.classification(self._records(exact=0.5))
        assert M.h_b1_1(cls)["verdict"] == "FAIL"
        cls = M.classification(self._records(trap_p=0.5))
        assert M.h_b1_2(cls)["verdict"] == "FAIL"
        cls = M.classification(self._records(agree=0.8))
        assert M.h_b1_3(cls)["verdict"] == "FAIL"

    def test_h_b1_4_factor_two_on_dwell_and_half_on_deadlock(self):
        v = M.h_b1_4(self._records())
        assert v["verdict"] == "PASS" and v["detail"]["BASE-61"]["n_onsets"] == 5  # law onset and the 140th-step sample excluded (A4.1)
        assert M.h_b1_4(self._records(obs_len=50, pred=200.0))["verdict"] == "FAIL"  # observed 50 vs predicted 200
        assert M.h_b1_4(self._records(dl_stoch=0.1))["verdict"] == "FAIL"  # 0.1 < 0.5 × 0.3
        assert M.h_b1_4(self._records(dl_det=0.0, dl_stoch=0.0))["clause1_pass"]["BASE-61"] is True  # A4: a zero deterministic rate passes trivially

    def test_record_status(self):
        n = M.cell_name("BASE-61", "det")
        rec = {"name": n, "policy": "BASE-61", "mode": "det", "n_episodes": 100, "seed0": M.SEED0, "episodes": [{}] * 100, "obs_width": 14}
        assert M.record_status(rec) == ("MEASURABLE", "")
        assert M.record_status(rec | {"obs_width": 11})[0] == "INVALID"
        assert M.record_status(rec | {"n_episodes": 99})[0] == "INVALID"


class TestSmoke:
    def test_one_episode_each_mode_into_a_tmp_dir(self, monkeypatch):
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="pb_geom_"))
        monkeypatch.setattr(M, "OUT", tmp)
        M.run_cells([M.cell_name("BASE-61", "det"), M.cell_name("BASE-61", "stoch")], "cpu", n_episodes=1)
        for mode in M.MODES:
            r = json.loads((tmp / "s26_b1_cells" / f"BASE-61_{mode}.json").read_text())
            assert r["n_episodes"] == 1 and r["violation_steps"] == 0 and r["obs_width"] == 14
            assert M.record_status(r)[0] == "INVALID"
        import shutil

        shutil.rmtree(tmp)
