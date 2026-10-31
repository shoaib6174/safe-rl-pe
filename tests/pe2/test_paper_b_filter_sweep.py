"""Paper B rounds B2–B4 runner — cell table, the three filters, the per-episode
recorder's engagement/cycle/escape logic, record validation, the identity gate
CG-B2-1, CG-B4-1, the verdict rules of the prereg (sections 2–4) and the
launcher's coverage. No real 500-episode cell."""
from __future__ import annotations

import importlib.util
import json
import pathlib
import tempfile

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("pbfs", REPO / "scripts/paper_b_filter_sweep.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)

from pe2.config import EnvConfig  # noqa: E402
from pe2.shield import CBFShield, PredictiveShield  # noqa: E402


def _ind(n, k):
    return [True] * k + [False] * (n - k)


def _discordant(n, a_only, b_only, both=0):
    a = [True] * both + [True] * a_only + [False] * b_only + [False] * (n - both - a_only - b_only)
    b = [True] * both + [False] * a_only + [True] * b_only + [False] * (n - both - a_only - b_only)
    return a, b


def _rec(name, cap=None, dl=None, **extra):
    s = M.CELLS[name]
    cap = cap if cap is not None else _ind(500, 100)
    dl = dl if dl is not None else _ind(500, 0)
    return {"name": name, "policy": s["policy"], "opponent": s["opponent"], "setting": s["setting"], "group": s["group"],
            "n_episodes": 500, "seed0": M.GATE_SEED0, "obs_width": M.expected_width(s["policy"]), "capture_ind": cap, "deadlock_ind": dl,
            "capture_rate": sum(cap) / 500, "deadlock_episode_rate": sum(dl) / 500, "immobile_step_rate": 0.05, "violation_steps": 0,
            "recoveries": 100, "escape_times": [{"t": x, "remaining": 1000} for x in (5, 7, 9, 12)], "n_engagements": 4, "n_cycles": 0, "cg_b4": None} | extra


class TestCells:
    def test_eighty_six_cells_in_three_groups(self):
        assert len(M.CELLS) == 86 == len(set(M.CELL_ORDER))
        groups = {g: sum(1 for c in M.CELLS.values() if c["group"] == g) for g in ("B2", "B3", "B4")}
        assert groups == {"B2": 70, "B3": 8, "B4": 8}
        assert M.T_TURN == 45 and M.GATE_SEED0 == 1_740_000 and M.GATE_N == 500

    def test_record_status(self):
        n = M.cell_name("STR14", "GREEDY", "RECF-10")
        ok = _rec(n)
        assert M.record_status(ok) == ("MEASURABLE", "")
        assert M.record_status(ok | {"n_episodes": 499})[0] == "INVALID"
        assert M.record_status(ok | {"obs_width": 14})[0] == "INVALID"  # STR14 is 11-dim
        assert M.record_status(ok | {"setting": "PLAIN"})[0] == "INVALID"
        assert M.record_status({"name": n, "verdict": M.UNDETERMINED})[0] == M.UNDETERMINED


class TestFilters:
    def test_factory_builds_the_declared_filter_per_setting(self):
        cfg = EnvConfig()
        got = {s: M.shield_factory(s, cfg)(1_740_000) for s in M.SETTINGS}
        assert type(got["PLAIN"]) is CBFShield
        for n in M.RECF_NS:
            sh = got[f"RECF-{n}"]
            assert type(sh).__name__ == "RecoveryShieldV2" and sh.recovery_after == n and sh.move_eps is None
        assert type(got["PERT-20"]) is M.PerturbationShield and got["PERT-20"].recovery_after == 20
        assert type(got["PRED-PLAIN"]) is PredictiveShield and got["PRED-PLAIN"].horizon == 4
        pr = got["PRED-RECF-20"]
        assert [c.__name__ for c in type(pr).__mro__[:4]] == ["PredictiveRecoveryShield", "RecoveryShieldV2", "PredictiveShield", "CBFShield"]
        assert pr.horizon == 4 and pr.recovery_after == 20
        with pytest.raises(ValueError):
            M.shield_factory("RECF-X-9", cfg)

    def test_perturbation_shield_executes_only_filtered_commands_and_is_seeded(self):
        cfg = EnvConfig()
        obs = [{"x": 0.0, "y": 0.0, "radius": 0.5}]
        state = np.array([-0.655, 0.0, 0.0])  # body 5 mm from the surface, heading into it: forward is blocked
        plain = CBFShield(cfg)
        outs = []
        for seed in (1, 1, 2):
            sh = M.PerturbationShield(cfg, seed=seed, recovery_after=3)
            seq = []
            for _ in range(8):
                u, info = sh.filter((1.0, 0.0), state, obs, v_max=cfg.v_max)
                seq.append((round(float(u[0]), 6), round(float(u[1]), 6), info["recovery_kind"]))
                # every executed command is a CBFShield output for SOME nominal: re-filtering it is the identity
                u2, _ = plain.filter((float(u[0]), float(u[1])), state, obs, v_max=cfg.v_max)
                assert np.allclose(u2, u, atol=1e-9)
            outs.append(seq)
        assert outs[0] == outs[1]  # same seed, same draws
        assert any(k == "perturbation" for *_u, k in outs[0]) or sh.n_recoveries == 0  # fires when a draw moves
        u_plain, _ = plain.filter((1.0, 0.0), state, obs, v_max=cfg.v_max)  # before recovery_after: the ordinary projection
        assert outs[0][:2] == [(round(float(u_plain[0]), 6), round(float(u_plain[1]), 6), "none")] * 2

    def test_predictive_recovery_composition_fires_through_the_predictive_filter(self):
        """CG-B4-1 at unit level: on a firing step the executed command is one
        the predictive filter returns for the winning candidate, and a fresh
        PredictiveShield accepts it."""
        cfg = EnvConfig()
        obs = [{"x": 0.0, "y": 0.0, "radius": 0.5}]
        state = np.array([-0.655, 0.0, 0.0])
        sh = M.PredictiveRecoveryShield(cfg, recovery_after=2, horizon=4)
        fresh = PredictiveShield(cfg, horizon=4)
        fired = False
        for _ in range(6):
            u, info = sh.filter((1.0, 0.0), state, obs, v_max=cfg.v_max)
            assert info["method"] in ("exact", "backup")  # predictive tiers only: no `relaxed`
            if info["recovery_fired"]:
                fired = True
                assert fresh._accept(np.asarray(u, dtype=np.float64), state, obs)
                cand = {"forward": (cfg.v_max, 0.0), "rotation": (0.0, cfg.omega_max if u[1] > 0 else -cfg.omega_max)}.get(info["recovery_kind"])
                if cand is not None and info["recovery_kind"] == "rotation":
                    u_c, _ = fresh.filter(cand, state, obs, v_max=cfg.v_max)
                    assert np.allclose(u_c, u)
        assert fired and sh.n_recoveries >= 1


class _FakeShield:
    """A scripted filter: `plan[t] = (v_exec, fired, kind)`."""

    cfg = EnvConfig()

    def __init__(self, plan, step=0.02):
        self.plan, self.step, self.n_recoveries = plan, step, 0

    def filter(self, action, state, obstacles, v_max=None):
        v, fired, kind = self.plan.pop(0)
        self.n_recoveries += int(fired)
        return np.array([v, 0.0]), {"v_nom": 1.0, "immobile": v <= 0.05, "stalled": v <= 0.05, "method": "exact", "recovery_fired": fired, "recovery_kind": kind}


class TestEpisodeLog:
    def test_engagements_cycles_and_escape_times(self):
        # 10 immobile steps, then an engagement of 3 firings that moves the body 0.05/step from the 2nd firing,
        # a 3-step gap (same engagement), a firing, then 50 quiet steps, then a new engagement 2 steps later = NOT a cycle,
        # then a quiet gap of 10 (< 40) and a third engagement = a cycle.
        plan = [(0.0, False, "none")] * 10 + [(0.0, True, "forward"), (1.0, True, "forward"), (1.0, True, "forward")] + [(1.0, False, "none")] * 3 \
            + [(0.0, True, "rotation")] + [(0.0, False, "none")] * 50 + [(1.0, True, "forward")] + [(0.0, False, "none")] * 10 + [(1.0, True, "forward")] + [(1.0, False, "none")] * 6
        log = M.EpisodeLog(_FakeShield(list(plan)), keep_trace=True)
        x = 0.0
        for t in range(len(plan)):
            u, _ = log.filter((1.0, 0.0), np.array([x, 0.0, 0.0]), [], v_max=1.0)
            x += 0.05 * float(u[0])
        out = log.finish()
        assert out["n_engagements"] == 3 and out["n_cycles"] == 1
        assert out["kinds"] == {"forward": 5, "policy": 0, "rotation": 1, "perturbation": 0}
        # first engagement: trigger at t = 10 (x = 0); the body is 0.05 away when step 12 starts and 0.10 away when
        # step 13 starts -- the recorder reads the pose at the START of a step, so the escape time is 3
        assert out["escape_times"][0]["t"] == 3 and out["escape_times"][0]["remaining"] == len(plan) - 10
        # the rotation firing (v = 0) plus the 50 quiet immobile steps form the longest stall run: 51
        assert out["longest_stall_run"] == 51 and out["n_steps"] == len(plan) and len(log.trace) == len(plan)

    def test_recorder_is_observation_free(self):
        cfg = EnvConfig()
        obs = [{"x": 0.0, "y": 0.0, "radius": 0.5}]
        state = np.array([-1.5, 0.3, 0.2])
        a, b = CBFShield(cfg), M.EpisodeLog(CBFShield(cfg), keep_trace=False)
        for act in ((1.0, 0.0), (0.3, 2.0), (0.0, -1.0)):
            ua, ia = a.filter(act, state, obs, v_max=1.0)
            ub, ib = b.filter(act, state, obs, v_max=1.0)
            assert np.array_equal(ua, ub) and ia == ib


class TestGatesAndVerdicts:
    def test_identity_pairs_point_at_existing_s17_records(self):
        pairs = M.identity_pairs()
        assert len(pairs) == 12
        for name, (path, opp) in pairs.items():
            assert path.exists() and opp in ("ATK4-45", "GREEDY") and M.CELLS[name]["setting"] in ("PLAIN", "RECF-20")

    def test_cg_b2_1_refuses_on_one_flipped_episode(self):
        pairs = M.identity_pairs()
        name, (path, opp) = next(iter(pairs.items()))
        stored = json.loads(path.read_text())["final"]["gate"][opp]
        same = _rec(name, cap=[bool(x) for x in stored["capture_ind"]], dl=[bool(x) for x in stored["deadlock_ind"]])
        recs = {name: same}
        v = M.cg_b2_1(recs)
        assert v["verdict"] == "INCOMPLETE" and name not in v["missing"]
        flipped = dict(same, deadlock_ind=[not same["deadlock_ind"][0]] + same["deadlock_ind"][1:])
        assert name in M.cg_b2_1({name: flipped})["offenders"]

    def test_cg_b4_1_refuses_on_any_failure(self):
        recs = {}
        for p in M.B34_POLICIES:
            for o in M.OPPONENTS:
                n = M.cell_name(p, o, "PRED-RECF-20")
                recs[n] = _rec(n, cg_b4={"checked": 30, "failures": 0})
        assert M.cg_b4_1(recs)["verdict"] == "PASS"
        n0 = next(iter(recs))
        recs[n0]["cg_b4"] = {"checked": 30, "failures": 1}
        assert M.cg_b4_1(recs)["verdict"] == "REFUSE"

    def _b2_records(self, gain=40, pert_dead=0.0):
        recs = {}
        for p in M.B2_POLICIES:
            base_cap, _ = _discordant(500, 0, 0, both=200)
            for s in M.B2_SETTINGS:
                up, _ = _discordant(500, 0 if s == "PLAIN" else gain, 0, both=200)
                cap = up
                for o in M.OPPONENTS:
                    n = M.cell_name(p, o, s)
                    dl_rate = 0.25 if s == "PLAIN" else (pert_dead if s == "PERT-20" else (0.2 if s == "RECF-80" else 0.0))
                    recs[n] = _rec(n, cap=cap if o == "GREEDY" else _ind(500, 60), dl=_ind(500, int(round(dl_rate * 500))),
                                   recoveries=(5000 if s == "PERT-20" else 1000), immobile_step_rate=0.05,
                                   escape_times=[{"t": x, "remaining": 1000} for x in (5, 8, 12, 20, 30)] + [{"t": None, "remaining": 1000}] if s.startswith("RECF") else [])
        return recs

    def test_h_b2_1_and_h_b2_4_on_synthetic_records(self):
        recs = self._b2_records()
        v = M.h_b2_1(recs)
        assert v["verdict"] == "PASS", v
        v4 = M.h_b2_4(recs)
        assert v4["verdict"] == "FAIL" and all(v4["pert_clears"].values()) and not any(v4["recf_beats_capture"].values())  # equal capture -> not beaten
        # a significant DROP at N = 40 fails H-B2-1
        p = M.B2_POLICIES[0]
        down, _ = _discordant(500, 0, 60, both=140)
        recs[M.cell_name(p, "GREEDY", "RECF-40")] = _rec(M.cell_name(p, "GREEDY", "RECF-40"), cap=down)
        assert M.h_b2_1(recs)["verdict"] == "FAIL"

    def test_h_b2_2_bound_and_h_b2_3_immobile_bar(self):
        recs = self._b2_records()
        v = M.h_b2_2(recs)
        assert v["verdict"] == "FAIL"  # one censored-but-decidable (1000 steps left) of six = 83 % within bound
        for n, r in recs.items():
            if r["setting"] == "RECF-20":
                r["escape_times"][-1] = {"t": None, "remaining": 30}  # A4.3: censored with < 65 steps left is undecidable, not a violation
        assert M.h_b2_2(recs)["verdict"] == "PASS"
        assert M.h_b2_3(recs)["verdict"] == "PASS"
        recs[M.cell_name("RC19", "ATK4-45", "RECF-20")]["immobile_step_rate"] = 0.2
        assert M.h_b2_3(recs)["verdict"] == "FAIL"

    def test_h_b3_1_and_h_b4_1(self):
        recs = {}
        for p in M.B34_POLICIES:
            for o in M.OPPONENTS:
                plain_cap, recf_cap = _discordant(500, 0, 40, both=200)  # RECF wins 40 more
                recs[M.cell_name(p, o, "PRED-PLAIN")] = _rec(M.cell_name(p, o, "PRED-PLAIN"), cap=plain_cap, dl=_ind(500, 60))
                recs[M.cell_name(p, o, "PRED-RECF-20")] = _rec(M.cell_name(p, o, "PRED-RECF-20"), cap=recf_cap, dl=_ind(500, 0))
        assert M.h_b3_1(recs)["verdict"] == "PASS" and M.h_b4_1(recs)["verdict"] == "PASS"
        for p in M.B34_POLICIES[:2]:
            recs[M.cell_name(p, "ATK4-45", "PRED-PLAIN")]["deadlock_episode_rate"] = 0.01
        assert M.h_b3_1(recs)["verdict"] == "FAIL"


class TestLauncher:
    def test_five_sweep_chains_cover_the_86_cells_once_and_the_guard_names_the_runner(self):
        src = (REPO / "scripts/launch_paper_b.sh").read_text()
        chains = [line.split("--cells", 1)[1].split(">")[0].split() for line in src.splitlines()
                  if "--cells" in line and line.lstrip().startswith("nohup") and "filter_sweep" in line]
        assert len(chains) == 5
        flat = [c for ch in chains for c in ch]
        assert sorted(flat) == sorted(M.CELL_ORDER)
        assert "paper_b_" in src.split("pgrep -f")[1].split("\n")[0] and "derisk_s2[345]_" in src


class TestSmoke:
    def test_two_episodes_of_one_cell_per_group_into_a_tmp_dir(self, monkeypatch):
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="pb_sweep_"))
        monkeypatch.setattr(M, "OUT", tmp)
        names = [M.cell_name("STR14", "GREEDY", "PERT-20"), M.cell_name("BASE-60", "GREEDY", "PRED-RECF-20")]
        M.run_cells(names, "cpu", n_episodes=2)
        for n in names:
            r = json.loads((tmp / "s26_b2_cells" / f"{n}.json").read_text())
            assert r["n_episodes"] == 2 and len(r["capture_ind"]) == 2 and r["violation_steps"] == 0
            assert M.record_status(r)[0] == "INVALID"  # never measurable at N = 2
        assert json.loads((tmp / "s26_b2_cells" / f"{names[1]}.json").read_text())["cg_b4"]["failures"] == 0
        import shutil

        shutil.rmtree(tmp)
