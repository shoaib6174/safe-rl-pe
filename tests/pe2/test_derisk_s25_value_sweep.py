"""S25 value sweep — the cell table, the two added components, the verdict
rules of `docs/pathway/value_sweep_prereg_s25_2026-11-01.md` section 4, CG-V1's
identity logic, and the launcher's chain coverage. No real 500-episode cell."""
from __future__ import annotations

import importlib.util
import json
import pathlib
import tempfile

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("s25", REPO / "scripts/derisk_s25_value_sweep.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


def _ind(n: int, k: int) -> list[bool]:
    return [True] * k + [False] * (n - k)


def _rec(name: str, ind: list[bool], **extra) -> dict:
    spec_ = M.CELLS[name]
    return {
        "name": name, "row": spec_["row"], "opponent": spec_["opponent"], "setting": spec_["setting"],
        "n_episodes": len(ind), "seed0": M.GATE_SEED0, "capture_ind": ind,
        "capture_rate": sum(ind) / len(ind), "violation_steps": 0, "cg_v3_ok": True,
        "shield": {"class": "NoOpControl" if spec_["margin"] is None else "RecoveryShieldV2"},
    } | extra


def _discordant(n: int, a_only: int, b_only: int, both: int = 0):
    a = [True] * both + [True] * a_only + [False] * b_only + [False] * (n - both - a_only - b_only)
    b = [True] * both + [False] * a_only + [True] * b_only + [False] * (n - both - a_only - b_only)
    return a, b


class TestCellTable:
    def test_eighty_cells_four_by_four_by_five(self):
        assert len(M.CELLS) == 80 == len(set(M.CELL_ORDER))
        for s in M.SETTINGS:
            assert sum(1 for n in M.CELLS if M.CELLS[n]["setting"] == s) == 16
        assert M.SETTINGS["NONE"] is None and M.SETTINGS["d0.00"] == 0.0 and M.SETTINGS["d0.20"] == 0.2

    def test_record_status_validates_declared_fields(self):
        name = M.cell_name("SEL-81", "EVS-80", "d0.10")
        ok = _rec(name, _ind(500, 50))
        assert M.record_status(ok) == ("MEASURABLE", "")
        assert M.record_status(ok | {"n_episodes": 499})[0] == "INVALID"
        assert M.record_status(ok | {"setting": "d0.20"})[0] == "INVALID"
        assert M.record_status(ok | {"verdict": M.UNDETERMINED})[0] == M.UNDETERMINED
        assert M.record_status({"name": "nope"})[0] == "INVALID"


class TestComponents:
    def test_shield_factory_builds_the_declared_filter(self):
        cfg = M.S1.cell_config()
        none = M.shield_factory("NONE")(cfg)
        assert isinstance(none, M.NoOpControl) and none.n_recoveries == 0
        for s, d in (("d0.00", 0.0), ("d0.05", 0.05), ("d0.20", 0.2)):
            sh = M.shield_factory(s)(cfg)
            assert type(sh).__name__ == "RecoveryShieldV2"
            assert sh.margin == pytest.approx(d) and sh.gamma == M.GAMMA and sh.recovery_after == M.RECOVERY_AFTER
        with pytest.raises(ValueError):
            M.shield_factory("d9.99")

    def test_build_cell_env_applies_the_setting_through_begin(self):
        cfg = M.S1.cell_config()
        for setting, cls, margin in (("NONE", "NoOpControl", 0.0), ("d0.20", "RecoveryShieldV2", 0.2)):
            for row in ("FULLOBS", "SEL-81"):
                env = M.build_cell_env(M.cell_name(row, "GREEDY", setting), cfg)
                game = env.game if hasattr(env, "game") else env
                game._begin(M.GATE_SEED0)
                sp = M.shield_spec(game.shield)
                assert sp["class"] == cls and sp["margin"] == pytest.approx(margin)

    def test_fullobs_tactic_is_normalised_and_never_done(self):
        cfg = M.S1.cell_config()
        env = M.build_cell_env(M.cell_name("FULLOBS", "HUG", "d0.00"), cfg)
        env._begin(M.GATE_SEED0)
        assert env.law.select.__name__ == "select"  # `fixed`
        for _ in range(5):
            a = env.law.act(env.inner._last_joint_obs["pursuer"])
            assert a is not None and a.shape == (2,) and float(a[0]) == pytest.approx(1.0) and abs(float(a[1])) <= 1.0
        assert env.law.current.name == "FULLOBS-GREEDY"


class TestVerdicts:
    def _records_for(self, none_k, d0_k, d20_k, d05_k=None, d10_k=None):
        recs = {}
        for r in M.PURSUERS:
            a, b = _discordant(500, none_k, 0, both=d0_k)  # NONE wins none_k more, same base
            recs[M.cell_name(r, "EVS-80", "NONE")] = _rec(M.cell_name(r, "EVS-80", "NONE"), a)
            recs[M.cell_name(r, "EVS-80", "d0.00")] = _rec(M.cell_name(r, "EVS-80", "d0.00"), b)
            lo, _ = _discordant(500, 0, d20_k, both=max(d0_k - d20_k, 0))
            recs[M.cell_name(r, "EVS-80", "d0.20")] = _rec(M.cell_name(r, "EVS-80", "d0.20"), lo)
            for s, k in (("d0.05", d05_k), ("d0.10", d10_k)):
                ind = b if k is None else _ind(500, k)
                recs[M.cell_name(r, "EVS-80", s)] = _rec(M.cell_name(r, "EVS-80", s), ind)
        return recs

    def test_h_val_1_passes_only_when_all_four_rows_rise_significantly(self):
        recs = self._records_for(none_k=40, d0_k=60, d20_k=20)
        assert M.h_val_1(recs)["verdict"] == "PASS"
        # one row flat -> FAIL with 3/4
        r = M.PURSUERS[0]
        recs[M.cell_name(r, "EVS-80", "NONE")] = recs[M.cell_name(r, "EVS-80", "d0.00")]
        v = M.h_val_1(recs)
        assert v["verdict"] == "FAIL" and v["n_pass"] == 3

    def test_h_val_2_requires_three_down_and_no_up(self):
        recs = self._records_for(none_k=40, d0_k=60, d20_k=25)
        assert M.h_val_2(recs)["verdict"] == "PASS"
        # an UP-significant row at d0.10 refutes it
        r = M.PURSUERS[1]
        up, _ = _discordant(500, 30, 0, both=60)
        recs[M.cell_name(r, "EVS-80", "d0.10")] = _rec(M.cell_name(r, "EVS-80", "d0.10"), up)
        v = M.h_val_2(recs)
        assert v["verdict"] == "FAIL" and v["up_significant"] == [f"{r}@d0.10"]

    def test_h_val_3_threshold_on_exact_counts(self):
        name = M.cell_name("FULLOBS", "EVS-80", "d0.00")
        assert M.h_val_3({name: _rec(name, _ind(500, 124))})["reading"] == "margin-only"
        assert M.h_val_3({name: _rec(name, _ind(500, 125))})["reading"] == "information-dependent"
        assert M.h_val_3({})["verdict"] == "INCOMPLETE"

    def test_h_val_4_solves_the_lp_per_setting_and_reports_order(self):
        recs = {}
        for s, scale in (("NONE", 0.9), ("d0.00", 0.5), ("d0.05", 0.45), ("d0.10", 0.4), ("d0.20", 0.3)):
            for i, p in enumerate(M.PURSUERS):
                for j, e in enumerate(M.EVADERS):
                    k = int(500 * scale * (0.5 + 0.1 * i) * (1.0 if e != "EVS-80" else 0.3))
                    n = M.cell_name(p, e, s)
                    recs[n] = _rec(n, _ind(500, min(k, 500)))
        v = M.h_val_4(recs)
        assert v["pre_stated_order_holds"] is True
        assert all("value" in v["per_setting"][s] for s in M.SETTINGS)
        assert v["evs_in_support_at_every_shielded_setting"] is True

    def test_h_safe_counts_shielded_cells_only(self):
        a = _rec(M.cell_name("SEL-81", "HUG", "d0.00"), _ind(500, 1))
        b = _rec(M.cell_name("SEL-81", "HUG", "NONE"), _ind(500, 1), violation_steps=900)
        v = M.h_safe({a["name"]: a, b["name"]: b})
        assert v["verdict"] == "HOLDS" and v["none_control_contact_steps"][b["name"]] == 900
        a["violation_steps"] = 1
        assert M.h_safe({a["name"]: a})["verdict"] == "FAILS"


class TestCgV1Identity:
    def test_identity_pairs_cover_the_twelve_reproduced_cells(self):
        pairs = M.identity_pairs()
        assert len(pairs) == 12 and all(M.CELLS[n]["setting"] == "d0.00" for n in pairs)
        for _n, (path, _mod) in pairs.items():
            assert path.exists(), path  # the stored S24 records this round must reproduce

    def test_cg_v1_refuses_on_a_single_flipped_episode(self, monkeypatch):
        pairs = M.identity_pairs()
        name, (path, _mod) = next(iter(pairs.items()))
        stored = json.loads(path.read_text())
        same = _rec(name, [bool(x) for x in stored["capture_ind"]])
        monkeypatch.setattr(M, "identity_pairs", lambda: {name: pairs[name]})
        assert M.cg_v1({name: same})["verdict"] == "PASS"
        flipped = dict(same)
        flipped["capture_ind"] = [not same["capture_ind"][0]] + same["capture_ind"][1:]
        assert M.cg_v1({name: flipped})["verdict"] == "REFUSE"
        assert M.cg_v1({})["verdict"] == "INCOMPLETE"

    def test_cg_v1_raises_inside_the_binding(self):
        with M._bound(), pytest.raises(AssertionError):
            M.cg_v1({})


class TestLauncher:
    def test_five_chains_cover_the_eighty_cells_once(self):
        src = (REPO / "scripts/launch_s25.sh").read_text()
        chains = [
            line.split("--cells", 1)[1].split(">")[0].split()
            for line in src.splitlines()
            if "--cells" in line and line.lstrip().startswith("nohup")
        ]
        assert len(chains) == 5
        flat = [c for ch in chains for c in ch]
        assert sorted(flat) == sorted(M.CELL_ORDER)
        assert "derisk_s2[345]_" in src and "OMP_NUM_THREADS=4" in src


class TestSmoke:
    def test_one_episode_per_row_kind_into_a_tmp_dir(self, monkeypatch):
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="s25_test_"))
        monkeypatch.setattr(M, "OUT", tmp)
        # all three row kinds AND both learned checkpoint resolvers (SEL-81 via S24.1, SEL3-80 via S24.2b):
        # the first launch died on a bound-name recursion in the SEL-81 path the smoke had not covered
        names = [M.cell_name("FULLOBS", "GREEDY", "NONE"), M.cell_name("SEL3-80", "TACTIC-V0", "d0.05"), M.cell_name("SEL-81", "HUG", "d0.20")]
        M.run_cells(names, "cpu", n_episodes=1)
        for n in names:
            r = json.loads((tmp / "s25_cells" / f"{n}.json").read_text())
            assert r.get("verdict") is None and r["n_episodes"] == 1 and r["cg_v3_ok"]
            assert r["shield"]["class"] == ("NoOpControl" if M.CELLS[n]["margin"] is None else "RecoveryShieldV2")
            assert M.record_status(r)[0] == "INVALID"  # never measurable at N = 1
        import shutil

        shutil.rmtree(tmp)
