"""The F6–F8 figure script runs on synthetic rounds shaped like the S26 runners' output, and refuses without a round file."""

import importlib.util
import json
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _mod():
    spec = importlib.util.spec_from_file_location("figs26", ROOT / "scripts" / "paper_b_figures_s26.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _state(rng, kind="stalled", run=40, branch="search", with_grid=True):
    n_adm = int(rng.integers(5, 60))
    return {"seed": 3_000_000, "t": int(rng.integers(50, 900)), "kind": kind, "run_at_sample": run, "run_length": run + int(rng.integers(0, 300)),
            "censored": False, "dwell_beyond_sample": None if rng.random() < 0.2 else int(rng.integers(1, 400)), "branch": branch,
            "v_nom": 1.0, "w_nom": float(rng.uniform(-2.8, 2.8)), "tier": "exact", "u_shield": [0.0, 0.3],
            "qp": {"found": True, "u": [0.0, 0.31], "dist_w": 1e-4, "agree": True, "forward": False},
            "trap_heading": True, "trap_position": False, "rot_admissible": True, "n_admissible_grid": n_adm,
            "admissible_grid": [int(i) for i in rng.choice(41 * 41, n_adm, replace=False)] if with_grid else None,
            "leave": {"p_leave": 0.05, "n": 2000, "predicted_dwell_steps": 200.0}}


def _write_fake_rounds(root: pathlib.Path, rng, b4_included=True):
    R = root / "results" / "derisk"
    (R / "s26_b1_cells").mkdir(parents=True)
    (R / "s26_b2_cells").mkdir(parents=True)
    pols4 = ("BASE-60", "BASE-61", "BASE-62", "RC19")
    cls = {p: {"n": 3, "n_exact": 3, "exact_tier": 1.0, "trap_heading": 0.9, "trap_position": 0.05, "pair_trap": 0.85, "rot_admissible": 0.8, "qp_agree": 1.0,
               "qp_forward": 0.0, "n_self_stop": 2, "law_branch_share": 0.1} for p in pols4}
    detail = {p: {"deadlock_det": 0.3, "deadlock_stoch": 0.2, "n_onsets": 3, "median_observed_beyond_onset": 150.0, "median_predicted": 200.0,
                  "predicted_infinite_share": 0.0, "censored_share": 0.2, "law_branch_onsets": 0} for p in pols4}
    (R / "s26_b1_round.json").write_text(json.dumps({"classification": cls, "gates": {"CG-B1-2": {"verdict": "PASS"}},
                                                     "hypotheses": {"H-B1-4": {"verdict": "PASS", "detail": detail}}}))
    for p in pols4:
        for mode in ("det", "stoch"):
            states = [_state(rng) for _ in range(3)] + [_state(rng, kind="self_stop")] + [_state(rng, branch="law")]
            (R / "s26_b1_cells" / f"{p}_{mode}.json").write_text(json.dumps({"name": f"{p}_{mode}", "policy": p, "mode": mode, "states": states}))
    pols5 = (*pols4[:3], "STR14", "RC19")
    sets = ["PLAIN", "RECF-5", "RECF-10", "RECF-20", "RECF-40", "RECF-80", "PERT-20", "PRED-PLAIN", "PRED-RECF-20"]
    for p in pols5:
        for o in ("ATK4-45", "GREEDY"):
            for s in sets:
                if s.startswith("PRED") and p == "STR14":
                    continue
                esc = [{"t": None if rng.random() < 0.05 else int(rng.integers(1, 120)), "remaining": int(rng.integers(0, 1200))} for _ in range(int(rng.integers(0, 40)))]
                rec = {"name": f"{p}_{o}_{s}", "policy": p, "opponent": o, "setting": s, "n_episodes": 500, "capture_rate": float(rng.uniform(0.2, 0.8)),
                       "deadlock_episode_rate": float(rng.uniform(0, 0.3)), "immobile_step_rate": float(rng.uniform(0, 0.5)), "recoveries": int(rng.integers(0, 5000)),
                       "n_engagements": int(rng.integers(1, 800)), "n_cycles": int(rng.integers(0, 200)), "escape_times": esc}
                (R / "s26_b2_cells" / f"{rec['name']}.json").write_text(json.dumps(rec))
    hyps = {h: {"verdict": "PASS"} for h in ("H-B2-1", "H-B2-2", "H-B2-3", "H-B2-4", "H-B2-5", "H-B3-1", "H-B4-1")}
    hyps["H-B2-2"] |= {"recf20_within_bound": {}, "vacuous_cells": []}
    (R / "s26_b2_round.json").write_text(json.dumps({"hypotheses": hyps, "gates": {"CG-B4-1": {"verdict": "PASS" if b4_included else "FAIL"}}, "b4_included": b4_included}))
    return R


@pytest.mark.parametrize("b4_included", [True, False])
def test_figures_render_from_synthetic_rounds(tmp_path, monkeypatch, b4_included):
    m = _mod()
    R = _write_fake_rounds(tmp_path, np.random.default_rng(0), b4_included)
    monkeypatch.setattr(m, "R", R)
    monkeypatch.setattr(m, "OUT", tmp_path / "figs")
    m.main()
    for f in ("F6_geometry.pdf", "F6_geometry.png", "F7_recovery_sweep.pdf", "F8_predictive.pdf", "figure_data_s26.json"):
        assert (tmp_path / "figs" / f).exists(), f
    data = json.loads((tmp_path / "figs" / "figure_data_s26.json").read_text())
    assert data["F8"]["settings"] == (["PLAIN", "RECF-20", "PRED-PLAIN", "PRED-RECF-20"] if b4_included else ["PLAIN", "RECF-20", "PRED-PLAIN"])
    assert len(data["F7"]["BASE-61"]["deadlock_ATK"]) == 6
    assert data["F6a"]["trap_heading"] is True
    assert data["F7"]["escape"]["n_decided"] > 0


def test_refuses_without_round_file(tmp_path, monkeypatch):
    m = _mod()
    monkeypatch.setattr(m, "R", tmp_path)
    with pytest.raises(SystemExit, match="REFUSED"):
        m.main()
