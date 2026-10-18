"""E.4 — the meta-game LP over synthetic matrices and the refusal path."""
import importlib.util
import pathlib

import numpy as np
import pytest

_REPO = pathlib.Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("s24_2_meta_game", _REPO / "scripts/s24_2_meta_game.py")
MG = importlib.util.module_from_spec(spec)
spec.loader.exec_module(MG)


def test_pure_saddle_point_gives_a_pure_equilibrium_with_zero_exploitability():
    # row 1 dominates; column 0 is the evader's best reply to it -> saddle at (1, 0)
    m = np.array([[0.2, 0.3, 0.4, 0.1], [0.5, 0.6, 0.7, 0.55], [0.1, 0.2, 0.3, 0.05], [0.4, 0.5, 0.6, 0.45]])
    g = MG.solve(m)
    assert g["pure_saddle"] and g["row_support"] == ["LEAD-1.0"] and g["col_support"] == ["GREEDY"]
    assert g["value"] == pytest.approx(0.5)
    assert g["exploitability_pursuer"]["LEAD-1.0"] == pytest.approx(0.0)
    assert g["exploitability_evader"]["GREEDY"] == pytest.approx(0.0)
    assert all(v >= -1e-9 for v in g["exploitability_pursuer"].values())
    assert all(v >= -1e-9 for v in g["exploitability_evader"].values())


def test_cyclic_matrix_gives_mixed_support_and_a_positive_exploitability_for_every_pure_policy():
    # matching pennies embedded in 4x4: no pure saddle
    m = np.array([[0.9, 0.1, 0.5, 0.5], [0.1, 0.9, 0.5, 0.5], [0.3, 0.3, 0.3, 0.3], [0.2, 0.2, 0.2, 0.2]])
    g = MG.solve(m)
    assert not g["pure_saddle"] and len(g["row_support"]) >= 2 and len(g["col_support"]) >= 2
    assert g["value"] == pytest.approx(0.5, abs=1e-6)
    assert g["duality_gap"] < 1e-6
    assert all(v > 0.0 for v in g["exploitability_pursuer"].values())


def test_refuses_on_a_missing_record(tmp_path, monkeypatch):
    monkeypatch.setattr(MG.S1, "cells_dir", lambda: tmp_path)
    with pytest.raises(SystemExit, match="REFUSED: missing record"):
        MG.capture_matrix("SEL3-80")
