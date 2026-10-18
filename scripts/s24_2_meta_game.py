"""S24.2 Step E.4 — the tactic meta-game (REPORT-ONLY, prereg section 4 + A2.4).

The 4 x 4 CAPTURE matrix (pursuer rows maximise, evader columns minimise) over
records that already exist — no cell is run here:
  rows  {RULES-V0, LEAD-1.0, SEL-81, SEL3-best}   cols {GREEDY, HUG, TACTIC-V0, EVS-80}
  S24.1 cells   : RULES-V0 / LEAD-1.0 / SEL-81  x  GREEDY / HUG / TACTIC-V0   (s24_1_cells)
  stage-1 cells : EVS-80  x  SEL-81 / RULES-V0 / LEAD-1.0                     (s24_2_cells)
  stage-2 cells : SEL3-best  x  GREEDY / HUG / TACTIC-V0 / EVS-80             (s24_2b_cells)
`SEL3-best` = the stage-2 pursuer seed with the highest mean capture over its
six gate evaders (A2.4's pre-stated rule, applied once, read from the stage-2
round JSON). Every record is validated by ITS OWN round's `record_status`.
The LP is `scripts/derisk_s14_matrix.py::meta_game` (scipy linprog), reused.
Exploitability of a pure pursuer policy i = value - min_j M[i, j]; of a pure
evader policy j = max_i M[i, j] - value (both >= 0; 0 iff the pure policy is
a maximin / minimax strategy). This script is the only reader of its numbers
and runs only after stages 1 and 2 are summarised.
"""
from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import sys

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np  # noqa: E402

_HERE = pathlib.Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _load(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S14 = _load("derisk_s14_matrix", _HERE / "derisk_s14_matrix.py")
S1 = _load("derisk_s24_1_gate", _HERE / "derisk_s24_1_gate.py")
E2 = _load("derisk_s24_2_gate", _HERE / "derisk_s24_2_gate.py")
E3 = _load("derisk_s24_2b_gate", _HERE / "derisk_s24_2b_gate.py")

OUT = pathlib.Path("results/derisk")
PURSUERS: tuple[str, ...] = ("RULES-V0", "LEAD-1.0", "SEL-81", "SEL3-best")
EVADERS: tuple[str, ...] = ("GREEDY", "HUG", "TACTIC-V0", "EVS-80")
SCRIPTED_EVADERS = ("GREEDY", "HUG", "TACTIC-V0")


def sel3_best(round_json: pathlib.Path = OUT / "s24_2b_round.json") -> str:
    """A2.4: the SEL3 seed with the highest mean capture over the six stage-2
    evaders, read from the stage-2 round JSON (never from a re-cut)."""
    blob = json.loads(round_json.read_text())
    cells = blob["cells"]
    means = {}
    for s in E3.SEEDS:
        row = f"SEL3-{s}"
        rates = [cells[E3.cell_name(row, o)]["capture_rate"] for o in E3.OPPONENTS]
        means[row] = float(np.mean(rates))
    best = max(means, key=means.get)
    return best, means


def _read(module, path: pathlib.Path) -> dict:
    """One record, validated by its own round's `record_status`."""
    if not path.exists():
        raise SystemExit(f"meta-game REFUSED: missing record {path}")
    rec = json.loads(path.read_text())
    status, why = module.record_status(rec)
    if status != "MEASURABLE":
        raise SystemExit(f"meta-game REFUSED: {path.name} is {status}: {why}")
    return rec


def cell_record(pursuer: str, evader: str, best: str) -> tuple[dict, str]:
    """The record for (pursuer, evader) and which round it comes from."""
    if pursuer == "SEL3-best":
        return _read(E3, E3.cells_dir() / f"{E3.cell_name(best, evader)}.json"), "s24_2b"
    if evader == "EVS-80":
        return _read(E2, E2.cells_dir() / f"{E2.cell_name('EVS-80', pursuer)}.json"), "s24_2"
    return _read(S1, S1.cells_dir() / f"{S1.cell_name(pursuer, evader)}.json"), "s24_1"


def capture_matrix(best: str) -> tuple[np.ndarray, dict]:
    m = np.full((len(PURSUERS), len(EVADERS)), np.nan)
    cells: dict = {}
    for i, p in enumerate(PURSUERS):
        for j, e in enumerate(EVADERS):
            rec, source = cell_record(p, e, best)
            k = int(sum(rec["capture_ind"]))
            n = int(rec["n_episodes"])
            m[i, j] = k / n
            cells[f"{p}|{e}"] = {
                "source": source,
                "record": rec["name"],
                "k": k,
                "n": n,
                "capture_rate": k / n,
                "wilson": S1._ci(k, n),
                "picks_share": (rec.get("aggregate") or {}).get("picks_share"),
                "usage_steps": rec.get("usage_steps"),
            }
    return m, cells


def solve(m: np.ndarray, rows=PURSUERS, cols=EVADERS) -> dict:
    """The zero-sum LP on the capture payoff (rows maximise) + exploitability."""
    game = S14.meta_game(m.tolist(), list(rows), list(cols), minimise_rows=False)
    value = float(game["value"])
    expl_p = {r: float(value - m[i].min()) for i, r in enumerate(rows)}
    expl_e = {c: float(m[:, j].max() - value) for j, c in enumerate(cols)}
    game["exploitability_pursuer"] = expl_p
    game["exploitability_evader"] = expl_e
    game["best_response_to_pursuer"] = {r: cols[int(m[i].argmin())] for i, r in enumerate(rows)}
    game["best_response_to_evader"] = {c: rows[int(m[:, j].argmax())] for j, c in enumerate(cols)}
    game["pure_saddle"] = bool(len(game["row_support"]) == 1 and len(game["col_support"]) == 1)
    return game


def write_figure(m: np.ndarray, path: pathlib.Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4.6))
    im = ax.imshow(m, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(EVADERS)), EVADERS)
    ax.set_yticks(range(len(PURSUERS)), PURSUERS)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", color="w" if m[i, j] < 0.6 else "k", fontsize=9)
    ax.set_xlabel("evader")
    ax.set_ylabel("pursuer")
    ax.set_title("S24.2 meta-game — capture (N = 500 paired episodes from 2400000)", fontsize=9)
    fig.colorbar(im, ax=ax, label="capture rate")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> dict:
    best, means = sel3_best()
    m, cells = capture_matrix(best)
    game = solve(m)
    blob = {
        "pursuers": list(PURSUERS),
        "evaders": list(EVADERS),
        "sel3_best": best,
        "sel3_means": means,
        "matrix": m.tolist(),
        "cells": cells,
        "game": game,
        "spec": "docs/pathway/tactics_prereg_s24_2_2026-10-17.md section 4 + A2.4",
    }
    S1._write_json(OUT / "s24_2_meta_game.json", blob)
    write_figure(m, OUT / "s24_2_meta_game.png")
    print(f"SEL3-best = {best} (means {means})")
    print("capture matrix (rows pursuers, cols evaders):")
    print("           " + "  ".join(f"{e:>9s}" for e in EVADERS))
    for i, p in enumerate(PURSUERS):
        print(f"{p:>10s} " + "  ".join(f"{m[i, j]:9.3f}" for j in range(len(EVADERS))))
    print(json.dumps({k: game[k] for k in ("value", "row_strategy", "col_strategy", "row_support", "col_support", "duality_gap", "pure_saddle", "exploitability_pursuer", "exploitability_evader")}, indent=1))
    return blob


if __name__ == "__main__":
    main()
