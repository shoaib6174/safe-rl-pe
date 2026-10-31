"""S25 — does the filter set the game's value? Margin sweep × full-observation
pursuer, EVALUATION ONLY (no training). Pre-registration:
`docs/pathway/value_sweep_prereg_s25_2026-11-01.md` (binding: sections 2–5).

80 cells = 4 pursuer rows × 4 evader columns × 5 filter settings, N = 500 paired
episodes on the S24 gate stream. Every policy is frozen; the filter is the only
thing that varies between settings, and it is swapped by rebinding the game
module's `build_shield` (section 5's registered reading: the assembly calls it
at construction and at every `_begin`, so every episode of a cell is built
under that cell's setting and nothing else in the assembly changes).

Rebinding derivative of the S24.1 gate (its `run_cell`/`run_cells`/
`scan_records`/`load_records`/`read_cell` are CALLED through `_bound()`, as the
S24.2b gate does), with this round's own cell table, seats, shield factory,
verdicts, GIF pass and summary. No rate is printed by `--cells`; `--summarize`
is the only reader of numbers and runs after `--gifs` (CG-V2).

Usage:
  ./venv/bin/python scripts/derisk_s25_value_sweep.py --cells <names...>
  ./venv/bin/python scripts/derisk_s25_value_sweep.py --gifs
  ./venv/bin/python scripts/derisk_s25_value_sweep.py --summarize
"""
from __future__ import annotations

import os

OMP_THREADS = "4"
os.environ.setdefault("OMP_NUM_THREADS", OMP_THREADS)

import argparse
import contextlib
import importlib.util
import json
import pathlib
import sys

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from pe2.config import EnvConfig
from pe2.derisk import paired_indicator_test
from pe2.env import wrap_angle
from pe2.scripted import K_OMEGA, _clip_omega
from pe2.shield import RECOVERY_AFTER, NoOpShield, RecoveryShieldV2
from pe2.structured import _normalize
from pe2.tactics import PursuerTactic, fixed


def _load(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S1 = _load("derisk_s24_1_gate", _HERE / "derisk_s24_1_gate.py")  # the gate template
G2 = _load("derisk_s24_2_gate", _HERE / "derisk_s24_2_gate.py")  # stage-1 records
T2 = G2.T2  # the EVS trainer: the authority on EVS-80's checkpoint path
T3 = _load("derisk_s24_2b_train", _HERE / "derisk_s24_2b_train.py")  # SEL3 ckpts
G3 = _load("derisk_s24_2b_gate", _HERE / "derisk_s24_2b_gate.py")  # stage-2 records
MG = _load("s24_2_meta_game", _HERE / "s24_2_meta_game.py")  # the LP (A5.1)
GAME = S1.GAME  # the tactic-game module S24.1 measured with

PREREG = "docs/pathway/value_sweep_prereg_s25_2026-11-01.md"
PREFIX = "s25"
OUT = pathlib.Path("results/derisk")
UNDETERMINED = S1.UNDETERMINED

# --- pre-registered. Do not tune. ---
GATE_SEED0 = 2_400_000
N_EPISODES = 500
P_GATE = 0.05
RATIO = 1.2
FULLOBS_THRESHOLD = 0.25  # H-VAL-3's pre-stated threshold (section 4)
PURSUERS: tuple[str, ...] = ("RULES-V0", "SEL-81", "SEL3-80", "FULLOBS")
EVADERS: tuple[str, ...] = ("GREEDY", "HUG", "TACTIC-V0", "EVS-80")
SETTINGS: dict[str, float | None] = {  # name -> margin delta (m); None = NONE control
    "NONE": None,
    "d0.00": 0.0,
    "d0.05": 0.05,
    "d0.10": 0.10,
    "d0.20": 0.20,
}
BASE_SETTING = "d0.00"  # the measured game (S24)
TOP_SETTING = "d0.20"  # H-VAL-2's comparison
GAMMA = 0.2
EVS_BEST = "EVS-80"

if S1.GATE_SEED0 != GATE_SEED0 or S1.N_EPISODES != N_EPISODES:
    raise SystemExit("S24.1's gate stream or N differ from this round's; the identity gate CG-V1 would be void")
if GAME.RECOVERY_AFTER != RECOVERY_AFTER:
    raise SystemExit("recovery_after moved; the measured game ran at the library default")


# ----------------------------------------------------------------------------
# the two components this round adds
# ----------------------------------------------------------------------------
class NoOpControl(NoOpShield):
    """Section 2's NONE setting: `NoOpShield` (pass-through, contacts counted
    not prevented, `method == "none"`) carrying the recovery counters the
    assembly's episode log reads, all identically zero — a control, not a
    safety claim."""

    n_recoveries = 0
    n_recoveries_forward = 0
    n_recoveries_rotation = 0


class FullObsGreedy(PursuerTactic):
    """Section 2's FULLOBS row: `pe2.scripted.GreedyPursuer`'s command — turn
    toward the TRUE evader pose, full speed — normalised as every law is
    (`pe2.structured._normalize`). Never DONE, no search, no tactics."""

    name = "FULLOBS-GREEDY"

    def act(self, obs):
        env = self.env
        p, e = env.pursuer_pose, env.evader_pose
        err = wrap_angle(float(np.arctan2(e[1] - p[1], e[0] - p[0]) - p[2]))
        self.steps += 1
        return _normalize(self.cfg.v_max, _clip_omega(K_OMEGA * err, self.cfg), self.cfg)


def shield_factory(setting: str):
    """The per-setting replacement for the game module's `build_shield`."""
    if setting not in SETTINGS:
        raise ValueError(f"unknown setting {setting!r}; this round's are {tuple(SETTINGS)}")
    delta = SETTINGS[setting]

    def build(cfg: EnvConfig):
        if delta is None:
            return NoOpControl(cfg)
        shield = RecoveryShieldV2(cfg, recovery_after=RECOVERY_AFTER, margin=float(delta))
        if (shield.gamma, shield.recovery_after) != (GAMMA, RECOVERY_AFTER):
            raise SystemExit("CBF gamma or recovery_after moved from the measured game's")
        return shield

    build.setting = setting  # type: ignore[attr-defined]
    return build


def shield_spec(shield) -> dict:
    return {
        "class": type(shield).__name__,
        "margin": float(getattr(shield, "margin", float("nan"))),
        "gamma": float(getattr(shield, "gamma", float("nan"))),
        "recovery_after": int(getattr(shield, "recovery_after", 0)),
    }


# ----------------------------------------------------------------------------
# the cell table
# ----------------------------------------------------------------------------
def cell_name(row: str, opponent: str, setting: str) -> str:
    return f"{row}_{opponent}_{setting}"


ROWS: dict[str, dict] = {
    "RULES-V0": {"kind": "scripted", "select": "PursuerRulesV0"},
    "SEL-81": {"kind": "learned", "arm": "SEL-81", "seed": 81, "memory": True},
    "SEL3-80": {"kind": "learned", "arm": "SEL3-80", "seed": 80, "memory": True},
    "FULLOBS": {"kind": "scripted", "select": "fixed(FullObsGreedy) = FULLOBS"},
}


def _build_cells() -> dict[str, dict]:
    cells: dict[str, dict] = {}
    for setting, margin in SETTINGS.items():
        for row in PURSUERS:
            for opp in EVADERS:
                cells[cell_name(row, opp, setting)] = {
                    "row": row,
                    "opponent": opp,
                    "setting": setting,
                    "margin": margin,
                    "kind": ROWS[row]["kind"],
                    "memory": ROWS[row].get("memory"),
                    "select": ROWS[row].get("select"),
                    "hold": GAME.HOLD,
                    "in_distribution": setting == BASE_SETTING,  # the trained setting
                }
    return cells


CELLS = _build_cells()
CELL_ORDER: tuple[str, ...] = tuple(CELLS)
assert len(CELLS) == 80


def cells_dir() -> pathlib.Path:
    return OUT / f"{PREFIX}_cells"


def cell_file(name: str) -> pathlib.Path:
    return cells_dir() / f"{name}.json"


def round_file(ext: str) -> pathlib.Path:
    return OUT / f"{PREFIX}_round.{ext}"


def ckpt_path(row: str) -> pathlib.Path:
    """Each learned row's checkpoint through ITS trainer's/gate's own function."""
    if row == "SEL-81":
        return S1.ckpt_path(row)
    if row == "SEL3-80":
        return T3.ckpt_path(row)
    raise KeyError(f"{row} is not a learned row of this round")


def record_status(rec: dict) -> tuple[str, str]:
    name = rec.get("name")
    if name not in CELLS:
        return "INVALID", f"name {name!r} is not a cell of this round"
    if rec.get("verdict") == UNDETERMINED:
        return UNDETERMINED, f"crashed cell: {rec.get('error')!r}"
    if rec.get("n_episodes") != N_EPISODES:
        return "INVALID", f"n_episodes {rec.get('n_episodes')!r} != {N_EPISODES}"
    if rec.get("seed0") != GATE_SEED0:
        return "INVALID", f"seed0 {rec.get('seed0')!r} != {GATE_SEED0}"
    spec = CELLS[name]
    for key in ("row", "opponent", "setting"):
        if rec.get(key) != spec[key]:
            return "INVALID", f"{key} {rec.get(key)!r} != the declared {spec[key]!r}"
    return "MEASURABLE", ""


def provenance(device: str = "cpu") -> dict:
    base = _ORIG["provenance"](device)  # the ORIGINAL: inside the binding S1.provenance is this function
    return base | {
        "round": "S25 value sweep",
        "spec": PREREG,
        "spec_sha256": S1._sha256(_REPO / PREREG) if hasattr(S1, "_sha256") else None,
        "settings": {k: v for k, v in SETTINGS.items()},
        "rows": list(PURSUERS),
        "evaders": list(EVADERS),
    }


def write_cell(record: dict) -> pathlib.Path:
    cells_dir().mkdir(parents=True, exist_ok=True)
    path = cell_file(record["name"])
    S1._write_json(path, record)
    return path


def undetermined_record(name: str, exc: BaseException, device: str = "cpu") -> dict:
    spec = CELLS[name]
    return {
        "name": name,
        "row": spec["row"],
        "opponent": spec["opponent"],
        "setting": spec["setting"],
        "verdict": UNDETERMINED,
        "error": f"{type(exc).__name__}: {exc}",
        "seed0": GATE_SEED0,
        "device": device,
        "provenance": provenance(device),
    }


# ----------------------------------------------------------------------------
# seats and the cell env
# ----------------------------------------------------------------------------
_EVS_MODELS: dict = {}


def _install_evader(game, rules, opponent: str, cfg: EnvConfig, device: str) -> None:
    """The evader seat: S24.1's `_set_opponent` for the scripted three
    (TACTIC-V0 is the assembly's default), `LearnedEvaderSelect` over an
    `EvaderSelectorMDP` built OVER this game for EVS-80 (the S24.2b pattern)."""
    if opponent == EVS_BEST:
        from sb3_contrib import MaskablePPO

        from pe2.evader_selector_mdp import EvaderSelectorMDP
        from pe2.learned_select import LearnedEvaderSelect

        key = (str(T2.ckpt_path(opponent)), device)
        if key not in _EVS_MODELS:
            _EVS_MODELS[key] = MaskablePPO.load(T2.ckpt_path(opponent), device=device)
        sel = EvaderSelectorMDP.over_game(game, rules, cfg=cfg)
        game.evader.select = LearnedEvaderSelect(sel, _EVS_MODELS[key], name=opponent)
        return
    S1._set_opponent(game, opponent, cfg, device)


def _install_shield(setting: str) -> None:
    """Section 5's registered reading: rebind the game module's `build_shield`
    so construction AND every `_begin` build this setting's filter."""
    GAME.build_shield = shield_factory(setting)


def build_cell_env(
    name: str,
    cfg: EnvConfig | None = None,
    device: str = "cpu",
    *,
    recorder: bool = False,
    label: str = "",
):
    """`SelectorMDP` for a learned row, the v0b `TacticGameMDP` for a scripted
    one (S24.1's two shapes), under this cell's filter setting, with this
    cell's evader installed and NOTHING else varied."""
    spec = CELLS[name]
    cfg = S1.cell_config() if cfg is None else cfg
    _install_shield(spec["setting"])
    if spec["kind"] == "learned":
        from pe2.selector_mdp import SelectorMDP

        env = SelectorMDP(
            cfg.evader_speed_ratio, memory=spec["memory"], recorder=recorder, label=label, cfg=cfg
        )
        game, rules = env.game, env.rules
    else:
        game, rules = GAME.build_game(cfg, recorder=recorder, label=label)
        if spec["row"] == "FULLOBS":
            tactic = FullObsGreedy(cfg)
            game.law.tactics.append(tactic)  # reset with the game each episode
            game.law.select = fixed(tactic)
        env = game
    _install_evader(game, rules, spec["opponent"], cfg, device)
    return env


def _shield_of(env):
    game = env.game if hasattr(env, "game") else env
    return game.shield


# ----------------------------------------------------------------------------
# the binding onto S24.1's gate
# ----------------------------------------------------------------------------
_BOUND: tuple[str, ...] = (
    "CELLS", "CELL_ORDER", "cells_dir", "cell_file", "record_status", "provenance",
    "write_cell", "undetermined_record", "ROWS", "build_cell_env", "ckpt_path", "run_cell",
)
_ORIG: dict = {
    k: getattr(S1, k)
    for k in (*_BOUND, "run_cells", "scan_records", "load_records", "read_cell")
}
_BINDING_DEPTH = 0


def _assert_outside_binding(what: str) -> None:
    if _BINDING_DEPTH:
        raise AssertionError(f"{what} must read other rounds' records OUTSIDE the binding")


@contextlib.contextmanager
def _bound():
    global _BINDING_DEPTH
    over = {k: globals()[k] for k in _BOUND}
    saved = {k: getattr(S1, k) for k in over}
    for k, v in over.items():
        setattr(S1, k, v)
    _BINDING_DEPTH += 1
    try:
        yield
    finally:
        _BINDING_DEPTH -= 1
        for k, v in saved.items():
            setattr(S1, k, v)


def run_cell(name: str, device: str = "cpu", n_episodes: int | None = None) -> dict:
    """S24.1's own `run_cell` (through the binding), plus this round's setting
    and the CG-V3 shield check: the filter actually built for this cell."""
    spec = CELLS[name]
    rec = _ORIG["run_cell"](name, device, n_episodes)
    probe = build_cell_env(name, S1.cell_config(), device)
    game = probe.game if hasattr(probe, "game") else probe
    game._begin(GATE_SEED0)  # the per-episode rebuild path
    built = shield_spec(game.shield)
    expect_cls = "NoOpControl" if spec["margin"] is None else "RecoveryShieldV2"
    ok = built["class"] == expect_cls and (
        spec["margin"] is None or abs(built["margin"] - spec["margin"]) < 1e-12
    )
    if not ok:
        raise SystemExit(f"CG-V3: cell {name} built {built}, expected {expect_cls} margin {spec['margin']}")
    return rec | {"setting": spec["setting"], "margin": spec["margin"], "shield": built, "cg_v3_ok": True}


def run_cells(names: list[str] | None = None, device: str = "cpu", n_episodes: int | None = None):
    if not T2.ckpt_path(EVS_BEST).exists():
        raise SystemExit(f"missing the frozen evader checkpoint at {T2.ckpt_path(EVS_BEST)}")
    with _bound():
        return _ORIG["run_cells"](names, device, n_episodes)


def scan_records():
    with _bound():
        return _ORIG["scan_records"]()


def load_records(*, quiet: bool = False) -> dict:
    with _bound():
        return _ORIG["load_records"](quiet=quiet)


def read_cell(name: str) -> dict:
    with _bound():
        return _ORIG["read_cell"](name)


# ----------------------------------------------------------------------------
# CG-V1 — identity at d0.00 against the stored S24 records
# ----------------------------------------------------------------------------
def _stored(path: pathlib.Path, module) -> dict | None:
    if not path.exists():
        return None
    rec = json.loads(path.read_text())
    status, _why = module.record_status(rec)
    return rec if status == "MEASURABLE" else None


def identity_pairs() -> dict[str, tuple[pathlib.Path, object]]:
    """This round's d0.00 cell -> (the stored record it must reproduce, the
    module whose `record_status` validates it)."""
    pairs: dict[str, tuple[pathlib.Path, object]] = {}
    for opp in ("GREEDY", "HUG", "TACTIC-V0"):
        pairs[cell_name("SEL-81", opp, BASE_SETTING)] = (_ORIG["cells_dir"]() / f"SEL-81_{opp}.json", S1)
        pairs[cell_name("RULES-V0", opp, BASE_SETTING)] = (_ORIG["cells_dir"]() / f"RULES-V0_{opp}.json", S1)
    for opp in EVADERS:
        pairs[cell_name("SEL3-80", opp, BASE_SETTING)] = (G3.cells_dir() / f"SEL3-80_{opp}.json", G3)
    for p in ("SEL-81", "RULES-V0"):
        pairs[cell_name(p, EVS_BEST, BASE_SETTING)] = (G2.cells_dir() / f"EVS-80_{p}.json", G2)
    return pairs


def cg_v1(records: dict) -> dict:
    _assert_outside_binding("CG-V1")
    out: dict = {}
    offenders: list[str] = []
    missing: list[str] = []
    for name, (path, module) in identity_pairs().items():
        if name not in records:
            missing.append(name)
            continue
        stored = _stored(path, module)
        if stored is None:
            offenders.append(f"{name}: stored record {path.name} absent or not measurable")
            continue
        same = list(map(bool, records[name]["capture_ind"])) == list(map(bool, stored["capture_ind"]))
        out[name] = {"stored": str(path), "identical": same,
                     "k_here": int(sum(records[name]["capture_ind"])), "k_stored": int(sum(stored["capture_ind"]))}
        if not same:
            offenders.append(name)
    return {
        "gate": "CG-V1 (identity at d0.00 with the stored S24 records, all 500 episodes)",
        "verdict": "INCOMPLETE" if missing else ("REFUSE" if offenders else "PASS"),
        "pairs": out, "offenders": offenders, "missing": missing,
    }


def cg_v3(records: dict) -> dict:
    bad = [n for n, r in records.items() if not r.get("cg_v3_ok")]
    none_bad = [
        n for n, r in records.items()
        if CELLS[n]["margin"] is None and r.get("shield", {}).get("class") != "NoOpControl"
    ]
    return {"gate": "CG-V3 (the filter built per cell)", "verdict": "REFUSE" if (bad or none_bad) else "PASS",
            "offenders": sorted(set(bad + none_bad))}


# ----------------------------------------------------------------------------
# verdicts (section 4)
# ----------------------------------------------------------------------------
def _pair(records: dict, a: str, b: str, direction: str) -> dict:
    """Paired indicator test that `a` captures MORE (`UP`) or LESS (`DOWN`) than
    `b` on the same episodes."""
    ra, rb = records.get(a), records.get(b)
    if ra is None or rb is None:
        return {"verdict": "INCOMPLETE", "a": a, "b": b, "missing": [n for n, r in ((a, ra), (b, rb)) if r is None]}
    x, y = (ra, rb) if direction == "UP" else (rb, ra)
    t = paired_indicator_test(x["capture_ind"], y["capture_ind"], P_GATE)
    return {
        "verdict": "PASS" if t["pass"] else "FAIL", "a": a, "b": b, "direction": direction,
        "capture_a": ra["capture_rate"], "capture_b": rb["capture_rate"],
        "delta": ra["capture_rate"] - rb["capture_rate"], "p": t["p"], "test": t,
    }


def h_val_1(records: dict) -> dict:
    rows = {r: _pair(records, cell_name(r, EVS_BEST, "NONE"), cell_name(r, EVS_BEST, BASE_SETTING), "UP") for r in PURSUERS}
    vs = [b["verdict"] for b in rows.values()]
    n = sum(v == "PASS" for v in vs)
    return {"hypothesis": "H-VAL-1 (the claim): NONE > d0.00 vs EVS-80, UP significant, 4/4 rows", "gated": True,
            "verdict": "INCOMPLETE" if "INCOMPLETE" in vs else ("PASS" if n == len(PURSUERS) else "FAIL"),
            "n_pass": n, "rows": rows}


def h_val_2(records: dict) -> dict:
    down = {r: _pair(records, cell_name(r, EVS_BEST, TOP_SETTING), cell_name(r, EVS_BEST, BASE_SETTING), "DOWN") for r in PURSUERS}
    up_any = {}
    for s in ("d0.05", "d0.10", TOP_SETTING):
        for r in PURSUERS:
            up_any[f"{r}@{s}"] = _pair(records, cell_name(r, EVS_BEST, s), cell_name(r, EVS_BEST, BASE_SETTING), "UP")
    vs = [b["verdict"] for b in list(down.values()) + list(up_any.values())]
    n_down = sum(b["verdict"] == "PASS" for b in down.values())
    ups = sorted(k for k, b in up_any.items() if b["verdict"] == "PASS")
    passed = n_down >= 3 and not ups
    return {"hypothesis": "H-VAL-2 (monotone): d0.20 < d0.00 vs EVS-80 on >= 3/4 rows, no UP-significant row at any delta > 0",
            "gated": True, "verdict": "INCOMPLETE" if "INCOMPLETE" in vs else ("PASS" if passed else "FAIL"),
            "n_down_pass": n_down, "up_significant": ups, "down": down, "up_checks": up_any,
            "reported_deltas": {s: {r: records.get(cell_name(r, EVS_BEST, s), {}).get("capture_rate") for r in PURSUERS}
                                for s in SETTINGS}}


def h_val_3(records: dict) -> dict:
    a = records.get(cell_name("FULLOBS", EVS_BEST, BASE_SETTING))
    none = records.get(cell_name("FULLOBS", EVS_BEST, "NONE"))
    if a is None:
        return {"hypothesis": "H-VAL-3", "gated": False, "verdict": "INCOMPLETE"}
    k = int(sum(a["capture_ind"]))
    margin_only = k < round(FULLOBS_THRESHOLD * N_EPISODES)  # exact counts
    return {"hypothesis": f"H-VAL-3 (information vs margin): FULLOBS x EVS-80 at d0.00 < {FULLOBS_THRESHOLD} => margin effect",
            "gated": False, "verdict": "REPORTED", "capture_fullobs_d0": a["capture_rate"], "k": k,
            "threshold_k": round(FULLOBS_THRESHOLD * N_EPISODES), "reading": "margin-only" if margin_only else "information-dependent",
            "capture_fullobs_none": None if none is None else none["capture_rate"],
            "fullobs_vs_scripted_d0": {o: records.get(cell_name("FULLOBS", o, BASE_SETTING), {}).get("capture_rate") for o in EVADERS}}


def h_val_4(records: dict) -> dict:
    """The LP value and saddle per setting (A5.1's definitions via `MG.solve`)."""
    per: dict = {}
    for s in SETTINGS:
        m = np.full((len(PURSUERS), len(EVADERS)), np.nan)
        for i, p in enumerate(PURSUERS):
            for j, e in enumerate(EVADERS):
                r = records.get(cell_name(p, e, s))
                if r is not None:
                    m[i, j] = r["capture_rate"]
        if np.isnan(m).any():
            per[s] = {"verdict": "INCOMPLETE", "matrix": m.tolist()}
            continue
        g = MG.solve(m, PURSUERS, EVADERS)
        per[s] = {"matrix": m.tolist(), "value": g["value"], "row_support": g["row_support"],
                  "col_support": g["col_support"], "pure_saddle": g["pure_saddle"],
                  "exploitability_pursuer": g["exploitability_pursuer"], "exploitability_evader": g["exploitability_evader"]}
    vals = {s: per[s].get("value") for s in SETTINGS}
    ordered = (vals["NONE"] is not None and vals[BASE_SETTING] is not None and vals[TOP_SETTING] is not None
               and vals["NONE"] > vals[BASE_SETTING] > vals[TOP_SETTING])
    evs_in_support = all(EVS_BEST in per[s].get("col_support", []) for s in SETTINGS if s != "NONE" and "value" in per[s])
    return {"hypothesis": "H-VAL-4 (value curve, reported)", "gated": False, "verdict": "REPORTED",
            "values": vals, "pre_stated_order_holds": ordered, "evs_in_support_at_every_shielded_setting": evs_in_support,
            "per_setting": per}


def h_safe(records: dict) -> dict:
    shielded = {n: int(r["violation_steps"]) for n, r in records.items() if CELLS[n]["margin"] is not None}
    none = {n: int(r["violation_steps"]) for n, r in records.items() if CELLS[n]["margin"] is None}
    offenders = sorted(n for n, v in shielded.items() if v)
    return {"hypothesis": "H-SAFE (shielded cells only)", "verdict": "FAILS" if offenders else "HOLDS",
            "shielded_violation_steps": int(sum(shielded.values())), "offenders": offenders,
            "none_control_contact_steps": none, "n_shielded_cells": len(shielded)}


def direction(v: dict) -> str:
    parts = [f"H-SAFE {v['h_safe']['verdict']} over {v['h_safe']['n_shielded_cells']} shielded cells."]
    parts.append(f"CG-V1 {v['cg_v1']['verdict']}; CG-V3 {v['cg_v3']['verdict']}.")
    t1, t2, t3, t4 = v["h_val_1"], v["h_val_2"], v["h_val_3"], v["h_val_4"]
    parts.append(f"H-VAL-1 {t1['verdict']} ({t1['n_pass']}/4 rows rise without the filter).")
    parts.append(f"H-VAL-2 {t2['verdict']} ({t2['n_down_pass']}/4 rows fall at d0.20; UP-significant: {t2['up_significant'] or 'none'}).")
    if t3["verdict"] == "REPORTED":
        parts.append(f"H-VAL-3: FULLOBS vs EVS-80 at d0.00 = {t3['capture_fullobs_d0']:.3f} -> {t3['reading']} (threshold {FULLOBS_THRESHOLD}).")
    parts.append(f"H-VAL-4 values: {', '.join(f'{s} {x:.3f}' for s, x in t4['values'].items() if x is not None)}; pre-stated order {'holds' if t4['pre_stated_order_holds'] else 'does NOT hold'}.")
    if t1["verdict"] == "PASS" and t2["verdict"] == "PASS":
        parts.append("Reading: the filter sets the value — removing it raises every pursuer's capture against the learned evader and tightening it lowers it.")
    elif t1["verdict"] == "FAIL" and t1["n_pass"] == 0:
        parts.append("Reading WITHDRAWN: no pursuer captures more without the filter; the evader's orbit beats them on geometry alone.")
    else:
        parts.append("Reading QUALIFIED: the filter's role holds for some rows/settings only (see per-row tests).")
    return " ".join(parts)


def verdicts(records: dict) -> dict:
    v = {"cg_v1": cg_v1(records), "cg_v3": cg_v3(records), "h_safe": h_safe(records),
         "h_val_1": h_val_1(records), "h_val_2": h_val_2(records), "h_val_3": h_val_3(records),
         "h_val_4": h_val_4(records), "n_cells": len(CELLS), "n_records": len(records),
         "missing": [n for n in CELL_ORDER if n not in records], "spec": PREREG}
    v["direction"] = direction(v)
    return v


# ----------------------------------------------------------------------------
# GIFs (CG-V2) and the summary
# ----------------------------------------------------------------------------
GIF_CELLS: tuple[tuple[str, str, str], ...] = (
    ("FULLOBS", EVS_BEST, BASE_SETTING), ("FULLOBS", EVS_BEST, "NONE"),
    ("SEL-81", EVS_BEST, "NONE"), ("SEL-81", EVS_BEST, TOP_SETTING),
)
GIF_N = 3


def gifs_dir() -> pathlib.Path:
    return OUT / f"{PREFIX}_gifs"


def write_gifs(device: str = "cpu", fps: float = 15.0) -> list[pathlib.Path]:
    from pe2.render import save_gif

    with _bound():
        measurable, _rej, _by = _ORIG["scan_records"]()
    for row, opp, setting in GIF_CELLS:
        name = cell_name(row, opp, setting)
        if name not in measurable:
            raise SystemExit(f"CG-V2 refuses: no measurable record for {name}")
        if ROWS[row]["kind"] == "learned" and not ckpt_path(row).exists():
            raise SystemExit(f"CG-V2 refuses: no checkpoint for {row}")
    gifs_dir().mkdir(parents=True, exist_ok=True)
    cfg = S1.cell_config()
    written = []
    for row, opp, setting in GIF_CELLS:
        name = cell_name(row, opp, setting)
        for i in range(GIF_N):
            seed = GATE_SEED0 + i
            path = gifs_dir() / f"{name}_{seed}.gif"
            if path.exists():
                continue
            title = f"{row} vs {opp} · filter {setting} · episode {seed}"
            env = build_cell_env(name, cfg, device, recorder=True, label=title)
            game = env.game if ROWS[row]["kind"] == "learned" else env
            game.every = int(S1.GIF_EVERY)
            if ROWS[row]["kind"] == "learned":
                from sb3_contrib import MaskablePPO

                log = env.play_episode(MaskablePPO.load(ckpt_path(row), device=device).predict, seed)
            else:
                with _bound():
                    log = S1.scripted_log(env, cfg, seed)
            game.finish(log["outcome"], log["steps"])
            save_gif(game.frames, str(path), fps=fps)
            print(f"wrote {path} ({len(game.frames)} frames; {log['outcome']} @ {log['steps']})", flush=True)
            written.append(path)
    return written


def write_figure(records: dict, v: dict, path: pathlib.Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = list(SETTINGS)
    xs = list(range(len(order)))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for r in PURSUERS:
        ys = [records.get(cell_name(r, EVS_BEST, s), {}).get("capture_rate", np.nan) for s in order]
        axes[0].plot(xs, ys, marker="o", label=r)
    axes[0].set_xticks(xs, order)
    axes[0].set_ylabel("capture vs EVS-80")
    axes[0].set_xlabel("filter setting (NONE = pass-through control)")
    axes[0].set_title("frozen pursuers vs the learned evader across the filter", fontsize=9)
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    vals = [v["h_val_4"]["values"].get(s) for s in order]
    axes[1].plot(xs, [np.nan if x is None else x for x in vals], marker="s", color="k")
    axes[1].set_xticks(xs, order)
    axes[1].set_ylabel("LP value of the 4 x 4 capture game")
    axes[1].set_title("the game's value across the filter (H-VAL-4)", fontsize=9)
    axes[1].grid(alpha=0.3)
    fig.suptitle(f"S25 — N = {N_EPISODES} paired episodes from {GATE_SEED0}, ratio {RATIO}, policies frozen", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


_BULK = ("capture_ind", "steps", "violations", "n_switches", "n_zero_length_spans", "n_decisions",
         "inapplicable_picks_per_episode", "tactics", "branch")


def summarize(device: str = "cpu") -> dict:
    records = load_records()
    v = verdicts(records)
    for gate in ("cg_v1", "cg_v3"):
        if v[gate]["verdict"] != "PASS":
            raise SystemExit(f"{gate.upper()} {v[gate]['verdict']}: {v[gate].get('offenders') or v[gate].get('missing')}. No rate is read.")
    if v["h_safe"]["verdict"] != "HOLDS":
        raise SystemExit(f"H-SAFE FAILS on shielded cells: {v['h_safe']['offenders']}")
    if v["missing"]:
        raise SystemExit(f"INCOMPLETE: {len(v['missing'])} cells missing: {v['missing'][:6]}...")
    with _bound():
        _m, rejected, _b = _ORIG["scan_records"]()
    blob = {"verdicts": v,
            "cells": {n: {k: val for k, val in r.items() if k not in _BULK} for n, r in records.items()},
            "rejected_records": rejected, "provenance": provenance(device)}
    S1._write_json(round_file("json"), blob)
    write_figure(records, v, round_file("png"))
    print(json.dumps({k: v[k] for k in ("h_safe", "cg_v1", "cg_v3")}, indent=1)[:4000])
    print(v["direction"])
    print(f"cells {len(records)}/{len(CELLS)}")
    return blob


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cells", nargs="*", default=None)
    p.add_argument("--gifs", action="store_true")
    p.add_argument("--summarize", action="store_true")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    did = False
    if args.cells is not None:
        run_cells(args.cells or None, args.device)
        did = True
    if args.gifs:
        write_gifs(args.device)
        did = True
    if args.summarize:
        summarize(args.device)
        did = True
    if not did:
        raise SystemExit("nothing to do: pass --cells [names...], --gifs and/or --summarize")


if __name__ == "__main__":
    main()
