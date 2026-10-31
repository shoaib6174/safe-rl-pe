"""Paper B rounds B2, B3, B4 (S26) — frozen policies × opponent × filter setting.

Spec: docs/pathway/paper_b_prereg_rounds_2026-11-02.md sections 0, 2, 3, 4, 8.
Nothing here may change a threshold, a seed, a stream, a rule or a cell list
after a number is seen.

Cells `{POLICY}_{OPPONENT}_{SETTING}`, N = 500 paired on stream 1 740 000+ (S17's
gate stream, so the 12 `BASE-6x_{ATK4-45,GREEDY}_{PLAIN,RECF-20}` cells MUST
reproduce S17's stored records bit for bit — CG-B2-1, blocking):

  B2  5 policies (BASE-60/61/62, STR14, RC19) × 2 opponents × 7 settings
      PLAIN | RECF-5 RECF-10 RECF-20 RECF-40 RECF-80 | PERT-20           = 70
  B3  4 policies (BASE-60/61/62, RC19) × 2 opponents × PRED-PLAIN         =  8
  B4  the same 4 × 2 × PRED-RECF-20 (RecoveryShieldV2 composed onto the
      predictive filter by the method resolution order, CG-B4-1)          =  8

Evaluation path = S17's (`derisk_s17_fixes_b.evaluate_arm_policy`): one
`SearchMDP` per cell, a FRESH shield per episode, `play_episode` per seed. The
recording wrapper is observation-free (it returns the filter's `(u, info)`
untouched), which is what makes the identity gate possible.

Usage:
  ./venv/bin/python scripts/paper_b_filter_sweep.py --cells NAME [NAME ...]
  ./venv/bin/python scripts/paper_b_filter_sweep.py --gifs        (CG-B2-3, before --summarize)
  ./venv/bin/python scripts/paper_b_filter_sweep.py --summarize   (the only reader of numbers)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import subprocess

import numpy as np

from pe2.config import EnvConfig
from pe2.derisk import paired_indicator_test
from pe2.metrics import DEADLOCK_STEPS, summarize_shield
from pe2.obs import obs_dim
from pe2.scripted import GreedyEvader
from pe2.shield import STALL_V_FRAC, CBFShield, PredictiveShield, RecoveryShieldV2

SPEC = "docs/pathway/paper_b_prereg_rounds_2026-11-02.md"
PREFIX = "s26_b2"
OUT = pathlib.Path("results/derisk")
S17_PREFIX = "s17"

GATE_SEED0 = 1_740_000  # S17's gate stream (spec section 2: paired, identity with S17)
GATE_N = 500
HOLD = 10
EVADER_HOLD = 10
SHIELD_OBS_DEFAULT = True
TRACE_EPISODES = 3
CG_B4_EPISODES = 5  # runtime acceptance check on the first episodes of every PRED-RECF cell
P_GATE = 0.05
DEADLOCK_BAR = 0.05
K_DEADLOCK = DEADLOCK_STEPS  # 40
ESCAPE_DIST = 0.10  # m, spec section 0
T_TURN = math.ceil(2 * math.pi / (EnvConfig().omega_max * EnvConfig().dt))  # 45 steps
ENGAGE_GAP = 5  # steps without a firing that end an engagement
CYCLE_WINDOW = K_DEADLOCK  # an engagement starting within this many steps of the last one's end is a cycle
PRED_HORIZON = 4
PERT_TRIES = 8
IMMOBILE_BAR_RECF20 = 0.12  # H-B2-3
STALL_THRESHOLDS = (40, 80, 160)
ATK_CKPT = OUT / "ckpts_ATK4-45" / "ATK4-45_10000000.zip"

POLICIES: dict[str, dict] = {
    "BASE-60": {"ckpt": OUT / "ckpts_BASE-60" / "BASE-60_5000000.zip", "shield_obs": True, "s17": "BASE-60"},
    "BASE-61": {"ckpt": OUT / "ckpts_BASE-61" / "BASE-61_5000000.zip", "shield_obs": True, "s17": "BASE-61"},
    "BASE-62": {"ckpt": OUT / "ckpts_BASE-62" / "BASE-62_5000000.zip", "shield_obs": True, "s17": "BASE-62"},
    "STR14": {"ckpt": OUT / "ckpts_STR14" / "STR14_10000000.zip", "shield_obs": False, "s17": None},
    "RC19": {"ckpt": OUT / "ckpts_RC19" / "RC19_10000000.zip", "shield_obs": True, "s17": None},
}
B2_POLICIES = ("BASE-60", "BASE-61", "BASE-62", "STR14", "RC19")
B34_POLICIES = ("BASE-60", "BASE-61", "BASE-62", "RC19")
OPPONENTS = ("ATK4-45", "GREEDY")
CAPTURE_OPPONENT = "GREEDY"
DEADLOCK_OPPONENT = "ATK4-45"
RECF_NS = (5, 10, 20, 40, 80)
B2_SETTINGS = ("PLAIN", *(f"RECF-{n}" for n in RECF_NS), "PERT-20")
B3_SETTINGS = ("PRED-PLAIN",)
B4_SETTINGS = ("PRED-RECF-20",)
SETTINGS = (*B2_SETTINGS, *B3_SETTINGS, *B4_SETTINGS)
KINDS = ("forward", "policy", "rotation", "perturbation")
GIF_SEEDS = (GATE_SEED0, GATE_SEED0 + 1)
GIF_CELLS = (  # CG-B2-3 (spec section 2)
    ("STR14", "ATK4-45", "PLAIN", (GATE_SEED0,)),
    ("STR14", "ATK4-45", "RECF-20", (GATE_SEED0,)),
    ("RC19", "ATK4-45", "PLAIN", (GATE_SEED0,)),
    ("RC19", "ATK4-45", "RECF-20", (GATE_SEED0,)),
    ("BASE-61", "ATK4-45", "PERT-20", GIF_SEEDS),
    ("BASE-61", "ATK4-45", "RECF-80", GIF_SEEDS),
)
UNDETERMINED = "UNDETERMINED"


def cell_name(policy: str, opponent: str, setting: str) -> str:
    return f"{policy}_{opponent}_{setting}"


def _build_cells() -> dict[str, dict]:
    cells: dict[str, dict] = {}
    for group, pols, settings in (("B2", B2_POLICIES, B2_SETTINGS), ("B3", B34_POLICIES, B3_SETTINGS), ("B4", B34_POLICIES, B4_SETTINGS)):
        for p in pols:
            for o in OPPONENTS:
                for s in settings:
                    cells[cell_name(p, o, s)] = {"policy": p, "opponent": o, "setting": s, "group": group}
    return cells


CELLS = _build_cells()
CELL_ORDER = list(CELLS)


def cells_dir() -> pathlib.Path:
    return OUT / f"{PREFIX}_cells"


def gifs_dir() -> pathlib.Path:
    return OUT / f"{PREFIX}_gifs"


def _round_file(ext: str) -> pathlib.Path:
    return OUT / f"{PREFIX}_round.{ext}"


# ----------------------------------------------------------------------------
# the filters (spec sections 0, 2, 4)
# ----------------------------------------------------------------------------
class PerturbationShield(CBFShield):
    """The random-perturbation baseline (spec section 2, `PERT-20`).

    After `recovery_after` consecutive immobile steps, draw up to `tries`
    actions uniformly from the box with the per-episode generator, filter each
    through the ordinary CBF shield and execute the first whose filtered speed
    exceeds `STALL_V_FRAC * v_max`; otherwise the ordinary filtered result
    stands. Every executed command is a `CBFShield.filter` output, so H-SAFE
    covers it exactly as it covers `RecoveryShieldV2`.
    """

    def __init__(self, cfg: EnvConfig, seed: int, recovery_after: int = 20, tries: int = PERT_TRIES, **kw):
        super().__init__(cfg, **kw)
        self.recovery_after = int(recovery_after)
        self.tries = int(tries)
        self.rng = np.random.default_rng(int(seed))
        self._immobile_run = 0
        self.n_recoveries = 0
        self.n_recoveries_forward = 0
        self.n_recoveries_rotation = 0
        self.n_recoveries_perturbation = 0

    def filter(self, action, state, obstacles, v_max=None):
        u, info = super().filter(action, state, obstacles, v_max)
        info["recovery_fired"] = False
        info["recovery_kind"] = "none"
        if not info["immobile"]:
            self._immobile_run = 0
            return u, info
        self._immobile_run += 1
        if self._immobile_run < self.recovery_after:
            return u, info
        v_cap = self.cfg.v_max if v_max is None else v_max
        eps = STALL_V_FRAC * v_cap
        v_nom, w_nom = self._clip_nominal(action, v_cap)
        for _ in range(self.tries):
            cand = (float(self.rng.uniform(0.0, v_cap)), float(self.rng.uniform(-self.cfg.omega_max, self.cfg.omega_max)))
            u_try, info_try = super().filter(cand, state, obstacles, v_max)
            if float(u_try[0]) > eps:
                out = self._info(info_try["method"], info_try["feasible"], u_try, v_nom, w_nom, self.constraints(state, obstacles), v_cap)
                out["recovery_fired"] = True
                out["recovery_kind"] = "perturbation"
                self.n_recoveries += 1
                self.n_recoveries_perturbation += 1
                return u_try, out
        return u, info


class PredictiveRecoveryShield(RecoveryShieldV2, PredictiveShield):
    """B4's composition (spec section 4): NO overrides. By the MRO,
    `RecoveryShieldV2.filter`'s `super().filter` is `PredictiveShield.filter`, so
    the nominal and all five recovery candidates are judged by the predictive
    rollout and every executed command is one the predictive filter returned."""


def shield_factory(setting: str, cfg: EnvConfig):
    """`setting -> (episode_seed -> shield)`. A fresh filter per episode."""
    if setting == "PLAIN":
        return lambda seed: CBFShield(cfg)
    if setting.startswith("RECF-"):
        n = int(setting.split("-")[1])
        return lambda seed: RecoveryShieldV2(cfg, recovery_after=n)
    if setting == "PERT-20":
        return lambda seed: PerturbationShield(cfg, seed=seed, recovery_after=20)
    if setting == "PRED-PLAIN":
        return lambda seed: PredictiveShield(cfg, horizon=PRED_HORIZON)
    if setting == "PRED-RECF-20":
        return lambda seed: PredictiveRecoveryShield(cfg, recovery_after=20, horizon=PRED_HORIZON)
    raise ValueError(f"unknown setting {setting!r}")


def shield_spec(sh) -> dict:
    return {
        "class": type(sh).__name__,
        "mro": [c.__name__ for c in type(sh).__mro__[:-1]],
        "recovery_after": getattr(sh, "recovery_after", None),
        "horizon": getattr(sh, "horizon", None),
        "margin": float(sh.margin),
        "gamma": float(sh.gamma),
    }


# ----------------------------------------------------------------------------
# the observation-free per-episode recorder (spec section 2 "per cell")
# ----------------------------------------------------------------------------
class EpisodeLog:
    """Wraps one episode's shield; returns the filter's `(u, info)` untouched.

    Computes, from the executed commands and the true pose it is handed:
    stall runs, firings by kind, engagements, cycles and escape times (spec
    section 0 definitions). `check_pred` (CG-B4-1) re-filters every fired
    command through a fresh `PredictiveShield._accept`."""

    def __init__(self, shield, keep_trace: bool, check_pred: PredictiveShield | None = None):
        self.shield = shield
        self.keep_trace = bool(keep_trace)
        self.check_pred = check_pred
        self.n_steps = 0
        self.n_immobile = 0
        self.n_stalled = 0
        self.kinds = dict.fromkeys(KINDS, 0)
        self.longest_stall = 0
        self._stall_run = 0
        self.engagements: list[dict] = []  # {"start", "end", "escape_time" | None}
        self._last_fire_step: int | None = None
        self._open: dict | None = None
        self.pending_escapes: list[dict] = []  # triggers whose escape has not happened yet
        self.cg_b4_failures = 0
        self.cg_b4_checked = 0
        self.trace: list[dict] = []

    def __getattr__(self, name):
        if name == "shield":
            raise AttributeError(name)
        return getattr(self.shield, name)

    def filter(self, action, state, obstacles, v_max=None):
        u, info = self.shield.filter(action, state, obstacles, v_max)
        t = self.n_steps
        self.n_steps += 1
        pos = (float(state[0]), float(state[1]))
        immobile = bool(info["immobile"])
        stalled = bool(info["stalled"])
        fired = bool(info.get("recovery_fired", False))
        kind = str(info.get("recovery_kind", "none"))
        self.n_immobile += int(immobile)
        self.n_stalled += int(stalled)
        self._stall_run = self._stall_run + 1 if stalled else 0
        self.longest_stall = max(self.longest_stall, self._stall_run)
        if fired:
            self.kinds[kind] = self.kinds.get(kind, 0) + 1
            if self.check_pred is not None:
                self.cg_b4_checked += 1
                if not self.check_pred._accept(np.asarray(u, dtype=np.float64), state, obstacles):
                    self.cg_b4_failures += 1
        # engagements and cycles
        if fired:
            if self._open is None or (self._last_fire_step is not None and t - self._last_fire_step > ENGAGE_GAP):
                self._close_open(t)
                prev_end = self.engagements[-1]["end"] if self.engagements else None
                self._open = {"start": t, "end": None, "cycle": bool(prev_end is not None and t - prev_end <= CYCLE_WINDOW)}
                self.pending_escapes.append({"start": t, "pos": pos, "eng": self._open})
            self._last_fire_step = t
        elif self._open is not None and self._last_fire_step is not None and t - self._last_fire_step > ENGAGE_GAP:
            self._close_open(t)
        # escape times
        still = []
        for p in self.pending_escapes:
            if math.hypot(pos[0] - p["pos"][0], pos[1] - p["pos"][1]) >= ESCAPE_DIST:
                p["eng"]["escape_time"] = t - p["start"]
            else:
                still.append(p)
        self.pending_escapes = still
        if self.keep_trace:
            self.trace.append({
                "v_nom": float(info["v_nom"]), "w_nom": float(np.clip(float(action[1]), -self.shield.cfg.omega_max, self.shield.cfg.omega_max)),
                "v_exec": float(u[0]), "w_exec": float(u[1]), "x": pos[0], "y": pos[1], "theta": float(state[2]),
                "immobile": immobile, "stalled": stalled, "fired": fired, "kind": kind, "method": info["method"],
            })
        return u, info

    def _close_open(self, t: int) -> None:
        if self._open is not None:
            self._open["end"] = self._last_fire_step if self._last_fire_step is not None else t
            self._open.setdefault("escape_time", None)
            self.engagements.append(self._open)
            self._open = None

    def finish(self) -> dict:
        self._close_open(self.n_steps)
        for p in self.pending_escapes:
            p["eng"].setdefault("escape_time", None)
        self.pending_escapes = []
        return {
            "n_steps": self.n_steps, "n_immobile": self.n_immobile, "n_stalled": self.n_stalled,
            "longest_stall_run": self.longest_stall, "kinds": dict(self.kinds),
            "n_engagements": len(self.engagements), "n_cycles": int(sum(e["cycle"] for e in self.engagements)),
            "escape_times": [e["escape_time"] for e in self.engagements],
            "recoveries": int(getattr(self.shield, "n_recoveries", 0)),
            "cg_b4": {"checked": self.cg_b4_checked, "failures": self.cg_b4_failures} if self.check_pred is not None else None,
        }


# ----------------------------------------------------------------------------
# models, opponents, evaluation
# ----------------------------------------------------------------------------
_MODELS: dict[str, object] = {}


def _model(path: pathlib.Path, device: str):
    from stable_baselines3 import PPO

    key = str(path)
    if key not in _MODELS:
        _MODELS[key] = PPO.load(path, device=device)
    return _MODELS[key]


def expected_width(policy: str) -> int:
    return obs_dim(EnvConfig()) + 3 * int(POLICIES[policy]["shield_obs"])


def check_width(policy: str, model) -> int:
    """CG-B2-2: the checkpoint's input width must be the width this runner feeds it."""
    got = int(model.observation_space.shape[0])
    want = expected_width(policy)
    if got != want:
        raise RuntimeError(f"CG-B2-2: {policy} expects {got} inputs, runner would feed {want}")
    return got


def opponent_factory(name: str, device: str):
    if name == "GREEDY":
        return GreedyEvader
    if name == "ATK4-45":
        from pe2.evade import D_POS, StructuredEvaderPolicy

        cfg = EnvConfig()
        model = _model(ATK_CKPT, device)
        return lambda: StructuredEvaderPolicy(model, cfg, hold=EVADER_HOLD, d_pos=D_POS)
    raise ValueError(name)


def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def provenance(device: str) -> dict:
    import stable_baselines3
    import torch

    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        commit = "unknown"
    return {"git_commit": commit, "sb3_version": stable_baselines3.__version__, "torch_threads": torch.get_num_threads(),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"), "device": device}


def evaluate_cell(name: str, device: str = "cpu", n_episodes: int = GATE_N, seed0: int = GATE_SEED0) -> dict:
    """One cell: S17's evaluation path with a fresh, seeded, recording shield per episode."""
    from pe2.structured import SearchMDP

    spec = CELLS[name]
    pspec = POLICIES[spec["policy"]]
    cfg = EnvConfig()
    model = _model(pspec["ckpt"], device)
    width = check_width(spec["policy"], model)
    factory = shield_factory(spec["setting"], cfg)
    mdp = SearchMDP(cfg, opponent_factory(spec["opponent"], device), hold=HOLD, shield=None, shield_obs=pspec["shield_obs"], escape_obs=False)
    logs, eps = [], []
    traces = []
    pred_checker = PredictiveShield(cfg, horizon=PRED_HORIZON) if spec["setting"].startswith("PRED-RECF") else None
    sample_shield = None
    for i in range(n_episodes):
        seed = seed0 + i
        sh = factory(seed)
        sample_shield = sample_shield or sh
        rec = EpisodeLog(sh, keep_trace=i < TRACE_EPISODES, check_pred=pred_checker if i < CG_B4_EPISODES else None)
        mdp.shield = rec
        log = mdp.play_episode(model.predict, seed)
        logs.append(log)
        eps.append(rec.finish() | {"seed": seed, "outcome": log["outcome"], "steps": log["steps"], "violations": log["violations"]})
        if rec.keep_trace:
            traces.append(rec.trace)
    sh_sum = summarize_shield(logs)
    n_steps = sum(e["n_steps"] for e in eps)
    esc = [t for e in eps for t in e["escape_times"]]
    kinds = dict.fromkeys(KINDS, 0)
    for e in eps:
        for k, v in e["kinds"].items():
            kinds[k] = kinds.get(k, 0) + v
    cg_b4 = None
    if pred_checker is not None:
        cg_b4 = {"checked": int(sum(e["cg_b4"]["checked"] for e in eps if e["cg_b4"])),
                 "failures": int(sum(e["cg_b4"]["failures"] for e in eps if e["cg_b4"]))}
    return {
        "name": name, "policy": spec["policy"], "opponent": spec["opponent"], "setting": spec["setting"], "group": spec["group"],
        "n_episodes": n_episodes, "seed0": seed0, "hold": HOLD, "obs_width": width, "ckpt": str(pspec["ckpt"]), "ckpt_sha256": _sha256(pspec["ckpt"]),
        "shield": shield_spec(sample_shield),
        "capture_ind": [lg["outcome"] == "capture" for lg in logs],
        "deadlock_ind": [bool(lg["shield"]["deadlock"]) for lg in logs],
        "capture_rate": float(np.mean([lg["outcome"] == "capture" for lg in logs])),
        "deadlock_episode_rate": float(sh_sum["deadlock_episode_rate"]),
        "immobile_step_rate": float(sh_sum["immobile_step_rate"]),
        "stall_step_rate": float(sh_sum["stall_step_rate"]),
        "intervention_rate": float(sh_sum["intervention_rate"]),
        "backup_rate": float(sh_sum["backup_rate"]),
        "n_steps": int(n_steps),
        "violation_steps": int(sh_sum["violation_steps"]),
        "longest_stall_runs": [e["longest_stall_run"] for e in eps],
        "stall_run_rates": {str(k): float(np.mean([e["longest_stall_run"] >= k for e in eps])) for k in STALL_THRESHOLDS},
        "recoveries": int(sum(e["recoveries"] for e in eps)),
        "recovery_kinds": kinds,
        "n_engagements": int(sum(e["n_engagements"] for e in eps)),
        "n_cycles": int(sum(e["n_cycles"] for e in eps)),
        "engagements_per_episode": [e["n_engagements"] for e in eps],
        "escape_times": esc,  # None = censored at episode end
        "cg_b4": cg_b4,
        "traces": traces,
        "provenance": provenance(device),
    }


# ----------------------------------------------------------------------------
# records, guards
# ----------------------------------------------------------------------------
def _write_json(path: pathlib.Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=float))
    os.replace(tmp, path)


def record_status(rec: dict) -> tuple[str, str]:
    name = rec.get("name")
    if name not in CELLS:
        return "INVALID", f"unknown cell {name!r}"
    if rec.get("verdict") == UNDETERMINED:
        return UNDETERMINED, rec.get("reason", "")
    spec = CELLS[name]
    for k in ("policy", "opponent", "setting"):
        if rec.get(k) != spec[k]:
            return "INVALID", f"{k} mismatch"
    if rec.get("n_episodes") != GATE_N or rec.get("seed0") != GATE_SEED0:
        return "INVALID", "n/seed0 mismatch"
    if len(rec.get("capture_ind", [])) != GATE_N or len(rec.get("deadlock_ind", [])) != GATE_N:
        return "INVALID", "indicator length"
    if rec.get("obs_width") != expected_width(spec["policy"]):
        return "INVALID", "obs width"
    return "MEASURABLE", ""


def run_cells(names: list[str] | None, device: str = "cpu", n_episodes: int = GATE_N) -> None:
    names = names or CELL_ORDER
    bad = [n for n in names if n not in CELLS]
    if bad:
        raise SystemExit(f"unknown cells: {bad}")
    cells_dir().mkdir(parents=True, exist_ok=True)
    for name in names:
        path = cells_dir() / f"{name}.json"
        if path.exists():
            print(f"[{name}] record exists — stop rule: not re-run", flush=True)
            continue
        print(f"[{name}] running N = {n_episodes}", flush=True)
        try:
            rec = evaluate_cell(name, device, n_episodes)
        except Exception as exc:  # noqa: BLE001 — the crash policy: UNDETERMINED, never re-run
            spec = CELLS[name]
            rec = {"name": name, **spec, "verdict": UNDETERMINED, "reason": f"{type(exc).__name__}: {exc}", "provenance": provenance(device)}
            _write_json(path, rec)
            print(f"[{name}] UNDETERMINED: {rec['reason']}", flush=True)
            raise
        _write_json(path, rec)
        print(f"[{name}] capture {rec['capture_rate']:.3f} deadlock {rec['deadlock_episode_rate']:.3f} immobile {rec['immobile_step_rate']:.3f} "
              f"firings {rec['recoveries']} eng {rec['n_engagements']} cycles {rec['n_cycles']} viol {rec['violation_steps']}", flush=True)


def scan_records() -> tuple[dict, dict, dict]:
    measurable, rejected, undetermined = {}, {}, {}
    if not cells_dir().exists():
        return measurable, rejected, undetermined
    for p in sorted(cells_dir().glob("*.json")):
        rec = json.loads(p.read_text())
        st, why = record_status(rec)
        if st == "MEASURABLE":
            measurable[rec["name"]] = rec
        elif st == UNDETERMINED:
            undetermined[rec["name"]] = why
        else:
            rejected[p.name] = why
    return measurable, rejected, undetermined


# ----------------------------------------------------------------------------
# construct gates and hypotheses (spec sections 2–4)
# ----------------------------------------------------------------------------
def identity_pairs() -> dict[str, tuple[pathlib.Path, str]]:
    """CG-B2-1: our cell -> (S17 arm record, opponent key inside it)."""
    pairs = {}
    for p in ("BASE-60", "BASE-61", "BASE-62"):
        for opp in OPPONENTS:
            pairs[cell_name(p, opp, "PLAIN")] = (OUT / f"{S17_PREFIX}_arm_{p}.json", opp)
            pairs[cell_name(p, opp, "RECF-20")] = (OUT / f"{S17_PREFIX}_arm_RECF-{p}.json", opp)
    return pairs


def cg_b2_1(records: dict) -> dict:
    out, offenders, missing = {}, [], []
    for name, (path, opp) in identity_pairs().items():
        if name not in records:
            missing.append(name)
            continue
        if not path.exists():
            offenders.append(f"{name}: S17 record {path.name} absent")
            continue
        stored = json.loads(path.read_text())["final"]["gate"][opp]
        same_c = list(map(bool, records[name]["capture_ind"])) == list(map(bool, stored["capture_ind"]))
        same_d = list(map(bool, records[name]["deadlock_ind"])) == list(map(bool, stored["deadlock_ind"]))
        out[name] = {"stored": str(path), "capture_identical": same_c, "deadlock_identical": same_d}
        if not (same_c and same_d):
            offenders.append(name)
    return {"gate": "CG-B2-1 (bit identity with S17 on the 12 shared cells)", "verdict": "INCOMPLETE" if missing else ("REFUSE" if offenders else "PASS"),
            "pairs": out, "offenders": offenders, "missing": missing}


def cg_b2_3() -> dict:
    expected = [gifs_dir() / f"{cell_name(p, o, s)}_{seed}.gif" for p, o, s, seeds in GIF_CELLS for seed in seeds]
    missing = [str(p) for p in expected if not p.exists()]
    return {"gate": "CG-B2-3 (8 GIFs written and watched before the summary)", "verdict": "REFUSE" if missing else "PASS", "missing": missing, "n": len(expected)}


def cg_b4_1(records: dict) -> dict:
    rows = {n: r["cg_b4"] for n, r in records.items() if r["setting"] == "PRED-RECF-20"}
    bad = [n for n, c in rows.items() if c is None or c["failures"] > 0]
    return {"gate": "CG-B4-1 (every fired command accepted by a fresh PredictiveShield)", "verdict": "REFUSE" if bad else ("INCOMPLETE" if len(rows) < 8 else "PASS"), "rows": rows, "offenders": bad}


def hsafe(records: dict) -> dict:
    off = {n: r["violation_steps"] for n, r in records.items() if r["violation_steps"]}
    return {"verdict": "FAILS" if off else "HOLDS", "offenders": off, "cells": len(records)}


def _rec(records: dict, p: str, o: str, s: str) -> dict | None:
    return records.get(cell_name(p, o, s))


def _paired(records, p, o, s_a, s_b, key="capture_ind") -> dict | None:
    a, b = _rec(records, p, o, s_a), _rec(records, p, o, s_b)
    if a is None or b is None:
        return None
    return paired_indicator_test(a[key], b[key], P_GATE)


def h_b2_1(records: dict) -> dict:
    """Sensitivity: capture never significantly below PLAIN for N <= 40; significantly above on >= 4/5 for N in {10, 20};
    deadlock < 0.05 for N <= 20 on 5/5."""
    below, above, dead = {}, {}, {}
    for n in RECF_NS:
        s = f"RECF-{n}"
        for p in B2_POLICIES:
            t_up = _paired(records, p, CAPTURE_OPPONENT, s, "PLAIN")
            t_dn = _paired(records, p, CAPTURE_OPPONENT, "PLAIN", s)
            if t_up is None:
                continue
            above[f"{p}@{s}"] = t_up["pass"]
            below[f"{p}@{s}"] = t_dn["pass"]
            r = _rec(records, p, DEADLOCK_OPPONENT, s)
            if r is not None:
                dead[f"{p}@{s}"] = r["deadlock_episode_rate"]
    complete = len(above) == len(RECF_NS) * len(B2_POLICIES) and len(dead) == len(above)
    no_drop = all(not v for k, v in below.items() if int(k.split("RECF-")[1]) <= 40)
    up_ok = all(sum(above[f"{p}@RECF-{n}"] for p in B2_POLICIES) >= 4 for n in (10, 20)) if complete else False
    dead_ok = all(dead[f"{p}@RECF-{n}"] < DEADLOCK_BAR for n in RECF_NS if n <= 20 for p in B2_POLICIES) if complete else False
    return {"hypothesis": "H-B2-1", "verdict": "INCOMPLETE" if not complete else ("PASS" if (no_drop and up_ok and dead_ok) else "FAIL"),
            "no_significant_drop_N<=40": no_drop, "significant_gain_4of5_at_10_20": up_ok, "deadlock_below_bar_N<=20": dead_ok,
            "above": above, "below": below, "deadlock": dead}


def h_b2_2(records: dict) -> dict:
    """Bound: >= 95 % of RECF-20 triggers escape within 20 + T_TURN on every cell; median escape < N + T_TURN generally."""
    per, medians = {}, {}
    for p in B2_POLICIES:
        for o in OPPONENTS:
            for n in RECF_NS:
                r = _rec(records, p, o, f"RECF-{n}")
                if r is None:
                    continue
                esc = r["escape_times"]
                if not esc:
                    continue
                within = sum(1 for t in esc if t is not None and t <= n + T_TURN) / len(esc)
                med = float(np.median([t if t is not None else np.inf for t in esc]))
                medians[f"{p}_{o}@RECF-{n}"] = {"median": med, "bound": n + T_TURN, "n_triggers": len(esc), "censored": sum(t is None for t in esc), "within_bound": within}
                if n == 20:
                    per[f"{p}_{o}"] = within
    complete = len(per) == len(B2_POLICIES) * len(OPPONENTS)
    ok = complete and all(v >= 0.95 for v in per.values())
    return {"hypothesis": "H-B2-2", "verdict": "INCOMPLETE" if not complete else ("PASS" if ok else "FAIL"), "recf20_within_bound": per, "bound_recf20": 20 + T_TURN, "all": medians}


def h_b2_3(records: dict) -> dict:
    rows = {}
    for p in B2_POLICIES:
        for o in OPPONENTS:
            r = _rec(records, p, o, "RECF-20")
            if r is not None:
                rows[f"{p}_{o}"] = {"immobile": r["immobile_step_rate"], "engagements": r["n_engagements"], "cycles": r["n_cycles"],
                                     "cycle_rate": (r["n_cycles"] / r["n_engagements"]) if r["n_engagements"] else None}
    complete = len(rows) == len(B2_POLICIES) * len(OPPONENTS)
    ok = complete and all(v["immobile"] <= IMMOBILE_BAR_RECF20 for v in rows.values())
    return {"hypothesis": "H-B2-3", "verdict": "INCOMPLETE" if not complete else ("PASS" if ok else "FAIL"), "bar": IMMOBILE_BAR_RECF20, "rows": rows}


def h_b2_4(records: dict) -> dict:
    """Baseline: PERT-20 clears deadlock on >= 4/5; RECF-20 beats PERT-20 on capture on >= 3/5 and firings on 5/5."""
    clears, beats, fewer = {}, {}, {}
    for p in B2_POLICIES:
        r = _rec(records, p, DEADLOCK_OPPONENT, "PERT-20")
        if r is not None:
            clears[p] = r["deadlock_episode_rate"] < DEADLOCK_BAR
        t = _paired(records, p, CAPTURE_OPPONENT, "RECF-20", "PERT-20")
        if t is not None:
            beats[p] = t["pass"]
        a, b = _rec(records, p, CAPTURE_OPPONENT, "RECF-20"), _rec(records, p, CAPTURE_OPPONENT, "PERT-20")
        if a is not None and b is not None:
            fewer[p] = a["recoveries"] < b["recoveries"]
    complete = len(clears) == len(beats) == len(fewer) == len(B2_POLICIES)
    ok = complete and sum(clears.values()) >= 4 and sum(beats.values()) >= 3 and all(fewer.values())
    return {"hypothesis": "H-B2-4", "verdict": "INCOMPLETE" if not complete else ("PASS" if ok else "FAIL"), "pert_clears": clears, "recf_beats_capture": beats, "recf_fewer_firings": fewer}


def h_b2_5(records: dict) -> dict:
    rows = {}
    for p in ("STR14", "RC19"):
        t = _paired(records, p, CAPTURE_OPPONENT, "RECF-20", "PLAIN")
        a, b = _rec(records, p, DEADLOCK_OPPONENT, "PLAIN"), _rec(records, p, DEADLOCK_OPPONENT, "RECF-20")
        if t is None or a is None or b is None:
            continue
        rel = 1.0 - b["immobile_step_rate"] / a["immobile_step_rate"] if a["immobile_step_rate"] > 0 else None
        rows[p] = {"capture_plain": _rec(records, p, CAPTURE_OPPONENT, "PLAIN")["capture_rate"], "capture_recf20": _rec(records, p, CAPTURE_OPPONENT, "RECF-20")["capture_rate"],
                   "p": t["p"], "capture_up": t["pass"], "immobile_plain": a["immobile_step_rate"], "immobile_recf20": b["immobile_step_rate"], "relative_cut": rel,
                   "pass": bool(t["pass"] and rel is not None and rel >= 0.80)}
    complete = len(rows) == 2
    return {"hypothesis": "H-B2-5", "verdict": "INCOMPLETE" if not complete else ("PASS" if all(r["pass"] for r in rows.values()) else "FAIL"), "rows": rows}


def h_b3_1(records: dict) -> dict:
    rows = {p: _rec(records, p, DEADLOCK_OPPONENT, "PRED-PLAIN")["deadlock_episode_rate"] for p in B34_POLICIES if _rec(records, p, DEADLOCK_OPPONENT, "PRED-PLAIN")}
    complete = len(rows) == len(B34_POLICIES)
    n = sum(v >= DEADLOCK_BAR for v in rows.values())
    return {"hypothesis": "H-B3-1", "verdict": "INCOMPLETE" if not complete else ("PASS" if n >= 3 else "FAIL"), "deadlock_pred_plain": rows, "n_at_or_above_bar": n}


def h_b4_1(records: dict) -> dict:
    dead, no_drop, up = {}, {}, {}
    for p in B34_POLICIES:
        r = _rec(records, p, DEADLOCK_OPPONENT, "PRED-RECF-20")
        if r is not None:
            dead[p] = r["deadlock_episode_rate"]
        t_dn = _paired(records, p, CAPTURE_OPPONENT, "PRED-PLAIN", "PRED-RECF-20")
        t_up = _paired(records, p, CAPTURE_OPPONENT, "PRED-RECF-20", "PRED-PLAIN")
        if t_dn is not None:
            no_drop[p] = not t_dn["pass"]
            up[p] = t_up["pass"]
    complete = len(dead) == len(no_drop) == len(B34_POLICIES)
    ok = complete and all(v < DEADLOCK_BAR for v in dead.values()) and all(no_drop.values()) and sum(up.values()) >= 2
    return {"hypothesis": "H-B4-1", "verdict": "INCOMPLETE" if not complete else ("PASS" if ok else "FAIL"), "deadlock": dead, "no_significant_drop": no_drop, "significant_gain": up}


# ----------------------------------------------------------------------------
# GIFs (CG-B2-3) and the summary
# ----------------------------------------------------------------------------
def gif_recorder_class():
    """`SearchMDP` subclass that renders every `every`-th step (S18's recorder);
    built lazily so matplotlib is imported only when GIFs are written. Shared
    with `paper_b_geometry.py`."""
    from pe2.render import FovOverlay, Renderer, ShieldOverlay, Trails
    from pe2.structured import SearchMDP

    class GifRecorder(SearchMDP):
        def __init__(self, *a, label: str, every: int, **kw):
            super().__init__(*a, **kw)
            self.label, self.every, self.frames = label, every, []
            self._r = Renderer(self.cfg, dpi=80, figsize=(5.5, 5.0))
            self._t = Trails()
            self._live = {"nominal": None, "executed": None, "info": None}
            self._ov = [FovOverlay("pursuer"), ShieldOverlay("pursuer", None, self._live)]
            self.trace = []

        def _draw(self, title):
            self._ov[1].shield = self.shield.shield if hasattr(self.shield, "shield") else self.shield
            self.frames.append(self._r.draw(self.inner.env, trails=self._t, overlays=self._ov, title_prefix=title))

        def _step_inner(self, action, branch="law"):
            cfg, env = self.cfg, self.inner.env
            self._t.append("pursuer", *env.pursuer_pose[:2])
            self._t.append("evader", *env.evader_pose[:2])
            a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
            nominal = ((float(a[0]) + 1.0) / 2.0 * cfg.v_max, float(a[1]) * cfg.omega_max)
            out = super()._step_inner(action, branch)
            v, w = self.trace[-1][2], self.trace[-1][3]
            self._live.update(nominal=nominal, executed=((v + 1.0) / 2.0 * cfg.v_max, w * cfg.omega_max), info=self._last_sinfo)
            if env.current_step % self.every == 0:
                self._draw(f"[{branch}] {self.label}")
            return out

        def finish(self, outcome, step, hold_frames=10):
            for _ in range(hold_frames):
                self._draw(f"[{outcome.upper()} @ {step}] {self.label}")
            self._r.close()

    return GifRecorder


def write_gifs(device: str = "cpu", fps: float = 15.0, every: int = 4) -> list[pathlib.Path]:
    from pe2.render import save_gif

    Recorder = gif_recorder_class()
    measurable, _r, _u = scan_records()
    gifs_dir().mkdir(parents=True, exist_ok=True)
    written = []
    cfg = EnvConfig()
    for p, o, s, seeds in GIF_CELLS:
        name = cell_name(p, o, s)
        if name not in measurable:
            raise SystemExit(f"CG-B2-3 refuses: no measurable record for {name}")
        for seed in seeds:
            path = gifs_dir() / f"{name}_{seed}.gif"
            if path.exists():
                continue
            model = _model(POLICIES[p]["ckpt"], device)
            check_width(p, model)
            rec = Recorder(cfg, opponent_factory(o, device), label=f"{p} vs {o} · {s} · ep {seed}", every=every, hold=HOLD, shield=None, shield_obs=POLICIES[p]["shield_obs"])
            rec.shield = EpisodeLog(shield_factory(s, cfg)(seed), keep_trace=False)
            log = rec.play_episode(model.predict, seed)
            rec.finish(log["outcome"], log["steps"])
            save_gif(rec.frames, str(path), fps=fps)
            print(f"wrote {path} ({len(rec.frames)} frames; {log['outcome']} @ {log['steps']}; violations {log['violations']})", flush=True)
            written.append(path)
    return written


def write_figure(records: dict, path: pathlib.Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    xs = [0, *RECF_NS]
    labels = ["PLAIN", *[str(n) for n in RECF_NS]]
    for p in B2_POLICIES:
        cap = [records.get(cell_name(p, CAPTURE_OPPONENT, s), {}).get("capture_rate", np.nan) for s in ("PLAIN", *(f"RECF-{n}" for n in RECF_NS))]
        dl = [records.get(cell_name(p, DEADLOCK_OPPONENT, s), {}).get("deadlock_episode_rate", np.nan) for s in ("PLAIN", *(f"RECF-{n}" for n in RECF_NS))]
        axes[0].plot(range(len(xs)), cap, marker="o", label=p)
        axes[1].plot(range(len(xs)), dl, marker="o", label=p)
        pert = records.get(cell_name(p, CAPTURE_OPPONENT, "PERT-20"))
        if pert:
            axes[0].scatter([3], [pert["capture_rate"]], marker="x", s=60, color="k")
    for ax, ttl in ((axes[0], f"capture vs {CAPTURE_OPPONENT} (x = PERT-20)"), (axes[1], f"deadlock episodes vs {DEADLOCK_OPPONENT}")):
        ax.set_xticks(range(len(xs)), labels)
        ax.set_xlabel("recovery_after N")
        ax.set_title(ttl, fontsize=9)
        ax.grid(alpha=0.3)
    axes[1].axhline(DEADLOCK_BAR, ls=":", color="k")
    axes[0].legend(fontsize=7)
    esc = [t for p in B2_POLICIES for o in OPPONENTS for t in (records.get(cell_name(p, o, "RECF-20"), {}).get("escape_times") or []) if t is not None]
    if esc:
        axes[2].hist(esc, bins=60, color="#444")
        axes[2].axvline(20 + T_TURN, color="r", ls="--", label=f"N + one turn = {20 + T_TURN}")
        axes[2].set_xlabel("escape time after trigger (steps), RECF-20, all cells")
        axes[2].legend(fontsize=8)
    fig.suptitle("S26 B2–B4 — recovery_after sweep (pre-registered; summary only after CG-B2-1/3 and CG-B4-1)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def summarize() -> dict:
    measurable, rejected, undetermined = scan_records()
    if rejected:
        raise SystemExit(f"REFUSED: invalid records (reject, do not re-run): {rejected}")
    missing = [n for n in CELL_ORDER if n not in measurable and n not in undetermined]
    if missing:
        raise SystemExit(f"REFUSED: {len(missing)} cells missing: {missing[:8]}{'...' if len(missing) > 8 else ''}")
    g1, g3, g4 = cg_b2_1(measurable), cg_b2_3(), cg_b4_1(measurable)
    for g in (g1, g3):
        if g["verdict"] != "PASS":
            raise SystemExit(f"REFUSED by {g['gate']}: {g}")
    hs = hsafe(measurable)
    out = {
        "spec": SPEC, "n_measurable": len(measurable), "undetermined": undetermined,
        "gates": {"CG-B2-1": g1, "CG-B2-3": g3, "CG-B4-1": g4}, "hsafe": hs,
        "hypotheses": {h["hypothesis"]: h for h in (h_b2_1(measurable), h_b2_2(measurable), h_b2_3(measurable), h_b2_4(measurable), h_b2_5(measurable), h_b3_1(measurable), h_b4_1(measurable))},
        "cells": {n: {k: r[k] for k in ("capture_rate", "deadlock_episode_rate", "immobile_step_rate", "stall_step_rate", "recoveries", "recovery_kinds", "n_engagements", "n_cycles", "stall_run_rates", "violation_steps")} for n, r in measurable.items()},
        "b4_included": g4["verdict"] == "PASS",
    }
    if hs["verdict"] != "HOLDS":
        out["direction"] = "H-SAFE FAILS: stop; the paper's safety sentence is wrong."
    _write_json(_round_file("json"), out)
    write_figure(measurable, _round_file("png"))
    for k, h in out["hypotheses"].items():
        print(f"{k}: {h['verdict']}")
    print(f"H-SAFE: {hs['verdict']}; CG-B4-1: {g4['verdict']} (B4 included: {out['b4_included']})")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="*", default=None)
    ap.add_argument("--gifs", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-episodes", type=int, default=GATE_N)
    args = ap.parse_args()
    if args.cells is not None:
        run_cells(args.cells, args.device, args.n_episodes)
    if args.gifs:
        write_gifs(args.device)
    if args.summarize:
        summarize()
    if args.cells is None and not args.gifs and not args.summarize:
        raise SystemExit("nothing to do: pass --cells [names...], --gifs and/or --summarize")


if __name__ == "__main__":
    main()
