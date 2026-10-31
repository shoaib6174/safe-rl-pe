"""Paper B round B1 (S26) — stall geometry and dwell (Prop 1 measured).

Spec: docs/pathway/paper_b_prereg_rounds_2026-11-02.md sections 0, 1, 8 (+A1).
Archives, under pre-registration, what D126 (V1/V2), D127 (V3) and D130 measured
as LOG-ONLY diagnostics.

Cells `{POLICY}_{MODE}`: policies BASE-60/61/62 and RC19 under the plain
`CBFShield` vs `ATK4-45`, N = 100 episodes on stream 3 000 000+, MODE `det`
(deterministic actions) and `stoch` (the SAME seeds, actions sampled from the
policy's Gaussian, torch seeded with the episode seed).

Per episode the recording shield (observation-free) stores every step's pose,
nominal, executed command and tier, and the predict wrapper stores the
decision observation the policy saw. From those, the sampling units of
section 0 are cut: a STALLED STATE at the 40th step of a stall run (and the
140th if still stalled) and a SELF-STOP STATE at the 40th step of an immobile-
not-stalled run. Each is probed (section 1 (a)–(f)):

  tier            the filter's tier when the recorded nominal is re-filtered
  qp              exact weighted projection onto the linearised rows ∩ box
                  (SLSQP, 4 starts) vs the shield's output
  trap_heading    no grid action with v > 0.05 v_max passes the EXACT one-step
                  test at the current pose (41 × 41 grid)
  trap_position   the same at all 72 headings 5° apart
  rot_admissible  some grid action with v <= 0.05 v_max, |ω| >= 0.5 ω_max passes
  p_leave         of 2 000 actions sampled from the policy's Gaussian at the
                  decision observation, held 10 steps through the filter with
                  `pe2.env.pursuer_step`, the fraction that moves the body
                  >= 0.10 m; predicted dwell beyond onset = 10 / p_leave steps

Usage:
  ./venv/bin/python scripts/paper_b_geometry.py --cells BASE-61_det ...
  ./venv/bin/python scripts/paper_b_geometry.py --gifs        (CG-B1-2, before --summarize)
  ./venv/bin/python scripts/paper_b_geometry.py --summarize
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import pathlib

import numpy as np

from pe2.config import EnvConfig
from pe2.env import pursuer_step
from pe2.metrics import DEADLOCK_STEPS
from pe2.shield import MOVE_V_FRAC, STALL_V_FRAC, CBFShield

SPEC = "docs/pathway/paper_b_prereg_rounds_2026-11-02.md"
PREFIX = "s26_b1"
OUT = pathlib.Path("results/derisk")
SEED0 = 3_000_000  # spec section 1: fresh stream
N_EPISODES = 100
HOLD = 10
OPPONENT = "ATK4-45"
POLICIES = ("BASE-60", "BASE-61", "BASE-62", "RC19")
MODES = ("det", "stoch")
K = DEADLOCK_STEPS  # 40: the stalled-state onset
LATE = 140  # the second sample point inside a long run
GRID = 41
HEADINGS = 72
N_SAMPLES = 2000
LEAVE_DIST = 0.10
W_TOL = 1e-3  # QP agreement in the weighted norm
UNDETERMINED = "UNDETERMINED"
BAR_TIER, BAR_TRAP, BAR_POS, BAR_QP, BAR_QP_FWD = 0.90, 0.90, 0.10, 0.99, 0.01
GIF_EVERY = 4

_SWEEP = None


def sweep():
    """The B2 runner's shared pieces (policies, opponent, model cache, GIF recorder)."""
    global _SWEEP
    if _SWEEP is None:
        p = pathlib.Path(__file__).resolve().parent / "paper_b_filter_sweep.py"
        spec = importlib.util.spec_from_file_location("paper_b_filter_sweep", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _SWEEP = mod
    return _SWEEP


def cell_name(policy: str, mode: str) -> str:
    return f"{policy}_{mode}"


CELLS = {cell_name(p, m): {"policy": p, "mode": m} for p in POLICIES for m in MODES}
CELL_ORDER = list(CELLS)


def cells_dir() -> pathlib.Path:
    return OUT / f"{PREFIX}_cells"


def gifs_dir() -> pathlib.Path:
    return OUT / f"{PREFIX}_gifs"


def _round_file(ext: str) -> pathlib.Path:
    return OUT / f"{PREFIX}_round.{ext}"


# ----------------------------------------------------------------------------
# recording (observation-free shield wrapper + predict wrapper)
# ----------------------------------------------------------------------------
class StepStore:
    """Returns the filter's `(u, info)` untouched; keeps per-step rows."""

    def __init__(self, shield: CBFShield, decision: dict):
        self.shield = shield
        self.decision = decision  # {"obs": latest decision observation, "idx": decision count}
        self.rows: list[dict] = []
        self.obstacles = None

    def __getattr__(self, name):
        if name == "shield":
            raise AttributeError(name)
        return getattr(self.shield, name)

    def filter(self, action, state, obstacles, v_max=None):
        u, info = self.shield.filter(action, state, obstacles, v_max)
        if self.obstacles is None:
            self.obstacles = [dict(o) for o in obstacles]
        self.rows.append({
            "pose": [float(state[0]), float(state[1]), float(state[2])],
            "v_nom": float(info["v_nom"]), "w_nom": float(np.clip(float(action[1]), -self.shield.cfg.omega_max, self.shield.cfg.omega_max)),
            "v_exec": float(u[0]), "w_exec": float(u[1]), "method": info["method"],
            "immobile": bool(info["immobile"]), "stalled": bool(info["stalled"]),
            "decision_idx": self.decision["idx"],
        })
        return u, info


def make_predict(model, mode: str, decision: dict, seed: int):
    """`predict(obs, deterministic=...)` as `SearchMDP.play_episode` calls it; the
    wrapper decides the mode itself (stochastic sampling seeded per episode)."""
    import torch

    if mode == "stoch":
        torch.manual_seed(int(seed))

    def predict(obs, deterministic=True):  # noqa: ARG001 — the mode is the cell's, not the caller's
        decision["obs"] = np.asarray(obs, dtype=np.float32).copy()
        decision["idx"] += 1
        decision["history"].append(decision["obs"])
        return model.predict(obs, deterministic=(mode == "det"))

    return predict


def play_recorded(model, cfg: EnvConfig, opp_factory, shield_obs: bool, mode: str, seed: int) -> dict:
    from pe2.structured import SearchMDP

    decision = {"obs": None, "idx": 0, "history": []}
    store = StepStore(CBFShield(cfg), decision)
    mdp = SearchMDP(cfg, opp_factory, hold=HOLD, shield=store, shield_obs=shield_obs, escape_obs=False)
    mdp.trace = []  # (visible, branch, a0, a1) per env step -- appended once per `_step_inner`, i.e. once per filter call
    log = mdp.play_episode(make_predict(model, mode, decision, seed), seed)
    if len(mdp.trace) != len(store.rows):
        raise RuntimeError(f"branch trace ({len(mdp.trace)}) and shield rows ({len(store.rows)}) disagree")
    for r, (visible, branch, _a0, _a1) in zip(store.rows, mdp.trace, strict=True):
        r["branch"] = branch  # "law" (intercept law drove the step) or "search" (the policy's held decision)
        r["visible"] = bool(visible)
    return {"log": log, "rows": store.rows, "obstacles": store.obstacles, "decisions": decision["history"]}


def cut_states(rows: list[dict]) -> list[dict]:
    """Section 0's sampling units from one episode's rows."""
    out = []
    stall_run = immobile_only_run = 0
    for t, r in enumerate(rows):
        stall_run = stall_run + 1 if r["stalled"] else 0
        immobile_only_run = immobile_only_run + 1 if (r["immobile"] and not r["stalled"]) else 0
        if stall_run in (K, LATE):
            out.append({"kind": "stalled", "t": t, "run_at_sample": stall_run})
        if immobile_only_run in (K, LATE):
            out.append({"kind": "self_stop", "t": t, "run_at_sample": immobile_only_run})
    # the length of the run each sample belongs to, and whether it was censored by the episode end
    for s in out:
        key = "stalled" if s["kind"] == "stalled" else "immobile"
        t = s["t"]
        end = t
        while end + 1 < len(rows) and rows[end + 1][key] and (s["kind"] == "stalled" or not rows[end + 1]["stalled"]):
            end += 1
        s["run_length"] = end - (t - s["run_at_sample"] + 1) + 1
        s["censored"] = end == len(rows) - 1
        # A4.5: the dwell the leave probability predicts -- steps from this sample until the body has moved
        # >= LEAVE_DIST from the sampled pose (None = censored at the episode end); the same event p_leave counts
        x0, y0 = rows[t]["pose"][0], rows[t]["pose"][1]
        s["dwell_beyond_sample"] = next((u - t for u in range(t + 1, len(rows)) if math.hypot(rows[u]["pose"][0] - x0, rows[u]["pose"][1] - y0) >= LEAVE_DIST), None)
    return out


# ----------------------------------------------------------------------------
# geometry probes (section 1 (a)–(f))
# ----------------------------------------------------------------------------
def admissible_mask(shield: CBFShield, state, obstacles, U: np.ndarray) -> np.ndarray:
    """The filter's EXACT one-step test, row for row, on a batch of actions."""
    cons = shield.constraints(state, obstacles)
    H0 = np.array([c.h for c in cons])
    H1 = shield.successor_barriers(state, obstacles, U)
    return (H1 >= (1.0 - shield.gamma) * H0).all(axis=1)


def oracle_mask(shield: CBFShield, state, obstacles, U: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """D126's admissibility oracle: `u` is admissible iff `filter(u)` returns `u`
    unchanged (any tier). A superset of `admissible_mask` (the exact tier)."""
    out = np.zeros(len(U), dtype=bool)
    for i, (v, w) in enumerate(U):
        u, _ = shield.filter((float(v), float(w)), state, obstacles, v_max=shield.cfg.v_max)
        out[i] = abs(float(u[0]) - v) <= tol and abs(float(u[1]) - w) <= tol
    return out


COARSE = 21  # the oracle grid (one filter call per point)


def action_grid(cfg: EnvConfig, n: int = GRID) -> np.ndarray:
    v = np.linspace(0.0, cfg.v_max, n)
    w = np.linspace(-cfg.omega_max, cfg.omega_max, n)
    V, W = np.meshgrid(v, w, indexing="ij")
    return np.column_stack([V.ravel(), W.ravel()])


def qp_projection(shield: CBFShield, state, obstacles, v_nom: float, w_nom: float) -> tuple[np.ndarray | None, bool]:
    """Exact weighted projection onto {c0 v + c1 ω ≥ c2} ∩ box (SLSQP, 4 starts).
    Returns (u*, feasible_found)."""
    from scipy.optimize import minimize

    cfg = shield.cfg
    rows = shield._dcbf_rows(shield.constraints(state, obstacles))
    cons = [{"type": "ineq", "fun": (lambda u, a=c0, b=c1, c=c2: a * u[0] + b * u[1] - c)} for c0, c1, c2 in rows]
    bounds = [(0.0, cfg.v_max), (-cfg.omega_max, cfg.omega_max)]
    obj = lambda u: shield.w_v * (u[0] - v_nom) ** 2 + shield.w_omega * (u[1] - w_nom) ** 2  # noqa: E731
    best, best_val = None, np.inf
    for x0 in ([v_nom, w_nom], [0.0, 0.0], [0.0, cfg.omega_max], [0.0, -cfg.omega_max]):
        res = minimize(obj, np.array(x0, dtype=float), method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": 200, "ftol": 1e-12})
        ok = all(c0 * res.x[0] + c1 * res.x[1] - c2 >= -1e-9 for c0, c1, c2 in rows)
        if ok and res.fun < best_val:
            best, best_val = np.array(res.x, dtype=float), float(res.fun)
    return best, best is not None


def leave_probability(model, shield: CBFShield, state, obstacles, obs: np.ndarray, seed: int, n: int = N_SAMPLES) -> dict:
    """Section 1 (f): sample the policy's Gaussian at `obs`, hold each action
    10 steps through the filter from `state`, count the bodies that moved."""
    import torch

    cfg = shield.cfg
    lim = cfg.arena_size / 2.0 - cfg.agent_radius
    g = torch.Generator().manual_seed(int(seed))
    with torch.no_grad():
        dist = model.policy.get_distribution(torch.as_tensor(np.asarray(obs, dtype=np.float32)[None])).distribution
        mean = np.asarray(dist.mean.cpu().numpy(), dtype=float).reshape(-1)
        scale = np.asarray(dist.scale.cpu().numpy(), dtype=float).reshape(-1)
        if scale.shape != mean.shape:  # one shared log_std broadcast over the batch
            scale = np.broadcast_to(scale, mean.shape).copy()
        eps = torch.randn((n, mean.shape[0]), generator=g).numpy()
    acts = np.clip(mean + scale * eps, -1.0, 1.0)
    left = 0
    p0 = np.asarray(state, dtype=float)
    for a in acts:
        v_nom = (float(a[0]) + 1.0) / 2.0 * cfg.v_max
        w_nom = float(a[1]) * cfg.omega_max
        pose = p0.copy()
        for _ in range(HOLD):
            (v, w), _info = shield.filter((v_nom, w_nom), pose, obstacles, v_max=cfg.v_max)
            pose, _c1, _c2 = pursuer_step(pose, float(v), float(w), cfg, obstacles, (-lim, lim), (-lim, lim))
            if math.hypot(pose[0] - p0[0], pose[1] - p0[1]) >= LEAVE_DIST:
                left += 1
                break
    p = left / n
    return {"p_leave": p, "n": n, "mean_action": [float(x) for x in mean], "sigma": [float(x) for x in scale],
            "predicted_dwell_steps": (HOLD / p) if p > 0 else None}


def probe_state(model, shield: CBFShield, cfg: EnvConfig, row: dict, obstacles, obs: np.ndarray, seed: int, with_leave: bool) -> dict:
    """Section 1 (a)–(f) at one recorded state. Admissibility is reported under
    BOTH definitions (A3): `exact` = every DCBF row satisfied on the true
    successor (the projection tier of Prop 1); `oracle` = the filter returns the
    command unchanged (D126's definition; what will actually be executed). The
    trap classification and `rot_admissible` use the oracle; the exact figures
    are reported beside them."""
    state = np.array(row["pose"], dtype=float)
    u_sh, info = shield.filter((row["v_nom"], row["w_nom"]), state, obstacles, v_max=cfg.v_max)
    v_eps = STALL_V_FRAC * cfg.v_max
    U = action_grid(cfg)
    adm = admissible_mask(shield, state, obstacles, U)
    fwd = U[:, 0] > v_eps
    Uc = action_grid(cfg, COARSE)
    orc = oracle_mask(shield, state, obstacles, Uc)
    fwd_c = Uc[:, 0] > v_eps
    rot_c = (Uc[:, 0] == 0.0) & (np.abs(Uc[:, 1]) >= 0.5 * cfg.omega_max)  # a pure turn: v = 0 exactly (spec section 1 (e))
    trap_heading_exact = not bool((adm & fwd).any())
    # A4.6: one rule for both tests -- a heading admits forward motion if the exact 41 x 41 grid or the oracle grid does
    trap_heading = not (bool((orc & fwd_c).any()) or bool((adm & fwd).any()))
    headings_fwd_exact = headings_fwd = 0
    Uq = action_grid(cfg, 11)
    fwd_q = Uq[:, 0] > v_eps
    for k in range(HEADINGS):
        s_k = np.array([state[0], state[1], -math.pi + 2 * math.pi * k / HEADINGS])
        ex = bool((admissible_mask(shield, s_k, obstacles, U) & fwd).any())
        headings_fwd_exact += int(ex)
        # exact ⊆ oracle, so a heading with exact forward motion needs no oracle pass
        headings_fwd += int(ex or bool((oracle_mask(shield, s_k, obstacles, Uq) & fwd_q).any()))
    qp_u, qp_ok = qp_projection(shield, state, obstacles, row["v_nom"], row["w_nom"])
    d_w = None
    if qp_ok:
        d_w = math.sqrt(shield.w_v * (qp_u[0] - u_sh[0]) ** 2 + shield.w_omega * (qp_u[1] - u_sh[1]) ** 2)
    out = {
        "tier": info["method"], "u_shield": [float(u_sh[0]), float(u_sh[1])],
        "qp": {"found": qp_ok, "u": None if qp_u is None else [float(qp_u[0]), float(qp_u[1])], "dist_w": d_w,
               "agree": bool(qp_ok and d_w <= W_TOL), "forward": bool(qp_ok and qp_u[0] > v_eps)},
        "trap_heading": trap_heading, "trap_heading_exact": trap_heading_exact,
        "trap_position": headings_fwd == 0, "trap_position_exact": headings_fwd_exact == 0,
        "headings_with_forward": headings_fwd, "headings_with_forward_exact": headings_fwd_exact,
        "rot_admissible": bool((orc & rot_c).any()), "rot_admissible_exact": bool((adm & (U[:, 0] == 0.0) & (np.abs(U[:, 1]) >= 0.5 * cfg.omega_max)).any()),
        "n_admissible_grid": int(adm.sum()), "n_oracle_grid": int(orc.sum()),
        "admissible_grid": [int(i) for i in np.flatnonzero(adm)] if len(np.flatnonzero(adm)) <= 400 else None,
        "oracle_grid": [int(i) for i in np.flatnonzero(orc)],
    }
    if with_leave:
        out["leave"] = leave_probability(model, shield, state, obstacles, obs, seed)
    return out


# ----------------------------------------------------------------------------
# cells
# ----------------------------------------------------------------------------
def evaluate_cell(name: str, device: str = "cpu", n_episodes: int = N_EPISODES, seed0: int = SEED0) -> dict:
    S = sweep()
    spec = CELLS[name]
    pspec = S.POLICIES[spec["policy"]]
    cfg = EnvConfig()
    model = S._model(pspec["ckpt"], device)
    width = S.check_width(spec["policy"], model)
    opp = S.opponent_factory(OPPONENT, device)
    shield = CBFShield(cfg)
    episodes, states = [], []
    for i in range(n_episodes):
        seed = seed0 + i
        ep = play_recorded(model, cfg, opp, pspec["shield_obs"], spec["mode"], seed)
        rows = ep["rows"]
        n_imm = sum(r["immobile"] for r in rows)
        n_st = sum(r["stalled"] for r in rows)
        samples = cut_states(rows)
        for s in samples:
            r = rows[s["t"]]
            obs = ep["decisions"][r["decision_idx"] - 1] if r["decision_idx"] > 0 else None
            probe = probe_state(model, shield, cfg, r, ep["obstacles"], obs, seed * 10_000 + s["t"], with_leave=obs is not None)
            states.append({"seed": seed, **s, "pose": r["pose"], "v_nom": r["v_nom"], "w_nom": r["w_nom"], "method_at_step": r["method"],
                           "branch": r.get("branch"), "visible": r.get("visible"),
                           "obstacles": ep["obstacles"], "has_decision_obs": obs is not None, **probe})
        episodes.append({"seed": seed, "outcome": ep["log"]["outcome"], "steps": ep["log"]["steps"], "violations": ep["log"]["violations"],
                         "n_immobile": n_imm, "n_stalled": n_st, "n_steps": len(rows),
                         "longest_stall_run": int(ep["log"]["shield"]["longest_stall_run"]), "deadlock": bool(ep["log"]["shield"]["deadlock"]),
                         "stall_runs": _runs(rows, "stalled"), "n_samples": len(samples)})
        print(f"[{name}] ep {i} {ep['log']['outcome']} @ {ep['log']['steps']} immobile {n_imm} stalled {n_st} samples {len(samples)}", flush=True)
    n_steps = sum(e["n_steps"] for e in episodes)
    return {
        "name": name, "policy": spec["policy"], "mode": spec["mode"], "opponent": OPPONENT, "n_episodes": n_episodes, "seed0": seed0,
        "hold": HOLD, "obs_width": width, "ckpt": str(pspec["ckpt"]), "ckpt_sha256": S._sha256(pspec["ckpt"]),
        "deadlock_episode_rate": float(np.mean([e["deadlock"] for e in episodes])),
        "immobile_step_rate": float(sum(e["n_immobile"] for e in episodes) / n_steps),
        "stall_step_rate": float(sum(e["n_stalled"] for e in episodes) / n_steps),
        "self_stop_share_of_immobile": float(1.0 - sum(e["n_stalled"] for e in episodes) / max(1, sum(e["n_immobile"] for e in episodes))),
        "capture_rate": float(np.mean([e["outcome"] == "capture" for e in episodes])),
        "violation_steps": int(sum(e["violations"] for e in episodes)),
        "episodes": episodes, "states": states, "provenance": S.provenance(device),
    }


def _runs(rows: list[dict], key: str) -> list[dict]:
    """Maximal runs of `key`; each carries the branch driving the step at which the run reaches K (A2)."""
    out, run, start = [], 0, None
    for t, r in enumerate(rows):
        if r[key]:
            if run == 0:
                start = t
            run += 1
        elif run:
            out.append({"start": start, "length": run, "censored": False, "branch": rows[min(start + K - 1, t - 1)].get("branch")})
            run = 0
    if run:
        out.append({"start": start, "length": run, "censored": True, "branch": rows[min(start + K - 1, len(rows) - 1)].get("branch")})
    return out


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
    if rec.get("policy") != spec["policy"] or rec.get("mode") != spec["mode"]:
        return "INVALID", "policy/mode mismatch"
    if rec.get("n_episodes") != N_EPISODES or rec.get("seed0") != SEED0 or len(rec.get("episodes", [])) != N_EPISODES:
        return "INVALID", "n/seed0 mismatch"
    if rec.get("obs_width") != sweep().expected_width(spec["policy"]):
        return "INVALID", "obs width"
    return "MEASURABLE", ""


_REAL_OUT = pathlib.Path("results/derisk")


def run_cells(names: list[str] | None, device: str = "cpu", n_episodes: int = N_EPISODES) -> None:
    if n_episodes != N_EPISODES and OUT.resolve() == _REAL_OUT.resolve():
        raise SystemExit(f"REFUSED: N = {n_episodes} against the real record root would write an INVALID record that blocks the cell")
    names = names or CELL_ORDER
    bad = [n for n in names if n not in CELLS]
    if bad:
        raise SystemExit(f"unknown cells: {bad}")
    crashes = 0
    for name in names:
        path = cells_dir() / f"{name}.json"
        if path.exists():
            print(f"[{name}] record exists — stop rule: not re-run", flush=True)
            continue
        try:
            rec = evaluate_cell(name, device, n_episodes)
        except Exception as exc:  # noqa: BLE001 — A4.9: UNDETERMINED, never re-run, the chain goes on
            _write_json(path, {"name": name, **CELLS[name], "verdict": UNDETERMINED, "reason": f"{type(exc).__name__}: {exc}", "provenance": sweep().provenance(device)})
            print(f"[{name}] UNDETERMINED: {type(exc).__name__}: {exc}", flush=True)
            crashes += 1
            if crashes >= 2:
                raise SystemExit("two consecutive crashed cells: aborting the chain") from exc
            continue
        crashes = 0
        _write_json(path, rec)
        print(f"[{name}] deadlock {rec['deadlock_episode_rate']:.3f} immobile {rec['immobile_step_rate']:.3f} stalled {rec['stall_step_rate']:.3f} states {len(rec['states'])} viol {rec['violation_steps']}", flush=True)


def scan_records() -> tuple[dict, dict, dict]:
    measurable, rejected, undetermined = {}, {}, {}
    if not cells_dir().exists():
        return measurable, rejected, undetermined
    S = sweep()
    hashes = {name: S._sha256(spec["ckpt"]) for name, spec in S.POLICIES.items() if spec["ckpt"].exists()}
    for p in sorted(cells_dir().glob("*.json")):
        rec = json.loads(p.read_text())
        st, why = record_status(rec)
        if p.stem != rec.get("name"):
            st, why = "INVALID", f"file name {p.stem} != record name {rec.get('name')!r}"
        elif st == "MEASURABLE" and rec.get("ckpt_sha256") != hashes.get(rec["policy"]):
            st, why = "INVALID", "checkpoint hash differs from the frozen checkpoint"
        if st == "MEASURABLE":
            measurable[rec["name"]] = rec
        elif st == UNDETERMINED:
            undetermined[rec["name"]] = why
        else:
            rejected[p.name] = why
    return measurable, rejected, undetermined


# ----------------------------------------------------------------------------
# hypotheses (section 1)
# ----------------------------------------------------------------------------
def _stalled(rec: dict) -> list[dict]:
    return [s for s in rec["states"] if s["kind"] == "stalled"]


def classification(records: dict) -> dict:
    """Per policy, over the deterministic arm's stalled states (the sampling unit of H-B1-1..3)."""
    out = {}
    for p in POLICIES:
        rec = records.get(cell_name(p, "det"))
        if rec is None:
            continue
        st = _stalled(rec)
        n = len(st)
        if n == 0:
            out[p] = {"n": 0}
            continue
        out[p] = {
            "n": n,
            "exact_tier": sum(s["tier"] == "exact" for s in st) / n,
            "trap_heading": sum(s["trap_heading"] for s in st) / n,
            "trap_position": sum(s["trap_position"] for s in st) / n,
            "pair_trap": sum(s["trap_heading"] and not s["trap_position"] for s in st) / n,
            "rot_admissible": sum(s["rot_admissible"] for s in st) / n,
            # A4.7: the projection characterisation is for the exact tier, so QP agreement is pooled over exact-tier states
            "n_exact": sum(s["tier"] == "exact" for s in st),
            "qp_agree": (sum(s["qp"]["agree"] for s in st if s["tier"] == "exact") / max(1, sum(s["tier"] == "exact" for s in st))),
            "qp_forward": (sum(s["qp"]["forward"] for s in st if s["tier"] == "exact") / max(1, sum(s["tier"] == "exact" for s in st))),
            "n_self_stop": sum(s["kind"] == "self_stop" for s in rec["states"]),
            "law_branch_share": sum(s.get("branch") == "law" for s in st) / n,  # A2: law pins (D142) vs policy stalls
        }
    return out


def h_b1_1(cls: dict) -> dict:
    rows = {p: c.get("exact_tier") for p, c in cls.items() if c.get("n")}
    pooled = _pooled(cls, "exact_tier")
    return {"hypothesis": "H-B1-1", "verdict": "INCOMPLETE" if len(cls) < len(POLICIES) else ("PASS" if pooled >= BAR_TIER else "FAIL"), "pooled": pooled, "per_policy": rows}


def h_b1_2(cls: dict) -> dict:
    th, tp = _pooled(cls, "trap_heading"), _pooled(cls, "trap_position")
    return {"hypothesis": "H-B1-2", "verdict": "INCOMPLETE" if len(cls) < len(POLICIES) else ("PASS" if (th >= BAR_TRAP and tp < BAR_POS) else "FAIL"),
            "trap_heading_pooled": th, "trap_position_pooled": tp, "per_policy": {p: {k: c.get(k) for k in ("trap_heading", "trap_position", "pair_trap", "n")} for p, c in cls.items()}}


def h_b1_3(cls: dict) -> dict:
    ag, fw = _pooled(cls, "qp_agree"), _pooled(cls, "qp_forward")
    return {"hypothesis": "H-B1-3", "verdict": "INCOMPLETE" if len(cls) < len(POLICIES) else ("PASS" if (ag >= BAR_QP and fw <= BAR_QP_FWD) else "FAIL"), "agree_pooled": ag, "forward_pooled": fw}


def _pooled(cls: dict, key: str) -> float:
    nkey = "n_exact" if key.startswith("qp_") else "n"  # A4.7
    num = sum(c[key] * c[nkey] for c in cls.values() if c.get(nkey))
    den = sum(c[nkey] for c in cls.values() if c.get(nkey))
    return num / den if den else float("nan")


def h_b1_4(records: dict) -> dict:
    """Clause 1: stochastic deadlock rate >= 0.5 x deterministic on >= 3/4 (a deterministic rate of 0 passes trivially).
    Clause 2 (A1, A2, A4.1, A4.5): over the STOCHASTIC arm's own search-branch stall ONSETS (run_at_sample == K), the
    median observed dwell beyond the sample (steps until the body moved >= 0.10 m; censored at the episode end counted
    at the censored length) is within a factor 2 of the median predicted 10/p_leave at those same states, on >= 3/4."""
    c1, c2, detail = {}, {}, {}
    for p in POLICIES:
        d, s = records.get(cell_name(p, "det")), records.get(cell_name(p, "stoch"))
        if d is None or s is None:
            continue
        c1[p] = True if d["deadlock_episode_rate"] == 0 else bool(s["deadlock_episode_rate"] >= 0.5 * d["deadlock_episode_rate"])
        onsets = [st for st in _stalled(s) if st.get("run_at_sample") == K and st.get("branch", "search") == "search" and st.get("leave")]
        obs = []
        for st in onsets:
            dw = st.get("dwell_beyond_sample")
            obs.append(dw if dw is not None else (st["run_length"] - st["run_at_sample"]))  # censored: at least the rest of the run
        pred = [st["leave"]["predicted_dwell_steps"] for st in onsets]
        pred_f = [np.inf if x is None else x for x in pred]
        mo = float(np.median(obs)) if obs else None
        mp = float(np.median(pred_f)) if pred_f else None
        ok = None
        if mo is not None and mp is not None:
            ok = bool(0.5 * mp <= mo <= 2.0 * mp) if np.isfinite(mp) else None
        c2[p] = ok
        detail[p] = {"deadlock_det": d["deadlock_episode_rate"], "deadlock_stoch": s["deadlock_episode_rate"], "n_onsets": len(onsets),
                     "median_observed_beyond_onset": mo, "median_predicted": None if mp is None or not np.isfinite(mp) else mp,
                     "predicted_infinite_share": float(np.mean([not np.isfinite(x) for x in pred_f])) if pred_f else None,
                     "censored_share": float(np.mean([st.get("dwell_beyond_sample") is None for st in onsets])) if onsets else None,
                     "law_branch_onsets": sum(1 for st in _stalled(s) if st.get("run_at_sample") == K and st.get("branch") == "law")}
    complete = len(c1) == len(POLICIES)
    n1 = sum(bool(v) for v in c1.values())
    n2 = sum(bool(v) for v in c2.values())
    return {"hypothesis": "H-B1-4", "verdict": "INCOMPLETE" if not complete else ("PASS" if (n1 >= 3 and n2 >= 3) else "FAIL"),
            "clause1_pass": c1, "clause2_pass": c2, "n1": n1, "n2": n2, "detail": detail}


def hsafe(records: dict) -> dict:
    off = {n: r["violation_steps"] for n, r in records.items() if r["violation_steps"]}
    return {"verdict": "FAILS" if off else "HOLDS", "offenders": off}


def cg_b1_2() -> dict:
    missing = [str(gifs_dir() / f"{p}_det.gif") for p in POLICIES if not (gifs_dir() / f"{p}_det.gif").exists()]
    return {"gate": "CG-B1-2 (4 GIFs watched before the summary)", "verdict": "REFUSE" if missing else "PASS", "missing": missing}


def write_gifs(device: str = "cpu", fps: float = 15.0) -> list[pathlib.Path]:
    """One deadlock episode per policy (deterministic arm): the first recorded
    episode whose longest stall run >= K."""
    from pe2.render import save_gif

    S = sweep()
    Recorder = S.gif_recorder_class()
    measurable, _r, _u = scan_records()
    gifs_dir().mkdir(parents=True, exist_ok=True)
    cfg = EnvConfig()
    written = []
    for p in POLICIES:
        rec = measurable.get(cell_name(p, "det"))
        if rec is None:
            raise SystemExit(f"CG-B1-2 refuses: no measurable record for {cell_name(p, 'det')}")
        path = gifs_dir() / f"{p}_det.gif"
        if path.exists():
            continue
        ep = next((e for e in rec["episodes"] if e["deadlock"]), None) or max(rec["episodes"], key=lambda e: e["longest_stall_run"])  # A4.8
        model = S._model(S.POLICIES[p]["ckpt"], device)
        r = Recorder(cfg, S.opponent_factory(OPPONENT, device), label=f"{p} vs {OPPONENT} · plain CBF · ep {ep['seed']}", every=GIF_EVERY, hold=HOLD, shield=CBFShield(cfg), shield_obs=S.POLICIES[p]["shield_obs"])
        log = r.play_episode(model.predict, ep["seed"])
        r.finish(log["outcome"], log["steps"])
        save_gif(r.frames, str(path), fps=fps)
        print(f"wrote {path} ({len(r.frames)} frames; {log['outcome']} @ {log['steps']})", flush=True)
        written.append(path)
    return written


def write_figure(records: dict, cls: dict, h4: dict, path: pathlib.Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    # (1) one stalled state's admissible set
    st = next((s for p in POLICIES for s in _stalled(records.get(cell_name(p, "det"), {"states": []})) if s.get("admissible_grid") is not None), None)
    cfg = EnvConfig()
    U = action_grid(cfg)
    if st is not None:
        adm = np.zeros(len(U), dtype=bool)
        adm[st["admissible_grid"]] = True
        axes[0].scatter(U[~adm, 1], U[~adm, 0], s=4, color="#ddd", label="inadmissible")
        axes[0].scatter(U[adm, 1], U[adm, 0], s=6, color="#2a9d8f", label="admissible (exact one-step test)")
        axes[0].scatter([st["w_nom"]], [st["v_nom"]], marker="*", s=140, color="k", label="policy's nominal")
        axes[0].scatter([st["u_shield"][1]], [st["u_shield"][0]], marker="x", s=80, color="r", label="filter output")
        axes[0].set_xlabel("ω (rad/s)")
        axes[0].set_ylabel("v (m/s)")
        axes[0].set_title(f"stalled state ({st['kind']}, seed {st['seed']}, step {st['t']}): trap at heading = {st['trap_heading']}, position trap = {st['trap_position']}", fontsize=8)
        axes[0].legend(fontsize=7)
    # (2) classification
    keys = ("exact_tier", "trap_heading", "trap_position", "pair_trap", "rot_admissible", "qp_agree")
    xs = np.arange(len(keys))
    for i, p in enumerate(POLICIES):
        c = cls.get(p, {})
        if c.get("n"):
            axes[1].bar(xs + 0.2 * i - 0.3, [c[k] for k in keys], width=0.2, label=f"{p} (n={c['n']})")
    axes[1].set_xticks(xs, keys, rotation=30, fontsize=8)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("stalled-state classification (deterministic arm)", fontsize=9)
    axes[1].legend(fontsize=7)
    # (3) dwell
    for p in POLICIES:
        d = h4["detail"].get(p)
        if d and d["median_predicted"] is not None and d["median_observed_beyond_onset"] is not None:
            axes[2].scatter([d["median_predicted"]], [d["median_observed_beyond_onset"]], s=60, label=p)
    lim = [1, 2000]
    axes[2].plot(lim, lim, "k--", lw=0.8)
    axes[2].plot(lim, [2 * x for x in lim], "k:", lw=0.6)
    axes[2].plot(lim, [0.5 * x for x in lim], "k:", lw=0.6)
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("predicted dwell beyond onset, 10 / p_leave (steps)")
    axes[2].set_ylabel("observed median, stochastic arm (steps)")
    axes[2].set_title("H-B1-4 dwell prediction", fontsize=9)
    axes[2].legend(fontsize=7)
    fig.suptitle("S26 B1 — stall geometry and dwell (pre-registered)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def summarize() -> dict:
    measurable, rejected, undetermined = scan_records()
    if rejected:
        raise SystemExit(f"REFUSED: invalid records: {rejected}")
    missing = [n for n in CELL_ORDER if n not in measurable and n not in undetermined]
    if missing:
        raise SystemExit(f"REFUSED: cells missing: {missing}")
    g = cg_b1_2()
    if g["verdict"] != "PASS":
        raise SystemExit(f"REFUSED by {g['gate']}: {g['missing']}")
    cls = classification(measurable)
    h4 = h_b1_4(measurable)
    out = {"spec": SPEC, "gates": {"CG-B1-2": g}, "hsafe": hsafe(measurable), "classification": cls,
           "hypotheses": {h["hypothesis"]: h for h in (h_b1_1(cls), h_b1_2(cls), h_b1_3(cls), h4)},
           "cells": {n: {k: r[k] for k in ("deadlock_episode_rate", "immobile_step_rate", "stall_step_rate", "self_stop_share_of_immobile", "capture_rate", "violation_steps")} | {"n_states": len(r["states"])} for n, r in measurable.items()},
           "undetermined": undetermined}
    _write_json(_round_file("json"), out)
    write_figure(measurable, cls, h4, _round_file("png"))
    for k, h in out["hypotheses"].items():
        print(f"{k}: {h['verdict']}")
    print(f"H-SAFE: {out['hsafe']['verdict']}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="*", default=None)
    ap.add_argument("--gifs", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-episodes", type=int, default=N_EPISODES)
    args = ap.parse_args()
    if args.cells is not None:
        run_cells(args.cells, args.device, args.n_episodes)
    if args.gifs:
        write_gifs(args.device)
    if args.summarize:
        summarize()
    if args.cells is None and not args.gifs and not args.summarize:
        raise SystemExit("nothing to do")


if __name__ == "__main__":
    main()
