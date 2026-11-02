"""Paper B round N1 (internal S27): which component commands the absorbing stall?

Spec: docs/pathway/paper_b_prereg_n1_branch_attribution_2026-11-03.md.

Replays the archived Experiment 4 (S17) bare-arm gate cells and the Experiment 8
(S26 B2) plain cells of STR14 and RC19 through `pe2.recorder.BranchTaggedSearchMDP`,
checks each replay against the archive episode by episode (CG-N1-1, CG-N1-2,
CG-N1-4), and attributes every deadlock episode to the component that commanded
the step at which its first stall run reached K = 40 steps: `chase` (the
intercept law, evader visible), `last_seen` (the intercept law driving to its
remembered point, evader out of view) or `head` (the learned search policy).

Evaluation only. Nothing is trained; no policy, filter, opponent or seed changes.

  ./venv/bin/python scripts/paper_b_branch_attribution.py --cells BASE-60_ATK4-45 OBS-61_HUG7
  ./venv/bin/python scripts/paper_b_branch_attribution.py --gifs
  ./venv/bin/python scripts/paper_b_branch_attribution.py --summarize
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import subprocess
import sys

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from pe2.config import EnvConfig  # noqa: E402
from pe2.eval import win_count  # noqa: E402
from pe2.metrics import DEADLOCK_STEPS, summarize_shield  # noqa: E402

SPEC = pathlib.Path("docs/pathway/paper_b_prereg_n1_branch_attribution_2026-11-03.md")
OUT = pathlib.Path("results/derisk")
CELLS_DIR = OUT / "s27_n1_cells"
GIFS_DIR = OUT / "s27_n1_gifs"
ROUND = OUT / "s27_n1_round.json"
K = DEADLOCK_STEPS
GATE_N = 500
SEED0 = 1_740_000
COMPONENTS = ("chase", "last_seen", "head")
RATE_KEYS = ("capture_rate", "deadlock_episode_rate", "stall_step_rate", "immobile_step_rate")
WATCH_MARK = "CG-N1-3 WATCHED"

SET_A_ARMS = tuple(f"{f}-{s}" for f in ("BASE", "OBS", "GAM", "OBSGAM") for s in (60, 61, 62))
SET_A_OPPS = ("ATK4-45", "HUG7", "GREEDY")
SET_B = (("STR14", "ATK4-45"), ("RC19", "ATK4-45"), ("STR14", "GREEDY"), ("RC19", "GREEDY"))
CELLS: dict[str, dict] = {
    **{f"{a}_{o}": {"set": "A", "policy": a, "opponent": o} for a in SET_A_ARMS for o in SET_A_OPPS},
    **{f"{p}_{o}": {"set": "B", "policy": p, "opponent": o} for p, o in SET_B},
}
GIF_CELLS = ("BASE-61_ATK4-45", "OBSGAM-61_ATK4-45", "STR14_ATK4-45", "RC19_ATK4-45")
FAMILIES = ("BASE", "OBS", "GAM", "OBSGAM")


# ----------------------------------------------------------------------------
# the archived runners, loaded from this tree (the frozen worktree when launched)
# ----------------------------------------------------------------------------
_MODULES: dict[str, object] = {}


def _runner(name: str):
    if name not in _MODULES:
        spec = importlib.util.spec_from_file_location(f"n1_{name}", _HERE / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _MODULES[name] = mod
    return _MODULES[name]


def s17():
    return _runner("derisk_s17_fixes_b")


def sweep():
    return _runner("paper_b_filter_sweep")


_PPO: dict[str, object] = {}


def _ppo(path: pathlib.Path):
    from stable_baselines3 import PPO

    key = str(path)
    if key not in _PPO:
        _PPO[key] = PPO.load(path, device="cpu")
    return _PPO[key]


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


# ----------------------------------------------------------------------------
# attribution (spec sections 4-5)
# ----------------------------------------------------------------------------
def component(rec: dict) -> str:
    """Spec section 4: the component that commanded one recorded step."""
    if rec["branch"] == "search":
        return "head"
    if rec["branch"] == "law":
        return "chase" if rec["visible"] else "last_seen"
    raise ValueError(f"unknown branch {rec['branch']!r}")


def stall_runs(stalled: list[bool]) -> list[tuple[int, int, bool]]:
    """Maximal runs of True as (start, length, censored)."""
    runs, start = [], None
    for i, s in enumerate(stalled):
        if s and start is None:
            start = i
        elif not s and start is not None:
            runs.append((start, i - start, False))
            start = None
    if start is not None:
        runs.append((start, len(stalled) - start, True))
    return runs


def _count(comps: list[str]) -> dict[str, int]:
    return {c: sum(x == c for x in comps) for c in COMPONENTS}


def episode_attribution(steps: list[dict]) -> dict:
    """One episode's per-step records -> its stall runs of length >= K, the
    component at each run's K-th step, and step counts by component."""
    comps = [component(r) for r in steps]
    stalled = [bool(r["stalled"]) for r in steps]
    k_runs = []
    for start, length, censored in stall_runs(stalled):
        if length < K:
            continue
        run = comps[start : start + length]
        k_runs.append(
            {
                "start": start,
                "length": length,
                "censored": censored,
                "component_at_k": comps[start + K - 1],
                "crossing": len(set(run)) > 1,
                "composition": _count(run),
            }
        )
    return {
        "deadlock": bool(k_runs),
        "first_k": k_runs[0] if k_runs else None,
        "k_runs": k_runs,
        "any_head_k": any(r["component_at_k"] == "head" for r in k_runs),
        "steps_by_component": _count(comps),
        "stalled_by_component": _count([c for c, s in zip(comps, stalled, strict=True) if s]),
        "immobile_by_component": _count([c for c, r in zip(comps, steps, strict=True) if r["immobile"]]),
    }


def aggregate(eps: list[dict]) -> dict:
    dl = [e for e in eps if e["deadlock"]]
    first_k = {c: sum(e["first_k"]["component_at_k"] == c for e in dl) for c in COMPONENTS}
    k_runs = {c: sum(r["component_at_k"] == c for e in eps for r in e["k_runs"]) for c in COMPONENTS}

    def pooled(key: str) -> dict[str, int]:
        return {c: sum(e[key][c] for e in eps) for c in COMPONENTS}

    stalled = pooled("stalled_by_component")
    n_stalled = sum(stalled.values())
    return {
        "n_episodes": len(eps),
        "n_deadlock": len(dl),
        "first_k": first_k,
        "first_k_share": {c: first_k[c] / len(dl) for c in COMPONENTS} if dl else None,
        "k_runs": k_runs,
        "n_k_runs": sum(k_runs.values()),
        "crossing_k_runs": sum(r["crossing"] for e in eps for r in e["k_runs"]),
        "deadlock_with_any_head_k": sum(e["any_head_k"] for e in dl),
        "stalled_by_component": stalled,
        "stalled_share": {c: stalled[c] / n_stalled for c in COMPONENTS} if n_stalled else None,
        "immobile_by_component": pooled("immobile_by_component"),
        "steps_by_component": pooled("steps_by_component"),
    }


# ----------------------------------------------------------------------------
# the archive and the reproduction check (CG-N1-1)
# ----------------------------------------------------------------------------
def archived(cell: str) -> dict:
    spec = CELLS[cell]
    if spec["set"] == "A":
        path = OUT / f"s17_arm_{spec['policy']}.json"
        rec = json.loads(path.read_text())
        g = rec["final"]["gate"][spec["opponent"]]
        return {
            "capture_ind": [bool(x) for x in g["capture_ind"]],
            "deadlock_ind": [bool(x) for x in g["deadlock_ind"]],
            "capture_rate": g["capture_rate"],
            **{k: g["pursuer"][k] for k in RATE_KEYS[1:]},
            "sha256": _archived_sha_a(rec, spec["policy"]),
            "source": str(path),
        }
    path = OUT / "s26_b2_cells" / f"{spec['policy']}_{spec['opponent']}_PLAIN.json"
    rec = json.loads(path.read_text())
    return {
        "capture_ind": [bool(x) for x in rec["capture_ind"]],
        "deadlock_ind": [bool(x) for x in rec["deadlock_ind"]],
        **{k: rec[k] for k in RATE_KEYS},
        "sha256": rec["ckpt_sha256"],
        "source": str(path),
    }


def _archived_sha_a(rec: dict, arm: str) -> str:
    """Amendment A1: an S17 arm that TRAINED (OBSGAM) gated its own checkpoint and
    archived no hash of it; the RECF arm of the same family and seed gated the
    same file and archived it as `source_sha256`."""
    if rec.get("source_sha256"):
        return rec["source_sha256"]
    twin = json.loads((OUT / f"s17_arm_RECF-{arm}.json").read_text())
    return twin["source_sha256"]


def compare_reproduction(replayed: dict, arch: dict, prefix: int | None = None) -> dict:
    """Element-wise indicators over the first `prefix` episodes (all when None)
    and, for a full cell, exact equality of the four rates."""
    n = len(arch["capture_ind"]) if prefix is None else int(prefix)
    out: dict = {"n_compared": n}
    ok = True
    for key in ("capture_ind", "deadlock_ind"):
        rep = list(replayed[key])
        ok &= len(rep) == n
        out[f"{key}_mismatch"] = [i for i in range(min(n, len(rep))) if bool(rep[i]) != bool(arch[key][i])]
        ok &= not out[f"{key}_mismatch"]
    if prefix is None:
        out["rate_mismatch"] = {k: [replayed[k], arch[k]] for k in RATE_KEYS if replayed[k] != arch[k]}
        ok &= not out["rate_mismatch"]
    out["pass"] = bool(ok)
    return out


# ----------------------------------------------------------------------------
# the replay (spec section 3)
# ----------------------------------------------------------------------------
def _setup(cell: str) -> dict:
    spec = CELLS[cell]
    if spec["set"] == "A":
        m = s17()
        arm = m.ARMS[spec["policy"]]
        return {
            "ckpt": m.source_ckpt(spec["policy"]),
            "shield_obs": bool(m.SHIELD_OBS),
            "escape_obs": bool(arm["escape_obs"]),
            "opponent_factory": m.gate_opponent_factory(spec["opponent"], "cpu"),
        }
    m = sweep()
    pol = m.POLICIES[spec["policy"]]
    return {
        "ckpt": pol["ckpt"],
        "shield_obs": bool(pol["shield_obs"]),
        "escape_obs": False,
        "opponent_factory": m.opponent_factory(spec["opponent"], "cpu"),
    }


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True, cwd=_HERE
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def replay(cell: str, n_episodes: int = GATE_N) -> dict:
    from pe2.recorder import BranchTaggedSearchMDP
    from pe2.shield import CBFShield
    from pe2.structured import HOLD

    cfg = EnvConfig()
    st = _setup(cell)
    model = _ppo(st["ckpt"])
    mdp = BranchTaggedSearchMDP(
        cfg,
        st["opponent_factory"],
        hold=HOLD,
        shield=None,
        shield_obs=st["shield_obs"],
        escape_obs=st["escape_obs"],
    )
    width_model = int(model.observation_space.shape[0])
    width_env = int(mdp.observation_space.shape[0])
    if width_model != width_env:
        raise RuntimeError(f"{cell}: checkpoint expects {width_model} inputs, the replay feeds {width_env}")
    logs, eps, mismatch = [], [], []
    for i in range(n_episodes):
        seed = SEED0 + i
        mdp.shield = CBFShield(cfg)  # fresh per episode, as both archived gates
        log = mdp.play_episode(model.predict, seed)
        att = episode_attribution(mdp.step_records)
        if att["deadlock"] != bool(log["shield"]["deadlock"]):
            mismatch.append(i)
        logs.append(log)
        eps.append({"i": i, "seed": seed, "outcome": log["outcome"], **att})
    k, n = win_count(logs, "pursuer")
    sh = summarize_shield(logs)
    replayed = {
        "capture_ind": [lg["outcome"] == "capture" for lg in logs],
        "deadlock_ind": [bool(lg["shield"]["deadlock"]) for lg in logs],
        "capture_rate": k / n,
        **{key: sh[key] for key in RATE_KEYS[1:]},
    }
    arch_sha = archived(cell)["sha256"]
    sha = _sha256(st["ckpt"])
    import torch

    return {
        "replayed": replayed,
        "episodes": eps,
        "cg_n1_2": {"mismatch": mismatch},
        "cg_n1_4": {"pass": sha == arch_sha, "ckpt_sha256": sha, "archived_sha256": arch_sha, "device": "cpu"},
        "provenance": {
            "git_commit": _git_commit(),
            "spec": str(SPEC),
            "spec_sha256": _sha256(SPEC) if SPEC.exists() else None,
            "ckpt": str(st["ckpt"]),
            "shield_obs": st["shield_obs"],
            "escape_obs": st["escape_obs"],
            "hold": HOLD,
            "seed0": SEED0,
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "torch_threads": torch.get_num_threads(),
            "python": sys.version.split()[0],
        },
    }


def _write_json(path: pathlib.Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)


def run_cell(cell: str) -> dict:
    """One cell, written once (stop rule). A cell that fails a construct gate is
    written UNDETERMINED with no attribution field."""
    path = CELLS_DIR / f"{cell}.json"
    if path.exists():
        raise SystemExit(f"{cell} already has a record ({path}); the stop rule forbids re-running it")
    out = replay(cell)
    arch = archived(cell)
    cg1 = compare_reproduction(out["replayed"], arch)
    ok = cg1["pass"] and not out["cg_n1_2"]["mismatch"] and out["cg_n1_4"]["pass"]
    spec = CELLS[cell]
    rec = {
        "name": cell,
        **spec,
        "n_episodes": GATE_N,
        "status": "PASS" if ok else "UNDETERMINED",
        "archive": arch["source"],
        "cg_n1_1": cg1,
        "cg_n1_2": out["cg_n1_2"],
        "cg_n1_4": out["cg_n1_4"],
        "provenance": out["provenance"],
    }
    if ok:
        rec["episodes"] = out["episodes"]
        rec["aggregate"] = aggregate(out["episodes"])
    _write_json(path, rec)
    print(f"{cell}: {rec['status']} (CG-N1-1 {cg1['pass']}, CG-N1-2 {not out['cg_n1_2']['mismatch']}, CG-N1-4 {out['cg_n1_4']['pass']})", flush=True)
    return rec


# ----------------------------------------------------------------------------
# CG-N1-3: the designated episodes, rendered with the component in the title
# ----------------------------------------------------------------------------
def _record(cell: str) -> dict:
    path = CELLS_DIR / f"{cell}.json"
    if not path.exists():
        raise SystemExit(f"no record for {cell}")
    return json.loads(path.read_text())


def designated() -> list[dict]:
    out = []
    for cell in GIF_CELLS:
        rec = _record(cell)
        if rec["status"] != "PASS":
            raise SystemExit(f"CG-N1-3 refuses: {cell} is {rec['status']}")
        for comp in ("last_seen", "head"):
            hit = next(
                (e for e in rec["episodes"] if e["deadlock"] and e["first_k"]["component_at_k"] == comp), None
            )
            if hit is not None:
                out.append({"cell": cell, "seed": hit["seed"], "component": comp, "first_k": hit["first_k"], "outcome": hit["outcome"]})
    return out


def write_gifs(every: int = 4, fps: float = 15.0) -> list[pathlib.Path]:
    from PIL import Image, ImageDraw

    from pe2.render import save_gif
    from pe2.shield import CBFShield
    from pe2.structured import HOLD

    Base = sweep().gif_recorder_class()

    class Labelled(Base):
        def __init__(self, *a, base_label: str, **kw):
            super().__init__(*a, label=base_label, **kw)
            self.base_label, self.k, self.frame_steps = base_label, 0, []

        def _step_inner(self, action, branch="law"):
            comp = "head" if branch == "search" else ("chase" if self._visible else "last_seen")
            ls = getattr(self.law, "last_seen", None)
            point = f" to ({ls[0]:.1f}, {ls[1]:.1f})" if comp == "last_seen" and ls is not None else ""
            self.label = f"{comp}{point} | step {self.k} | {self.base_label}"
            n0 = len(self.frames)
            out = super()._step_inner(action, branch)
            if len(self.frames) > n0:
                self.frame_steps.append(self.k)
            self.k += 1
            return out

    GIFS_DIR.mkdir(parents=True, exist_ok=True)
    cfg = EnvConfig()
    written = []
    for d in designated():
        cell, seed, fk = d["cell"], d["seed"], d["first_k"]
        stem = GIFS_DIR / f"{cell}_{seed}_{d['component']}"
        st = _setup(cell)
        model = _ppo(st["ckpt"])
        rec = Labelled(
            cfg, st["opponent_factory"], base_label=f"{cell} ep {seed}", every=every,
            hold=HOLD, shield=None, shield_obs=st["shield_obs"], escape_obs=st["escape_obs"],
        )
        rec.shield = CBFShield(cfg)
        log = rec.play_episode(model.predict, seed)
        if log["outcome"] != d["outcome"]:
            raise SystemExit(f"CG-N1-3: the GIF replay of {cell} ep {seed} ended {log['outcome']}, the record says {d['outcome']}")
        rec.finish(log["outcome"], log["steps"])
        save_gif(rec.frames, f"{stem}.gif", fps=fps)
        # contact sheet: before the stall, its start, its K-th step, 200 steps later
        last = rec.frame_steps[-1]
        targets = [max(0, fk["start"] - 30), fk["start"], fk["start"] + K - 1, min(fk["start"] + K - 1 + 200, last)]
        tiles = []
        for t in targets:
            j = min(range(len(rec.frame_steps)), key=lambda x: abs(rec.frame_steps[x] - t))
            img = Image.fromarray(rec.frames[j]).convert("RGB")
            ImageDraw.Draw(img).text((6, img.height - 16), f"target step {t} (frame at {rec.frame_steps[j]})", fill=(0, 0, 0))
            tiles.append(img)
        w, h = tiles[0].size
        sheet = Image.new("RGB", (2 * w, 2 * h), "white")
        for idx, img in enumerate(tiles):
            sheet.paste(img, ((idx % 2) * w, (idx // 2) * h))
        sheet.save(f"{stem}_sheet.png")
        print(f"wrote {stem}.gif and _sheet.png ({log['outcome']} @ {log['steps']}; first K-run start {fk['start']}, length {fk['length']})", flush=True)
        written.append(pathlib.Path(f"{stem}.gif"))
    return written


# ----------------------------------------------------------------------------
# the round (spec sections 6 and 8)
# ----------------------------------------------------------------------------
def _pooled(records: dict, names: list[str]) -> dict | None:
    ok = [records[n] for n in names if n in records and records[n]["status"] == "PASS"]
    if not ok:
        return None
    fk = {c: sum(r["aggregate"]["first_k"][c] for r in ok) for c in COMPONENTS}
    n_dl = sum(r["aggregate"]["n_deadlock"] for r in ok)
    st = {c: sum(r["aggregate"]["stalled_by_component"][c] for r in ok) for c in COMPONENTS}
    n_st = sum(st.values())
    return {
        "cells": [r["name"] for r in ok],
        "n_cells": len(ok),
        "n_episodes": sum(r["aggregate"]["n_episodes"] for r in ok),
        "n_deadlock": n_dl,
        "first_k": fk,
        "first_k_share": {c: fk[c] / n_dl for c in COMPONENTS} if n_dl else None,
        "stalled_by_component": st,
        "stalled_share": {c: st[c] / n_st for c in COMPONENTS} if n_st else None,
        "crossing_k_runs": sum(r["aggregate"]["crossing_k_runs"] for r in ok),
        "n_k_runs": sum(r["aggregate"]["n_k_runs"] for r in ok),
    }


def summarize() -> dict:
    if WATCH_MARK not in pathlib.Path(SPEC).read_text():
        raise SystemExit(f"--summarize refuses: no '{WATCH_MARK}' note in {SPEC} (CG-N1-3)")
    missing = [c for c in CELLS if not (CELLS_DIR / f"{c}.json").exists()]
    if missing:
        raise SystemExit(f"--summarize refuses: {len(missing)} cells have no record: {missing}")
    recs = {c: _record(c) for c in CELLS}
    undetermined = [c for c, r in recs.items() if r["status"] != "PASS"]

    def share(cell: str, comp: str):
        r = recs[cell]
        if r["status"] != "PASS" or r["aggregate"]["first_k_share"] is None:
            return None
        return r["aggregate"]["first_k_share"][comp]

    fam = {}
    for f in FAMILIES:
        for o in SET_A_OPPS:
            fam[f"{f}_{o}"] = _pooled(recs, [f"{f}-{s}_{o}" for s in (60, 61, 62)])

    # P-N1-1: BASE vs ATK4-45, last_seen share >= 0.80 on each seed
    p1 = [share(f"BASE-{s}_ATK4-45", "last_seen") for s in (60, 61, 62)]
    p1v = "UNDETERMINED" if None in p1 else ("PASS" if all(x >= 0.80 for x in p1) else "FAIL")
    # P-N1-2: every Set A ATK4-45 cell, chase share <= 0.05 (cells without deadlock episodes are named, not counted)
    p2 = {f"{a}_ATK4-45": share(f"{a}_ATK4-45", "chase") for a in SET_A_ARMS}
    p2v = "PASS" if all(x is not None and x <= 0.05 for x in p2.values()) else (
        "FAIL" if any(x is not None and x > 0.05 for x in p2.values()) else "UNDETERMINED"
    )
    # P-N1-3: RC19 vs ATK4-45, head share >= 0.20
    p3 = share("RC19_ATK4-45", "head")
    p3v = "UNDETERMINED" if p3 is None else ("PASS" if p3 >= 0.20 else "FAIL")

    # decision rule (section 8)
    h = {}
    for f in FAMILIES:
        pooled = fam[f"{f}_ATK4-45"]
        h[f] = None if (pooled is None or pooled["n_cells"] < 2 or pooled["first_k_share"] is None) else pooled["first_k_share"]["head"]
    known = [v for v in h.values() if v is not None]
    h_max = max(known) if known else None
    rule = None if h_max is None else ("D-1" if h_max < 0.10 else ("D-2" if h_max < 0.50 else "D-3"))
    trained = [fam[f"{f}_ATK4-45"] for f in FAMILIES if fam[f"{f}_ATK4-45"] is not None]
    sentence = {
        "x_law_last_seen": sum(p["first_k"]["last_seen"] for p in trained),
        "x_law_chase": sum(p["first_k"]["chase"] for p in trained),
        "y_head": sum(p["first_k"]["head"] for p in trained),
        "n": sum(p["n_deadlock"] for p in trained),
    }
    out = {
        "spec": str(SPEC),
        "spec_sha256": _sha256(SPEC),
        "n_cells": len(CELLS),
        "undetermined": undetermined,
        "cells": {c: (r["aggregate"] if r["status"] == "PASS" else {"status": r["status"]}) for c, r in recs.items()},
        "families": fam,
        "single_policies": {c: (recs[c]["aggregate"] if recs[c]["status"] == "PASS" else None) for c in CELLS if CELLS[c]["set"] == "B"},
        "predictions": {
            "P-N1-1": {"verdict": p1v, "last_seen_share_by_seed": dict(zip(("60", "61", "62"), p1, strict=True))},
            "P-N1-2": {"verdict": p2v, "chase_share": p2},
            "P-N1-3": {"verdict": p3v, "head_share": p3},
        },
        "decision": {"h_by_family": h, "h_max": h_max, "rule": rule, "h_free": share("STR14_ATK4-45", "head"), "h_aware": p3, "registered_sentence_numbers": sentence},
    }
    _write_json(ROUND, out)
    _report(out)
    return out


def _fmt(sh: dict | None) -> str:
    return "  (no deadlock episodes)" if sh is None else "  ".join(f"{c} {sh[c]:.3f}" for c in COMPONENTS)


def _report(out: dict) -> None:
    print(f"round N1: {out['n_cells']} cells, undetermined {out['undetermined'] or 'none'}")
    for key, p in out["families"].items():
        if p is not None:
            print(f"  {key:16s} cells {p['n_cells']} deadlock {p['n_deadlock']:4d}  first-K: {_fmt(p['first_k_share'])}  | stalled steps: {_fmt(p['stalled_share'])}")
    for c, a in out["single_policies"].items():
        if a is not None:
            print(f"  {c:16s} deadlock {a['n_deadlock']:4d}  first-K: {_fmt(a['first_k_share'])}  | stalled steps: {_fmt(a['stalled_share'])}")
    for k, v in out["predictions"].items():
        print(f"  {k}: {v['verdict']}")
    d = out["decision"]
    print(f"  decision: h_max {d['h_max']} -> {d['rule']}; numbers for the registered sentence {d['registered_sentence_numbers']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cells", nargs="+", help="cells to run, in order (literal names from CELLS)")
    ap.add_argument("--gifs", action="store_true", help="CG-N1-3: render the designated episodes")
    ap.add_argument("--summarize", action="store_true", help="the round summary (refuses before CG-N1-3)")
    args = ap.parse_args()
    if args.cells:
        unknown = [c for c in args.cells if c not in CELLS]
        if unknown:
            raise SystemExit(f"unknown cells {unknown}")
        for c in args.cells:
            if (CELLS_DIR / f"{c}.json").exists():
                print(f"{c}: record exists, skipped", flush=True)
                continue
            run_cell(c)
    if args.gifs:
        write_gifs()
    if args.summarize:
        summarize()


if __name__ == "__main__":
    main()
