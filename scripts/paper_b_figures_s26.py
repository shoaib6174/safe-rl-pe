"""Paper B figures F6–F8 from the S26 rounds (B1 stall geometry; B2–B4 filter sweep).

Reads `results/derisk/s26_b1_round.json` + `s26_b1_cells/*.json` and
`s26_b2_round.json` + `s26_b2_cells/*.json`; REFUSES to run until both round
files exist, so the pre-registered `--summarize` stays the first reader of any
number. Writes `docs/paper_b/figures/F6_geometry`, `F7_recovery_sweep`,
`F8_predictive` (pdf + png) and `figure_data_s26.json` with every number drawn.
Nothing under `results/` is written.

Colour follows the entity: the BASE family is blue (seeds by marker), STR14
orange, RC19 aqua, references neutral gray; the recovery is a line style.

  ./venv/bin/python scripts/paper_b_figures_s26.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from pe2.config import EnvConfig  # noqa: E402

R = pathlib.Path("results/derisk")
OUT = pathlib.Path("docs/paper_b/figures")
C = {"BASE": "#2a78d6", "STR14": "#eb6834", "RC19": "#1baf7a", "ref": "#52514e", "ink": "#0b0b0b", "muted": "#8a8984", "grid": "#e6e5e1", "light": "#d9d8d3"}
POL = ("BASE-60", "BASE-61", "BASE-62", "STR14", "RC19")
B34 = ("BASE-60", "BASE-61", "BASE-62", "RC19")
MARK = {"BASE-60": "o", "BASE-61": "s", "BASE-62": "^", "STR14": "D", "RC19": "v"}
ALPHA = {"BASE-60": 0.55, "BASE-61": 0.75, "BASE-62": 1.0, "STR14": 1.0, "RC19": 1.0}
RECF_NS = (5, 10, 20, 40, 80)
XS = ("PLAIN", *(f"RECF-{n}" for n in RECF_NS))
XLAB = ("plain", "5", "10", "20", "40", "80")
T_TURN = 45
DEADLOCK_BAR = 0.05
GRID, COARSE = 41, 21
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#52514e", "axes.linewidth": 0.6,
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.labelcolor": "#0b0b0b", "grid.color": C["grid"], "grid.linewidth": 0.5,
                     "legend.frameon": False, "pdf.fonttype": 42})
DATA: dict = {}


def colour(p: str) -> str:
    return C["BASE"] if p.startswith("BASE") else C[p]


def _save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / f"{name}.pdf")


def load_round(prefix: str) -> tuple[dict, dict]:
    rnd = R / f"{prefix}_round.json"
    if not rnd.exists():
        raise SystemExit(f"REFUSED: {rnd} does not exist — run the pre-registered --summarize first")
    cells = {}
    for p in sorted((R / f"{prefix}_cells").glob("*.json")):
        rec = json.loads(p.read_text())
        if rec.get("status") == "UNDETERMINED":
            continue
        cells[rec["name"]] = rec
    return json.loads(rnd.read_text()), cells


def action_grid(cfg: EnvConfig, n: int) -> np.ndarray:
    v = np.linspace(0.0, cfg.v_max, n)
    w = np.linspace(-cfg.omega_max, cfg.omega_max, n)
    V, W = np.meshgrid(v, w, indexing="ij")
    return np.column_stack([V.ravel(), W.ravel()])


# ---------------------------------------------------------------- F6 stall geometry (B1)
def f6(rnd: dict, cells: dict) -> None:
    cfg = EnvConfig()
    cls = rnd["classification"]
    h4 = rnd["hypotheses"]["H-B1-4"]
    fig, axes = plt.subplots(1, 3, figsize=(7.8, 2.6), gridspec_kw={"wspace": 0.45, "width_ratios": [1.05, 1.3, 1.0]})
    # (a) one real stalled state's admissible set in (v, omega)
    cand = [s for p in ("BASE-61", "BASE-60", "BASE-62", "RC19") for s in cells.get(f"{p}_det", {}).get("states", [])
            if s["kind"] == "stalled" and s.get("admissible_grid") is not None]
    st = next((s for s in cand if s["run_at_sample"] == 40 and s["trap_heading"]), cand[0] if cand else None)
    ax = axes[0]
    if st is not None:
        U = action_grid(cfg, GRID)
        adm = np.zeros(len(U), dtype=bool)
        adm[st["admissible_grid"]] = True
        ax.scatter(U[~adm, 1], U[~adm, 0], s=2.5, color=C["light"], label="inadmissible (exact one-step test)", rasterized=True)
        ax.scatter(U[adm, 1], U[adm, 0], s=3.5, color=C["BASE"], label="admissible", rasterized=True)
        ax.scatter([st["w_nom"]], [st["v_nom"]], marker="*", s=90, color=C["ink"], zorder=5, label="policy's command")
        ax.scatter([st["u_shield"][1]], [st["u_shield"][0]], marker="x", s=55, color=C["STR14"], zorder=6, label="filter output")
        if st["qp"].get("u") is not None:
            ax.scatter([st["qp"]["u"][1]], [st["qp"]["u"][0]], marker="+", s=70, color=C["RC19"], zorder=6, label="weighted QP projection")
        ax.set_xlabel("ω (rad/s)")
        ax.set_ylabel("v (m/s)")
        ax.set_xlim(-cfg.omega_max * 1.05, cfg.omega_max * 1.05)
        ax.set_ylim(-0.04, cfg.v_max * 1.05)
        pol = next(p for p in B34 if any(s is st for s in cells[f"{p}_det"]["states"]))
        ax.set_title(f"(a) a stalled state ({pol}, step {st['t']}): tier {st['tier']}, trap at heading {st['trap_heading']}, position trap {st['trap_position']}", loc="left", fontsize=7)
        ax.legend(fontsize=6, loc="upper right", handletextpad=0.3, labelspacing=0.3)
        DATA["F6a"] = {"policy": pol, "seed": st["seed"], "t": st["t"], "tier": st["tier"], "v_nom": st["v_nom"], "w_nom": st["w_nom"], "u_shield": st["u_shield"],
                       "qp": st["qp"], "n_admissible_grid": st["n_admissible_grid"], "trap_heading": st["trap_heading"], "trap_position": st["trap_position"]}
    # (b) classification of the deterministic arm's stalled states
    ax = axes[1]
    keys = ("exact_tier", "trap_heading", "trap_position", "pair_trap", "rot_admissible", "qp_agree")
    names = ("exact\ntier", "trap at\nheading", "position\ntrap", "pair\ntrap", "rotation\nadmissible", "QP\nagrees")
    xs = np.arange(len(keys))
    w = 0.19
    for i, p in enumerate(B34):
        c = cls.get(p, {})
        if not c.get("n"):
            continue
        ax.bar(xs + (i - 1.5) * w, [c[k] for k in keys], width=w - 0.02, color=colour(p), alpha=ALPHA[p], label=f"{p} (n = {c['n']})")
    ax.set_xticks(xs, names, fontsize=6.5)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("share of stalled states")
    ax.set_title("(b) classification, deterministic arm (traps under the oracle test)", loc="left", fontsize=7)
    ax.legend(fontsize=6, loc="upper right", ncol=2, handletextpad=0.3, columnspacing=0.6)
    ax.grid(axis="y")
    DATA["F6b"] = {p: {k: cls[p].get(k) for k in (*keys, "n", "n_exact", "law_branch_share", "n_self_stop")} for p in B34 if cls.get(p, {}).get("n")}
    # (c) dwell: predicted 10/p_leave vs observed, stochastic arm's search-branch onsets
    ax = axes[2]
    allp, allo = [], []
    for p in B34:
        s = cells.get(f"{p}_stoch")
        if s is None:
            continue
        on = [x for x in s["states"] if x["kind"] == "stalled" and x.get("run_at_sample") == 40 and x.get("branch", "search") == "search" and x.get("leave")]
        pr = [x["leave"]["predicted_dwell_steps"] for x in on]
        ob = [x["dwell_beyond_sample"] if x["dwell_beyond_sample"] is not None else (x["run_length"] - x["run_at_sample"]) for x in on]
        fin = [(a, b) for a, b in zip(pr, ob, strict=True) if a is not None]
        if fin:
            ax.scatter([a for a, _ in fin], [max(b, 1) for _, b in fin], s=7, color=colour(p), alpha=0.35, marker=MARK[p], linewidths=0, rasterized=True)
        allp += pr
        allo += ob
        d = h4["detail"].get(p)
        if d and d["median_predicted"] is not None and d["median_observed_beyond_onset"] is not None:
            ax.scatter([d["median_predicted"]], [max(d["median_observed_beyond_onset"], 1)], s=48, color=colour(p), marker=MARK[p], edgecolor=C["ink"], linewidths=0.6, zorder=5, label=f"{p} median (n = {d['n_onsets']})")
    lim = np.array([1.0, 5000.0])
    ax.fill_between(lim, 0.5 * lim, 2.0 * lim, color=C["light"], alpha=0.5, lw=0, label="within factor 2")
    ax.plot(lim, lim, color=C["ref"], lw=0.7, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel("predicted dwell beyond onset, 10 / p_leave (steps)")
    ax.set_ylabel("observed dwell (steps)")
    ax.set_title("(c) stochastic arm: dwell at stall onsets", loc="left", fontsize=7)
    ax.legend(fontsize=5.5, loc="lower right", handletextpad=0.3, labelspacing=0.25)
    DATA["F6c"] = {"detail": h4["detail"], "verdict": h4["verdict"], "n_points": len(allp)}
    _save(fig, "F6_geometry")


# ---------------------------------------------------------------- F7 recovery sweep (B2)
def cell(cells: dict, p: str, o: str, s: str) -> dict | None:
    return cells.get(f"{p}_{o}_{s}")


def f7(rnd: dict, cells: dict) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(7.8, 4.6), gridspec_kw={"wspace": 0.42, "hspace": 0.55})
    ax_dead, ax_cap, ax_fire, ax_cyc, ax_imm, ax_esc = axes.ravel()
    x = np.arange(len(XS))
    DATA["F7"] = {}
    for p in POL:
        dead = [(cell(cells, p, "ATK4-45", s) or {}).get("deadlock_episode_rate", np.nan) for s in XS]
        cap = [(cell(cells, p, "GREEDY", s) or {}).get("capture_rate", np.nan) for s in XS]
        imm = [(cell(cells, p, "ATK4-45", s) or {}).get("immobile_step_rate", np.nan) for s in XS]
        fire, cyc = [], []
        for s in XS:
            rs = [cell(cells, p, o, s) for o in ("ATK4-45", "GREEDY")]
            if any(r is None for r in rs):
                fire.append(np.nan)
                cyc.append(np.nan)
                continue
            fire.append(sum(r["recoveries"] for r in rs) / sum(r["n_episodes"] for r in rs))
            ne, nc = sum(r["n_engagements"] for r in rs), sum(r["n_cycles"] for r in rs)
            cyc.append(nc / ne if ne else np.nan)
        kw = {"color": colour(p), "alpha": ALPHA[p], "marker": MARK[p], "ms": 3.6, "lw": 1.1, "label": p}
        ax_dead.plot(x, dead, **kw)
        ax_cap.plot(x, cap, **kw)
        ax_fire.plot(x, fire, **kw)
        ax_cyc.plot(x, cyc, **kw)
        ax_imm.plot(x, imm, **kw)
        pert_d, pert_c = cell(cells, p, "ATK4-45", "PERT-20"), cell(cells, p, "GREEDY", "PERT-20")
        if pert_d and pert_c:
            kp = {"color": colour(p), "alpha": ALPHA[p], "marker": MARK[p], "s": 26, "facecolors": "none", "zorder": 5}
            ax_dead.scatter([3.35], [pert_d["deadlock_episode_rate"]], **kp)
            ax_cap.scatter([3.35], [pert_c["capture_rate"]], **kp)
            ax_imm.scatter([3.35], [pert_d["immobile_step_rate"]], **kp)
            rs = [pert_d, pert_c]
            ax_fire.scatter([3.35], [sum(r["recoveries"] for r in rs) / sum(r["n_episodes"] for r in rs)], **kp)
            ne, nc = sum(r["n_engagements"] for r in rs), sum(r["n_cycles"] for r in rs)
            ax_cyc.scatter([3.35], [nc / ne if ne else np.nan], **kp)
        DATA["F7"][p] = {"deadlock_ATK": dead, "capture_GREEDY": cap, "firings_per_episode": fire, "cycles_per_engagement": cyc, "immobile_ATK": imm,
                         "PERT-20": None if not (pert_d and pert_c) else {"deadlock": pert_d["deadlock_episode_rate"], "capture": pert_c["capture_rate"], "immobile": pert_d["immobile_step_rate"],
                                                                        "firings_per_episode": sum(r["recoveries"] for r in (pert_d, pert_c)) / sum(r["n_episodes"] for r in (pert_d, pert_c))}}
    for ax, ttl, yl in ((ax_dead, "(a) deadlock episodes vs the learned attacker", "deadlock episode rate"), (ax_cap, "(b) capture vs GREEDY", "capture rate"),
                        (ax_fire, "(c) recovery firings per episode (both opponents)", "firings / episode"), (ax_cyc, "(d) cycles per engagement", "cycle share"),
                        (ax_imm, "(e) immobile steps vs the learned attacker", "immobile step rate")):
        ax.set_xticks(x, XLAB)
        ax.set_xlabel("recovery_after N (open marker: perturbation baseline, N = 20)", fontsize=6.5)
        ax.set_ylabel(yl)
        ax.set_title(ttl, loc="left", fontsize=7)
        ax.grid(axis="y")
    ax_dead.axhline(DEADLOCK_BAR, color=C["ref"], lw=0.7, ls=":")
    ax_dead.annotate("bar 0.05", xy=(4.6, DEADLOCK_BAR), xytext=(0, 2), textcoords="offset points", fontsize=6, color=C["ref"])
    ax_fire.set_yscale("symlog", linthresh=1.0)
    ax_cap.legend(fontsize=6, loc="lower right", ncol=2, handletextpad=0.3, columnspacing=0.6)
    # (f) escape time after a trigger, RECF-20, all ten cells, with the N + one-turn bound
    esc = [e["t"] for p in POL for o in ("ATK4-45", "GREEDY") for e in ((cell(cells, p, o, "RECF-20") or {}).get("escape_times") or []) if e["t"] is not None]
    cens = sum(1 for p in POL for o in ("ATK4-45", "GREEDY") for e in ((cell(cells, p, o, "RECF-20") or {}).get("escape_times") or []) if e["t"] is None)
    if esc:
        bins = np.arange(0, max(200, min(max(esc), 400)) + 5, 5)
        ax_esc.hist(np.clip(esc, 0, bins[-1]), bins=bins, color=C["BASE"], alpha=0.85)
        ax_esc.axvline(20 + T_TURN, color=C["STR14"], lw=1.0, ls="--")
        ax_esc.annotate(f"N + one turn = {20 + T_TURN}", xy=(20 + T_TURN, 1), xytext=(4, 0), textcoords="offset points", fontsize=6, color=C["STR14"], rotation=90, va="bottom")
        ax_esc.set_yscale("log")
    ax_esc.set_xlabel("steps from trigger to a 0.10 m displacement (RECF-20, all cells)", fontsize=6.5)
    ax_esc.set_ylabel("triggers")
    ax_esc.set_title("(f) escape time after a recovery trigger", loc="left", fontsize=7)
    h22 = rnd["hypotheses"].get("H-B2-2", {})
    DATA["F7"]["escape"] = {"n_decided": len(esc), "n_censored": cens, "median": float(np.median(esc)) if esc else None, "bound": 20 + T_TURN,
                            "share_within_bound": float(np.mean([t <= 20 + T_TURN for t in esc])) if esc else None, "H-B2-2": {k: h22.get(k) for k in ("verdict", "recf20_within_bound", "vacuous_cells")}}
    DATA["F7"]["verdicts"] = {k: v.get("verdict") for k, v in rnd["hypotheses"].items()}
    _save(fig, "F7_recovery_sweep")


# ---------------------------------------------------------------- F8 the second filter (B3/B4)
def f8(rnd: dict, cells: dict) -> None:
    sets = ["PLAIN", "RECF-20", "PRED-PLAIN"] + (["PRED-RECF-20"] if rnd.get("b4_included") else [])
    lab = {"PLAIN": "CBF\nplain", "RECF-20": "CBF +\nrecovery", "PRED-PLAIN": "predictive\nplain", "PRED-RECF-20": "predictive +\nrecovery"}
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 2.5), gridspec_kw={"wspace": 0.35})
    x = np.arange(len(sets))
    DATA["F8"] = {"settings": sets, "b4_included": rnd.get("b4_included")}
    for p in B34:
        dead = [(cell(cells, p, "ATK4-45", s) or {}).get("deadlock_episode_rate", np.nan) for s in sets]
        cap = [(cell(cells, p, "GREEDY", s) or {}).get("capture_rate", np.nan) for s in sets]
        kw = {"color": colour(p), "alpha": ALPHA[p], "marker": MARK[p], "ms": 4, "lw": 1.1, "label": p}
        axes[0].plot(x, dead, **kw)
        axes[1].plot(x, cap, **kw)
        DATA["F8"][p] = {"deadlock_ATK": dead, "capture_GREEDY": cap}
    for ax, ttl, yl in ((axes[0], "(a) deadlock episodes vs the learned attacker", "deadlock episode rate"), (axes[1], "(b) capture vs GREEDY", "capture rate")):
        ax.set_xticks(x, [lab[s] for s in sets], fontsize=7)
        ax.set_ylabel(yl)
        ax.set_title(ttl, loc="left", fontsize=7)
        ax.grid(axis="y")
    axes[0].axhline(DEADLOCK_BAR, color=C["ref"], lw=0.7, ls=":")
    axes[1].legend(fontsize=6.5, loc="best", ncol=2, handletextpad=0.3, columnspacing=0.6)
    h3, h4 = rnd["hypotheses"].get("H-B3-1", {}), rnd["hypotheses"].get("H-B4-1", {})
    DATA["F8"]["verdicts"] = {"H-B3-1": h3.get("verdict"), "H-B4-1": h4.get("verdict"), "CG-B4-1": rnd["gates"]["CG-B4-1"].get("verdict")}
    _save(fig, "F8_predictive")


def main() -> None:
    b1_rnd, b1_cells = load_round("s26_b1")
    b2_rnd, b2_cells = load_round("s26_b2")
    f6(b1_rnd, b1_cells)
    f7(b2_rnd, b2_cells)
    f8(b2_rnd, b2_cells)
    DATA["sources"] = {"B1": "results/derisk/s26_b1_round.json + s26_b1_cells/*.json", "B2-B4": "results/derisk/s26_b2_round.json + s26_b2_cells/*.json"}
    (OUT / "figure_data_s26.json").write_text(json.dumps(DATA, indent=1, default=float))
    print("wrote", OUT / "figure_data_s26.json")


if __name__ == "__main__":
    main()
