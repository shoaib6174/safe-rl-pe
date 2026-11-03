"""Paper B revision (round-1 roadmap R9): main-text figures at their PRINTED size.

Every panel is drawn at the width it occupies on the page (\\textwidth of
l4dc2027.cls = 433.62 pt = 6.0 in) with 7-8 pt type, from ARCHIVED records
only; nothing under `results/` is written and no cell is re-run. The v1 scripts
(`paper_b_figures.py`, `paper_b_figures_s26.py`) are left untouched so the first
draft's figures stay reproducible; this script writes `R*_*.{pdf,png}` and
`figure_data_rev.json` beside them.

  R2_stall      (a) admissible set at one stalled state (Exp. 7 = S26 B1)
                (b) which component commanded the 40th stalled step (N1 = S27)
  R3_mechanism  Exp. 2 (S15), the discounted escape advantage as an EXPECTATION
                over the archived delays (Remark 1, roadmap R3)
  R4_recovery   (a) Exp. 4 (S17) deadlock vs capture, (b)-(c) Exp. 8 (S26 B2)
  R5_dissoc     Exp. 3 (S16) stall penalty, Exp. 5 (S18) correction penalty

  ./venv/bin/python scripts/paper_b_figures_rev.py
"""

from __future__ import annotations

import glob
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
TW = 433.62 / 72.27  # inches
INK, MUTED, GRID, LIGHT = "#0b0b0b", "#8a8984", "#e6e5e1", "#d9d8d3"
# one colour per policy family, used in every revision figure
FC = {"Base": "#52514e", "Obs": "#2a78d6", "Hor": "#eb6834", "ObsHor": "#1baf7a", "Free": "#7b4fd0", "Aware": "#c2417f"}
# one colour per component of the pursuer
CC = {"last_seen": "#3d5a80", "chase": "#98c1d9", "head": "#e0a526"}
CNAME = {"last_seen": "law, driving to the remembered point", "chase": "law, chasing the visible evader", "head": "learned search head"}
FAMKEY = {"Base": "BASE", "Obs": "OBS", "Hor": "GAM", "ObsHor": "OBSGAM"}
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#52514e",
                     "axes.linewidth": 0.6, "grid.color": GRID, "grid.linewidth": 0.5, "legend.frameon": False, "pdf.fonttype": 42,
                     "font.family": "serif"})
DATA: dict = {}


def _save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name}.png", dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", OUT / f"{name}.pdf")


def _j(p) -> dict:
    return json.loads(pathlib.Path(p).read_text())


# ------------------------------------------------------------------ R2 the stall
def r2() -> None:
    cfg = EnvConfig()
    ref = _j(OUT / "figure_data_s26.json")["F6a"]
    cell = _j(R / "s26_b1_cells" / f"{ref['policy']}_det.json")
    st = next(s for s in cell["states"] if s["seed"] == ref["seed"] and s["t"] == ref["t"] and s["kind"] == "stalled")
    n1 = _j(R / "s27_n1_round.json")
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(TW, 2.05), gridspec_kw={"width_ratios": [1.0, 2.15], "wspace": 0.55})

    n = 41
    V, W = np.meshgrid(np.linspace(0, cfg.v_max, n), np.linspace(-cfg.omega_max, cfg.omega_max, n), indexing="ij")
    U = np.column_stack([V.ravel(), W.ravel()])
    adm = np.zeros(len(U), bool)
    adm[st["admissible_grid"]] = True
    axa.scatter(U[~adm, 1], U[~adm, 0] / cfg.v_max, s=1.6, color=LIGHT, rasterized=True, label="blocked by the filter")
    axa.scatter(U[adm, 1], U[adm, 0] / cfg.v_max, s=5, color=FC["Obs"], rasterized=True, label="allowed")
    axa.scatter([st["w_nom"]], [st["v_nom"] / cfg.v_max], marker="*", s=70, color=INK, zorder=5, label="what the pursuer asks for")
    axa.scatter([st["u_shield"][1]], [st["u_shield"][0] / cfg.v_max], marker="x", s=40, color=FC["Hor"], zorder=6, lw=1.2, label="what the filter executes")
    axa.set_xlabel(r"turn rate $\omega$ (rad/s)")
    axa.set_ylabel(r"speed $v/v_{\max}$")
    axa.set_xlim(-3.0, 3.0)
    axa.set_ylim(-0.05, 1.08)
    axa.set_title("(a) commands at one stalled state", loc="left")
    axa.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2, handletextpad=0.2, columnspacing=0.6, markerscale=1.3, fontsize=6.5)

    rows = [("Base", "BASE"), ("Obs", "OBS"), ("Hor", "GAM"), ("ObsHor", "OBSGAM"), ("Free", "STR14"), ("Aware", "RC19")]
    opps = (("ATK4-45", "learned attacker"), ("GREEDY", "greedy evader"))
    y, labels, ys = 0.0, [], []
    out = {}
    for name, key in rows:
        for oi, (opp, olab) in enumerate(opps):
            src = n1["families"] if name in FAMKEY else n1["single_policies"]
            rec = src[f"{key}_{opp}"]
            sh, nd = rec["first_k_share"], rec["n_deadlock"]
            left = 0.0
            for comp in ("last_seen", "chase", "head"):
                axb.barh(y, sh[comp], left=left, height=0.38, color=CC[comp], alpha=1.0 if oi == 0 else 0.55, edgecolor="white", lw=0.4)
                left += sh[comp]
            axb.text(1.01, y, f"{sh['head']:.2f}  (n = {nd})", va="center", fontsize=6.5, color=INK if oi == 0 else MUTED)
            out[f"{name}_{opp}"] = {"share": sh, "n_deadlock": nd}
            ys.append(y)
            labels.append(f"{name}, {'attacker' if oi == 0 else 'greedy'}")
            y += 0.42
        y += 0.32
    axb.set_yticks(ys, labels, fontsize=6.5)
    axb.invert_yaxis()
    axb.set_xlim(0, 1)
    axb.set_xlabel("share of deadlock episodes, by who commanded the 40th stalled step")
    axb.text(1.01, -0.55, "head share", fontsize=6.5, color=INK)
    axb.set_title("(b) who is stuck: the scripted law or the learned head", loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, color=CC[c]) for c in ("last_seen", "chase", "head")]
    axb.legend(handles, [CNAME[c] for c in ("last_seen", "chase", "head")], loc="upper center", bbox_to_anchor=(0.45, -0.22), ncol=3,
               handlelength=1.0, columnspacing=0.8, fontsize=6.5)
    for s in ("left",):
        axb.spines[s].set_visible(False)
    DATA["R2"] = {"a_state": {k: ref[k] for k in ("policy", "seed", "t", "v_nom", "w_nom", "u_shield", "n_admissible_grid", "trap_heading", "trap_position")},
                  "b_first_k": out, "source": "s26_b1_cells (a); s27_n1_round.json families / single_policies (b)"}
    _save(fig, "R2_stall")


# ------------------------------------------------------------------ R4 the recovery
def r4() -> None:
    fig, (axa, axb, axc) = plt.subplots(1, 3, figsize=(TW, 2.0), gridspec_kw={"width_ratios": [1.25, 1, 1], "wspace": 0.5})
    pts = {}
    for name, fam in FAMKEY.items():
        for rec in ((False,) if fam == "GAM" else (False, True)):
            for s in (60, 61, 62):
                a = _j(R / f"s17_arm_{'RECF-' if rec else ''}{fam}-{s}.json")["final"]["gate"]
                x, yv = a["ATK4-45"]["pursuer"]["deadlock_episode_rate"], a["GREEDY"]["capture_rate"]
                pts[f"{'RECF-' if rec else ''}{name}-{s}"] = (x, yv)
                axa.scatter(x, yv, s=20, marker="D" if rec else "o", facecolors="white" if rec else FC[name], edgecolors=FC[name], lw=1.0, zorder=3)
    for name in ("Base", "Obs", "ObsHor"):
        for s in (60, 61, 62):
            (x0, y0), (x1, y1) = pts[f"{name}-{s}"], pts[f"RECF-{name}-{s}"]
            axa.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops={"arrowstyle": "-|>", "color": FC[name], "lw": 0.5, "alpha": 0.6, "mutation_scale": 6})
    axa.axvline(0.05, color=MUTED, lw=0.7, ls=":")
    axa.set_xlabel("deadlock episodes (vs attacker)")
    axa.set_ylabel("capture (vs greedy evader)")
    axa.set_xlim(-0.03, 0.65)
    axa.set_ylim(0.2, 0.76)
    axa.grid()
    axa.set_title("(a) Exp. 6: recovery off / on", loc="left")
    fam_h = [plt.Line2D([], [], marker="o", ls="", color=FC[n], ms=4, label=n) for n in FAMKEY]
    shp = [plt.Line2D([], [], marker="o", ls="", color=INK, ms=4, label="no recovery"),
           plt.Line2D([], [], marker="D", ls="", mfc="white", mec=INK, ms=4, label="+ recovery")]
    axa.legend(handles=fam_h + shp, loc="lower right", ncol=2, fontsize=6, handletextpad=0.1, columnspacing=0.4, labelspacing=0.15)

    cells = {}
    for p in sorted((R / "s26_b2_cells").glob("*.json")):
        rec = _j(p)
        if rec.get("status") != "UNDETERMINED":
            cells[rec["name"]] = rec
    pol = {"Base 1": ("BASE-60", FC["Base"], "o"), "Base 2": ("BASE-61", FC["Base"], "s"), "Base 3": ("BASE-62", FC["Base"], "^"),
           "Free": ("STR14", FC["Free"], "D"), "Aware": ("RC19", FC["Aware"], "v")}
    xs = ("PLAIN", "RECF-5", "RECF-10", "RECF-20", "RECF-40", "RECF-80")
    x = np.arange(len(xs))
    out = {}
    for lab, (key, col, mk) in pol.items():
        dead = [cells[f"{key}_ATK4-45_{s}"]["deadlock_episode_rate"] for s in xs]
        cap = [cells[f"{key}_GREEDY_{s}"]["capture_rate"] for s in xs]
        axb.plot(x[:4], dead[:4], color=col, marker=mk, ms=3, lw=0.9, label=lab)  # K = 40 indicator is uninformative for N >= 40
        axc.plot(x, cap, color=col, marker=mk, ms=3, lw=0.9, label=lab)
        pd_, pc = cells[f"{key}_ATK4-45_PERT-20"]["deadlock_episode_rate"], cells[f"{key}_GREEDY_PERT-20"]["capture_rate"]
        axb.scatter([3.3], [pd_], s=16, marker=mk, facecolors="none", edgecolors=col, lw=0.8, zorder=4)
        axc.scatter([3.3], [pc], s=16, marker=mk, facecolors="none", edgecolors=col, lw=0.8, zorder=4)
        out[lab] = {"deadlock_ATK": dead, "capture_GREEDY": cap, "pert20": [pd_, pc]}
    for ax, yl, t in ((axb, "deadlock (vs attacker)", "(b) Exp. 7: deadlock"), (axc, "capture (vs greedy)", "(c) Exp. 7: capture")):
        ax.set_xticks(x if ax is axc else x[:4], ["off", "5", "10", "20", "40", "80"] if ax is axc else ["off", "5", "10", "20"])
        ax.set_xlabel("recovery trigger $N$ (steps)")
        ax.set_ylabel(yl)
        ax.grid(axis="y")
        ax.set_title(t, loc="left")
    axb.axhline(0.05, color=MUTED, lw=0.7, ls=":")
    h, l = axc.get_legend_handles_labels()
    h.append(plt.Line2D([], [], marker="o", ls="", mfc="none", mec=MUTED, ms=4))
    l.append("random perturbation, $N$ = 20")
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.66, -0.1), ncol=6, fontsize=6.5, handlelength=1.4, columnspacing=0.8)
    DATA["R4"] = {"a_exp4": pts, "bc_exp8": out, "source": "s17_arm_*.json final.gate (a); s26_b2_cells (b, c)"}
    _save(fig, "R4_recovery")


# ------------------------------------------------------------------ R5 dissociations
def r5() -> None:
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(TW * 0.62, 1.85), gridspec_kw={"wspace": 0.5, "width_ratios": [1.15, 1]})
    names = ["stalled\nsteps", "immobile\nsteps", "capture"]
    out = {}
    for j, s in enumerate((60, 61, 62)):
        b = _j(R / f"s16_arm_BASE-{s}.json")["final"]["gate"]
        p = _j(R / f"s16_arm_PEN-{s}.json")["final"]["gate"]
        vb = [b["ATK4-45"]["pursuer"]["stall_step_rate"], b["ATK4-45"]["pursuer"]["immobile_step_rate"], b["GREEDY"]["capture_rate"]]
        vp = [p["ATK4-45"]["pursuer"]["stall_step_rate"], p["ATK4-45"]["pursuer"]["immobile_step_rate"], p["GREEDY"]["capture_rate"]]
        for i in range(3):
            xx = i + 0.2 * (j - 1)
            axa.annotate("", xy=(xx, vp[i]), xytext=(xx, vb[i]), arrowprops={"arrowstyle": "-|>", "color": FC["Hor"], "lw": 0.7, "mutation_scale": 6})
            axa.scatter([xx], [vb[i]], s=10, color=FC["Base"], zorder=3)
        out[f"exp3_seed{s}"] = {"Base": vb, "penalty": vp}
    axa.set_xticks(range(3), names)
    axa.set_ylim(0, 0.9)
    axa.set_ylabel("rate")
    axa.grid(axis="y")
    axa.set_title("(a) stall penalty (Exp. 5)", loc="left")
    s18 = _j(R / "s18_round.json")["arms"]
    for s in (70, 71, 72):
        b, p = s18[f"DVARY-BASE-{s}"]["final"]["gate"]["GREEDY"], s18[f"DVARY-PEN-{s}"]["final"]["gate"]["GREEDY"]
        xb, yb, xp, yp = b["pursuer"]["deadlock_episode_rate"], b["capture_rate"], p["pursuer"]["deadlock_episode_rate"], p["capture_rate"]
        axb.annotate("", xy=(xp, yp), xytext=(xb, yb), arrowprops={"arrowstyle": "-|>", "color": FC["Hor"], "lw": 0.7, "mutation_scale": 6})
        axb.scatter([xb], [yb], s=10, color=FC["Base"], zorder=3)
        out[f"exp5_seed{s}"] = {"Base": [xb, yb], "penalty": [xp, yp]}
    axb.set_xlabel("deadlock episodes")
    axb.set_ylabel("capture")
    axb.set_xlim(0.05, 0.26)
    axb.set_ylim(0.15, 0.6)
    axb.grid()
    axb.set_title("(b) correction penalty (Exp. 9)", loc="left")
    axa.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=FC["Base"], ms=3.5, label="without penalty"),
                        plt.Line2D([], [], color=FC["Hor"], lw=0.8, marker=">", ms=3, label="with penalty")],
               loc="upper left", fontsize=6.5, handlelength=1.2)
    DATA["R5"] = {**out, "source": "s16_arm_{BASE,PEN}-*.json (a); s18_round.json DVARY arms, GREEDY gate (b)"}
    _save(fig, "R5_dissoc")


if __name__ == "__main__":
    r2()
    r4()
    r5()
    f = OUT / "figure_data_rev.json"
    old = json.loads(f.read_text()) if f.exists() else {}  # keeps R0, written by paper_b_game_figure.py
    f.write_text(json.dumps(old | DATA, indent=1, default=float))
    print("wrote", OUT / "figure_data_rev.json")
