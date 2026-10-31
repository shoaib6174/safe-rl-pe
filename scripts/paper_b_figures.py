"""Paper B figures from ARCHIVED records only (evidence audit §4, F1–F5).

Reads `results/derisk/s15_*`, `s16_arm_*`, `s17_arm_*`, `s18_round.json`,
`s19_round.json`; writes `docs/paper_b/figures/F*.{pdf,png}` and a
`figure_data.json` with every number drawn, so each panel is traceable to its
record. No round is re-run and nothing under `results/` is written.

Colour: three validated categorical hues (blue/orange/aqua) for the learned
remedies, neutral gray for the reference arm; the recovery is a marker/line
style, never a fourth hue. Pursuer RED / evader BLUE conventions of the GIFs are
not used here (no agents are drawn).

  ./venv/bin/python scripts/paper_b_figures.py
"""

from __future__ import annotations

import glob
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

R = pathlib.Path("results/derisk")
OUT = pathlib.Path("docs/paper_b/figures")
C = {"BASE": "#52514e", "OBS": "#2a78d6", "GAM": "#eb6834", "OBSGAM": "#1baf7a", "ink": "#0b0b0b", "muted": "#8a8984", "grid": "#e6e5e1", "bar": "#b0afa9"}
FAM = ("BASE", "OBS", "GAM", "OBSGAM")
SEEDS = (60, 61, 62)
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#52514e", "axes.linewidth": 0.6,
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.labelcolor": "#0b0b0b", "grid.color": C["grid"], "grid.linewidth": 0.5,
                     "legend.frameon": False, "pdf.fonttype": 42})
DATA: dict = {}


def _save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / f"{name}.pdf")


def arm(name: str) -> dict:
    return json.loads((R / f"s17_arm_{name}.json").read_text())


# ---------------------------------------------------------------- F1 frozen input
def f1() -> None:
    bare = arm("OBS-61")["final"]["gate"]["HUG7"]["traces"][3]
    recf = arm("RECF-OBS-61")["final"]["gate"]["HUG7"]["traces"][3]
    fig, axes = plt.subplots(2, 1, figsize=(6.8, 3.4), sharex=True, gridspec_kw={"height_ratios": [1, 1]})
    for ax, tr, title in ((axes[0], bare, "plain CBF filter (OBS-61, episode 1 740 003 vs HUG7)"), (axes[1], recf, "same policy, same episode, forward-offering recovery (N = 20)")):
        t = np.arange(len(tr))
        vn = np.array([s["v_nom"] for s in tr])
        ve = np.array([s["v_exec"] for s in tr])
        ax.plot(t, vn, color=C["muted"], lw=1.0, label="commanded speed v_nom")
        ax.plot(t, ve, color=C["OBS"], lw=1.4, label="executed speed v_exec")
        fired = np.array([s["recovery_kind"] != "none" for s in tr])
        if fired.any():
            ax.scatter(t[fired], np.full(fired.sum(), -0.06), s=6, color=C["GAM"], marker="|", label="recovery fires")
        ax.set_ylim(-0.12, 1.08)
        ax.set_ylabel("speed (m/s)")
        ax.set_title(title, loc="left", fontsize=8)
        ax.grid(axis="y")
    axes[0].annotate("stalled 1 106 steps: v_nom = 1.000 ± 0.000, v_exec = 0", xy=(600, 0.5), fontsize=7.5, color=C["ink"])
    axes[1].annotate(f"captured at step {len(recf)}", xy=(len(recf) + 10, 0.5), fontsize=7.5, color=C["ink"])
    axes[1].set_xlabel("environment step (dt = 0.05 s)")
    axes[1].legend(loc="upper right", fontsize=7, ncol=3)
    axes[1].set_xlim(0, 1200)
    DATA["F1"] = {"source": "s17_arm_OBS-61.json / s17_arm_RECF-OBS-61.json, gate HUG7, traces[3]", "bare_len": len(bare), "recf_len": len(recf),
                  "bare_immobile_frac": float(np.mean([s["immobile"] for s in bare])), "recf_firings": int(sum(s["recovery_kind"] != "none" for s in recf))}
    _save(fig, "F1_frozen_input")


# ---------------------------------------------------------------- F2 mechanism
def f2() -> None:
    rnd = json.loads((R / "s15_round.json").read_text())["verdicts"]["m1"]
    entries = [e for p in sorted(glob.glob(str(R / "s15_cells" / "*.json"))) for e in json.loads(open(p).read())["entries"]]
    esc = [e for e in entries if e["escaped"]]
    fig, axes = plt.subplots(1, 3, figsize=(7.8, 2.4), gridspec_kw={"wspace": 0.5})
    # (a) capture after escape vs not
    vals = [rnd["capture_rate_escaped"], rnd["capture_rate_not_escaped"]]
    bars = axes[0].bar(["escaped the stall", "did not"], vals, color=[C["OBS"], C["bar"]], width=0.55)
    for b, v in zip(bars, vals, strict=True):
        axes[0].text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=7.5)
    axes[0].set_ylabel("P(capture | stall entry)")
    axes[0].set_title(f"(a) escaping multiplies capture {rnd['capture_rate_escaped'] / rnd['capture_rate_not_escaped']:.1f}×\nS15, {rnd['n_entries']} stall entries", fontsize=7.5, loc="left")
    axes[0].set_ylim(0, 0.36)
    # (b) delay distribution
    delays = np.array([e["steps_remaining"] for e in esc])
    axes[1].hist(delays, bins=24, color=C["OBS"], edgecolor="white", linewidth=0.4)
    axes[1].axvline(rnd["mean_steps_remaining_escaped"], color=C["ink"], lw=0.8, ls="--")
    axes[1].text(rnd["mean_steps_remaining_escaped"] - 15, axes[1].get_ylim()[1] * 0.9, f"mean {rnd['mean_steps_remaining_escaped']:.0f}", fontsize=7, ha="right")
    axes[1].set_xlabel("steps from escape to episode end")
    axes[1].set_ylabel("stall entries (all 8 cells)")
    axes[1].set_title(f"(b) the consequence arrives late\n{len(delays)} escapes, 8-cell pool", fontsize=7.5, loc="left")
    # (c) discounted advantage vs gamma
    A, T = 0.257, rnd["mean_steps_remaining_escaped"]
    g = np.linspace(0.985, 0.9995, 400)
    axes[2].plot(g, A * g ** T, color=C["OBS"], lw=1.4)
    axes[2].axhline(0.05, color=C["muted"], lw=0.8, ls=":")
    axes[2].text(0.9852, 0.055, "0.05 bar", fontsize=7, color=C["muted"])
    for gg, lab in ((0.99, "γ = 0.99\n(training)"), (0.999, "γ = 0.999\n(GAM arm)")):
        axes[2].scatter([gg], [A * gg ** T], s=22, color=C["ink"], zorder=3)
        axes[2].annotate(lab, (gg, A * gg ** T), textcoords="offset points", xytext=(4, 6), fontsize=7)
    axes[2].set_xlabel("discount γ")
    axes[2].set_ylabel("discounted escape advantage")
    axes[2].set_title(f"(c) advantage 0.257 · γ^{T:.0f}\nvs the discount", fontsize=7.5, loc="left")
    for ax in axes:
        ax.grid(axis="y")
    DATA["F2"] = {"source": "s15_round.json verdicts.m1 (hold-10 cells) + s15_cells/*.json entries (8-cell pool)", "capture_escaped": vals[0], "capture_not": vals[1],
                  "n_entries_m1": rnd["n_entries"], "mean_delay_m1": T, "n_escapes_pool": int(len(delays)), "pool_mean_delay": float(delays.mean()),
                  "advantage_at_0.99": float(A * 0.99 ** T), "advantage_at_0.999": float(A * 0.999 ** T)}
    _save(fig, "F2_mechanism")


# ---------------------------------------------------------------- F3 remedies
def f3() -> None:
    fig, ax = plt.subplots(figsize=(4.4, 3.3))
    pts = {}
    for fam in FAM:
        for rec in ((False,) if fam == "GAM" else (False, True)):  # S17 had no RECF-GAM arm
            name = f"RECF-{fam}" if rec else fam
            for s in SEEDS:
                a = arm(f"{name}-{s}")["final"]["gate"]
                x = a["ATK4-45"]["pursuer"]["deadlock_episode_rate"]
                y = a["GREEDY"]["capture_rate"]
                pts[f"{name}-{s}"] = (x, y)
                ax.scatter(x, y, s=34, color=C[fam], marker="o" if not rec else "D", facecolors=C[fam] if not rec else "white", edgecolors=C[fam], linewidths=1.2, zorder=3)
    # arrows from each bare seed to its recovered twin
    for fam in FAM:
        if fam == "GAM":
            continue
        for s in SEEDS:
            x0, y0 = pts[f"{fam}-{s}"]
            x1, y1 = pts[f"RECF-{fam}-{s}"]
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops={"arrowstyle": "-", "color": C[fam], "lw": 0.6, "alpha": 0.5})
    ax.axvline(0.05, color=C["muted"], lw=0.8, ls=":")
    ax.text(0.055, 0.33, "deadlock bar 0.05", fontsize=7, color=C["muted"], rotation=90, va="bottom")
    ax.set_xlabel("deadlock-episode rate vs learned attacker (ATK4-45)")
    ax.set_ylabel("capture rate vs GreedyEvader")
    ax.set_xlim(-0.02, 0.65)
    ax.set_ylim(0.3, 0.76)
    ax.grid()
    # legends: hue = family (direct labels), shape = recovery
    for fam, (lx, ly) in zip(FAM, ((0.30, 0.37), (0.40, 0.52), (0.50, 0.44), (0.40, 0.60)), strict=True):
        ax.text(lx, ly, {"BASE": "BASE (reference)", "OBS": "OBS: admissible-turn features", "GAM": "GAM: γ = 0.999", "OBSGAM": "OBSGAM: both"}[fam], color=C[fam], fontsize=7)
    h1 = ax.scatter([], [], marker="o", color=C["ink"], s=30, label="bare policy")
    h2 = ax.scatter([], [], marker="D", facecolors="white", edgecolors=C["ink"], s=30, label="+ forward-offering recovery (N = 20)")
    ax.legend(handles=[h1, h2], loc="upper right", fontsize=7)
    ax.set_title("S17: 7 families × 3 seeds, N = 500 paired, 0 violations", fontsize=8, loc="left")
    DATA["F3"] = {"source": "s17_arm_*.json final.gate", "points": pts}
    _save(fig, "F3_remedies")


# ---------------------------------------------------------------- F4 firings (layering)
def f4() -> None:
    fig, ax = plt.subplots(figsize=(4.4, 2.4))
    labels, fw, rot, cols = [], [], [], []
    for fam in FAM:
        if fam == "GAM":
            continue  # no RECF-GAM arm in S17
        for s in SEEDS:
            k = arm(f"RECF-{fam}-{s}")["final"]["recovery_kinds"]
            labels.append(f"{fam}-{s}")
            fw.append(k["forward"])
            rot.append(k["rotation"])
            cols.append(C[fam])
    x = np.arange(len(labels))
    ax.bar(x, fw, color=cols, width=0.62, label="forward escape")
    ax.bar(x, rot, bottom=fw, color=cols, alpha=0.35, width=0.62, hatch="////", edgecolor="white", linewidth=0, label="rotation")
    ax.set_yscale("log")
    ax.set_ylim(10, 5e6)
    ax.set_xticks(x, labels, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("recovery firings per 1 500 episodes")
    for i, (a, b) in enumerate(zip(fw, rot, strict=True)):
        ax.text(i, (a + b) * 1.25, f"{a + b:,}", ha="center", fontsize=6.5)
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.set_title("the learned half cuts how often the escape is needed (RECF-OBSGAM-61: 363)", fontsize=8, loc="left")
    ax.grid(axis="y")
    DATA["F4"] = {"source": "s17_arm_RECF-*.json final.recovery_kinds", "forward": dict(zip(labels, fw, strict=True)), "rotation": dict(zip(labels, rot, strict=True))}
    _save(fig, "F4_firings")


# ---------------------------------------------------------------- F5 three dissociations
def f5() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.8, 2.5), gridspec_kw={"wspace": 0.45})
    # (a) S16 PEN vs BASE as dumbbells: the penalised quantity falls, immobility and capture move the wrong way
    s16 = {a: json.loads((R / f"s16_arm_{a}.json").read_text())["final"]["gate"] for a in [f"{f}-{s}" for f in ("BASE", "PEN") for s in SEEDS]}
    names = ["stall-step\n(penalised)", "immobile-step", "capture\nvs GREEDY"]
    for j, s in enumerate(SEEDS):
        b, p = s16[f"BASE-{s}"], s16[f"PEN-{s}"]
        vb = [b["ATK4-45"]["pursuer"]["stall_step_rate"], b["ATK4-45"]["pursuer"]["immobile_step_rate"], b["GREEDY"]["capture_rate"]]
        vp = [p["ATK4-45"]["pursuer"]["stall_step_rate"], p["ATK4-45"]["pursuer"]["immobile_step_rate"], p["GREEDY"]["capture_rate"]]
        for i in range(3):
            x = i + 0.22 * (j - 1)
            axes[0].plot([x, x], [vb[i], vp[i]], color=C["muted"], lw=0.8, zorder=1)
            axes[0].scatter([x], [vb[i]], s=18, color=C["BASE"], zorder=3)
            axes[0].scatter([x], [vp[i]], s=18, color=C["GAM"], zorder=3)
        DATA.setdefault("F5a", {})[f"seed{s}"] = {"BASE": vb, "PEN": vp}
    axes[0].scatter([], [], s=18, color=C["BASE"], label="BASE")
    axes[0].scatter([], [], s=18, color=C["GAM"], label="PEN (10× stall penalty)")
    axes[0].legend(fontsize=6.5, loc="upper left")
    axes[0].set_xticks(range(3), names, fontsize=6.5)
    axes[0].set_ylim(0, 0.95)
    axes[0].set_title("(a) S16: penalising the stall, 3 seeds", fontsize=7.5, loc="left")
    # (b) S18 DVARY: deadlock down, capture down
    s18 = json.loads((R / "s18_round.json").read_text())["arms"]
    for s in (70, 71, 72):
        b, p = s18[f"DVARY-BASE-{s}"]["final"]["gate"]["GREEDY"], s18[f"DVARY-PEN-{s}"]["final"]["gate"]["GREEDY"]
        xb, yb, xp, yp = b["pursuer"]["deadlock_episode_rate"], b["capture_rate"], p["pursuer"]["deadlock_episode_rate"], p["capture_rate"]
        axes[1].plot([xb, xp], [yb, yp], color=C["muted"], lw=0.8, zorder=1)
        axes[1].scatter([xb], [yb], s=18, color=C["BASE"], zorder=3)
        axes[1].scatter([xp], [yp], s=18, color=C["GAM"], zorder=3)
        DATA.setdefault("F5b", {})[f"seed{s}"] = {"BASE": [xb, yb], "PEN": [xp, yp]}
    axes[1].scatter([], [], s=18, color=C["BASE"], label="BASE")
    axes[1].scatter([], [], s=18, color=C["GAM"], label="+ intervention penalty")
    axes[1].legend(fontsize=6.5, loc="lower right")
    axes[1].set_xlabel("deadlock-episode rate")
    axes[1].set_ylabel("capture vs GREEDY")
    axes[1].set_xlim(0.05, 0.26)
    axes[1].set_ylim(0.15, 0.6)
    axes[1].set_title("(b) S18: deadlock improves, capture falls", fontsize=7.5, loc="left")
    # (c) S19 law pins removed, capture flat
    v = json.loads((R / "s19_round.json").read_text())["verdicts"]
    opps = ("GREEDY", "HUG", "ATK")
    pin_l = [v["h_plan_3"]["opponents"][o]["frac_line"] for o in opps]
    pin_p = [v["h_plan_3"]["opponents"][o]["frac_plan"] for o in opps]
    cap_l = [v["h_plan_1"]["opponents"][o]["capture_line"] for o in opps]
    cap_p = [v["h_plan_1"]["opponents"][o]["capture_plan"] for o in opps]
    x = np.arange(3)
    axes[2].bar(x - 0.2, pin_l, width=0.18, color=C["bar"], label="law-pinned episodes, LINE")
    axes[2].bar(x - 0.0, pin_p, width=0.18, color=C["OBS"], label="law-pinned episodes, PLAN")
    axes[2].scatter(x + 0.22, cap_l, marker="_", s=120, color=C["BASE"], label="capture, LINE")
    axes[2].scatter(x + 0.22, cap_p, marker="_", s=120, color=C["GAM"], label="capture, PLAN")
    axes[2].set_xticks(x, opps)
    axes[2].set_ylim(0, 0.62)
    axes[2].legend(fontsize=5.8, loc="upper right", ncol=2)
    axes[2].set_title("(c) S19: pins −94…−99 %, capture flat", fontsize=7.5, loc="left")
    DATA["F5c"] = {"pin_line": pin_l, "pin_plan": pin_p, "capture_line": cap_l, "capture_plan": cap_p}
    for ax in axes:
        ax.grid(axis="y")
    _save(fig, "F5_dissociations")


if __name__ == "__main__":
    f1()
    f2()
    f3()
    f4()
    f5()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figure_data.json").write_text(json.dumps(DATA, indent=1, default=float))
    print("wrote", OUT / "figure_data.json")
