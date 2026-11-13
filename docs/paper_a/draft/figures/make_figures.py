"""Paper A figures, rendered from the archived round records only (results/derisk/*.json).

F2  baselines.pdf   capture rate per pursuer family x scripted evader (S24.1 + S24-FLAT), Wilson 95% CIs
F3  metagame.pdf    4x4 tactic meta-game matrix with the pure saddle (S24.2 stage 3)
F4  valuesweep.pdf  S25: capture vs the learned evader across filter settings; LP value curve

Run: ./venv/bin/python docs/paper_a/draft/figures/make_figures.py
"""
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[4]
D = ROOT / "results" / "derisk"
OUT = pathlib.Path(__file__).resolve().parent

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#777777", "axes.linewidth": 0.6, "xtick.color": "#555555", "ytick.color": "#555555",
    "grid.color": "#dddddd", "grid.linewidth": 0.5, "pdf.fonttype": 42,
})
# Okabe-Ito, fixed order (validated with the dataviz palette script, light surface).
C = {"blue": "#0072B2", "verm": "#D55E00", "green": "#009E73", "purple": "#CC79A7", "ink": "#222222", "muted": "#8a8a8a"}


def load(name):
    return json.load(open(D / f"{name}.json"))


def ci(cell):
    w = cell["wilson_capture"]
    return w["rate"], w["wilson"][0], w["wilson"][1]


# ----------------------------------------------------------------------------- F2
def fig_baselines():
    s1, sf = load("s24_1_round")["cells"], load("s24_flat_round")["cells"]
    cells = {**s1, **sf}
    evaders = [("GREEDY", "Greedy evader"), ("HUG", "Obstacle-hugging evader"), ("TACTIC-V0", "Scripted tactical evader")]
    # (label, row prefixes in the records, seeds)
    rows = [
        ("Flat PPO, 40-step hold", ["FLAT40-80", "FLAT40-81", "FLAT40-82"]),
        ("Flat PPO, 10-step hold", ["FLAT-80", "FLAT-81", "FLAT-82"]),
        ("Structured search + law", ["STRUCT-80", "STRUCT-81", "STRUCT-82"]),
        ("Best fixed law (LEAD)", ["LEAD-1.0"]),
        ("Hand-written rules", ["RULES-V0"]),
        ("Selector, no memory", ["NOMEM-80", "NOMEM-81", "NOMEM-82"]),
        ("Selector (ours)", ["SEL-80", "SEL-81", "SEL-82"]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.5), sharey=True)
    for ax, (ev, title) in zip(axes, evaders):
        ax.set_title(title, loc="left", color=C["ink"])
        ax.grid(axis="x")
        ax.set_axisbelow(True)
        for yi, (label, prefixes) in enumerate(rows):
            col = C["verm"] if label.startswith("Selector (ours)") else C["blue"]
            for j, p in enumerate(prefixes):
                r, lo, hi = ci(cells[f"{p}_{ev}"])
                off = (j - (len(prefixes) - 1) / 2) * 0.22
                ax.plot([lo, hi], [yi + off, yi + off], color=col, lw=0.9, alpha=0.55, solid_capstyle="butt")
                ax.plot(r, yi + off, "o", ms=3.6, color=col, mec="white", mew=0.5)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    axes[1].set_xlabel("capture rate over 500 paired episodes (dot = one seed; bar = Wilson 95% interval)")
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels([r[0] for r in rows])
    axes[0].set_ylim(-0.6, len(rows) - 0.4)
    fig.tight_layout(w_pad=1.0)
    fig.savefig(OUT / "baselines.pdf")
    plt.close(fig)


# ----------------------------------------------------------------------------- F3
def fig_metagame():
    m = load("s24_2_meta_game")
    M = np.array(m["matrix"])
    rows = ["Rules", "Lead law", "Selector", "Selector,\nretrained"]
    cols = ["Greedy", "Hugger", "Scripted\ntactical", "Learned\nselector"]
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            v = M[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8,
                    color="white" if v > 0.55 else C["ink"])
    ri = m["pursuers"].index(m["game"]["row_support"][0])
    cj = m["evaders"].index(m["game"]["col_support"][0])
    ax.add_patch(plt.Rectangle((cj - 0.5, ri - 0.5), 1, 1, fill=False, ec=C["verm"], lw=2.0))
    ax.set_xticks(range(4)); ax.set_xticklabels(cols)
    ax.set_yticks(range(4)); ax.set_yticklabels(rows)
    ax.set_xlabel("evader")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("pursuer capture rate")
    cb.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "metagame.pdf")
    plt.close(fig)


# ----------------------------------------------------------------------------- F4
def fig_valuesweep():
    r = load("s25_round")
    cells, hv4 = r["cells"], r["verdicts"]["h_val_4"]
    settings = ["NONE", "d0.00", "d0.05", "d0.10", "d0.20"]
    xt = ["no filter", "δ = 0", "0.05", "0.10", "0.20 m"]
    purs = [("SEL3-80", "Selector, retrained", C["blue"], "o"), ("SEL-81", "Selector", C["verm"], "s"),
            ("RULES-V0", "Rules", C["green"], "^"), ("FULLOBS", "Full-state greedy", C["purple"], "D")]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.16, 2.6), gridspec_kw={"width_ratios": [1.25, 1]})
    x = np.arange(len(settings))
    for key, label, col, mk in purs:
        ys, los, his = zip(*[ci(cells[f"{key}_EVS-80_{s}"]) for s in settings])
        a.fill_between(x, los, his, color=col, alpha=0.12, lw=0)
        a.plot(x, ys, "-" + mk, color=col, lw=1.6, ms=4.2, mec="white", mew=0.5, label=label)
        nudge = {"RULES-V0": 5, "SEL-81": -5}.get(key, 0)  # the two end within 0.01 of each other
        a.annotate(label, (x[-1], ys[-1]), xytext=(5, nudge), textcoords="offset points", va="center", fontsize=7, color=C["ink"])
    a.set_xticks(x); a.set_xticklabels(xt)
    a.set_ylim(0, 0.25); a.set_ylabel("capture rate vs the learned evader")
    a.set_xlabel("filter setting (obstacle margin δ)")
    a.grid(axis="y"); a.set_axisbelow(True)
    a.set_xlim(-0.2, len(settings) - 1 + 1.9)
    a.set_title("(a) frozen pursuers against the learned evader", loc="left")

    vals = [hv4["values"][s] for s in settings]
    b.plot(x, vals, "-s", color=C["ink"], lw=1.6, ms=4.5, mec="white", mew=0.5)
    for xi, s in zip(x, settings):
        sup = hv4["per_setting"][s]["col_support"]
        name = {"EVS-80": "learned", "HUG": "hugger"}.get(sup[0], sup[0]) if len(sup) == 1 else "mixed"
        b.annotate(name, (xi, vals[list(settings).index(s)]), xytext=(0, 7), textcoords="offset points",
                   ha="center", fontsize=7, color=C["muted"])
    b.set_xticks(x); b.set_xticklabels(xt)
    b.set_ylim(0, 0.25); b.set_ylabel("value of the 4×4 capture game")
    b.set_xlabel("filter setting (obstacle margin δ)")
    b.grid(axis="y"); b.set_axisbelow(True)
    b.set_title("(b) game value and the evader in its support", loc="left")
    fig.tight_layout(w_pad=1.5)
    fig.savefig(OUT / "valuesweep.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig_baselines()
    fig_metagame()
    fig_valuesweep()
    print("wrote", sorted(p.name for p in OUT.glob("*.pdf")))
