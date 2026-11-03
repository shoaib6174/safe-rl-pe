"""Paper B Figure 1: the game, drawn from a replay of the archived Figure-1 episode.

The episode is Free (STR14) against the greedy evader, Experiment-8 stream
seed 1 740 044 (the revision's numbering: Experiment 7), whose first 40-step
stall the N1 round attributed to the LEARNED HEAD (start 8, length 1 192). It is
replayed through `pe2.recorder.BranchTaggedSearchMDP` exactly as the N1 round
replayed it, once under the plain filter and once under the recovery at N = 20
(`paper_b_filter_sweep.shield_factory`). The replay must reproduce the archived
capture indicators of `s26_b2_cells/STR14_GREEDY_{PLAIN,RECF-20}.json` and N1's
first K-run, or the script refuses to draw. Evaluation only; nothing is
written to `results/`.

Usage: ./venv/bin/python scripts/paper_b_game_figure.py
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
ROOT = _HERE.parent
sys.path.insert(0, str(ROOT))

from pe2.config import EnvConfig  # noqa: E402

OUT = ROOT / "docs" / "paper_b" / "figures"
POLICY, OPP, EP_I, SEED0 = "STR14", "GREEDY", 44, 1_740_000
R = ROOT / "results" / "derisk"
ANNOT_DX, ANNOT_DY = 10, -58  # label offset for the stalled pose (points)
TEXTWIDTH_IN = 433.62 / 72.27  # l4dc2027.cls \textwidth


def _sweep():
    spec = importlib.util.spec_from_file_location("gf_sweep", _HERE / "paper_b_filter_sweep.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def play(recovery: bool) -> dict:
    from stable_baselines3 import PPO

    from pe2.recorder import BranchTaggedSearchMDP
    from pe2.structured import HOLD

    class PoseRecorder(BranchTaggedSearchMDP):
        def _step_inner(self, action, branch="law"):
            env = self.inner.env
            ls = getattr(self.law, "last_seen", None)
            pre = (np.array(env.pursuer_pose, float), np.array(env.evader_pose, float), None if ls is None else np.array(ls, float))
            out = super()._step_inner(action, branch)
            rec = self.step_records[-1]
            rec["p"], rec["e"], rec["last_seen"] = pre
            rec["fired"] = bool(self._last_sinfo.get("recovery_fired", False)) if self._last_sinfo else False
            return out

    m = _sweep()
    cfg = EnvConfig()
    pol = m.POLICIES[POLICY]
    model = PPO.load(pol["ckpt"], device="cpu")
    mdp = PoseRecorder(cfg, m.opponent_factory(OPP, "cpu"), hold=HOLD, shield=None,
                       shield_obs=bool(pol["shield_obs"]), escape_obs=False)
    seed = SEED0 + EP_I
    mdp.shield = m.shield_factory("RECF-20" if recovery else "PLAIN", cfg)(seed)
    log = mdp.play_episode(model.predict, seed)
    env = mdp.inner.env
    return {"log": log, "steps": mdp.step_records, "obstacles": list(env.obstacles),
            "final_p": np.array(env.pursuer_pose, float), "final_e": np.array(env.evader_pose, float), "cfg": cfg}


def first_stall(steps: list[dict], k: int = 40) -> tuple[int, int] | None:
    run = 0
    for i, s in enumerate(steps):
        run = run + 1 if s["stalled"] else 0
        if run == k:
            j = i
            while j + 1 < len(steps) and steps[j + 1]["stalled"]:
                j += 1
            return i - k + 1, j - (i - k + 1) + 1
    return None


def draw(bare: dict, recf: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon, Rectangle, Wedge

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "legend.fontsize": 7, "font.family": "serif"})
    BLUE, ORANGE, RED, GREY, GREEN = "#2a6fb0", "#d9822b", "#c0392b", "#9aa0a6", "#2e8b57"
    MUTED_INSET = "#8a8984"
    cfg = bare["cfg"]
    half = cfg.arena_size / 2.0

    fig = plt.figure(figsize=(TEXTWIDTH_IN, 2.45))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.05], height_ratios=[1, 1], wspace=0.24, hspace=0.75, left=0.0, right=1.0)
    axa, axb = fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[:, 1])
    axc1, axc2 = fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, 2], sharex=None)

    def body(ax, pose, color, size=0.32):
        x, y, th = pose
        tri = np.array([[size, 0], [-size * 0.6, size * 0.5], [-size * 0.6, -size * 0.5]])
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        ax.add_patch(Polygon(tri @ R.T + [x, y], closed=True, fc=color, ec="black", lw=0.5, zorder=5))

    def arena(ax, ep, title, upto):
        ax.add_patch(Rectangle((-half, -half), 2 * half, 2 * half, fill=False, lw=1.0, ec="black"))
        for o in ep["obstacles"]:
            ax.add_patch(Circle((o["x"], o["y"]), o["radius"], fc=GREY, ec="none", zorder=2))
        st = ep["steps"][:upto]
        P = np.array([s["p"][:2] for s in st] + [ep["final_p"][:2]])
        E = np.array([s["e"][:2] for s in st] + [ep["final_e"][:2]])
        ax.plot(E[:, 0], E[:, 1], color=ORANGE, lw=0.9, zorder=3)
        ax.plot(P[:, 0], P[:, 1], color=BLUE, lw=1.1, zorder=4)
        ax.plot(*P[0], "o", ms=3.5, mfc="white", mec=BLUE, zorder=6)
        ax.plot(*E[0], "o", ms=3.5, mfc="white", mec=ORANGE, zorder=6)
        ax.set_xlim(-half - 0.2, half + 0.2)
        ax.set_ylim(-half - 0.2, half + 0.2)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(title, loc="left")
        return P, E

    # (a) plain filter: the pursuer stops against the obstacle and never moves again
    stall = first_stall(bare["steps"])
    s0, slen = stall
    P, E = arena(axa, bare, "(a) CBF filter: never caught", len(bare["steps"]))
    ps = bare["steps"][s0]["p"]
    fov = cfg.sensing.fov_angle
    wedge = axa.add_patch(Wedge(ps[:2], cfg.sensing.fov_range, np.degrees(ps[2] - fov / 2), np.degrees(ps[2] + fov / 2),
                                fc=BLUE, alpha=0.08, ec="none", zorder=1))
    wedge.set_clip_path(axa.patches[0])  # the arena wall rectangle
    body(axa, ps, BLUE)
    body(axa, bare["final_e"], ORANGE)
    comp = bare["steps"][s0 + 39]
    ls = comp["last_seen"]
    if comp["branch"] == "law" and not comp["visible"] and ls is not None:
        axa.plot(*ls, marker="*", ms=8, mfc="white", mec=RED, mew=1.0, zorder=7)
        axa.annotate("remembered point", ls, xytext=(8, -10), textcoords="offset points", color=RED, fontsize=7)
    end = "to the end" if s0 + slen >= len(bare["steps"]) else f"for {slen} steps"
    off = (ANNOT_DX, ANNOT_DY)
    axa.text(0.97, 0.95, f"stuck from step {s0}\n{end} ({slen} steps)", transform=axa.transAxes, fontsize=7, ha="right", va="top", color=RED)

    # inset: the stalled pose up close, with what the policy asks for
    ins = axa.inset_axes([0.04, 0.05, 0.48, 0.5])
    zx, zy, zr = ps[0], ps[1], 0.9
    ins.add_patch(Rectangle((-half, -half), 2 * half, 2 * half, fill=False, lw=1.6, ec="black"))
    for o in bare["obstacles"]:
        ins.add_patch(Circle((o["x"], o["y"]), o["radius"], fc=GREY, ec="none"))
    body(ins, ps, BLUE, size=0.16)
    v_req = (bare["steps"][s0 + 39]["nominal"][0] + 1) / 2
    ins.annotate("", xy=(ps[0] + 0.55 * v_req * np.cos(ps[2]), ps[1] + 0.55 * v_req * np.sin(ps[2])), xytext=(ps[0], ps[1]),
                 arrowprops={"arrowstyle": "-|>", "color": RED, "lw": 1.0, "mutation_scale": 7})
    ins.text(0.5, 0.1, "keeps asking to\ndrive forward", transform=ins.transAxes, ha="center", fontsize=6.5, color=RED)
    ins.set_xlim(zx - zr, zx + zr)
    ins.set_ylim(zy - 1.45 * zr, zy + 0.45 * zr)
    ins.set_aspect("equal")
    ins.set_xticks([])
    ins.set_yticks([])
    for sp in ins.spines.values():
        sp.set_edgecolor(MUTED_INSET)
        sp.set_linewidth(0.6)
    axa.indicate_inset_zoom(ins, edgecolor=MUTED_INSET, lw=0.6)

    # (b) recovery: one firing, then a capture
    arena(axb, recf, f"(b) + recovery: caught at step {recf['log']['steps']}", len(recf["steps"]))
    fires = [i for i, s in enumerate(recf["steps"]) if s["fired"]]
    if fires:
        pf = recf["steps"][fires[0]]["p"]
        axb.plot(*pf[:2], marker="D", ms=4.5, mfc=GREEN, mec="black", mew=0.4, zorder=7)
        axb.annotate("recovery fires", pf[:2], xytext=(4, -16), textcoords="offset points", color=GREEN, fontsize=7)
    body(axb, recf["final_p"], BLUE)
    body(axb, recf["final_e"], ORANGE)

    # (c) commanded versus executed speed
    for ax, ep, lab in ((axc1, bare, "CBF filter"), (axc2, recf, "with recovery")):
        st = ep["steps"]
        t = np.arange(len(st))
        nom = [(s["nominal"][0] + 1) / 2 for s in st]
        exe = [(s["executed"][0] + 1) / 2 for s in st]
        ax.plot(t, nom, color="black", lw=0.8, ls="--", label="commanded speed")
        ax.plot(t, exe, color=BLUE, lw=1.0, label="executed speed")
        ax.set_xlim(0, 1200)
        ax.set_ylim(-0.08, 1.12)
        ax.set_yticks([0, 1])
        if ax is axc1:
            ax.set_ylabel(r"$v/v_{\max}$", labelpad=1)
        ax.set_title(f"(c) speed, {lab}" if ax is axc1 else f"speed, {lab}", loc="left")
        for i in [i for i, s in enumerate(st) if s.get("fired")][:1]:
            ax.axvline(i, color=GREEN, lw=0.8)
    axc1.axvspan(s0, s0 + slen, color=RED, alpha=0.10, lw=0)
    axc1.legend(loc="center right", frameon=False, handlelength=1.6)
    axc2.set_xlabel("step (0.05 s; limit 1 200)")
    axc2.text(recf["log"]["steps"] + 15, 0.5, "captured", fontsize=7, color=BLUE, va="center")

    handles = [plt.Line2D([], [], color=BLUE, lw=1.1, label="pursuer (filtered)"),
               plt.Line2D([], [], color=ORANGE, lw=0.9, label="evader"),
               plt.Line2D([], [], marker="o", ls="", mfc="white", mec="black", ms=3.5, label="start"),
               Circle((0, 0), 1, fc=GREY, ec="none", label="obstacle")]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.33, 0.1), columnspacing=1.2)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "R0_game.pdf", bbox_inches="tight")
    fig.savefig(OUT / "R0_game.png", dpi=220, bbox_inches="tight")


def main() -> None:
    arch = {k: json.loads((R / "s26_b2_cells" / f"{POLICY}_{OPP}_{k}.json").read_text())["capture_ind"][EP_I] for k in ("PLAIN", "RECF-20")}
    n1 = next(e for e in json.loads((R / "s27_n1_cells" / f"{POLICY}_{OPP}.json").read_text())["episodes"] if e["i"] == EP_I)["first_k"]
    bare, recf = play(False), play(True)
    stall = first_stall(bare["steps"])
    comp = bare["steps"][stall[0] + 39]
    got = {"plain_capture": bare["log"]["outcome"] == "capture", "recf_capture": recf["log"]["outcome"] == "capture",
           "first_K_run": stall, "component_at_K": "head" if comp["branch"] == "search" else ("chase" if comp["visible"] else "last_seen"),
           "plain_steps": len(bare["steps"]), "recf_steps": recf["log"]["steps"],
           "recf_firings": sum(s["fired"] for s in recf["steps"]), "violations": [bare["log"].get("violations"), recf["log"].get("violations")]}
    want = {"plain_capture": arch["PLAIN"], "recf_capture": arch["RECF-20"], "first_K_run": (n1["start"], n1["length"]), "component_at_K": n1["component_at_k"]}
    if any(got[k] != v for k, v in want.items()):
        raise SystemExit(f"replay does not match the archive: {got} vs {want}")
    print(json.dumps(got, default=str))
    draw(bare, recf)
    data = json.loads((OUT / "figure_data_rev.json").read_text()) if (OUT / "figure_data_rev.json").exists() else {}
    data["R0"] = {"source": f"replay of s26_b2 {POLICY}_{OPP} episode {EP_I} (seed {SEED0 + EP_I}), PLAIN and RECF-20; N1 first K-run", **got}
    (OUT / "figure_data_rev.json").write_text(json.dumps(data, indent=1, default=str))


if __name__ == "__main__":
    main()
