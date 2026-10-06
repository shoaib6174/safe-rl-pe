"""Paper Phase 0 (construct_validity_paper_plan_2026-10-03.md): aggregate the 36
S18 replay cells (`results/derisk/s18_branch_split_{ARM}_{OPP}.json`, D140) into
the Case-5 table and figure -- what share of the safety filter's work, and of
the stalls the deadlock metric counted, belonged to the scripted intercept LAW
rather than the learner. Reads only; every number comes from the artefacts.
"""
from __future__ import annotations

import glob
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = pathlib.Path("results/derisk")
FIG = pathlib.Path("paper/figures/case5_law_share.png")
COLS = ("SPARSE", "DRARE", "DVARY")


def main() -> None:
    rows = []
    for f in sorted(glob.glob(str(OUT / "s18_branch_split_*.json"))):
        d = json.load(open(f))
        a = d["aggregate"]
        ib = a["intervention_by_branch"]
        sb = a["stalled_by_branch"]
        tot_i = ib["law"] + ib["search"]
        tot_s = sb["law"] + sb["search"]
        rows.append(
            {
                "arm": d["arm"],
                "opponent": d["opponent"],
                "reproduction": d["reproduction"]["verdict"],
                "law_share_intervention": ib["law"] / tot_i if tot_i else None,
                "law_share_stalled": sb["law"] / tot_s if tot_s else None,
                "episode_frac_law_pin_ge_40": a["episode_frac_law_pin_ge_40"],
                "episode_frac_search_pin_ge_40": a["episode_frac_search_pin_ge_40"],
                "pre_decision_share": a["pre_decision_intervention"] / tot_i if tot_i else None,
            }
        )
    assert len(rows) == 36, len(rows)
    assert all(r["reproduction"] == "PASS" for r in rows)
    li = [r["law_share_intervention"] for r in rows]
    ls = [r["law_share_stalled"] for r in rows]
    summary = {
        "n_cells": 36,
        "all_reproduce_exactly": True,
        "law_share_intervention": {"min": min(li), "median": sorted(li)[18], "max": max(li)},
        "law_share_stalled": {"min": min(ls), "median": sorted(ls)[18], "max": max(ls)},
        "cells": rows,
    }
    (OUT / "s18_branch_split_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"law share of intervention magnitude: min {min(li):.3f} median {sorted(li)[18]:.3f} max {max(li):.3f}")
    print(f"law share of stalled steps:          min {min(ls):.3f} median {sorted(ls)[18]:.3f} max {max(ls):.3f}")
    print(f"{'arm':16s} {'opp':6s} {'law%int':>8s} {'law%stall':>9s} {'lawpin>=40':>10s} {'srchpin>=40':>11s}")
    for r in rows:
        print(f"{r['arm']:16s} {r['opponent']:6s} {r['law_share_intervention']:8.3f} {r['law_share_stalled']:9.3f} {r['episode_frac_law_pin_ge_40']:10.3f} {r['episode_frac_search_pin_ge_40']:11.3f}")
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for k, (key, title) in enumerate(
        (("law_share_intervention", "share of filter intervention magnitude"),
         ("law_share_stalled", "share of stalled steps"))
    ):
        for j, col in enumerate(COLS):
            for row_, mk in (("BASE", "o"), ("PEN", "s")):
                ys = [r[key] for r in rows if r["arm"].startswith(f"{col}-{row_}")]
                ax[k].scatter([j + (0.15 if row_ == "PEN" else -0.15)] * len(ys), ys, marker=mk, s=28,
                              label=row_ if (j == 0) else None)
        ax[k].set_xticks(range(3)); ax[k].set_xticklabels(COLS); ax[k].set_ylim(0, 1.02)
        ax[k].set_ylabel("scripted LAW's share"); ax[k].set_title(title, fontsize=9); ax[k].axhline(0.5, ls=":", lw=0.8, c="grey")
    ax[0].legend(fontsize=8, title="penalty", title_fontsize=8)
    fig.suptitle("Case 5: the filter's work is the scripted law's, not the learner's (36 S18 cells, exact replay)", fontsize=9)
    fig.tight_layout(); FIG.parent.mkdir(parents=True, exist_ok=True); fig.savefig(FIG, dpi=160)
    print("wrote", FIG)


if __name__ == "__main__":
    main()
