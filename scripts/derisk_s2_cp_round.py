"""S2 — the CP lead: is the mid-run improvement real, selectable and holdable?
Spec: docs/pathway/cp_selection_anchor_prereg_2026-09-13.md. Nothing here may
change a threshold, a seed, a rule or a budget after a number is seen.

Arms (1M env steps each unless noted; S1's CP config = PPO_KWARGS + lambda=1.0
from the PPO clone of SearchingPursuer):
  SEL{7,8,9}     beta=0 (KL measured only)   <- H-SEL: reproduce, select, gate
  ANC01_{7,8,9}  beta=0.1                     <- H-ANC at the stronger weight
  ANC001_{7,8,9} beta=0.01                    <- H-ANC at the weaker weight
  NULL7          beta=0, all rewards zeroed   <- negative control for SELECTION
  SBLONG         S1's SB for 10M env steps    <- side arm, no claim (amendment A1)

Usage:
  ./venv/bin/python scripts/derisk_s2_cp_round.py --measure-start   (bars on the gate stream)
  ./venv/bin/python scripts/derisk_s2_cp_round.py --arms SEL7       (one process per arm)
  ./venv/bin/python scripts/derisk_s2_cp_round.py --summarize
"""

from __future__ import annotations

import argparse
import json
import pathlib

from pe2.config import EnvConfig
from pe2.derisk import gate_pass, select_checkpoint, structured_verdict
from pe2.eval import wilson_ci, win_count
from pe2.scripted import GreedyEvader, SearchingPursuer

OUT = pathlib.Path("results/derisk")
START_PATH = OUT / "s2_start.json"
PPO_CLONE = OUT / "ppo_bc_searching_pursuer.zip"
SAC_CLONE = OUT / "sac_bc_searching_pursuer.zip"
S1_INIT = OUT / "s1_ppo_init.zip"
S1_START = OUT / "s1_start.json"
S1_CP_PROBE = OUT / "s1_probe_CP.json"

# --- pre-registered (spec sections 3-5). Do not tune. ---
TOTAL = 1_000_000
EVERY = 100_000
EVAL_N = 200
GATE_N = 500
SEL_SEED0 = 1_040_000  # selection/probe stream (fresh)
GATE_SEED0 = 1_050_000  # gate stream (fresh)
S1_EVAL_SEED0 = 1_010_000  # SBLONG probes on S1's stream so S1's bars apply
SEEDS = (7, 8, 9)
BETAS = {"ANC01": 0.1, "ANC001": 0.01}
CP_LAMBDA = 1.0
P_GATE = 0.05
TOTAL_LONG = 10_000_000
EVERY_LONG = 1_000_000
REFS = ("PPO_CLONE", "SAC_CLONE", "TEACHER")

ARMS: dict = {}
for _s in SEEDS:
    ARMS[f"SEL{_s}"] = {"kind": "ppo", "beta": 0.0, "seed": _s, "zero_reward": False}
    for _name, _b in BETAS.items():
        ARMS[f"{_name}_{_s}"] = {
            "kind": "ppo",
            "beta": _b,
            "seed": _s,
            "zero_reward": False,
        }
ARMS["NULL7"] = {"kind": "ppo", "beta": 0.0, "seed": 7, "zero_reward": True}
ARMS["SBLONG"] = {"kind": "sblong", "seed": 7}


def _ci(k: int, n: int) -> dict:
    lo, hi = wilson_ci(k, n)
    return {"k": k, "n": n, "rate": k / n, "wilson": [lo, hi]}


# ----------------------------------------------------------------------------
# guards
# ----------------------------------------------------------------------------
def require_not_measured() -> None:
    if START_PATH.exists():
        raise SystemExit(
            f"{START_PATH} already exists: the bars are already measured and the stop "
            "rule (spec section 6) forbids re-measuring them."
        )
    if any(OUT.glob("s2_arm_*.json")) or any(
        (OUT / f"ckpts_{a}").exists() for a in ARMS
    ):
        raise SystemExit(
            "S2 arms have already run; the stop rule forbids re-measuring."
        )


def require_arm_not_run(arm: str) -> None:
    if (OUT / f"s2_arm_{arm}.json").exists() or (OUT / f"ckpts_{arm}").exists():
        raise SystemExit(
            f"arm {arm} has already run (s2_arm_{arm}.json or ckpts_{arm} exists); "
            "the stop rule (spec section 6) forbids re-running an arm."
        )


def load_start() -> dict:
    if not START_PATH.exists():
        raise SystemExit(
            f"gate bars missing: {START_PATH}. Run --measure-start FIRST; the "
            "pre-registration requires it before any arm trains."
        )
    s = json.loads(START_PATH.read_text())
    for k in REFS:
        if k not in s or "logs" not in s[k]:
            raise SystemExit(
                f"{k} (with paired logs) missing from {START_PATH}; re-run --measure-start."
            )
    return s


# ----------------------------------------------------------------------------
# bars on the gate stream (spec section 3), BEFORE any arm
# ----------------------------------------------------------------------------
def measure_start(device: str) -> dict:
    from stable_baselines3 import PPO, SAC

    from pe2.exploitability import evaluate_sb3_policy
    from pe2.structured import evaluate_scripted

    require_not_measured()
    cfg = EnvConfig()
    out: dict = {"gate_seed0": GATE_SEED0, "gate_n": GATE_N}
    refs = {
        "PPO_CLONE": lambda: evaluate_sb3_policy(
            PPO.load(PPO_CLONE, device=device),
            cfg,
            "pursuer",
            GreedyEvader,
            GATE_N,
            GATE_SEED0,
        ),
        "SAC_CLONE": lambda: evaluate_sb3_policy(
            SAC.load(SAC_CLONE, device=device),
            cfg,
            "pursuer",
            GreedyEvader,
            GATE_N,
            GATE_SEED0,
        ),
        "TEACHER": lambda: evaluate_scripted(
            cfg, SearchingPursuer, GreedyEvader, GATE_N, GATE_SEED0
        ),
    }
    for name, fn in refs.items():
        logs = fn()
        out[name] = _ci(*win_count(logs, "pursuer")) | {"logs": logs}
        lo, hi = out[name]["wilson"]
        print(
            f"{name} = {out[name]['k']}/{GATE_N} = {out[name]['rate']:.4f}, "
            f"Wilson [{lo:.4f}, {hi:.4f}]",
            flush=True,
        )
    START_PATH.write_text(json.dumps(out, indent=2))
    return out


# ----------------------------------------------------------------------------
# arms
# ----------------------------------------------------------------------------
def select_and_gate(
    rows: list[dict], final_ckpt, starts: dict, cfg, device: str
) -> dict:
    """Spec section 3: the selected checkpoint AND the final one, each gated
    on the gate stream against every reference. Returns the two records."""
    from stable_baselines3 import PPO

    from pe2.exploitability import evaluate_sb3_policy

    def gate(ckpt_path) -> dict:
        model = PPO.load(ckpt_path, device=device)
        logs = evaluate_sb3_policy(
            model, cfg, "pursuer", GreedyEvader, GATE_N, GATE_SEED0
        )
        return {
            "ckpt": str(ckpt_path),
            "score": _ci(*win_count(logs, "pursuer")),
            "gate": {
                ref: gate_pass(logs, starts[ref]["logs"], "pursuer", P_GATE)
                for ref in REFS
            },
        }

    sel_row = select_checkpoint(rows)
    return {
        "selected": gate(sel_row["ckpt"])
        | {"steps": sel_row["steps"], "probe_win_rate": sel_row["win_rate"]},
        "final": gate(final_ckpt),
    }


def run_ppo_arm(
    arm: str, starts: dict, device: str, holder: dict | None = None
) -> dict:
    from stable_baselines3.common.callbacks import CallbackList

    from pe2.anchored_sac import AnchorKLProbe
    from pe2.exploitability import CheckpointEvalProbe, finetune_ppo_anchored
    from pe2.rewards import ZeroRewardWrapper

    spec = ARMS[arm]
    cfg = EnvConfig()
    d = OUT / f"ckpts_{arm}"
    d.mkdir(parents=True, exist_ok=True)
    probe = CheckpointEvalProbe(
        cfg,
        "pursuer",
        GreedyEvader,
        every=EVERY,
        eval_n=EVAL_N,
        seed0=SEL_SEED0,
        ckpt_dir=d,
        report_path=OUT / f"s2_probe_{arm}.json",
        label=arm,
    )
    if holder is not None:
        holder["rows"] = probe.rows
    kl_probe = AnchorKLProbe(
        probe.rows, EVERY
    )  # AFTER the checkpoint probe, same `every`
    finetune_ppo_anchored(
        cfg,
        "pursuer",
        GreedyEvader,
        PPO_CLONE,
        TOTAL,
        seed=spec["seed"],
        beta=spec["beta"],
        device=device,
        callback=CallbackList([probe.callback, kl_probe.callback]),
        env_wrap=ZeroRewardWrapper if spec["zero_reward"] else None,
        custom_objects={"gae_lambda": CP_LAMBDA},
    )
    result = {
        "arm": arm,
        "kind": "ppo",
        "settings": spec,
        "timesteps": TOTAL,
        "eval_n": EVAL_N,
        "sel_seed0": SEL_SEED0,
        "gate_seed0": GATE_SEED0,
        "checkpoints": probe.rows,
    }
    final_ckpt = d / f"{arm}_{TOTAL}.zip"
    if not final_ckpt.exists():
        result["verdict"] = "UNDETERMINED"
        result["reason"] = f"final checkpoint missing: {final_ckpt} was never written"
        return result
    result |= select_and_gate(probe.rows, final_ckpt, starts, cfg, device)
    sel, fin = (
        result["selected"]["gate"]["PPO_CLONE"],
        result["final"]["gate"]["PPO_CLONE"],
    )
    result["verdict"] = (
        f"selected {'PASS' if sel['pass'] else 'FAIL'} / "
        f"final {'PASS' if fin['pass'] else 'FAIL'} vs PPO clone"
    )
    result["reason"] = (
        f"selected @{result['selected']['steps']} (probe {result['selected']['probe_win_rate']:.3f}): gate "
        f"{sel['k']}/{sel['n']} = {sel['rate']:.3f}, McNemar p={sel['p']:.2e} (b={sel['b']}, c={sel['c']}) vs clone {sel['ref_rate']:.3f}; "
        f"final: {fin['k']}/{fin['n']} = {fin['rate']:.3f}, p={fin['p']:.2e} (b={fin['b']}, c={fin['c']}); "
        f"vs teacher: selected {'PASS' if result['selected']['gate']['TEACHER']['pass'] else 'FAIL'}, "
        f"final {'PASS' if result['final']['gate']['TEACHER']['pass'] else 'FAIL'}"
    )
    return result


def run_sblong(device: str, holder: dict | None = None) -> dict:
    """Side arm: S1's SB (terminal only, structured) for TOTAL_LONG env steps,
    probed on S1's stream so S1's bars in s1_start.json apply. No claim."""
    from stable_baselines3 import PPO

    from pe2.structured import HOLD, SearchMDP, StructuredEvalProbe, evaluate_structured

    if not S1_INIT.exists() or not S1_START.exists():
        raise SystemExit("SBLONG needs S1's s1_ppo_init.zip and s1_start.json")
    s1 = json.loads(S1_START.read_text())
    cfg = EnvConfig()
    arm = "SBLONG"
    d = OUT / f"ckpts_{arm}"
    d.mkdir(parents=True, exist_ok=True)
    probe = StructuredEvalProbe(
        cfg,
        GreedyEvader,
        every=EVERY_LONG,
        total=TOTAL_LONG,
        eval_n=EVAL_N,
        seed0=S1_EVAL_SEED0,
        ckpt_dir=d,
        report_path=OUT / f"s2_probe_{arm}.json",
        label=arm,
        hold=HOLD,
    )
    if holder is not None:
        holder["rows"] = probe.rows
    env = SearchMDP(
        cfg, GreedyEvader, seed0=7 * 1000, hold=HOLD, r_acq=0.0, zero_reward=False
    )
    model = PPO.load(S1_INIT, env=env, device=device, seed=7)
    model.learn(total_timesteps=TOTAL_LONG, callback=probe.callback)
    result = {
        "arm": arm,
        "kind": "sblong",
        "settings": ARMS[arm],
        "env_steps": int(env.env_steps),
        "decisions": int(model.num_timesteps),
        "eval_seed0": S1_EVAL_SEED0,
        "checkpoints": probe.rows,
        "s1_bars": {k: s1[k]["rate"] for k in ("BAR_ORBIT", "FLOOR_RAND", "START_S")},
    }
    ckpt = d / f"{arm}_{TOTAL_LONG}.zip"
    if not ckpt.exists():
        result["verdict"], result["reason"] = (
            "UNDETERMINED",
            f"final checkpoint missing: {ckpt}",
        )
        return result
    logs = evaluate_structured(
        PPO.load(ckpt, device=device),
        cfg,
        GreedyEvader,
        GATE_N,
        S1_EVAL_SEED0,
        hold=HOLD,
    )
    result["final"] = _ci(*win_count(logs, "pursuer")) | {"ckpt": str(ckpt)}
    result["verdict"], result["reason"] = structured_verdict(
        result["final"]["k"],
        result["final"]["n"],
        s1["BAR_ORBIT"]["rate"],
        s1["FLOOR_RAND"]["wilson"][1],
        s1["START_S"]["wilson"][1],
    )
    return result


def run_arm(arm: str, device: str) -> dict:
    starts = load_start()
    require_arm_not_run(arm)
    holder: dict = {}
    try:
        result = (
            run_sblong(device, holder)
            if ARMS[arm]["kind"] == "sblong"
            else run_ppo_arm(arm, starts, device, holder)
        )
    except Exception as e:
        (OUT / f"s2_arm_{arm}.json").write_text(
            json.dumps(
                {
                    "arm": arm,
                    "kind": ARMS[arm]["kind"],
                    "verdict": "UNDETERMINED",
                    "reason": f"crashed: {type(e).__name__}: {e}",
                    "checkpoints": holder.get("rows", []),
                },
                indent=2,
            )
        )
        raise
    (OUT / f"s2_arm_{arm}.json").write_text(json.dumps(result, indent=2))
    print(f"[{arm}] {result['verdict']}: {result['reason']}", flush=True)
    return result


# ----------------------------------------------------------------------------
# summary (spec sections 5-6), fixed before the run
# ----------------------------------------------------------------------------
def _passes(results: dict, arms: list[str], which: str) -> int | None:
    if any(a not in results or which not in results[a] for a in arms):
        return None
    return sum(bool(results[a][which]["gate"]["PPO_CLONE"]["pass"]) for a in arms)


def hypothesis_verdicts(results: dict) -> dict:
    sel_arms = [f"SEL{s}" for s in SEEDS]
    n_sel = _passes(results, sel_arms, "selected")
    null = results.get("NULL7")
    null_v = (
        "INCOMPLETE"
        if null is None or "selected" not in null
        else ("PASS" if null["selected"]["gate"]["PPO_CLONE"]["pass"] else "FAIL")
    )
    if n_sel is None:
        h_sel = "INCOMPLETE"
    elif null_v == "PASS":
        h_sel = "UNDETERMINED (selection manufactures improvement)"
    else:
        h_sel = "PASS" if n_sel >= 2 else "FAIL"
    h_anc = {}
    for name in BETAS:
        n = _passes(results, [f"{name}_{s}" for s in SEEDS], "final")
        h_anc[name] = "INCOMPLETE" if n is None else ("PASS" if n >= 2 else "FAIL")
    return {"H_SEL": h_sel, "H_SEL_passes": n_sel, "H_ANC": h_anc, "NULL": null_v}


def determinism_note(rows_new: list[dict], rows_s1: list[dict]) -> str:
    a = [(r["steps"], round(r["win_rate"], 6)) for r in rows_new]
    b = [(r["steps"], round(r["win_rate"], 6)) for r in rows_s1]
    if a == b:
        return "reproduced"
    diff = [s for (s, x), (_, y) in zip(a, b, strict=False) if x != y]
    return f"not reproduced (first difference at {diff[0] if diff else 'length'})"


def direction(h: dict) -> str:
    anc_pass = [n for n, v in h["H_ANC"].items() if v == "PASS"]
    if h["H_SEL"] == "INCOMPLETE" or any(
        v == "INCOMPLETE" for v in h["H_ANC"].values()
    ):
        return "incomplete: all SEL, ANC and NULL arms must run"
    if anc_pass:
        return (
            f"H-ANC PASSES at {', '.join(anc_pass)} (beta {[BETAS[n] for n in anc_pass]}): a learned pursuer that "
            "improves on its clone with no selection exists. The artefact for the paper is the anchored PPO "
            "pursuer; if it also beats the teacher that is the headline number. Route C withdrawn. Next "
            "pre-registration: the shield on this pursuer (Step 8). Prefer the beta whose final checkpoint "
            f"scores higher; report both. H-SEL: {h['H_SEL']}."
        )
    if h["H_SEL"] == "PASS":
        return (
            "H-SEL PASSES, H-ANC fails at both beta: the rise is real and reproducible but the anchor did not "
            "hold it. Report the selection-based artefact honestly (it needs the gate) and pre-register the "
            "next remedy from the ANC drift traces. Route C withdrawn on H-SEL."
        )
    if h["NULL"] == "PASS":
        return (
            "NULL PASSES: selection manufactures improvement at this probe density; H-SEL is not read; "
            "H-ANC failed at both beta, so no learned pursuer is claimed. Route C stands re-scoped."
        )
    return (
        "Both fail, NULL fails: the 0.645 was seed-7-specific or a stream artefact. Report it as such; "
        "Route C stands re-scoped (D108/D109). SBLONG is read on its own."
    )


def summarize(results: dict, starts: dict) -> dict:
    h = hypothesis_verdicts(results)
    d = direction(h)
    note = None
    if "SEL7" in results and S1_CP_PROBE.exists():
        note = determinism_note(
            results["SEL7"]["checkpoints"],
            json.loads(S1_CP_PROBE.read_text())["checkpoints"],
        )
    sb = results.get("SBLONG")
    return {
        "spec": "docs/pathway/cp_selection_anchor_prereg_2026-09-13.md",
        "hypotheses": h,
        "bars": {k: starts[k]["rate"] for k in REFS},
        "arms": {
            a: {"verdict": r.get("verdict"), "reason": r.get("reason")}
            for a, r in results.items()
        },
        "determinism_sel7_vs_s1_cp": note,
        "sblong": None
        if sb is None
        else {"verdict": sb.get("verdict"), "reason": sb.get("reason")},
        "complete": all(a in results for a in ARMS),
        "direction": d,
    }


def load_results() -> dict:
    return {
        p.stem.removeprefix("s2_arm_"): json.loads(p.read_text())
        for p in sorted(OUT.glob("s2_arm_*.json"))
    }


def write_figure(results: dict, starts: dict, summary: dict, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fam_color = {
        "SEL": "tab:blue",
        "ANC01": "tab:green",
        "ANC001": "tab:olive",
        "NULL": "tab:red",
    }
    ax = axes[0]
    for arm, r in results.items():
        if r.get("kind") != "ppo" or not r.get("checkpoints"):
            continue
        fam = arm.rsplit("_", 1)[0] if "_" in arm else arm[:-1]
        rows = r["checkpoints"]
        ax.plot(
            [x["steps"] for x in rows],
            [x["win_rate"] for x in rows],
            marker=".",
            color=fam_color.get(fam, "grey"),
            alpha=0.8,
            label=arm,
        )
        if "selected" in r:
            ax.plot(
                [r["selected"]["steps"]],
                [r["selected"]["probe_win_rate"]],
                marker="*",
                ms=12,
                color=fam_color.get(fam, "grey"),
            )
    for ref, c in zip(REFS, ("tab:red", "tab:purple", "k"), strict=True):
        ax.axhline(
            starts[ref]["rate"],
            ls=":",
            color=c,
            lw=1.2,
            label=f"{ref} (gate) = {starts[ref]['rate']:.3f}",
        )
    ax.set_xlabel("env steps")
    ax.set_ylabel(f"probe win rate (selection seeds {SEL_SEED0}+)")
    ax.set_title("S2 — 1M arms (stars = selected)")
    ax.legend(fontsize=6, ncol=2)
    ax = axes[1]
    for arm, r in results.items():
        if r.get("kind") != "ppo":
            continue
        rows = [x for x in r.get("checkpoints", []) if x.get("anchor_kl") is not None]
        if rows:
            fam = arm.rsplit("_", 1)[0] if "_" in arm else arm[:-1]
            ax.plot(
                [x["steps"] for x in rows],
                [x["anchor_kl"] for x in rows],
                marker=".",
                color=fam_color.get(fam, "grey"),
                alpha=0.8,
                label=arm,
            )
    ax.set_yscale("log")
    ax.set_xlabel("env steps")
    ax.set_ylabel("KL(pi || clone), training batches")
    ax.set_title("anchor divergence")
    ax.legend(fontsize=6, ncol=2)
    ax = axes[2]
    sb = results.get("SBLONG")
    if sb and sb.get("checkpoints"):
        rows = sb["checkpoints"]
        ax.plot(
            [x["steps"] for x in rows],
            [x["win_rate"] for x in rows],
            marker="o",
            color="tab:orange",
            label=f"SBLONG ({sb.get('verdict')})",
        )
        for k, c in (
            ("BAR_ORBIT", "tab:green"),
            ("FLOOR_RAND", "tab:orange"),
            ("START_S", "tab:purple"),
        ):
            ax.axhline(
                sb["s1_bars"][k],
                ls=":",
                color=c,
                lw=1.2,
                label=f"{k} = {sb['s1_bars'][k]:.3f}",
            )
    ax.set_xlabel("env steps")
    ax.set_title("SBLONG vs S1 bars (side arm)")
    ax.legend(fontsize=7)
    fig.suptitle(
        "S2 — CP selection + KL anchor"
        + ("" if summary["complete"] else " — INCOMPLETE")
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--measure-start", action="store_true")
    p.add_argument("--arms", nargs="*", choices=list(ARMS), default=[])
    p.add_argument("--summarize", action="store_true")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.measure_start:
        measure_start(args.device)
        return
    for arm in args.arms:
        run_arm(arm, args.device)
    if args.summarize:
        results, starts = load_results(), load_start()
        summary = summarize(results, starts)
        (OUT / "s2_round.json").write_text(
            json.dumps({"summary": summary, "arms": results}, indent=2)
        )
        write_figure(results, starts, summary, OUT / "s2_round.png")
        print(json.dumps(summary["hypotheses"], indent=2))
        print(json.dumps(summary["arms"], indent=2))
        print(f"\ndeterminism SEL7 vs S1 CP: {summary['determinism_sel7_vs_s1_cp']}")
        print(f"\n{summary['direction']}")


if __name__ == "__main__":
    main()
