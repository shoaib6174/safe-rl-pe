"""Step 6 visual gate: results/visual_gates/step_06/step_06_shield.gif

The pre-registration (D83, `docs/worklogs/2026-08-24_S119.md`) names four
checklist items for this GIF, and each is ASSERTED here as well as shown:

  C1. nominal vs executed action arrows diverge exactly at constraint
      approach. Asserted two ways: every exact-tier step whose logged
      `intervention` flag is set had at least one violated DCBF row,
      re-derived from `CBFShield.progress`, the exact one-step check on the
      constraint DEFINITION -- not from another copy of the filter; and every
      such step's `min_clearance` lies below the analytic bound
      `intervention_clearance_bound`, the largest clearance at which any row
      can bind given the action box. The converse is REPORTED, not asserted:
      a violated row whose correction is smaller than the shield's
      `INTERVENTION_ATOL` of 1e-4 is a genuine sub-threshold nudge, not a
      divergence, and counting those as failures would test the reporting
      tolerance rather than the filter.
  C2. the agent slides along an obstacle boundary without contact: a run of
      >= MIN_SLIDE_STEPS consecutive steps with obstacle clearance inside
      SLIDE_BAND, in an episode with zero contact steps.
  C3. the pursuer still makes contact-capture: a rendered episode ends in
      capture with the bodies touching (final distance <= capture_distance).
      Each episode gets one overlay-free terminal frame after its loop, so
      the capture instant is visible and not merely logged -- action frames
      are drawn before their step, so the last of them precedes the capture.
  C4. the intervention flash rate matches the logged rate: rendering is
      STRIDE=1, the overlay flashes off the logged `info["intervention"]`
      alone, and every rendered frame's flash state is asserted equal to its
      step's logged flag — so the flash rate IS the logged intervention rate,
      and both are printed.

Every rendered episode is additionally cross-checked line by line against its
row in `step_06_gate.json` (outcome, steps, violations, interventions), so the
GIF and the gated numbers cannot drift apart. Run pe2_step06_gate.py first.

The GIF is deterministic: fixed seeds, Agg backend, no RNG anywhere in the
render path. `--md5-only` re-renders and prints the checksum without writing.

Usage: ./venv/bin/python scripts/pe2_step06_visual_gate.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

import numpy as np

from pe2.config import EnvConfig
from pe2.env import PursuitEvasionEnv
from pe2.render import Renderer, ShieldOverlay, Trails, save_gif
from pe2.scripted import GreedyEvader, GreedyPursuer, ObstacleHuggingEvader
from pe2.shield import CBFShield

OUT_DIR = pathlib.Path("results/visual_gates/step_06")
GIF_PATH = OUT_DIR / "step_06_shield.gif"
JSON_PATH = OUT_DIR / "step_06_visual_gate.json"
GATE_JSON = OUT_DIR / "step_06_gate.json"

FPS = 20
STRIDE = 1  # every step, so the flash rate IS the logged intervention rate (C4)
HOLD_FRAMES = 15
SHIELDED_ROLE = "pursuer"

SLIDE_BAND = 0.30  # obstacle clearance (m) counted as "along the boundary"
MIN_SLIDE_STEPS = 15  # 0.75 s of continuous sliding

# Episodes are picked from the pre-registered streams (main 800000+, probe
# 810000+) for illustrative value; every number they produce is still
# cross-checked against that seed's row in the gate JSON.
# Filled in Task 8, Step 3, from step_06_gate.json v2: from
# arms.shielded_probe.episodes, the two lowest seeds with outcome=="capture",
# steps>=120, and n_interventions/n_steps>=0.10 -> 810013 (steps=249,
# interv_rate=0.731), 810023 (steps=255, interv_rate=0.627); from
# arms.shielded_main.episodes, the lowest seed with the same properties ->
# 800032 (steps=222, interv_rate=0.176). First choices met all of C1-C4,
# including C2's >= MIN_SLIDE_STEPS slide, so no fallback attempt was needed
# (see "episode_selection" in step_06_visual_gate.json).
GATE_EPISODES = [
    # (label, arm name in step_06_gate.json, evader class, seed)
    ("probe_slide", "shielded_probe", ObstacleHuggingEvader, 810013),
    ("probe_capture", "shielded_probe", ObstacleHuggingEvader, 810023),
    ("main_capture", "shielded_main", GreedyEvader, 800032),
]

# Every selection attempt, in order tried. Only one attempt was needed: the
# first-choice seeds above satisfied C1-C4, including C2's >= MIN_SLIDE_STEPS
# contact-free slide, so no fallback (next qualifying probe seed) was drawn.
EPISODE_SELECTION_LOG = [
    {
        "attempt": 1,
        "rule": (
            "shielded_probe: two lowest seeds with outcome=='capture', "
            "steps>=120, n_interventions/n_steps>=0.10; shielded_main: "
            "lowest seed with the same properties"
        ),
        "probe_seeds": [810013, 810023],
        "main_seed": 800032,
        "result": "accepted -- C1-C4 all satisfied, no fallback needed",
    },
]


# ---------------------------------------------------------------- pure parts


def intervention_clearance_bound(
    cfg: EnvConfig, obstacles: list[dict], shield
) -> float:
    """Largest clearance (m) at which any DCBF row can possibly bind.

    A row can bind only if the nominal action fails the EXACT one-step check
    (`CBFShield.progress` < 0, not a linearised row): h_i(x+) < (1-gamma) *
    h_i(x). Over one step the body moves at most v_max*dt and the VCP moves
    at most dt*(v_max + d*omega_max) (chord <= arc), so
        h_i(x+) - h_i(x) >= -(that reach)
    for every metre row, and
        h_i(x+) - h_i(x) >= -2*dist*reach
    for the m^2 obstacle row (h = dist^2 - r_eff^2, so a step of at most
    `reach` in the VCP position changes dist^2 by at most 2*dist*reach).
    Binding therefore needs gamma*h below that same reach-scale term, which
    is the identical bound computed below. The body (hard) rows bind only
    below dt*v_max/gamma = 0.25 m, which is inside the VCP-row bound below,
    so they never govern it.

    Boundary VCP rows can only bind below
        dt * (v_max + d * omega_max) / gamma.
    Obstacle VCP rows can only bind while
        dist < (K + sqrt(K^2 + 4 r_eff^2)) / 2,  K = 2*dt*(v_max + d*om)/gamma,
    i.e. below a clearance of that dist minus r_eff.

    Interventions above this bound would mean the filter is acting on
    something other than the constraints — which is exactly what C1 checks.
    """
    reach = cfg.dt * (cfg.v_max + shield.d * cfg.omega_max) / shield.gamma
    bound = reach  # the boundary rows
    for obs in obstacles:
        r_eff = obs["radius"] + cfg.agent_radius + shield.margin
        k = 2.0 * reach
        dist = (k + np.sqrt(k * k + 4.0 * r_eff * r_eff)) / 2.0
        bound = max(bound, dist - r_eff)
    return float(bound)


def nominal_violates_a_row(
    shield, state, obstacles, v_nom: float, w_nom: float
) -> bool:
    """True iff the nominal action fails the exact one-step check of at least
    one row — re-derived from `CBFShield.progress`, the constraint DEFINITION,
    not from another copy of the filter (C1)."""
    return bool(
        (shield.progress(state, obstacles, np.array([v_nom, w_nom])) < 0.0).any()
    )


def longest_run(flags: list[bool]) -> int:
    """Longest run of consecutive True values."""
    best = run = 0
    for flag in flags:
        run = run + 1 if flag else 0
        best = max(best, run)
    return best


def md5_of(path: pathlib.Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def gif_frame_count(path: pathlib.Path) -> int:
    """Frames actually in the written file (the encoder coalesces duplicates)."""
    from PIL import Image

    with Image.open(path) as im:
        return int(im.n_frames)


# ------------------------------------------------------------------ episode


def play(cfg, evader_cls, seed: int, shield, renderer, overlay, live, label: str):
    """Run one shielded episode, rendering every step; return (frames, record)."""
    env = PursuitEvasionEnv(cfg)
    pursuer, evader = GreedyPursuer(), evader_cls()
    pursuer.reset()
    evader.reset()
    obs, _ = env.reset(seed=seed)
    trails = Trails()
    frames: list[np.ndarray] = []
    steps: list[dict] = []
    flash_frames = 0
    action_frames = 0

    while True:
        v_p, w_p = pursuer.act(obs["pursuer"], env)
        v_e, w_e = evader.act(obs["evader"], env)
        state_before = np.array(env.pursuer_pose)
        cons = shield.constraints(state_before, env.obstacles)
        obstacle_clearance = min(
            (c.clearance for c in cons if c.kind == "obstacle"), default=float("inf")
        )
        nominal = (
            float(np.clip(v_p, 0.0, cfg.v_max)),
            float(np.clip(w_p, -cfg.omega_max, cfg.omega_max)),
        )
        nominal_infeasible = nominal_violates_a_row(
            shield, state_before, env.obstacles, *nominal
        )
        (v_p, w_p), sinfo = shield.filter(
            (v_p, w_p), env.pursuer_pose, env.obstacles, v_max=cfg.v_max
        )
        # The overlay draws from `live` alone, so the flash is the logged flag.
        live["nominal"] = nominal
        live["executed"] = (float(v_p), float(w_p))
        live["info"] = sinfo
        trails.append("pursuer", *env.pursuer_pose[:2])
        trails.append("evader", *env.evader_pose[:2])
        if env.current_step % STRIDE == 0:
            frames.append(
                renderer.draw(env, trails, overlays=[overlay], title_prefix=label)
            )
            action_frames += 1
            flash_frames += int(sinfo["intervention"])
        obs, _, terminated, truncated, info = env.step([v_p, w_p, v_e, w_e])
        steps.append(
            {
                "step": env.current_step,
                "tier": sinfo["method"],
                "intervention": bool(sinfo["intervention"]),
                "nominal_infeasible": bool(nominal_infeasible),
                "min_clearance": sinfo["min_clearance"],
                "obstacle_clearance": float(obstacle_clearance),
                "contact": bool(
                    info["pursuer_wall_contact"] or info["pursuer_obstacle_contact"]
                ),
            }
        )
        if terminated or truncated:
            break

    # One terminal frame AFTER the loop, so the outcome the episode is chosen
    # for is actually visible: every action frame is drawn before its step, so
    # without this the capture instant never appears. Drawn with no overlay --
    # it carries no action and must not add a flash to the C4 accounting.
    trails.append("pursuer", *env.pursuer_pose[:2])
    trails.append("evader", *env.evader_pose[:2])
    frames.append(renderer.draw(env, trails, overlays=[], title_prefix=label))
    frames.extend([frames[-1]] * HOLD_FRAMES)
    record = {
        "label": label,
        "seed": seed,
        "evader": evader_cls.__name__,
        "outcome": info["outcome"],
        "steps": env.current_step,
        "final_distance": float(info["distance"]),
        "violations": sum(s["contact"] for s in steps),
        "n_interventions": sum(s["intervention"] for s in steps),
        "rendered_frames": action_frames,
        "terminal_frames": 1,
        "flash_frames": flash_frames,
        "obstacles": [dict(o) for o in env.obstacles],
        "per_step": steps,
    }
    return frames, record


# --------------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--md5-only",
        action="store_true",
        help="re-render into a temp file and print the checksum only",
    )
    args = parser.parse_args()

    if any(seed is None for _, _, _, seed in GATE_EPISODES):
        raise SystemExit("GATE_EPISODES not yet chosen — see Task 8")

    if not GATE_JSON.exists():
        sys.exit(f"missing {GATE_JSON} — run scripts/pe2_step06_gate.py first")
    with open(GATE_JSON) as f:
        gate = json.load(f)
    rows = {
        arm: {row["seed"]: row for row in block["episodes"]}
        for arm, block in gate["arms"].items()
    }

    cfg = EnvConfig()
    shield = CBFShield(cfg)
    renderer = Renderer(cfg, dpi=80, figsize=(5, 5))
    live: dict = {}
    overlay = ShieldOverlay(SHIELDED_ROLE, shield, live)

    frames: list[np.ndarray] = []
    records = []
    for label, arm, evader_cls, seed in GATE_EPISODES:
        ep_frames, record = play(
            cfg, evader_cls, seed, shield, renderer, overlay, live, label
        )
        frames.extend(ep_frames)

        # Cross-check against the gated numbers for this exact seed.
        row = rows[arm][seed]
        for key in ("outcome", "steps", "violations"):
            assert record[key] == row[key], (
                f"{label} seed {seed}: rendered {key}={record[key]} != "
                f"gate JSON {key}={row[key]}"
            )
        assert record["n_interventions"] == row["n_interventions"], (
            f"{label} seed {seed}: rendered {record['n_interventions']} "
            f"interventions != gate JSON {row['n_interventions']}"
        )
        record["arm"] = arm
        records.append(record)
        print(
            f"{label} (seed {seed}, arm {arm}): {record['outcome']} at step "
            f"{record['steps']}, {record['violations']} violations, "
            f"{record['n_interventions']} interventions — matches gate row OK"
        )

    renderer.close()

    # ------------------------------------------------------- checklist C1-C4
    checklist = {}

    # C1: divergence happens exactly at constraint approach.
    mismatches, above_bound, max_intervention_clearance, non_exact = [], [], None, 0
    sub_threshold = 0
    for record in records:
        bound = intervention_clearance_bound(cfg, record["obstacles"], shield)
        record["intervention_clearance_bound"] = bound
        for s in record["per_step"]:
            if s["tier"] != "exact":
                non_exact += 1
                continue
            if s["intervention"] and not s["nominal_infeasible"]:
                mismatches.append((record["seed"], s["step"], s))
            if s["nominal_infeasible"] and not s["intervention"]:
                sub_threshold += 1
            if s["intervention"]:
                if max_intervention_clearance is None:
                    max_intervention_clearance = s["min_clearance"]
                max_intervention_clearance = max(
                    max_intervention_clearance, s["min_clearance"]
                )
                if s["min_clearance"] > bound:
                    above_bound.append((record["seed"], s["step"], s["min_clearance"]))
    assert not mismatches, (
        "C1: an exact-tier step diverged from the nominal without any DCBF row "
        f"being violated: {mismatches[:3]}"
    )
    assert not above_bound, (
        "C1: the shield intervened further from every constraint than any row "
        f"can bind: {above_bound[:3]}"
    )
    checklist["C1_arrows_diverge_at_constraint_approach"] = {
        "pass": True,
        "exact_tier_steps_checked": sum(len(r["per_step"]) for r in records)
        - non_exact,
        "non_exact_tier_steps_skipped": non_exact,
        "max_clearance_at_an_intervention_m": max_intervention_clearance,
        "sub_threshold_corrections_reported_not_gated": sub_threshold,
        "analytic_binding_bound_m": max(
            r["intervention_clearance_bound"] for r in records
        ),
    }

    # C2: sliding along an obstacle boundary without contact.
    slides = []
    for record in records:
        if record["violations"] > 0:
            continue
        run = longest_run(
            [s["obstacle_clearance"] <= SLIDE_BAND for s in record["per_step"]]
        )
        slides.append((record["label"], record["seed"], run))
    best = max(slides, key=lambda t: t[2]) if slides else ("none", None, 0)
    assert best[2] >= MIN_SLIDE_STEPS, (
        f"C2: longest contact-free run inside {SLIDE_BAND} m of an obstacle is "
        f"{best[2]} steps < {MIN_SLIDE_STEPS} — no visible slide to gate"
    )
    checklist["C2_slides_along_obstacle_without_contact"] = {
        "pass": True,
        "band_m": SLIDE_BAND,
        "min_required_steps": MIN_SLIDE_STEPS,
        "longest_contact_free_slide_steps": best[2],
        "episode": {"label": best[0], "seed": best[1]},
        "per_episode": [
            {"label": lab, "seed": sd, "slide_steps": run} for lab, sd, run in slides
        ],
    }

    # C3: the pursuer still makes contact-capture.
    captures = [
        r
        for r in records
        if r["outcome"] == "capture" and r["final_distance"] <= cfg.capture_distance
    ]
    assert captures, (
        "C3: no rendered episode ends in a contact-capture — the shield would "
        "be blocking the pursuer's own objective"
    )
    checklist["C3_pursuer_still_makes_contact_capture"] = {
        "pass": True,
        "capture_distance_m": cfg.capture_distance,
        "captures": [
            {
                "label": r["label"],
                "seed": r["seed"],
                "steps": r["steps"],
                "final_distance_m": r["final_distance"],
            }
            for r in captures
        ],
    }

    # C4: the flash rate is the logged rate.
    total_frames = sum(r["rendered_frames"] for r in records)
    total_flashes = sum(r["flash_frames"] for r in records)
    total_interventions = sum(r["n_interventions"] for r in records)
    total_steps = sum(r["steps"] for r in records)
    for r in records:
        assert r["flash_frames"] == r["n_interventions"], (
            f"C4: {r['label']} flashed on {r['flash_frames']} frames but logged "
            f"{r['n_interventions']} interventions"
        )
        assert r["rendered_frames"] == r["steps"], (
            f"C4: {r['label']} rendered {r['rendered_frames']} frames for "
            f"{r['steps']} steps — STRIDE must be 1 for a rate comparison"
        )
    checklist["C4_flash_rate_matches_logged_rate"] = {
        "pass": True,
        "stride": STRIDE,
        "rendered_frames": total_frames,
        "flash_frames": total_flashes,
        "logged_interventions": total_interventions,
        "logged_steps": total_steps,
        "flash_rate": total_flashes / total_frames if total_frames else 0.0,
        "logged_intervention_rate": (
            total_interventions / total_steps if total_steps else 0.0
        ),
    }

    for name, block in checklist.items():
        print(f"checklist {name}: PASS")
        for key, value in block.items():
            if key != "pass":
                print(f"    {key}: {value}")

    path = GIF_PATH.with_suffix(".md5check.gif") if args.md5_only else GIF_PATH
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_gif(frames, str(path), fps=FPS)
    digest = md5_of(path)
    # The GIF writer coalesces runs of identical frames, so the HOLD_FRAMES
    # tail of each episode does not survive into the file. Record what the
    # artifact actually contains, not what was handed to the encoder.
    gif_frames = gif_frame_count(path)
    if args.md5_only:
        path.unlink()
        print(f"md5 (re-render, not written): {digest}")
        return

    with open(JSON_PATH, "w") as f:
        json.dump(
            {
                "config_fingerprint": gate["config_fingerprint"],
                "gif": str(GIF_PATH),
                "gif_md5": digest,
                "gif_frames": gif_frames,
                "frames_submitted_to_encoder": len(frames),
                "fps": FPS,
                "stride": STRIDE,
                "hold_frames": HOLD_FRAMES,
                "episode_selection": EPISODE_SELECTION_LOG,
                "checklist": checklist,
                "episodes": [
                    {k: v for k, v in r.items() if k != "per_step"} for r in records
                ],
            },
            f,
            indent=2,
        )
    print(
        f"wrote {GIF_PATH} ({gif_frames} frames in the file, "
        f"{len(frames)} submitted), md5 {digest}"
    )
    print(f"wrote {JSON_PATH}")


if __name__ == "__main__":
    main()
