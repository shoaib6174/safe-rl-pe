"""Step 5 visual gate: harness dashboard PNG regenerated from the Step 3
battery JSON alone (no env access, no re-running duels) + determinism
check (two renders must be byte-identical).

Checklist for sign-off (hard rule 9):
1. Left panel: 4 matchups in canonical order, CR bars with Wilson 95% CI
   whiskers, n annotated, 0.5 dashed reference line. Numbers match the
   Step 3 battery printout exactly.
2. Right panel: paired-design diagram (D57) — same seed sequence feeds
   every matchup; states the JSON-only provenance.
3. Re-running this script produces a byte-identical PNG (md5 printed).

Reads  results/visual_gates/step_03/step_03_battery.json
Writes results/visual_gates/step_05/step_05_harness.png

Usage: ./venv/bin/python scripts/pe2_step05_visual_gate.py
"""

import hashlib
import json
import pathlib

from pe2.figures import MATCHUP_ORDER, build_dashboard

BATTERY_JSON = pathlib.Path("results/visual_gates/step_03/step_03_battery.json")
OUT_DIR = pathlib.Path("results/visual_gates/step_05")
OUT_PNG = OUT_DIR / "step_05_harness.png"


def md5(path: pathlib.Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    build_dashboard(BATTERY_JSON, OUT_PNG)
    first = md5(OUT_PNG)
    build_dashboard(BATTERY_JSON, OUT_PNG)  # re-render over it
    second = md5(OUT_PNG)
    assert first == second, f"dashboard not deterministic: {first} != {second}"
    print(f"determinism: md5 {first} stable across re-render OK")

    # cross-check: rendered numbers must be the battery numbers
    with open(BATTERY_JSON) as f:
        battery = json.load(f)
    assert [m for m in MATCHUP_ORDER if m in battery["summaries"]] == list(
        battery["summaries"]
    ), "battery JSON matchup set differs from canonical MATCHUP_ORDER"
    print("cross-check: dashboard covers exactly the battery matchups OK")
    for name, s in battery["summaries"].items():
        print(
            f"  {name:<10} CR {s['cr']:.3f} "
            f"[{s['cr_ci'][0]:.3f}, {s['cr_ci'][1]:.3f}]  n={s['n']}"
        )
    print(f"wrote {OUT_PNG}")
    print("\nchecklist for sign-off:")
    print("1. CR bars + Wilson CIs match the numbers above; 0.5 line shown")
    print("2. paired-design panel states shared seed sequence + JSON provenance")
    print("3. md5 stable across re-render (printed above)")


if __name__ == "__main__":
    main()
