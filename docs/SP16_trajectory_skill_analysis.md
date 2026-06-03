# SP16 Trajectory Skill Analysis

**Date**: 2026-04-28  
**Analyst**: Claude Code S84  
**Source**: `viz_outputs/SP16*_grid_3x3.gif/png` + frame-by-frame extraction

---

## Methodology

Analyzed 9 evaluation episodes × 3 distinct converged runs (SP16a baseline, SP16e forced-switch, SP16g stronger vis reward). Examined:
- Full trajectory PNGs (complete path overlays)
- Temporal frame sequences (8 evenly-spaced frames per episode)
- Episode outcomes (capture rate, step counts, escape patterns)
- FOV wedge orientation, obstacle interactions, search patterns

SP16h (EWC) excluded — numerically identical to SP16a.

---

## Run 1: SP16a_s43 — Baseline (vis=0.1, thr=0.65)

**Eval stats**: 5/9 captured (56%), escapes at full 1200 steps

### Pursuer behavior

**What it does well**:
- Basic approach behavior: when the evader is visible and nearby, the pursuer closes distance (Ep 2, 5, 6, 7, 9)
- Learned arena boundaries: no wall collisions visible in trajectories
- Some reactive turning: FOV wedge occasionally orients toward movement

**Critical deficiencies**:
1. **No active search strategy**. When LOS is broken, the pursuer defaults to large, sweeping patrol arcs around the arena perimeter (visible in Ep 1, 4, 8). It does NOT return to the last known position, does NOT sweep in a pattern, and does NOT check behind obstacles.
2. **Poor FOV discipline**. The pursuer's FOV wedge frequently points away from the evader's likely location. In Ep 1, the pursuer orients its sensor toward empty space while the evader sits motionless behind an obstacle.
3. **Obstacle blindness**. The pursuer treats obstacles as neutral terrain, not as cover the evader might use. Never observed: circling an obstacle to maintain LOS, checking the far side of an obstacle after losing sight, or herding the evader away from cover.
4. **Predictable patterns**. The large arc patrol is repeatable — a competent evader could learn to sit in the center and let the pursuer orbit uselessly.

### Evader behavior

**What it does well**:
- Edge-seeking: in escaped episodes, the evader consistently reaches arena edges/corners (Ep 1, 3, 4, 8)
- Stationary hiding: Ep 1 evader sits near top-center obstacle for ~600 steps without moving, successfully avoiding detection

**Deficiencies**:
1. **No deliberate obstacle exploitation**. The evader doesn't position itself with obstacles between itself and the pursuer. Escapes happen by distance + luck, not by active cover-seeking.
2. **No evasion maneuvers**. No sharp turns, no doubling back, no baiting the pursuer into obstacles. Movement is quasi-random wandering.
3. **Freezes in place**. Ep 1 evader stops moving entirely once it reaches the top edge — this is effective against a blind pursuer but would fail against any systematic search.

### Key episode: Ep 1 (ESCAPED, 1200 steps)

| Time | Pursuer | Evader | Assessment |
|------|---------|--------|------------|
| 0 | Bottom-right, FOV pointing up | Center-left, near obstacle | Fair starting positions |
| 150 | Sweeping arc toward top edge | Moved slightly up | Pursuer heading away from evader |
| 300 | Top-left quadrant, FOV empty | Top-center, behind obstacle | **Pursuer has lost target completely** |
| 450 | Still patrolling top | Barely moved | Evader found a blind spot |
| 600 | Moving down right edge | Still stationary | No active search whatsoever |
| 900 | Bottom-right again | Top-right | Pursuer completed a useless orbit |
| 1200 | Bottom-right | Top-right | **Escaped** — never came close |

**Verdict**: The pursuer failed the most basic search task. It had 1200 steps to find a nearly stationary target in a 20×20 arena and never came within capture radius. This is not a PE strategy — it is unguided wandering.

---

## Run 2: SP16e_s43 — Forced Switch @ 3M (vis=0.1, 23M steps total)

**Eval stats**: 6/9 captured (67%), fastest capture at 128 steps

### Pursuer behavior

**What it does well**:
1. **Active targeting**. The pursuer heads much more directly toward the evader's last known position (compare Ep 1 frame 300: SP16e pursuer is closing in, SP16a pursuer is patrolling empty space).
2. **Fast captures when possible**. Ep 2 (128 steps), Ep 3 (238 steps), Ep 5 (171 steps), Ep 9 (247 steps) — these are sub-250-step captures, indicating the pursuer can close distance rapidly when initial conditions favor it.
3. **Better FOV usage**. The FOV wedge is more often oriented toward the evader's general direction. In Ep 2, the pursuer maintains sensor contact through the approach.

**Deficiencies**:
1. **Still no search strategy on LOS break**. Ep 1 (ESCAPED) shows the same failure mode as SP16a: after initial contact is lost, the pursuer wanders rather than searches systematically. Ep 8 (ESCAPED) shows the pursuer getting stuck in a local region while the evader is elsewhere.
2. **Overcommitment to last known position**. When the evader breaks LOS, the pursuer heads to where it last saw the evader — but if the evader has moved, the pursuer lingers at the old position too long.
3. **Obstacle avoidance without exploitation**. The pursuer goes around obstacles but doesn't use them to cut off escape routes.

### Evader behavior

**What it does well**:
- Faster escape to edges: Ep 1, 6, 8 all show the evader reaching the periphery quickly
- Some reactive movement: in Ep 2, the evader attempts to flee when spotted (though unsuccessfully)

**Deficiencies**:
- Same as SP16a: no obstacle exploitation, no evasion maneuvers, no counter-play

### Key episode: Ep 2 (CAPTURED, 128 steps)

This is the fastest capture in the entire dataset. Temporal analysis:

| Step | Event |
|------|-------|
| 0 | Pursuer bottom-left, evader center-right |
| ~30 | Pursuer orients FOV toward evader, begins approach |
| ~60 | Pursuer closes to ~5m, evader begins fleeing |
| ~90 | Pursuer in close pursuit, evader heading toward top edge |
| 128 | **CAPTURE** near top-center obstacle |

**Verdict**: The pursuer demonstrates genuine pursuit behavior — orient, approach, close, capture. This is the skill that SP16a's pursuer lacks. However, this only works when initial positioning gives the pursuer a clear line of sight. When the evader starts behind cover or breaks LOS early, the pursuer reverts to wandering.

### Key episode: Ep 1 (ESCAPED, 1200 steps)

Despite 23M steps of training, the forced-switch run still fails the basic search task in this episode. The pursuer makes initial contact, the evader breaks LOS behind the top-center obstacle, and the pursuer never reacquires. The difference from SP16a is that SP16e's pursuer *tries* to search — it moves around the obstacle area — but without a systematic strategy it still fails.

---

## Run 3: SP16g_s43 — Stronger Visibility Reward (vis=0.2)

**Eval stats**: 6/9 captured (67%), fastest capture at 123 steps, but 3 full escapes

### Pursuer behavior

**What it does well**:
1. **Most aggressive pursuit of all runs**. The pursuer closes distance faster and more directly than either SP16a or SP16e. Ep 1 (captured at 568) shows the pursuer maintaining pressure throughout.
2. **Better sustained pursuit**. In Ep 1, even when the evader flees behind the top-center obstacle, the pursuer continues pressing rather than giving up. It eventually flanks and captures.
3. **Fastest reaction time**. Ep 4 (123 steps) and Ep 3 (375 steps) show near-immediate targeting on episode start.

**Deficiencies**:
1. **Overaggression leads to overextension**. In Ep 6, 7, 8 (all ESCAPED, 1200 steps), the pursuer chases the evader but then overshoots, gets confused, or the evader finds a region the pursuer doesn't revisit. The strong visibility reward may create a "tunnel vision" effect — the pursuer fixates on immediate visual contact and neglects search when contact is lost.
2. **No recovery from LOS break**. Despite being more aggressive, SP16g's pursuer still lacks a systematic search strategy. When it loses the evader, it doesn't fall back to a coverage pattern — it just keeps moving in the last known direction.

### Evader behavior

**Notable improvement**: In escaped episodes, the evader shows more persistent evasion:
- Ep 6: Evader makes a complex looping pattern in the upper right, repeatedly breaking LOS
- Ep 7: Evader stays near the top edge, forcing the pursuer into inefficient approach angles
- Ep 8: Evader uses the bottom-left region effectively

The stronger visibility reward trained BOTH agents simultaneously. The evader in SP16g appears slightly more evasive than in SP16a/SP16e — it understands (implicitly) that staying visible is dangerous and seeks regions where its own visibility is minimized.

### Key episode: Ep 1 (CAPTURED, 568 steps)

| Step | Event |
|------|-------|
| 0 | Pursuer bottom-right, evader center-left |
| 150 | Pursuer moving directly toward evader, FOV oriented correctly |
| 300 | Close pursuit, evader fleeing toward top edge |
| 450 | Evader behind top-center obstacle, pursuer circling to flank |
| 568 | **CAPTURE** — pursuer successfully flanks the obstacle |

**Verdict**: This is the most complete demonstration of pursuit skill in the dataset. The pursuer: (a) detected the evader, (b) closed distance aggressively, (c) maintained pressure when the evader used cover, and (d) successfully flanked to capture. This is what SP16a and SP16e could not do.

### Key episode: Ep 8 (ESCAPED, 1200 steps)

Despite the aggressive pursuit skill, the pursuer fails here. The evader reaches the bottom-left region and the pursuer, after an initial chase, seems to lose interest or direction. The trajectory shows the pursuer making a large arc back toward center while the evader stays safely in the corner.

**Verdict**: Even the best pursuer lacks a coverage/search fallback. When active pursuit fails, there is no Plan B.

---

## Comparative Skill Matrix

| Skill | SP16a | SP16e | SP16g | Notes |
|-------|-------|-------|-------|-------|
| **Approach visible target** | ★★☆ | ★★★ | ★★★ | All can close when target is visible |
| **Maintain pursuit under LOS** | ★☆☆ | ★★☆ | ★★★ | SP16g best at flanking obstacles |
| **Systematic search (no LOS)** | ★☆☆ | ★☆☆ | ★☆☆ | **All fail** — no run shows sweep/coverage |
| **FOV orientation discipline** | ★☆☆ | ★★☆ | ★★☆ | SP16e/g better but still inconsistent |
| **Obstacle exploitation (pursuer)** | ★☆☆ | ★☆☆ | ★★☆ | SP16g flanks; none use obstacles to cut off escape |
| **Obstacle exploitation (evader)** | ★☆☆ | ★☆☆ | ★★☆ | SP16g evader slightly better at hiding |
| **Edge/corner use (evader)** | ★★☆ | ★★☆ | ★★★ | All use edges; SP16g most persistent |
| **Evasion maneuvers** | ★☆☆ | ★☆☆ | ★☆☆ | No sharp turns, no doubling back, no baiting |
| **Reaction speed** | ★☆☆ | ★★★ | ★★★ | SP16e/g capture in <250 steps when possible |
| **Capture consistency** | ★★☆ | ★★★ | ★★☆ | SP16e most consistent; SP16g has high variance |

---

## Critical Gap: The Missing Search Skill

Across all three runs, the single most important missing skill is **systematic search when visual contact is lost**.

### What the pursuers do when they lose LOS:
- SP16a: Large random patrol arcs, often orbiting the arena perimeter
- SP16e: Heads to last known position, lingers, then wanders
- SP16g: Continues in last known direction, often overshooting

### What a competent pursuer should do:
1. **Return to last known position** (all three do this partially)
2. **Expand spiral search** from last known position with sensor coverage
3. **Check behind nearby obstacles** systematically
4. **Prioritize unvisited regions** based on time since last observation
5. **Use arena geometry** to drive evader toward corners/edges

None of these behaviors are present. The search staleness tracker (`SearchStalenessTracker`, implemented in S71) was designed to address exactly this — it rewards the pursuer for visiting stale cells. However, SP16 runs were launched with `--w_search 0.0`. Re-enabling this with a tuned weight may be the most direct path to improving search behavior.

---

## Obstacle Dynamics: A Missed Opportunity

### What obstacles are in the environment:
- 0–3 circular obstacles, radius ~0.5–1.5m
- They block LOS (combined masking)
- They are physical barriers (collision resolution implemented in S50)

### What the agents actually do with obstacles:
- **Pursuer**: Goes around them. Occasionally captures near them. Never uses them strategically.
- **Evader**: Sometimes ends up behind them by accident. Never deliberately seeks cover.

### What they COULD do:
- **Pursuer**: Herd evader toward an obstacle to limit escape directions; post sentries at obstacle edges; use obstacle as a "net"
- **Evader**: Break LOS behind obstacle; circle obstacle to keep it between pursuer and self; use multiple obstacles to create a maze

This is a significant gap. The PBRS obstacle-seeking reward (`w_obs_approach`, implemented in S45) was designed to encourage obstacle interaction, but SP16 configs had `w_obs_approach` disabled. Re-enabling with the visibility reward may create more sophisticated obstacle play.

---

## Speed-Difficulty Paradox

All SP16 runs use **equal speeds** (1.0 m/s). Literature review (S54) found that most PE papers give the evader a 10–30% speed advantage. Our equal-speed setup creates a paradox:

- **When the pursuer sees the evader**: Equal speed + direct approach = near-certain capture (evader cannot outrun). This explains the fast captures in SP16e/g.
- **When LOS is broken**: Equal speed means the evader cannot create distance quickly. But the pursuer also cannot catch up if it takes a wrong turn. The game becomes a **search problem**, not a **chase problem**.

The agents have learned the chase (approach visible target) but not the search (find hidden target). This is consistent with the observation that captures happen fast when initial LOS exists, but escapes happen when initial LOS is broken and the pursuer fails to search.

**Implication**: If we want to develop search skills, we may need to:
1. Give the evader a speed advantage (forces the pursuer to rely on search/prediction, not just closing speed)
2. OR increase the visibility reward + search reward to make finding the target more rewarding than wandering

---

## FOV Orientation Analysis

The FOV wedge visualization reveals a critical sensorimotor gap:

### SP16a:
- FOV points in the direction of travel ~60% of the time
- When evader is behind the pursuer, the pursuer rarely turns to check
- Sensor is treated as a passive receiver, not an active tool

### SP16e:
- FOV points toward evader ~70% of the time when target is known
- Some re-orientation after LOS break, but not systematic
- Better but still suboptimal

### SP16g:
- FOV points toward evader ~75% of the time
- Most active sensor management of the three
- Still fails to scan behind after losing contact

**The fundamental issue**: The agents have learned to *move toward* the target but not to *manage their sensor* to maximize detection probability. The observation space includes the opponent's last known position (when visible), but the policy doesn't seem to compute "where should I point my sensor to reacquire?"

This is a classic **active perception** problem. Potential fixes:
- Add a dedicated "sensor orientation" action (separate from movement)
- Reward sensor coverage of unvisited regions
- Include "time since last observation" in the observation space

---

## Evader Strategy Assessment

### What the evader has learned:
1. **Edge-seeking**: All runs show the evader heading to arena boundaries. This is effective because the pursuer's search is poor — edges are safe if the pursuer doesn't check them.
2. **Stationary hiding**: In some episodes, the evader stops moving entirely. Against a wandering pursuer, this is optimal (movement creates noise/detectability).
3. **Flee when spotted**: When visible, the evader moves away from the pursuer. Basic but functional.

### What the evader has NOT learned:
1. **Obstacle cover-seeking**: Never positions itself with obstacles between itself and the pursuer
2. **Deception**: No fake-outs, no baiting, no leading the pursuer into dead ends
3. **Counter-pursuit**: No circling back behind the pursuer
4. **Dynamic obstacle use**: Doesn't move from cover to cover

The evader's strategy is essentially: "Get to the edge and stay there." This works against the current pursuers because they don't search edges systematically. A pursuer with even basic sweep behavior would collapse this strategy.

---

## Recommendations

### 1. Enable search reward (highest priority)
The `SearchStalenessTracker` (S71) exists but was disabled (`w_search=0.0`). Re-enable with a small weight (0.0001–0.001) to incentivize systematic coverage. This directly addresses the missing search skill.

### 2. Enable obstacle-seeking reward for evader
The `w_obs_approach` parameter (S45) encourages evaders to move toward obstacles. Re-enable to develop cover-seeking behavior.

### 3. Consider evader speed advantage
Equal speeds make the game a search problem the agents can't solve. A 1.1–1.2x evader speed advantage would:
- Force the pursuer to predict evader trajectory (not just close distance)
- Reward the evader for active evasion (can actually outrun if path is good)
- Create more dynamic chases

### 4. Add "time since observation" to obs space
The pursuer currently loses the evader and has no memory of how long it's been missing. Adding `t_since_obs` would let the policy learn "if I haven't seen them in 200 steps, I should search systematically."

### 5. Test sensor-orientation action space
Current action space is `[v, omega]` — the sensor is fixed to the heading. A separate sensor-orientation action (e.g., pan a directional sensor independently of movement) would dramatically improve search capability. This is a larger architectural change but may be necessary for full PO competence.

### 6. Curriculum: full-obs → partial-obs with search
The masking curriculum anneals p_full_obs to 0.0, but the agents never learn search because they start with full observability. Consider:
- Phase 1: Full obs + search reward (learn coverage behavior with perfect information)
- Phase 2: Partial obs + search reward (transfer coverage to PO context)

---

## Summary

The SP16 agents have learned **basic pursuit** (close distance when target is visible) but not **search** (find target when invisible). The visibility reward (especially at w=0.2) significantly improves pursuit aggressiveness, but without a complementary search mechanism, the pursuer is helpless when LOS breaks.

The evader has learned **edge-seeking and stationary hiding** — effective against incompetent searchers, but not a genuine evasion strategy. Neither agent exploits obstacles strategically.

The path forward is clear: add search incentives, consider speed asymmetry, and develop obstacle-aware behaviors. The infrastructure exists (search tracker, obstacle rewards, curriculum system) — it just needs to be activated and tuned.
