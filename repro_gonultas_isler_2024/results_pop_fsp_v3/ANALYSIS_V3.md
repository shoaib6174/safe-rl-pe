# Pop-FSP v3 Analysis — Capped Mixing (80%)

## Configuration
- `pop_mixing_max: 0.8` — population sampling capped at 80%
- `pop_mixing_start: 100`, `pop_mixing_end: 500`
- arena_size: 16.0, evader v_max: 1.5
- All other Pop-FSP v2 parameters same (recency λ=0.15, asymmetric pops 16 vs 8)

## Results

| Seed | Total Episodes | First CR=1.0 | Mixing Status at Lock-in | Final CR |
|------|---------------|--------------|--------------------------|----------|
| 0 | 1102 | **ep 52** | Warmup (0% mixing) | 1.00 |
| 1 | 889 | **ep 192** | Early ramp (18% mixing) | 1.00 |
| 2 | 2849 | **ep 78** | Warmup (0% mixing) | 1.00 |

## Critical Finding

**Two of three seeds locked BEFORE population mixing even started.**

- Seed 0: CR=1.0 at ep 52 (mixing starts at ep 100)
- Seed 2: CR=1.0 at ep 78 (mixing starts at ep 100)

This means the capped mixing mechanism is **completely irrelevant** — the lock-in happens during vanilla self-play warmup, before any population diversity is introduced.

## Interpretation

The structural ∀/∃ asymmetry is so strong that even a few hundred episodes of vanilla self-play are sufficient for the pursuer to find a strategy that dominates the evader. Once this happens:

1. The evader's training data becomes overwhelmingly negative (always caught)
2. The evader cannot learn effective escape strategies from failure-only experience
3. The pursuer's strategy reinforces because it always wins
4. Population mixing cannot help because the equilibrium is already locked

## Conclusion

**Pop-FSP v3 (capped mixing) is fundamentally insufficient.** Any fix must either:

1. **Modify the game** to make the evader inherently more competitive (v4 approach)
2. **Prevent early lock-in** during warmup (e.g., force competitive opponents)
3. **Use a fundamentally different algorithm** that doesn't rely on self-play (e.g., PSRO with exploitability tracking)

The v4 approach (modified game: arena=20, evader=1.75) is the only remaining option in our current design space.
