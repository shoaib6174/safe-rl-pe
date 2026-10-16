"""S24-FLAT Step F.1 — the flat continuous-action learner on the frozen game.

Pre-registration: `docs/pathway/tactics_prereg_s24_flat_2026-10-14.md`
sections 2, 3 (arm `FLAT-s`) and 5 (CG-F1, CG-F3). D151's claim —
"hierarchical selection over validated tactics learns under the filter where
flat learning did not" — rests on OLD flat-RL collapses under other shields,
observations and budgets (D108). This env is the same learner class
(continuous PPO) on the EXACT frozen tactic game the selector plays on: same
`RecoveryShieldV2`, same `hold`, same dead-reckoning memory and recogniser
features in the observation, same scripted `TacticOpponent(EvaderRulesV0)`
evader, same terminal-only reward, same budget. One variable changes against
`SelectorMDP`: the ACTION — `(v, omega)` in `[-1, 1]^2` instead of a tactic —
and, as a consequence of there being no tactics, the 21 tactic one-hot
entries of the observation are identically 0.

Readings taken here, because the brief or the prereg can be read two ways.
The prereg wins wherever they differ and that is said at the line.

1. **The env is `SelectorMDP` with a different action, so it is a SUBCLASS of
   it and not a second copy of its assembly.** The brief offers factoring the
   observation builder into a shared function; `SelectorMDP`'s own test
   `test_the_cap_is_the_selectors_module_and_a_span_never_exceeds_it` asserts
   `"DECISION_PERIOD" in inspect.getsource(SelectorMDP._observe)`, so
   factoring would change a selector test, which the brief forbids. Inheriting
   reuses the builder BIT-FOR-BIT with `pe2/selector_mdp.py` untouched: the
   layout constants are the same objects, `_observe` is the same code, and
   the 21 tactic entries are zeroed at their source — `_spans()` returns no
   spans, because this pursuer runs no tactics.
2. **The action is held `hold` env steps regardless of visibility; one
   decision is one span** (prereg section 3: "held for 10 env steps (the same
   `hold`) ... at EVERY decision regardless of visibility"). This is the one
   place `SearchMDP.step`'s hold pattern is NOT copied: there the hold breaks
   early on an acquisition because the intercept law takes over, and FLAT has
   no law to hand control to. `hold` is read off the game (`SearchMDP.hold`,
   = `pe2.structured.HOLD`), not restated.
3. **The decode is `SearchMDP`'s, by handing the Box action to
   `_step_inner`** — `v = (a0 + 1) / 2 * v_max`, `omega = a1 * omega_max` is
   computed there, before the shield, so there is no second copy of it here.
4. **Every env step is tagged `branch="search"`** — the recorder's
   learner-driven branch (`pe2/recorder.py`) — so FLAT's immobile-step rate,
   pinned runs and intervention split are the same quantities, in the same
   fields, as the selector's learner steps.
5. **Reward is the env's terminal +-1 summed over the span and nothing else**
   (prereg section 3, as S24.1): `SearchMDP._shaped` is never reached, so
   `r_acq`, the stall penalty and the intervention penalty cannot enter.
6. **CG-F1's observation half runs on a fresh CLONE of this env plus a
   `SelectorMDP` built with the same arguments**, never on `self`: asserting
   it must not reset the training env, consume an episode of its seed stream
   or move its `env_steps` / `violations_total` counters, which the trainer's
   budget and H-SAFE are read off. The config and assembly halves are
   inspected on `self` directly.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from pe2.selector_mdp import N_LAST_TACTICS, OBS_SLICES, SelectorMDP, _game_module

PREREG = "docs/pathway/tactics_prereg_s24_flat_2026-10-14.md"

# Reading 1: no tactic ever runs, so the current-tactic block (one-hot 5 +
# time-in-tactic) and the last-3 block are the 21 entries that must read 0.
NO_SPANS: tuple = (None,) * (1 + N_LAST_TACTICS)
TACTIC_SLICES = (OBS_SLICES["current_tactic"], OBS_SLICES["last_tactics"])
TACTIC_INDICES = tuple(
    i for block in TACTIC_SLICES for i in range(block.start, block.stop)
)
N_TACTIC_ENTRIES = len(TACTIC_INDICES)  # 6 + 15 = 21

# CG-F1 is a CONSTRUCT check, so it runs on the construct stream the v0b game
# was watched on (`scripts/s24_tactic_game.py::CONSTRUCT_SEED0`) — disjoint
# from this round's training stream (seed * 1000), probe stream (2 550 000+)
# and gate stream (2 400 000+).
CG_SEED = 2_290_000
CG_DECISIONS = 12  # enough for the selector's last-3 block to fill


class FlatMDP(SelectorMDP):
    """One episode of the frozen v0b tactic game driven by a continuous
    `(v, omega)` command held `hold` env steps (reading 2).

    The pursuer has NO intercept law and NO tactics: `TacticLaw.act` is never
    called, and the `select` `SelectorMDP.__init__` injects stays installed as
    a GUARD — if anything ever did consult the law it would raise
    `_NeedDecision` out of `step` rather than silently play a tactic.

    `env_steps` counts UNDERLYING env steps, so budgets and probes stay in env
    steps as in every prior round. `inapplicable_picks` and `picks` are
    inherited and stay at 0 / empty: CG-F3 rules that CG-T1 and A2.1 do not
    apply to an env with no mask, and they are kept only so the log schema the
    gate runner reads is the selector's.
    """

    def __init__(
        self,
        speed_ratio: float = 1.2,
        *,
        memory: bool = True,
        seed0: int = 0,
        recorder: bool = False,
        label: str = "",
        cfg=None,
        hold: int | None = None,
    ) -> None:
        # Reading 6: the exact constructor arguments, so CG-F1 can build the
        # comparison pair with the same ones and not with re-derived defaults.
        self._args = {
            "speed_ratio": float(speed_ratio),
            "memory": bool(memory),
            "seed0": int(seed0),
            "recorder": bool(recorder),
            "label": str(label),
            "cfg": cfg,
            "hold": hold,
        }
        super().__init__(
            speed_ratio,
            memory=memory,
            seed0=seed0,
            recorder=recorder,
            label=label,
            cfg=cfg,
        )
        # Reading 2 / 3: the game's own hold and the inner wrapper's own Box,
        # exactly as `SearchMDP.__init__` takes them.
        # A1.2: FLAT40 holds the command for the selector's decision cap (40)
        # instead of the primitive's hold (10); the game's own hold otherwise.
        self.hold = int(hold) if hold is not None else int(self.game.hold)
        self.action_space = self.game.inner.action_space
        self.span_lengths: list[int] = []

    # -- the tactic blocks are dead (reading 1) ---------------------------
    def _spans(self) -> list:
        """No span ever: this pursuer runs no tactics, so the 21 tactic
        entries of `SelectorMDP._observe` are written 0 at their source."""
        return list(NO_SPANS)

    def action_masks(self):
        """CG-F3: FLAT has no mask. Raised rather than returned all-True so a
        maskable learner cannot be trained on this env by accident."""
        raise AssertionError(
            "FlatMDP has no action mask (CG-F3): the action is continuous and "
            "every command is admissible to the env; the shield, not a mask, "
            "is what refuses one"
        )

    # -- episode driving --------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self.span_lengths = []
        return super().reset(seed=seed, options=options)

    def step(self, action):
        a = np.asarray(action, dtype=np.float32)  # `_step_inner` clips it
        self.n_decisions += 1
        self._span_steps = 0
        # A2.4's span bookkeeping, as `SelectorMDP.step` does it: this env
        # drives `_step_inner` itself, and without these the recorder's
        # `in_span` reads False on every step and the filter's work cannot be
        # split (`pe2/recorder.py`).
        self.game._in_span = True
        self.game.span_intervention = 0.0
        self.game.span_stalled = 0
        reward, term, trunc = 0.0, False, False
        obs = self._obs_raw
        for _ in range(self.hold):  # reading 2: no early break on visibility
            obs, r, _onset, term, trunc = self.game._step_inner(a, "search")
            reward += r  # reading 5: the env's terminal +-1, nothing else
            self._span_steps += 1
            if term or trunc:
                break
        self.game._span_total += float(self.game.span_intervention)
        self._obs_raw = obs
        self.span_lengths.append(int(self._span_steps))
        info = dict(
            self.game._info,
            env_steps=self.env_steps,
            span_steps=self._span_steps,
            inapplicable_picks=self.inapplicable_picks,  # CG-F3: always 0
            violations=int(self.game.episode_violations),
            stalled_steps=int(self.game.span_stalled),
            intervention_magnitude=float(self.game.span_intervention),
        )
        return self._observe(obs), float(reward), term, trunc, info

    # -- evaluation -------------------------------------------------------
    def play_episode(self, predict, seed: int) -> dict:
        """One seeded episode, `predict(obs, deterministic=True) -> (action,
        _)` deciding at every decision point — the `PPO.predict` contract (no
        `action_masks`: CG-F3).

        The log is `SelectorMDP.play_episode`'s schema so the gate runner reads
        one shape for both arms, with the three tactic-selection fields
        answered as "there were none": `pursuer_spans` empty, `n_switches`
        None (not 0 — nothing could have switched), `picks` empty.
        """
        game = self.game
        obs, _info = self.reset(seed=seed)
        term = trunc = False
        while not (term or trunc):
            action, _ = predict(obs, deterministic=True)
            obs, _r, term, trunc, _i = self.step(action)
        branch = game.branch_summary()
        return {
            "seed": int(seed),
            "role": "pursuer",
            "outcome": game._info["outcome"],
            "steps": int(game.inner.env.current_step),
            # The brief's name for the same quantity, so the flat log answers
            # to both it and the selector's `steps`.
            "n_steps": int(game.inner.env.current_step),
            "final_distance": float(game._info["distance"]),
            "los_breaks": int(game._los_breaks),
            "shielded_role": "pursuer",
            "violations": int(game.episode_violations),
            "shield": game.shield_metrics.summary(),
            "branch": branch,
            "tactics": {
                # The pursuer's span log is empty by construction (its law is
                # never consulted); the evader's is the scripted v0 evader's.
                "evader": _history(game.evader),
                "pursuer": _history(game.law),
            },
            "usage": {role: dict(counts) for role, counts in game.usage.items()},
            "n_decisions": int(self.n_decisions),
            "mean_span_steps": (
                sum(self.span_lengths) / len(self.span_lengths)
                if self.span_lengths
                else 0.0
            ),
            # No tactic selection happened: see the docstring.
            "pursuer_spans": [],
            "n_zero_length_spans": 0,
            "n_switches": None,
            "picks": {},
            "inapplicable_picks": 0,
            "recovery": {
                "n_recoveries": int(game.shield.n_recoveries),
                "n_forward": int(game.shield.n_recoveries_forward),
                "n_rotation": int(game.shield.n_recoveries_rotation),
            },
        }

    # -- CG-F1 ------------------------------------------------------------
    def config_diff_vs_selector(self, allowed_assembly: tuple = ()) -> dict:
        """CG-F1 (prereg section 5): FLAT's env differs from the selector's
        ONLY in the action space and the zeroed tactic block.

        Three comparisons against a `SelectorMDP` built with THIS env's
        constructor arguments:

        - `config_diff` — the differing `EnvConfig` fields; must be `[]`;
        - `assembly_diff` — the differing fields of the game assembly the
          action space is not (shield and its gains, `hold`, the observation
          flags, the reward wrapper and every shaping weight, the evader, the
          rule tables, the memory, the seed stream, the observation width);
          must be `[]`;
        - the per-entry observation comparison on an identical reset state:
          every entry must agree, and the 21 tactic entries must stay 0 in
          FLAT while they go non-zero in the selector once it has run tactics
          (`CG_DECISIONS` decisions on the construct seed).

        Reading 6: the observation half uses a fresh clone, never `self`.
        """
        args = dict(self._args, recorder=False, label="")
        # `hold` is FlatMDP's own argument (A1.2); SelectorMDP's cadence is its DONE-or-40 cap
        flat = type(self)(**args)
        sel = SelectorMDP(**{k: v for k, v in args.items() if k != "hold"})
        config_diff = sorted(
            field
            for field, value in dataclasses.asdict(flat.cfg).items()
            if value != dataclasses.asdict(sel.cfg).get(field)
        )
        mine, theirs = _assembly(flat), _assembly(sel)  # review item 2: both clones
        allowed = set(allowed_assembly)
        assembly_diff = sorted(k for k, v in mine.items() if theirs[k] != v)

        flat_obs, _i = flat.reset(seed=CG_SEED)
        sel_obs, _i = sel.reset(seed=CG_SEED)
        diff = [int(i) for i in np.flatnonzero(np.asarray(flat_obs) != np.asarray(sel_obs))]
        index = list(TACTIC_INDICES)
        for _ in range(CG_DECISIONS):  # let the selector fill its tactic block
            mask = sel.action_masks()
            _o, _r, term, trunc, _i = sel.step(int(np.flatnonzero(mask)[0]))
            if term or trunc:
                break
        sel_live = sel._observe(sel._obs_raw)
        for _ in range(CG_DECISIONS):  # and FLAT drive the same number
            _o, _r, term, trunc, _i = flat.step(np.array([1.0, 0.0], dtype=np.float32))
            if term or trunc:
                break
        flat_live = flat._observe(flat._obs_raw)
        flat_nonzero = int(np.count_nonzero(np.asarray(flat_live)[index]))
        sel_nonzero = int(np.count_nonzero(np.asarray(sel_live)[index]))
        return {
            "gate": "CG-F1",
            "prereg": PREREG,
            "cg_seed": CG_SEED,
            "config_diff": config_diff,
            "assembly_diff": assembly_diff,
            "assembly": mine,
            "obs_len": [len(flat_obs), len(sel_obs)],
            "obs_diff_indices_on_identical_reset": diff,
            "tactic_indices": index,
            "n_tactic_entries": N_TACTIC_ENTRIES,
            "decisions_driven": CG_DECISIONS,
            "flat_tactic_entries_nonzero": flat_nonzero,
            "selector_tactic_entries_nonzero": sel_nonzero,
            "action_space": {
                "flat": str(flat.action_space),
                "selector": str(sel.action_space),
            },
            "note": (
                "the config and assembly halves are inspected on the built "
                "env; the observation half runs on a fresh clone plus a "
                "SelectorMDP with the same arguments, so asserting CG-F1 "
                "never consumes the training env's episode stream or its "
                "counters"
            ),
            "allowed_assembly_diff": sorted(allowed),
            "ok": bool(
                not config_diff
                and not (set(assembly_diff) - allowed)
                and not diff
                and flat_nonzero == 0
                and sel_nonzero > 0
                and len(flat_obs) == len(sel_obs)
            ),
        }


def _history(adapter) -> list:
    """The game module's own span-log formatter (`scripts/s24_tactic_game.py`),
    so the two arms' `tactics` blocks are written by one function."""
    return _game_module()._history(adapter)


def _assembly(env) -> dict:
    """The fields of the built game that CG-F1 requires to be identical: every
    ingredient of the frozen v0b assembly except the action space itself."""
    game = env.game
    shield = game.shield
    return {
        "game_class": type(game).__name__,
        "shield_class": type(shield).__name__,
        "shield_margin": float(shield.margin),
        "shield_gamma": float(shield.gamma),
        "shield_recovery_after": int(shield.recovery_after),
        "hold": int(game.hold),
        "shield_obs": bool(game.shield_obs),
        "escape_obs": bool(game.escape_obs),
        "no_reward_wrapper": game._driver is game.inner,
        "zero_reward": bool(game.zero_reward),
        "r_acq": float(game.r_acq),
        "stall_penalty": float(game.stall_penalty),
        "intervention_penalty": float(game.intervention_penalty),
        "law_class": type(game.law).__name__,
        "evader_class": type(game.evader).__name__,
        "evader_rules": type(game.rules["evader"]).__name__,
        "pursuer_rules": type(game.rules["pursuer"]).__name__,
        "pursuer_memory": type(game.rules["pursuer"].memory).__name__,
        "pursuer_tactics": [t.name for t in game.rules["pursuer"].tactics],
        "env_seed0": int(game.inner.seed0),
        "obs_len": int(env.observation_space.shape[0]),
        "memory_arm": bool(env.memory),
    }
