"""S24.2 Step E.1 — a frozen learned PURSUER selector as a `select` callable.

Spec: `docs/pathway/tactics_prereg_s24_2_2026-10-17.md` section 1 and amendment
A1.2 ("learned = a frozen MaskablePPO checkpoint driven through `SelectorMDP`'s
own `_observe`/`action_masks`/`_candidates`, library per checkpoint").

The point of this module is that NOTHING about the pursuer's seat is re-derived
for the evader's round. The observation, the mask, the action -> tactic map and
the per-decision bookkeeping are `pe2.selector_mdp.SelectorMDP`'s own methods,
called on a `SelectorMDP` built OVER the shared game (`SelectorMDP.over_game`),
so the pursuer the evader trains against is the artefact the S24.1 gate
measured and not a second implementation of it.

What is NOT done here is the env stepping: `TacticLaw` asks `select` for a
tactic at its own decision points, and the evader's env owns the steps. So this
callable does exactly the part of `SelectorMDP.step` that is not an env step —
the mask check, the pick counters and A2.4's span bookkeeping — and returns the
tactic.
"""

from __future__ import annotations

import numpy as np

from pe2.selector_mdp import SelectorMDP


class LearnedPursuerSelect:
    """`select(obs, env, history) -> Tactic` playing a frozen MaskablePPO
    selector over `sel`'s library.

    `sel` must be a `SelectorMDP` whose `game` IS the game being driven (build
    it with `SelectorMDP.over_game`), because the observation and the mask are
    read off that game's live state. `sel.game.law.select` is replaced by this
    object by whoever owns the game — `sel`'s own `_select`, installed by its
    constructor, is not used and would raise `_NeedDecision`.
    """

    def __init__(self, sel: SelectorMDP, model, name: str = "") -> None:
        self.sel = sel
        self.model = model
        self.name = str(name)
        if model.observation_space.shape != (sel.obs_len,):
            raise ValueError(
                f"checkpoint {self.name or '?'} has observation shape "
                f"{model.observation_space.shape}, library {sel.library!r} has "
                f"({sel.obs_len},): the library flag does not match the "
                "checkpoint it is replaying"
            )
        if int(model.action_space.n) != sel.n_actions:
            raise ValueError(
                f"checkpoint {self.name or '?'} has {model.action_space.n} "
                f"actions, library {sel.library!r} has {sel.n_actions}"
            )

    def reset(self) -> None:
        """Per-episode counters. The game's own reset is the caller's."""
        self.sel.picks.clear()
        self.sel.n_decisions = 0

    def __call__(self, obs, env, history):
        sel = self.sel
        sel._obs_raw = obs
        mask = sel.action_masks()
        index = int(
            self.model.predict(
                sel._observe(obs), action_masks=mask, deterministic=True
            )[0]
        )
        if not mask[index]:
            # The same refusal `SelectorMDP.step` makes, for the same reason
            # (CG-T1): a masked action reaching the library would mean
            # `MaskablePPO` ignored the mask it was handed.
            sel.inapplicable_picks += 1
            raise AssertionError(
                f"frozen selector {self.name or '?'} picked action {index} "
                f"({sel.actions[index]}), which the mask "
                f"{np.asarray(mask).astype(int).tolist()} refuses"
            )
        # `SelectorMDP.step`'s bookkeeping that is not an env step: the pick
        # counters and A2.4's span scope.
        sel.n_decisions += 1
        sel.picks[sel.actions[index]] += 1
        game = sel.game
        game._in_span = True
        game.span_intervention = 0.0
        game.span_stalled = 0
        tactic = sel._candidates(obs)[index]
        if tactic is None:  # unreachable: a None candidate is masked out
            raise AssertionError(
                f"action {index} ({sel.actions[index]}) resolves to no tactic "
                "instance yet the mask admitted it"
            )
        return tactic
