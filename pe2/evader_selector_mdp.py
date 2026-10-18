"""S24.2 Step E.1 — the EVADER's tactic-selection semi-MDP.

Spec: `docs/pathway/tactics_prereg_s24_2_2026-10-17.md` section 1 and amendment
A1 (A1.1 reward, A1.2 pursuer seats, A1.3 decision mechanics). Where section 1
and A1 differ, A1 wins; where the brief and the prereg differ, the prereg wins.

This is the mirror of `pe2.selector_mdp.SelectorMDP` in the other seat: ONE
field of the frozen v0b assembly changes, the evader adapter's `select`, and
the pursuer's seat is whatever A1.2 names — the game's own `TacticLaw` select
(`RULES-V0`), `fixed(rules["pursuer"].lead)` (`LEAD-1.0`), or a frozen
`MaskablePPO` checkpoint driven through `SelectorMDP`'s OWN observation, mask
and candidate code over the SHARED game (`pe2.learned_select`). Nothing about
either seat is re-derived here; what is new is only the evader's side of it.

Readings taken here, each because the brief or the prereg can be read two ways.

1. **Reward is `-(the reward `_step_inner` returns)`, summed over the span.**
   A1.1 says "the NEGATIVE of the pursuer's reward exactly as the game emits
   it ... pure zero-sum, no second reward definition", and its parenthesis
   names `r_acq`. But `SelectorMDP` — the pursuer whose reward this negates —
   sums `_step_inner`'s reward and nothing else (its reading 6: "no acquisition
   bonus"), so including `r_acq` here would BE a second reward definition, the
   one thing A1.1 forbids. The brief's operational line ("the reward returned
   by `_step_inner`") is therefore taken, and it is exactly zero-sum against
   the pursuer selector this round trains against.
2. **The pursuer's command for an env step is computed ONCE** (A1.3) and held
   across an evader decision point, because `TacticLaw.act` is not idempotent:
   it advances `step_index` and the current tactic's span counter, so a second
   call per env step would rewrite the pursuer's span log. CG-E1 is the test.
3. **A pursuer law with nothing to act on falls to the search policy for
   `game.hold` steps**, the action computed once at the state the law gave up
   on and held, breaking early on visibility or the episode's end — i.e.
   `SearchMDP._advance` + `SearchMDP.step` under `OrbitSearch.predict`, which
   is how the v0b game and the S24.1 gate's BASELINE row drive that case. The
   evader's decisions are per env step, so the hold cannot be a single
   `step()` call here; it is a counter instead, and the commands are the same.
   (Under a learned seat the branch is dead: A2.1's mask admits no tactic that
   is DONE on entry, so `TacticLaw.act` never returns None.)
4. **The evader decides BEFORE the env step** (A1.3) and its command reaches
   `SingleAgentWrapper.step` through a one-shot proxy. The inputs are
   identical to the in-step call's — the shield mutates its own state but no
   env state, and no env step intervenes — which is what CG-E1 proves.
5. **`hidden` is read from the PURSUER's observation** (`obs_p[6] <= 0.5`), not
   from the evader's: the question the evader's selector asks is whether it is
   unseen, and the evader's own `obs[6]` answers a different one (whether IT
   can see the pursuer). Everything else geometric is truth, because L1.5
   makes every scripted evader in this repo a full-state observer and
   `EvaderRulesV0` already reads `env` poses.
6. **The current-tactic block describes the span that is ENDING** and the
   last-3 block the three before it, as `SelectorMDP`'s reading 5.
"""

from __future__ import annotations

import pathlib
from collections import Counter

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from pe2.config import EnvConfig
from pe2.learned_select import LearnedPursuerSelect
from pe2.recogniser import EVADER_TACTICS, FEATURES, tactic_family
from pe2.scripted import HUG_THREAT_RANGE
from pe2.selector_mdp import (
    N_LAST_TACTICS,
    N_OBSTACLE_SLOTS,
    SelectorMDP,
    _block_slices,
    _game_module,
    _NeedDecision,
    count_switches,
    spans_with_length,
)
from pe2.selectors import DECISION_PERIOD, _surface_distances
from pe2.structured import OrbitSearch
from pe2.tactics import ARRIVAL_TOL, fixed

PREREG = "docs/pathway/tactics_prereg_s24_2_2026-10-17.md"

# Section 1's action space, imported and not restated: the recogniser's evader
# label space IS this list, exactly as `SelectorMDP.ACTIONS is PURSUER_TACTICS`.
EVADER_ACTIONS: tuple[str, ...] = EVADER_TACTICS
N_EV_ACTIONS = len(EVADER_ACTIONS)

# The mirror of `SelectorMDP.OBS_LAYOUT`, block by block: (name, width,
# in_memory_block). Three blocks of the pursuer's are absent and the prereg
# says why: no `dead_reckoning` (the evader carries no belief — it is never the
# occluded observer), and no `shield` / `escape` (the shield is on the pursuer;
# "the evader is unshielded").
EV_OBS_LAYOUT: tuple[tuple[str, int, bool], ...] = (
    ("hidden", 1, False),  # reading 5: the PURSUER's obs[6] <= 0.5
    ("pursuer_range_bearing", 2, False),  # body frame, range / half, bearing / pi
    ("own_pose", 4, False),  # x / half, y / half, cos theta, sin theta
    ("obstacles", 3 * N_OBSTACLE_SLOTS, False),  # surface dist / half, bearing, radius
    ("wall", 1, False),  # nearest-wall distance / half
    ("features", len(FEATURES) + 1, True),  # the PURSUER's motion features, then `valid`
    ("current_tactic", N_EV_ACTIONS + 1, True),  # one-hot, then time-in-tactic / T
    ("last_tactics", N_EV_ACTIONS * N_LAST_TACTICS, True),  # most recent first
    ("speed_ratio", 1, False),
)
EV_OBS_SLICES = _block_slices(EV_OBS_LAYOUT)
EV_MEMORY_SLICES = tuple(
    EV_OBS_SLICES[name] for name, _w, mem in EV_OBS_LAYOUT if mem
)
EV_OBS_LEN = sum(width for _name, width, _mem in EV_OBS_LAYOUT)

SEATS = ("RULES-V0", "LEAD-1.0")  # the two scripted seats of A1.2
DECISION_SOURCES = ("agent", "rules")


def done_on_entry(tactic, env) -> bool:
    """Would this evader tactic's `act` return None on its FIRST call?

    A2.1's invariant in the evader's seat: such a pick would leave a
    zero-length span and cost the evader a standing step (`TacticOpponent.act`
    returns (0, 0) when the tactic it just entered is DONE), so the mask
    refuses it. The question is answered from each tactic's OWN termination
    test — never a re-derivation of it:

    - `FLEE` is `GreedyEvader`, which always returns a command: never DONE.
    - `ORBIT(k)`'s `act` returns None when the pursuer is farther than
      `HUG_THREAT_RANGE`, or when the hug law has no direction about k (which
      is also its `applicable`).
    - `HOP(j)`'s `act` returns None when the pursuer cannot see the evader (the
      tactic's goal, already reached) or when the far-side point is within
      `ARRIVAL_TOL`.
    - `WALL-SLIDE`'s `act` returns None when the pursuer is farther than
      `HUG_THREAT_RANGE`, which is also its `applicable`.
    - `HOLD-HIDDEN`'s `act` returns None the step visibility returns, which is
      also the negation of its `applicable`.

    `_Capped`'s 40-step cap is NOT consulted: a span boundary `enter()`s the
    tactic, which zeroes that counter, so the cap can never make a tactic DONE
    on entry.
    """
    name = tactic_family(tactic.name)
    if name not in EVADER_ACTIONS:
        raise ValueError(f"{name} is not one of {EVADER_ACTIONS}")
    if name == "FLEE":
        return False
    threat = float(np.hypot(*(env.evader_pose[:2] - env.pursuer_pose[:2])))
    if name == "ORBIT":
        return threat > HUG_THREAT_RANGE or tactic._direction(env) is None
    if name == "WALL-SLIDE":
        return not tactic._threatened(env)
    if name == "HOLD-HIDDEN":
        return tactic._seen(env)
    if not tactic._seen(env):  # HOP: the pursuer cannot see the evader
        return True
    goal = tactic._goal_point(env)
    return float(np.hypot(*(goal - env.evader_pose[:2]))) <= ARRIVAL_TOL


class _OneShotEvader:
    """A1.3's hand-off: the evader command computed BEFORE the env step,
    returned once to `SingleAgentWrapper.step`.

    Installed as the wrapper's opponent factory, so it is the `_opponent` of
    every episode and `reset` / `uses_truth` are the real adapter's. It holds
    no policy and takes no decision; a second call inside one env step, or a
    call with nothing stashed, is a defect in the driving loop and raises.
    """

    def __init__(self, evader) -> None:
        self._evader = evader
        self._action: tuple[float, float] | None = None
        self.n_calls = 0

    @property
    def uses_truth(self) -> bool:
        return self._evader.uses_truth

    def reset(self, env=None) -> None:
        self._action = None
        self._evader.reset(env)

    def clear(self) -> None:
        self._action = None
        self.n_calls = 0

    def stash(self, action) -> None:
        if self._action is not None:
            raise AssertionError("an evader command is already pending")
        self._action = action

    def act(self, obs, env):
        if self._action is None:
            raise AssertionError(
                "the wrapper asked the evader for a command and none was "
                "stashed: the decision must be taken before the env step (A1.3)"
            )
        action, self._action = self._action, None
        self.n_calls += 1
        return action


class EvaderSelectorMDP(gym.Env):
    """One episode of the v0b tactic game with the EVADER's `select` replaced
    by the agent's masked discrete action.

    `env_steps` counts UNDERLYING env steps, so budgets and probes stay in env
    steps as in every prior round. `inapplicable_picks` is CG-T1's counter in
    this seat and must stay 0; an inapplicable action is also refused, because
    the mask makes it impossible by construction rather than caught afterwards.

    `pursuer` is one seat (A1.2) or a LIST of them, in which case one is drawn
    uniformly per episode from the episode RNG — seeded from the episode's own
    seed when `reset` is given one, so a seeded episode is reproducible by that
    seed alone, and from `seed0` otherwise.
    """

    metadata = {"render_modes": []}  # noqa: RUF012 — gym's own mutable field

    def __init__(
        self,
        speed_ratio: float = 1.2,
        *,
        pursuer="RULES-V0",
        memory: bool = True,
        seed0: int = 0,
        recorder: bool = False,
        label: str = "",
        cfg: EnvConfig | None = None,
        decision_source: str = "agent",
        device: str = "cpu",
    ) -> None:
        super().__init__()
        if decision_source not in DECISION_SOURCES:
            raise ValueError(
                f"decision_source must be one of {DECISION_SOURCES}, got "
                f"{decision_source!r}"
            )
        self.cfg = EnvConfig(evader_speed_ratio=float(speed_ratio)) if cfg is None else cfg
        self.memory = bool(memory)
        self.decision_source = str(decision_source)
        self.actions = EVADER_ACTIONS
        self.n_actions = N_EV_ACTIONS
        self.obs_layout = EV_OBS_LAYOUT
        self.obs_slices = EV_OBS_SLICES
        self.memory_slices = EV_MEMORY_SLICES
        self.obs_len = EV_OBS_LEN
        # The ONE assembly, `SelectorMDP`'s own module lookup.
        self.game, self.rules = _game_module().build_game(
            self.cfg, recorder=recorder, label=label
        )
        self.game.inner.seed0 = int(seed0)
        self.evader = self.rules["evader"]
        self.game.evader.select = self._select  # the ONE replaced field
        self._proxy = _OneShotEvader(self.game.evader)
        self.game.inner.opponent_factory = lambda: self._proxy  # A1.3, reading 4
        self._search = OrbitSearch(self.cfg)
        self.seats = self._build_seats(pursuer, device)
        # The episode RNG: seeded from `seed0` so an unseeded training stream is
        # deterministic, and reseeded from the episode seed by `gym.Env.reset`.
        self.np_random = np.random.default_rng(int(seed0))
        self.action_space = spaces.Discrete(self.n_actions)
        # A bound, not a normalisation claim — `SelectorMDP`'s own.
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(self.obs_len,), dtype=np.float32
        )
        self.inapplicable_picks = 0
        self.n_decisions = 0
        self.picks: Counter = Counter()
        self.pursuer_name = ""
        self._select_pursuer = None
        self._pending = None
        self._command = None  # (action, branch) for the next env step
        self._search_action = None
        self._search_left = 0
        self._span_steps = 0
        self._obs_p = np.zeros(self.game.inner.observation_space.shape, np.float32)

    @classmethod
    def over_game(
        cls, game, rules, *, cfg: EnvConfig, memory: bool = True
    ) -> EvaderSelectorMDP:
        """This class over a game that is ALREADY assembled — for a caller
        that owns the game and needs the EVADER's `_observe`/`action_masks`/
        `_candidates` code over it (E.3's `LearnedEvaderSelect`, the mirror of
        `LearnedPursuerSelect`'s own use of `SelectorMDP.over_game`), never a
        second game.

        UNLIKE `SelectorMDP.over_game` ("everything after `build_game` runs
        here, nothing else moves"), this is deliberately NOT the rest of
        `__init__`'s body. `SearchMDP` builds `self.inner` as
        `SingleAgentWrapper(cfg, "pursuer", opponent_factory)` in BOTH
        `SelectorMDP` and this class (`scripts/s24_tactic_game.py::build_game`
        is the one assembly), so `opponent_factory` governs the EVADER's
        identity either way. `SelectorMDP` never touches it, so its
        `over_game` really can run the whole constructor tail unchanged; this
        class's OWN `__init__` installs itself there on purpose — the
        one-shot proxy (A1.3) — because when `EvaderSelectorMDP` owns the
        game it also drives the evader's decisions as an external Gym agent.
        A caller using `over_game` drives nothing: it installs
        `game.evader.select` itself (never this object's own `_select`, which
        would raise `_NeedDecision`) and means to leave the game's opponent
        wiring exactly as it already found it. So only the mirror surface is
        built here — `_observe`, `action_masks`, `_candidates` and their
        per-decision bookkeeping (`picks`, `n_decisions`,
        `inapplicable_picks`) — and never `self._select`, `self._proxy`,
        `self.game.inner.opponent_factory` or the pursuer-seat machinery
        (`self.seats`, `self.np_random`), none of which `LearnedEvaderSelect`
        reads or writes.
        """
        self = cls.__new__(cls)
        self.cfg = cfg
        self.memory = bool(memory)
        self.decision_source = "agent"
        self.actions = EVADER_ACTIONS
        self.n_actions = N_EV_ACTIONS
        self.obs_layout = EV_OBS_LAYOUT
        self.obs_slices = EV_OBS_SLICES
        self.memory_slices = EV_MEMORY_SLICES
        self.obs_len = EV_OBS_LEN
        self.game, self.rules = game, rules
        self.evader = self.rules["evader"]
        self.inapplicable_picks = 0
        self.n_decisions = 0
        self.picks: Counter = Counter()
        self._pending = None
        return self

    # -- the pursuer's seat (A1.2) ----------------------------------------
    def _build_seats(self, pursuer, device: str) -> list[tuple[str, object]]:
        specs = list(pursuer) if isinstance(pursuer, list) else [pursuer]
        if not specs:
            raise ValueError("at least one pursuer seat is needed")
        return [self._build_seat(spec, device) for spec in specs]

    def _build_seat(self, spec, device: str) -> tuple[str, object]:
        if spec == "RULES-V0":  # the game's own default `TacticLaw` select
            return spec, self.rules["pursuer"]
        if spec == "LEAD-1.0":  # the S24.1 gate's BASELINE row, verbatim
            return spec, fixed(self.rules["pursuer"].lead)
        if not (isinstance(spec, tuple | list) and len(spec) == 3 and spec[0] == "ckpt"):
            raise ValueError(
                f"unknown pursuer seat {spec!r}; expected one of {SEATS} or "
                '("ckpt", path, library)'
            )
        from sb3_contrib import MaskablePPO

        _kind, path, library = spec
        path = pathlib.Path(path)
        if not path.exists():
            raise FileNotFoundError(f"no frozen pursuer checkpoint at {path}")
        sel = SelectorMDP.over_game(
            self.game, self.rules, cfg=self.cfg, library=str(library)
        )
        select = LearnedPursuerSelect(
            sel, MaskablePPO.load(path, device=device), name=path.stem
        )
        return path.stem, select

    # -- the injected selector --------------------------------------------
    def _select(self, obs, env, history):
        """The evader adapter's `select`. With `decision_source="rules"` this
        is `EvaderRulesV0` itself (CG-E1's replay); otherwise it hands back the
        agent's pending tactic and raises when there is none."""
        if self.decision_source == "rules":
            tactic = self.evader(obs, env, history)
            self.n_decisions += 1
            self.picks[tactic_family(tactic.name)] += 1
            return tactic
        if self._pending is None:
            raise _NeedDecision
        tactic, self._pending = self._pending, None
        return tactic

    # -- the action -> tactic map (section 1) -----------------------------
    @property
    def env_steps(self) -> int:
        return int(self.game.env_steps)

    @property
    def violations_total(self) -> int:
        return int(self.game.violations_total)

    def _obs_e(self) -> np.ndarray:
        return self.game.inner._last_joint_obs["evader"]

    def _candidates(self, obs=None) -> list:
        """The tactics this decision's actions resolve to, in `EVADER_ACTIONS`
        order; None where the family has no applicable instance.

        `EvaderRulesV0`'s own resolution, in its own words: ORBIT is the
        obstacle nearest the EVADER (rule 3's `argmin surfaces`), HOP is the
        first applicable of `_hops_from_farthest` (rule 2's ordering, farthest
        from the pursuer first).
        """
        rules, env = self.evader, self.game.inner.env
        obs = self._obs_e() if obs is None else obs
        surfaces = _surface_distances(env.evader_pose, env.obstacles)
        orbit = None
        if surfaces:
            k = int(np.argmin(surfaces))
            orbit = rules.orbits[k] if k < len(rules.orbits) else None
        hop = None
        for candidate in rules._hops_from_farthest(env):
            if candidate.applicable(obs, env):
                hop = candidate
                break
        return [rules.flee, orbit, hop, rules.wall_slide, rules.hold_hidden]

    def action_masks(self) -> np.ndarray:
        """`applicable()` of the tactic each action resolves to, at this
        decision, minus any tactic that would be DONE on entry (`done_on_entry`).

        Never empty: `FLEE` is `GreedyEvader`, always applicable and never DONE.
        """
        obs, env = self._obs_e(), self.game.inner.env
        mask = np.zeros(self.n_actions, dtype=bool)
        for i, tactic in enumerate(self._candidates(obs)):
            if tactic is None or done_on_entry(tactic, env):
                continue
            mask[i] = bool(tactic.applicable(obs, env))
        return mask

    # -- observation -------------------------------------------------------
    def _spans(self) -> list:
        """The current (ending) span and the three before it, most recent
        first, as `(family, length)` or None (reading 6)."""
        history = self.game.evader.history
        out = []
        for back in range(1 + N_LAST_TACTICS):
            if back >= len(history):
                out.append(None)
                continue
            name, start, end = history[-1 - back]
            stop = self.game.evader.step_index if end is None else end
            out.append((tactic_family(name), int(stop) - int(start)))
        return out

    def _observe(self, obs_p) -> np.ndarray:
        cfg, env = self.cfg, self.game.inner.env
        half = cfg.arena_size / 2.0
        own = np.asarray(env.evader_pose, dtype=float)
        c, s = np.cos(own[2]), np.sin(own[2])
        slices = self.obs_slices
        out = np.zeros(self.obs_len, dtype=np.float32)
        out[slices["hidden"]] = float(obs_p[6] <= 0.5)  # reading 5
        d = np.asarray(env.pursuer_pose, dtype=float)[:2] - own[:2]
        x_b, y_b = float(c * d[0] + s * d[1]), float(-s * d[0] + c * d[1])
        out[slices["pursuer_range_bearing"]] = (
            float(np.hypot(x_b, y_b)) / half,
            float(np.arctan2(y_b, x_b)) / np.pi,
        )
        out[slices["own_pose"]] = (own[0] / half, own[1] / half, float(c), float(s))
        start = slices["obstacles"].start
        order = np.argsort(_surface_distances(own, env.obstacles))
        for slot, j in enumerate(order[:N_OBSTACLE_SLOTS]):
            o = env.obstacles[int(j)]
            d = np.array([o["x"], o["y"]], dtype=float) - own[:2]
            out[start + 3 * slot : start + 3 * slot + 3] = (
                (float(np.hypot(*d)) - float(o["radius"])) / half,
                float(np.arctan2(-s * d[0] + c * d[1], c * d[0] + s * d[1])) / np.pi,
                float(o["radius"]) / cfg.obstacle_radius_max,
            )
        out[slices["wall"]] = (half - max(abs(own[0]), abs(own[1]))) / half
        f = self.evader.memory.features()  # the PURSUER's motion features (D147)
        out[slices["features"]] = [*(float(f[k]) for k in FEATURES), float(f["valid"])]
        spans = self._spans()
        if spans[0] is not None:
            family, length = spans[0]
            out[slices["current_tactic"].start + self.actions.index(family)] = 1.0
            out[slices["current_tactic"].stop - 1] = float(
                np.clip(length / DECISION_PERIOD, 0.0, 1.0)
            )
        start = slices["last_tactics"].start
        for back, span in enumerate(spans[1:]):
            if span is not None:
                out[start + self.n_actions * back + self.actions.index(span[0])] = 1.0
        out[slices["speed_ratio"]] = float(cfg.evader_speed_ratio)
        if not self.memory:
            for block in self.memory_slices:
                out[block] = 0.0
        return out

    # -- episode driving ---------------------------------------------------
    def _pursuer_command(self):
        """The pursuer's command for the NEXT env step, computed once and held
        across an evader decision point (reading 2), as `(action, branch)`."""
        if self._command is not None:
            return self._command
        if self._search_left > 0:  # reading 3: inside a held search span
            self._command = (self._search_action, "search")
            return self._command
        action = self.game.law.act(self._obs_p)
        if action is not None:
            self._command = (action, "law")
            return self._command
        # The law has nothing to act on: the search policy, computed at THIS
        # state and held for `hold` env steps (reading 3).
        self._search_action = self._search.predict(self._obs_p)[0]
        self._search_left = int(self.game.hold)
        self._command = (self._search_action, "search")
        return self._command

    def _advance(self):
        """Env steps from here until the evader's adapter needs a new decision
        or the episode ends. Returns (summed pursuer reward, terminated,
        truncated); A1.1's reward is the negative of the first."""
        game = self.game
        reward, term, trunc = 0.0, False, False
        while not (term or trunc):
            if (
                self.decision_source == "agent"
                and self._pending is None
                and game.evader.current is None
            ):
                break  # a decision point: the agent's turn
            action, branch = self._pursuer_command()  # A1.3 (1)
            try:
                ev_action = game.evader.act(self._obs_e(), game.inner.env)  # (2)
            except _NeedDecision:
                # The span just ended (`_close`d by the adapter) and no env
                # step has been taken from this state. The pursuer's command
                # stays held.
                game.evader.current = None
                break
            self._proxy.stash(ev_action)  # (3)
            self._obs_p, r, _o, term, trunc = game._step_inner(action, branch)  # (4)
            reward += r
            self._span_steps += 1
            self._command = None
            if branch == "search":
                self._search_left -= 1
                if term or trunc or game._visible:
                    self._search_left = 0
        return reward, term, trunc

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._pending = None
        self._command = None
        self._search_action = None
        self._search_left = 0
        self._span_steps = 0
        self.n_decisions = 0
        self.picks = Counter()
        index = int(self.np_random.integers(len(self.seats)))
        self.pursuer_name, self._select_pursuer = self.seats[index]
        self.game.law.select = self._select_pursuer
        if isinstance(self._select_pursuer, LearnedPursuerSelect):
            self._select_pursuer.reset()
        self._proxy.clear()
        obs = self.game._begin(seed)
        if self.game.inner._opponent is not self._proxy:  # A1.3's hand-off
            raise AssertionError("the one-shot proxy is not the wrapper's opponent")
        self._obs_p = obs
        return self._observe(obs), {
            "env_steps": self.env_steps,
            "pursuer": self.pursuer_name,
        }

    def step(self, action):
        if self.decision_source == "agent":
            index = int(action)
            mask = self.action_masks()
            if not mask[index]:
                self.inapplicable_picks += 1  # CG-T1: counted, then refused
                raise AssertionError(
                    f"action {index} ({self.actions[index]}) is not applicable at "
                    f"this decision (mask {mask.astype(int).tolist()}); CG-T1 "
                    "requires the mask to make an inapplicable pick impossible"
                )
            self._pending = self._candidates()[index]
            self.n_decisions += 1
            self.picks[self.actions[index]] += 1
        self._span_steps = 0
        # A2.4's span bookkeeping, as `SelectorMDP.step` does it: this env
        # drives `_step_inner` itself, so without it every step record reads
        # `in_span: False` and `span_intervention_total` is identically 0.
        self.game._in_span = True
        self.game.span_intervention = 0.0
        self.game.span_stalled = 0
        reward, term, trunc = self._advance()
        self.game._span_total += float(self.game.span_intervention)
        info = dict(
            self.game._info,
            env_steps=self.env_steps,
            span_steps=self._span_steps,
            inapplicable_picks=self.inapplicable_picks,
            violations=int(self.game.episode_violations),
            pursuer=self.pursuer_name,
        )
        return self._observe(self._obs_p), -float(reward), term, trunc, info

    # -- evaluation --------------------------------------------------------
    def spans(self) -> list[dict]:
        """The EVADER's span log with each span's length in env steps and the
        `zero_length` flag, the open span closed at `step_index`."""
        return spans_with_length(
            self.game.evader.history, self.game.evader.step_index
        )

    def n_switches(self) -> int:
        return count_switches(self.spans())

    def play_episode(self, predict, seed: int) -> dict:
        """One seeded episode in `SelectorMDP.play_episode`'s schema, plus
        `evader_spans` and `pursuer_name`. `predict` is unused (and may be
        None) when `decision_source="rules"`."""
        game = self.game
        obs, _info = self.reset(seed=seed)
        term = trunc = False
        while not (term or trunc):
            action = None
            if self.decision_source == "agent":
                action, _ = predict(
                    obs, deterministic=True, action_masks=self.action_masks()
                )
            obs, _r, term, trunc, _i = self.step(action)
        spans = self.spans()
        return {
            "seed": int(seed),
            "role": "evader",
            "pursuer_name": self.pursuer_name,
            "outcome": game._info["outcome"],
            "steps": int(game.inner.env.current_step),
            "final_distance": float(game._info["distance"]),
            "los_breaks": int(game._los_breaks),
            "shielded_role": "pursuer",
            "violations": int(game.episode_violations),
            "shield": game.shield_metrics.summary(),
            "branch": game.branch_summary(),
            "tactics": {
                "evader": _history(game.evader),
                "pursuer": _history(game.law),
            },
            "usage": {role: dict(counts) for role, counts in game.usage.items()},
            "n_decisions": int(self.n_decisions),
            "picks": dict(self.picks),
            "evader_spans": spans,
            "pursuer_spans": spans_with_length(game.law.history, game.law.step_index),
            "n_zero_length_spans": sum(s["zero_length"] for s in spans),
            "n_switches": int(self.n_switches()),
            "inapplicable_picks": int(self.inapplicable_picks),
            "recovery": {
                "n_recoveries": int(game.shield.n_recoveries),
                "n_forward": int(game.shield.n_recoveries_forward),
                "n_rotation": int(game.shield.n_recoveries_rotation),
            },
        }


def _history(adapter) -> list:
    """`scripts/s24_tactic_game.py::_history`, the record's own convention: the
    span log with the OPEN span closed at the adapter's step count."""
    return [
        [name, int(start), int(adapter.step_index if end is None else end)]
        for name, start, end in adapter.history
    ]
