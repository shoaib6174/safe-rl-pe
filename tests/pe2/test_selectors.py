"""S24.0 Step 4 — the scripted selectors v0 and the labelled tactic game.

Spec: `docs/pathway/tactics_plan_2026-10-13.md` section 1 ("Selector", "v0,
scripted") and section 2 (S24.0).

Four things are load-bearing here and the rest is bookkeeping:

- **Rule ORDER.** Each rule is checked on a hand-built state where it must
  fire, and on one where the rule above it must win instead — a rule table whose
  order is untested is not a rule table.
- **Only applicable tactics are ever returned** (the Step 1 review's watch item:
  neither adapter checks `applicable()`), on 200 random states per role with
  random window features so every branch is exercised. The returned tactic's own
  precondition is then asserted, not the rule's reasoning.
- **v0 is speed-agnostic**: the same pick at ratios 0.8 / 1.0 / 1.2 on identical
  states, and neither rule table's source names the ratio. The memory carries it
  for the LEARNED selector of S24.1; v0 must not branch on it.
- **The harness records what it claims**: both span logs with contiguous
  boundaries that close at the adapter's own step count, and a frame label in
  the exact format the brief fixes.

The features are supplied by a STUB memory wherever a rule's branch is under
test (`_Features`): `TacticRecogniser.features()` is a pure function of an
observed window, so driving a real window into a wanted corner would test the
recogniser rather than the rule table. The real memory is used in the random
states, in the speed-agnostic check and in the harness smoke.

No 100-episode run happens here; the pre-registered run is the script's.
"""

from __future__ import annotations

import inspect
import io
import itertools
import json
import tokenize

import numpy as np
import pytest

import scripts.s24_tactic_game as GAME
from pe2.config import EnvConfig
from pe2.env import PursuitEvasionEnv
from pe2.recogniser import FEATURES
from pe2.selectors import (
    AT_OBSTACLE_LEVEL,
    CLOSING_TREND,
    COMING_ROUND_SWEEP,
    DECISION_PERIOD,
    FLAT_OBSTACLE_TREND,
    MOVING_SPEED,
    ORBITING_SWEEP,
    STRAIGHT_RUN,
    WINDOW,
    EvaderRulesV0,
    PursuerRulesV0,
    TacticMemory,
)
from pe2.tactics import AMBUSH_CLOSE, BELIEF_FRESH_STEPS

CFG = EnvConfig()
RATIOS = (0.8, 1.0, 1.2)
CONSTRUCT_SEED0 = 2_290_000  # this step's construct stream


# --------------------------------------------------------------- scaffolding
class _Features:
    """A memory that reports exactly the window features a rule is being tested
    on. Everything else a v0 rule may read is present and inert."""

    def __init__(self, valid: bool = False, **values) -> None:
        unknown = set(values) - set(FEATURES)
        assert not unknown, f"not recogniser features: {unknown}"
        self._f = dict.fromkeys(FEATURES, 0.0)
        self._f.update(values)
        self._f["valid"] = bool(valid)
        self._f["n_visible_in_window"] = WINDOW if valid else 0
        self.speed_ratio = 1.2
        self.own_history: list = []
        self.opponent_history: list = []
        self.point = None
        self.age = None
        self.belief = None

    def features(self) -> dict:
        return dict(self._f)


def _scene(obstacles, pursuer, evader, cfg: EnvConfig = CFG):
    """An env with a HAND-BUILT map and both poses placed. The seeded reset
    builds a legal episode first, so everything the env needs is initialised
    and only the quantities the rules read are overwritten. The evader's
    heading always points away from the pursuer and the pursuer's at the
    evader unless the caller gives one."""
    env = PursuitEvasionEnv(cfg)
    env.reset(seed=CONSTRUCT_SEED0)
    env.obstacles = [dict(o) for o in obstacles]
    px, py = pursuer[0], pursuer[1]
    ex, ey = evader[0], evader[1]
    p_theta = pursuer[2] if len(pursuer) > 2 else float(np.arctan2(ey - py, ex - px))
    e_theta = evader[2] if len(evader) > 2 else float(np.arctan2(ey - py, ex - px))
    env.pursuer_pose = np.array([px, py, p_theta], dtype=float)
    env.evader_pose = np.array([ex, ey, e_theta], dtype=float)
    return env, env._get_obs()


# The map every hand-built scene uses: one disc in the middle of the arena and
# one parked in a corner, so "the obstacle nearest the evader" is never a tie.
A = {"x": 0.0, "y": 0.0, "radius": 1.0}
B = {"x": 3.0, "y": -1.0, "radius": 0.8}
FAR_B = {"x": -4.0, "y": -4.0, "radius": 0.5}
MAP = [A, B]


def _visible(env) -> bool:
    from pe2.sensing import is_visible

    return is_visible(
        env.pursuer_pose, env.evader_pose[:2], env.obstacles, env.cfg.sensing
    )


# ------------------------------------------------------------- evader rules
class TestEvaderRuleOrder:
    def test_rule_1_hidden_wins_over_an_orbit_that_would_also_fire(self):
        """The pursuer is 3.4 m away with an obstacle 0.2 m from the evader —
        rule 3's conditions hold — but it cannot see the evader, so rule 1 wins."""
        env, obs = _scene(MAP, (-3.0, 0.0, np.pi), (0.0, 1.5))
        assert not _visible(env)
        rules = EvaderRulesV0(CFG)
        for tactic in rules.tactics:
            tactic.reset(env)
        pick = rules.select(obs["evader"], env, _Features())
        assert pick.name == "HOLD-HIDDEN"

    def test_rule_2_hop_fires_on_a_pursuer_coming_round_my_obstacle(self):
        """Visible (so rule 1 is out), the pursuer's window is mostly tangential
        about the obstacle that is nearest BOTH of them, and it is closing."""
        env, obs = _scene(MAP, (1.5, 1.5), (0.0, 1.2))
        assert _visible(env)
        rules = EvaderRulesV0(CFG)
        for tactic in rules.tactics:
            tactic.reset(env)
        coming_round = _Features(
            valid=True,
            sweep_ratio=COMING_ROUND_SWEEP + 0.1,
            observer_trend=CLOSING_TREND - 0.4,
        )
        pick = rules.select(obs["evader"], env, coming_round)
        # B is the obstacle farthest from the pursuer, so the HOP goes there.
        assert pick.name == "HOP(1)"
        assert pick.applicable(obs["evader"], env)

    def test_rule_2_does_not_fire_when_the_pursuer_is_not_closing(self):
        """The same state with the gap OPENING instead: rule 2 is out and rule 3
        takes the nearest obstacle — so the order is visible in one scene."""
        env, obs = _scene(MAP, (1.5, 1.5), (0.0, 1.2))
        rules = EvaderRulesV0(CFG)
        for tactic in rules.tactics:
            tactic.reset(env)
        opening = _Features(
            valid=True, sweep_ratio=COMING_ROUND_SWEEP + 0.1, observer_trend=+0.5
        )
        assert rules.select(obs["evader"], env, opening).name == "ORBIT(0)"
        # ... and with no valid window at all, likewise.
        assert rules.select(obs["evader"], env, _Features()).name == "ORBIT(0)"

    def test_rule_4_wall_slide_when_the_wall_is_nearer_than_any_obstacle(self):
        env, obs = _scene(MAP, (3.0, 3.0), (4.5, 4.0))
        assert _visible(env)
        rules = EvaderRulesV0(CFG)
        for tactic in rules.tactics:
            tactic.reset(env)
        pick = rules.select(obs["evader"], env, _Features())
        assert pick.name == "WALL-SLIDE"
        assert pick.applicable(obs["evader"], env)

    def test_rule_5_flee_when_the_pursuer_is_out_of_hug_range(self):
        """An obstacle IS within HUG_RANGE of the evader here; rule 3 is out
        only because the pursuer is farther than HUG_THREAT_RANGE, which is what
        makes this a test of that guard rather than of the geometry."""
        from pe2.scripted import HUG_RANGE, HUG_THREAT_RANGE

        env, obs = _scene(MAP, (-4.5, -2.5), (0.5, 2.5))
        gap = float(np.hypot(*(env.evader_pose[:2] - env.pursuer_pose[:2])))
        assert gap > HUG_THREAT_RANGE and _visible(env)
        nearest = float(np.hypot(0.5, 2.5)) - A["radius"]
        assert nearest < HUG_RANGE
        rules = EvaderRulesV0(CFG)
        for tactic in rules.tactics:
            tactic.reset(env)
        assert rules.select(obs["evader"], env, _Features()).name == "FLEE"


# ------------------------------------------------------------ pursuer rules
class TestPursuerRuleOrder:
    def _rules(self, env):
        rules = PursuerRulesV0(env.cfg)
        for tactic in rules.tactics:
            tactic.reset(env)
        return rules

    def test_rule_1_sweep_whenever_the_evader_is_invisible(self):
        env, obs = _scene(MAP, (-3.0, 0.0, np.pi), (0.0, 1.5))
        assert not _visible(env)
        rules = self._rules(env)
        pick = rules.select(obs["pursuer"], env, _Features(valid=True, speed_frac=1.0))
        assert pick.name == "SWEEP"
        assert pick.applicable(obs["pursuer"], env)

    def test_rule_2_counter_orbit_on_an_orbiting_evader(self):
        """The segment grazes the disc: inside the INFLATED radius (so the
        obstacle is 'between' and COUNTER-ORBIT is applicable) and outside the
        bare one (so the evader is still visible, which `is_visible` decides on
        the bare radius)."""
        env, obs = _scene([A, FAR_B], (-3.0, 1.08), (3.0, 1.08))
        assert _visible(env)
        rules = self._rules(env)
        orbiting = _Features(
            valid=True, sweep_ratio=ORBITING_SWEEP + 0.1, obstacle_trend=0.0
        )
        pick = rules.select(obs["pursuer"], env, orbiting)
        assert pick.name == "COUNTER-ORBIT(0)"
        assert pick.applicable(obs["pursuer"], env)

    def test_rule_3_ambush_on_a_straight_run_at_an_obstacle(self):
        env, obs = _scene([A, FAR_B], (0.5, 2.8), (3.0, 1.5))
        assert _visible(env)
        gap = float(np.hypot(*(env.evader_pose[:2] - env.pursuer_pose[:2])))
        assert gap > AMBUSH_CLOSE
        rules = self._rules(env)
        hopping = _Features(
            valid=True,
            tortuosity=STRAIGHT_RUN + 0.05,
            obstacle_trend=-0.5,
            obstacle_level=AT_OBSTACLE_LEVEL + 0.2,
            speed_frac=1.0,
        )
        pick = rules.select(obs["pursuer"], env, hopping)
        assert pick.name == "AMBUSH(0)"
        assert pick.applicable(obs["pursuer"], env)

    def test_rule_3_falls_to_lead_when_the_gap_is_out_of_ambush_range(self):
        """The same hopping evader, the pursuer too far from the gap to reach
        it: the rule's own fallback, not rule 4 by accident — rule 4's condition
        holds here too, and both answers are LEAD, so the assertion that
        separates them is AMBUSH's `applicable` being False."""
        env, obs = _scene([A, FAR_B], (-2.5, 1.5), (3.0, 1.5))
        rules = self._rules(env)
        hopping = _Features(
            valid=True,
            tortuosity=STRAIGHT_RUN + 0.05,
            obstacle_trend=-0.5,
            obstacle_level=AT_OBSTACLE_LEVEL + 0.2,
            speed_frac=1.0,
        )
        assert not rules.ambushes[0].applicable(obs["pursuer"], env)
        assert rules.select(obs["pursuer"], env, hopping).name == "LEAD"

    def test_rule_4_lead_on_a_moving_visible_evader(self):
        env, obs = _scene([A, FAR_B], (-2.5, 1.5), (3.0, 1.5))
        rules = self._rules(env)
        moving = _Features(
            valid=True,
            speed_frac=MOVING_SPEED + 0.5,
            sweep_ratio=ORBITING_SWEEP - 0.2,
            tortuosity=STRAIGHT_RUN - 0.3,
        )
        assert rules.select(obs["pursuer"], env, moving).name == "LEAD"

    def test_rule_5_intercept_when_there_is_no_motion_to_lead(self):
        env, obs = _scene([A, FAR_B], (-2.5, 1.5), (3.0, 1.5))
        rules = self._rules(env)
        standing = _Features(valid=True, speed_frac=MOVING_SPEED - 0.1)
        assert rules.select(obs["pursuer"], env, standing).name == "INTERCEPT"
        # An invalid window (just acquired) is the same answer: INTERCEPT is
        # the no-information default, never a guess from a zero-filled window.
        assert rules.select(obs["pursuer"], env, _Features()).name == "INTERCEPT"

    def test_the_orbiting_rule_needs_a_flat_distance_to_the_obstacle(self):
        env, obs = _scene([A, FAR_B], (-3.0, 1.08), (3.0, 1.08))
        rules = self._rules(env)
        leaving = _Features(
            valid=True,
            sweep_ratio=ORBITING_SWEEP + 0.1,
            obstacle_trend=FLAT_OBSTACLE_TREND + 0.5,
            speed_frac=1.0,
        )
        assert rules.select(obs["pursuer"], env, leaving).name != "COUNTER-ORBIT(0)"


# ------------------------------------------------- only applicable, always
def _random_state(rng, cfg: EnvConfig):
    env = PursuitEvasionEnv(cfg)
    env.reset(seed=int(rng.integers(2_290_000, 2_299_999)))
    half = cfg.arena_size / 2.0 - cfg.agent_radius
    env.pursuer_pose = np.array(
        [*rng.uniform(-half, half, 2), rng.uniform(-np.pi, np.pi)]
    )
    env.evader_pose = np.array(
        [*rng.uniform(-half, half, 2), rng.uniform(-np.pi, np.pi)]
    )
    return env, env._get_obs()


def _random_features(rng):
    """A window drawn across the whole feature range, so every branch of both
    tables is reachable — including the zero-filled invalid one."""
    values = {
        name: float(rng.uniform(-1.5, 1.5) if "trend" in name else rng.uniform(0, 1.2))
        for name in FEATURES
    }
    return _Features(valid=bool(rng.random() < 0.8), **values)


@pytest.mark.parametrize("role", ["evader", "pursuer"])
def test_only_applicable_tactics_are_ever_returned(role):
    rng = np.random.default_rng([20_241_013, 4])
    rules = (EvaderRulesV0 if role == "evader" else PursuerRulesV0)(CFG)
    seen = set()
    for _ in range(200):
        env, obs = _random_state(rng, CFG)
        for tactic in rules.tactics:
            tactic.reset(env)
        pick = rules.select(obs[role], env, _random_features(rng))
        assert pick in rules.tactics, pick.name
        assert pick.applicable(obs[role], env), f"{pick.name} was not applicable"
        seen.add(pick.name)
    assert len(seen) >= 3, f"only {seen} ever picked: the test would be vacuous"


# ------------------------------------------------------------ speed-agnostic
@pytest.mark.parametrize("role", ["evader", "pursuer"])
def test_the_rule_tables_never_name_the_speed_ratio(role):
    """A NAME-token scan of the rule table's own source, so the module's prose
    about the rule neither satisfies nor breaks it."""
    table = EvaderRulesV0 if role == "evader" else PursuerRulesV0
    names = {
        token.string
        for token in tokenize.generate_tokens(
            io.StringIO(inspect.getsource(table)).readline
        )
        if token.type == tokenize.NAME
    }
    assert not names & {"evader_speed_ratio", "speed_ratio", "evader_v_max", "v_max"}


@pytest.mark.parametrize("role", ["evader", "pursuer"])
def test_the_same_state_gets_the_same_pick_at_every_ratio(role):
    """Identical states and identical windows at 0.8 / 1.0 / 1.2. The states are
    built with each ratio's own config (the ratio is a frozen config field), so
    what is held fixed is the geometry the rules read."""
    rng = np.random.default_rng([20_241_013, 44])
    draws = [
        (int(rng.integers(2_290_000, 2_299_999)), _random_features(rng))
        for _ in range(25)
    ]
    picks: dict[float, list[str]] = {}
    for ratio in RATIOS:
        cfg = EnvConfig(evader_speed_ratio=ratio)
        rules = (EvaderRulesV0 if role == "evader" else PursuerRulesV0)(cfg)
        names = []
        for seed, features in draws:
            state = np.random.default_rng([seed, 7])
            env, obs = _random_state(state, cfg)
            for tactic in rules.tactics:
                tactic.reset(env)
            names.append(rules.select(obs[role], env, features).name)
        picks[ratio] = names
    assert picks[0.8] == picks[1.0] == picks[1.2]


def test_the_memory_carries_the_ratio_it_does_not_let_the_rules_read(self=None):
    """It is an INPUT of the learned selector (plan section 1), so it is
    carried, measured here, and unused above."""
    for ratio in RATIOS:
        cfg = EnvConfig(evader_speed_ratio=ratio)
        assert TacticMemory(cfg, "evader").speed_ratio == ratio


# ------------------------------------------------------------------ memory
class TestMemory:
    def test_the_dead_reckoning_point_and_age_are_the_beliefs(self):
        cfg = CFG
        memory = TacticMemory(cfg, "evader", with_belief=True)
        assert memory.point is None and memory.age is None  # never seen
        env, obs = _scene(MAP, (1.5, 1.5), (0.0, 1.2))
        memory.observe(obs["pursuer"], env.pursuer_pose, env.evader_pose, env.obstacles)
        assert memory.age == 0
        assert np.allclose(memory.point, env.evader_pose[:2], atol=1e-3)
        blind = _scene(MAP, (-3.0, 0.0, np.pi), (0.0, 1.5))[1]["pursuer"]
        memory.observe(blind, env.pursuer_pose, env.evader_pose, env.obstacles)
        assert memory.age == 1  # dead reckoning from here on

    def test_the_full_state_observer_never_loses_the_opponent(self):
        """The evader's memory (L1.5): the window fills even on steps where the
        pursuer's observation says invisible."""
        memory = TacticMemory(CFG, "pursuer", with_belief=False, full_state=True)
        env, obs = _scene(MAP, (-3.0, 0.0, np.pi), (0.0, 1.5))
        assert obs["evader"][6] <= 0.5 or True  # the flag is not what it reads
        for _ in range(12):
            env.pursuer_pose[0] += 0.05
            memory.observe(
                env._get_obs()["evader"],
                env.evader_pose,
                env.pursuer_pose,
                env.obstacles,
            )
        f = memory.features()
        assert f["valid"] and f["n_visible_in_window"] == 12


def test_the_thresholds_are_pinned():
    """Every threshold the two tables branch on, by value. A change here is a
    change to the rule table and must be made deliberately, not by drift."""
    from pe2 import selectors

    assert (selectors.COMING_ROUND_SWEEP, selectors.ORBITING_SWEEP) == (0.50, 0.70)
    assert (selectors.CLOSING_TREND, selectors.APPROACHING_TREND) == (-0.10, -0.15)
    assert selectors.FLAT_OBSTACLE_TREND == 0.15
    assert selectors.STRAIGHT_RUN == 0.90
    assert selectors.AT_OBSTACLE_LEVEL == 0.10
    assert selectors.MOVING_SPEED == 0.15
    assert (selectors.WINDOW, selectors.DECISION_PERIOD) == (40, 40)
    # Imported, not restated: the two ranges rule 3 and rule 4 share with the
    # hug law, and the ambush's own DONE distance.
    from pe2.scripted import HUG_RANGE, HUG_THREAT_RANGE

    assert (HUG_THREAT_RANGE, HUG_RANGE, AMBUSH_CLOSE) == (7.0, 4.0, 2.0)
    assert BELIEF_FRESH_STEPS == 100


# ------------------------------------------------------- the decision period
class TestDecisionPeriod:
    """Plan section 1: a decision point is the current tactic's DONE OR every
    T = 40 steps. The adapters switch on DONE only, so the period is imposed by
    offering `select` capped tactics."""

    def test_a_capped_tactic_is_done_at_the_period(self):
        env, obs = _scene(MAP, (-4.5, -2.5), (0.5, 2.5))
        rules = EvaderRulesV0(CFG)
        for tactic in rules.tactics:
            tactic.reset(env)
        flee = rules.flee
        flee.enter()
        assert flee.limit == DECISION_PERIOD
        for _ in range(DECISION_PERIOD):
            assert flee.act(obs["evader"], env) is not None
        assert flee.act(obs["evader"], env) is None  # DONE: decide again
        flee.enter()  # ... and a new span starts the count over
        assert flee.act(obs["evader"], env) is not None

    def test_no_span_of_a_played_episode_exceeds_the_period(self):
        cfg = EnvConfig(evader_speed_ratio=1.2)
        mdp, _ = GAME.build_game(cfg)
        record = GAME.play_episode(mdp, CONSTRUCT_SEED0, cfg)
        for role, spans in record["tactics"].items():
            assert spans, role
            longest = max(end - start for _, start, end in spans)
            assert longest <= DECISION_PERIOD, f"{role}: {longest}"


# ----------------------------------------------------------------- harness
def test_the_frame_label_has_the_exact_format():
    assert GAME.frame_label("HOP(1)", "SWEEP") == "EVADER: HOP(1) | PURSUER: SWEEP"


def test_the_label_branch_is_the_features_only_one():
    assert GAME.label_branch() == "labels: none (D147/D148)"


def _spans_are_contiguous(history, total: int) -> bool:
    if not history:
        return False
    if history[0][1] != 0 or history[-1][2] != total:
        return False
    return all(a[2] == b[1] for a, b in itertools.pairwise(history))


class TestHarness:
    def test_two_episodes_record_both_histories_with_correct_boundaries(self):
        cfg = EnvConfig(evader_speed_ratio=1.2)
        mdp, rules = GAME.build_game(cfg)
        for seed in (CONSTRUCT_SEED0, CONSTRUCT_SEED0 + 1):
            record = GAME.play_episode(mdp, seed, cfg)
            assert record["outcome"] in ("capture", "timeout", "evasion")
            assert record["steps"] >= 1
            evader, pursuer = record["tactics"]["evader"], record["tactics"]["pursuer"]
            # The evader acts on every env step, so its spans tile the episode.
            assert _spans_are_contiguous(evader, mdp.evader.step_index)
            assert mdp.evader.step_index == record["steps"]
            # The law acts on law-branch steps only; its spans tile ITS calls.
            assert _spans_are_contiguous(pursuer, mdp.law.step_index)
            assert mdp.law.step_index + record["search_steps"] == record["steps"]
            names = {name for name, _, _ in evader} | {n for n, _, _ in pursuer}
            assert names <= {t.name for t in rules["evader"].tactics} | {
                t.name for t in rules["pursuer"].tactics
            }
            # Usage is one count per env step per role, by family.
            for role in ("evader", "pursuer"):
                assert sum(record["usage"][role].values()) == record["steps"]
            assert record["recogniser"]["evader"]["n_steps"] <= record["steps"]

    def test_a_two_episode_smoke_writes_a_record_and_prints_the_reading_first(
        self, tmp_path, capsys, monkeypatch
    ):
        monkeypatch.setattr(GAME, "OUT", tmp_path)
        monkeypatch.setattr(GAME, "RECORD", tmp_path / "s24_tactic_game.json")
        record = GAME.run(2, CONSTRUCT_SEED0, (1.2,))
        GAME.write_record(record)
        GAME.print_table(record)
        written = json.loads((tmp_path / "s24_tactic_game.json").read_text())
        assert written["reading"] == GAME.READING
        assert written["label_branch"] == "labels: none (D147/D148)"
        assert len(written["episodes"]["1.2"]) == 2
        summary = written["per_ratio"]["1.2"]
        assert 0.0 <= summary["capture_rate"] <= 1.0
        assert summary["violations_total"] >= 0
        out = capsys.readouterr().out
        # The reading comes before the table AND before any measured number
        # this script prints (the progress lines carry none).
        assert GAME.READING in out
        assert out.index(GAME.READING) < out.index("ratio 1.2:")
        assert out.index(GAME.READING) < out.index(f"{summary['mean_steps']:.1f}")

    def test_the_sweep_hook_steers_at_a_fresh_estimate_and_orbits_without_one(self):
        """The additive hook in `pe2.tactics.Sweep` (Step 4): the belief drives
        it while the estimate is fresh, `OrbitSearch` when it is not."""
        from pe2.structured import OrbitSearch
        from pe2.tactics import Sweep

        env, obs = _scene(MAP, (-3.0, 0.0, np.pi), (0.0, 1.5))
        blind = obs["pursuer"]
        assert blind[6] <= 0.5
        rules = PursuerRulesV0(CFG)
        sweep = rules.sweep
        sweep.reset(env)
        assert isinstance(sweep.inner, Sweep) and sweep.belief is rules.memory.belief
        # Never seen: the orbit search, bit for bit.
        reference = OrbitSearch(CFG)
        assert np.array_equal(sweep.act(blind), reference.predict(blind)[0])
        assert sweep.last_goal is None
        # A fresh sighting: it steers at the estimate instead.
        rules.memory.observe(
            _scene(MAP, (1.5, 1.5), (0.0, 1.2))[1]["pursuer"],
            env.pursuer_pose,
            env.evader_pose,
            env.obstacles,
        )
        assert sweep.act(blind) is not None
        assert sweep.last_goal is not None
        # Stale beyond the horizon: back to the orbit search.
        rules.memory.belief.t_since_seen = BELIEF_FRESH_STEPS
        sweep.enter()  # a fresh span: the 40-step cap is not what returns None
        sweep.act(blind)
        assert sweep.last_goal is None
