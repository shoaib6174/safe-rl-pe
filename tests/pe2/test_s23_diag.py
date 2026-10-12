"""`scripts/s23_diag_isolated.py` — the S23 step 2 isolated-obstacle diagnostic.

Plan: `docs/pathway/pursuit_strategy_plan_2026-10-12.md` section 3.

What is checked, and why each one is load-bearing:

- the single obstacle really is installed on EVERY episode, and the law read
  exactly it at `reset` (its inflated map carries one disc and nothing else) --
  without this the diagnostic would silently measure the default two-obstacle
  arena, which is the arena the full round measures;
- the three rows run the SAME seeds, so "LEAD captures more than PLAN" is a
  paired statement;
- `min_distance` is a real minimum: at most the capture distance on a captured
  episode, more than it on one that timed out (the env's own predicate);
- the Wilson interval is the S19 round's `_ci`, the SAME object, not a second
  interval written here;
- the whole thing runs end to end into a tmp dir on 2 episodes per law.

The real diagnostic (100 episodes x 3 laws) is NOT run here.
"""

import importlib.util
import json
import pathlib
import re
import sys

import numpy as np
import pytest

from pe2.eval import wilson_ci
from pe2.planner import PlannerInterceptLaw

_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _load(name: str):
    path = _ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _load("s23_diag_isolated")
# The runner imports the S19 round's helpers as `scripts.derisk_s19_planner`,
# so the test must import the SAME module object (a second load by file path
# would be a look-alike, and the identity check below would be vacuous).
import scripts.derisk_s19_planner as S19

SMOKE_N = 2


@pytest.fixture(scope="module")
def smoke(tmp_path_factory) -> dict:
    """The whole runner, 2 episodes per law, into a tmp dir."""
    out = tmp_path_factory.mktemp("s23diag") / "s23_diag_isolated.json"
    record = M.run(n=SMOKE_N, out=out)
    assert pathlib.Path(out).exists()
    record["_out"] = out
    return record


def _norm(text: str) -> str:
    """Whitespace- and markdown-insensitive, so "verbatim" can be checked
    against a reflowed, bolded source line."""
    return re.sub(r"\s+", " ", text.replace("*", "").replace("`", "")).strip()


# ----------------------------------------------------------------------------
# the arena
# ----------------------------------------------------------------------------
class TestTheIsolatedObstacle:
    def test_one_disc_at_the_centre_on_every_episode_and_the_law_saw_it(self):
        cfg = M.diag_config()
        mdp = M.build_env("LEAD-1.0", cfg)
        seen: list[tuple[list, list]] = []
        begin = mdp._begin

        def spy(seed):
            obs = begin(seed)
            seen.append(
                (
                    [dict(o) for o in mdp.inner.env.obstacles],
                    [dict(o) for o in mdp.law.obstacles],
                )
            )
            return obs

        mdp._begin = spy
        for seed in M.seeds(SMOKE_N):
            mdp.shield = S19.build_shield(cfg)
            mdp.play_episode(M.OrbitSearch(cfg).predict, seed)

        assert len(seen) == SMOKE_N
        for env_obs, law_obs in seen:
            assert len(env_obs) == 1
            assert env_obs[0]["x"] == 0.0
            assert env_obs[0]["y"] == 0.0
            assert env_obs[0]["radius"] == 0.8
            assert law_obs == env_obs  # the law read exactly the installed map

        # The planner's inflated map carries exactly one disc, and its blocked
        # mask is the mask of that disc alone.
        assert mdp.law._oc.shape == (1, 2)
        assert mdp.law._or == pytest.approx([0.8 + cfg.agent_radius])
        reference = PlannerInterceptLaw(cfg)
        reference.set_map([dict(M.OBSTACLE)])
        assert np.array_equal(mdp.law._blocked, reference._blocked)
        assert reference._blocked.any()

    def test_the_radius_is_inside_the_configured_obstacle_range(self):
        cfg = M.diag_config()
        assert (
            cfg.obstacle_radius_min <= M.OBSTACLE["radius"] <= cfg.obstacle_radius_max
        )

    def test_the_observation_layout_is_the_default_two_obstacle_one(self):
        """`n_physical_obstacles` (discs placed) and `n_obstacles` (the
        observation layout) do NOT have to agree; the diagnostic changes only
        the first, so the laws are the objects S19 measured."""
        cfg = M.diag_config()
        assert cfg.n_physical_obstacles == 1
        assert cfg.n_obstacles == 2

    def test_a_missing_installation_is_refused_not_ignored(self):
        """`_begin`'s assertion is what makes the installation a fact of the
        record rather than a hope: break it and the episode must not run."""
        cfg = M.diag_config()
        mdp = M.build_env("PLAN", cfg)
        mdp.obstacle = {"x": 1.0, "y": 0.0, "radius": 0.3}  # not what _begin wants
        mdp.inner.env.reset = mdp.inner.env.__class__.reset.__get__(mdp.inner.env)
        with pytest.raises(RuntimeError, match="isolated obstacle is not installed"):
            mdp.play_episode(M.OrbitSearch(cfg).predict, M.DIAG_SEED0)


# ----------------------------------------------------------------------------
# the pairing
# ----------------------------------------------------------------------------
class TestPairing:
    def test_the_three_laws_run_the_same_seeds(self, smoke):
        assert [row["law"] for row in smoke["rows"]] == list(M.LAWS)
        want = M.seeds(SMOKE_N)
        assert want == [M.DIAG_SEED0 + i for i in range(SMOKE_N)]
        for row in smoke["rows"]:
            assert row["seeds"] == want
            assert [e["seed"] for e in row["episodes"]] == want

    def test_only_the_law_differs_between_rows(self, smoke):
        keys = (
            "hold",
            "shield",
            "shield_obs",
            "escape_obs",
            "opponent",
            "search",
            "obstacle",
        )
        shared = {k: smoke["rows"][0][k] for k in keys}
        for row in smoke["rows"][1:]:
            assert {k: row[k] for k in keys} == shared
        assert [row["law_class"] for row in smoke["rows"]] == [
            "PlannerInterceptLaw",
            "LeadPlannerInterceptLaw",
            "LeadPlannerInterceptLaw",
        ]
        assert [row["tau"] for row in smoke["rows"]] == [None, 0.5, 1.0]


# ----------------------------------------------------------------------------
# the per-episode quantities
# ----------------------------------------------------------------------------
class TestMinDistance:
    def test_a_capture_has_min_distance_within_the_capture_radius(self, smoke):
        d_cap = M.diag_config().capture_distance
        captures = 0
        for row in smoke["rows"]:
            for e in row["episodes"]:
                if e["outcome"] == "capture":
                    captures += 1
                    assert e["min_distance"] <= d_cap
                else:
                    # the env terminates exactly when the distance reaches the
                    # radius, so a non-capture never got there
                    assert e["min_distance"] > d_cap
        assert captures > 0, "no captured episode in the smoke; nothing was checked"

    def test_min_distance_is_a_minimum_over_the_episode(self, smoke):
        for row in smoke["rows"]:
            for e in row["episodes"]:
                assert e["min_distance"] <= e["final_distance"] + 1e-12
                assert e["min_distance"] > 0.0

    def test_visible_run_is_bounded_by_the_visible_steps_and_the_episode(self, smoke):
        for row in smoke["rows"]:
            for e in row["episodes"]:
                assert 0 <= e["longest_visible_run"] <= e["visible_steps"] <= e["steps"]
                assert e["steps"] == e["branch"]["n_steps"]

    def test_the_lead_fraction_exists_only_on_the_lead_rows(self, smoke):
        plan, *leads = smoke["rows"]
        for e in plan["episodes"]:
            assert e["lead_v_est_nonzero_frac"] is None
            assert e["lead_visible_law_steps"] == 0
        assert plan["summary"]["lead_v_est_nonzero_frac_mean"] is None
        for row in leads:
            for e in row["episodes"]:
                if e["lead_visible_law_steps"]:
                    assert 0.0 <= e["lead_v_est_nonzero_frac"] <= 1.0
                else:
                    assert e["lead_v_est_nonzero_frac"] is None


# ----------------------------------------------------------------------------
# the summary
# ----------------------------------------------------------------------------
class TestSummary:
    def test_the_wilson_ci_is_the_s19_runners_helper_and_not_a_new_one(self):
        assert M._ci is S19._ci  # the same object, imported

    def test_the_capture_block_is_that_helper_on_this_rows_episodes(self, smoke):
        for row in smoke["rows"]:
            eps = row["episodes"]
            k = sum(e["outcome"] == "capture" for e in eps)
            assert row["summary"]["capture"] == M._ci(k, len(eps))
            lo, hi = wilson_ci(k, len(eps))
            assert row["summary"]["capture"]["wilson"] == [lo, hi]

    def test_the_reported_aggregates_recompute_from_the_episodes(self, smoke):
        for row in smoke["rows"]:
            eps, s = row["episodes"], row["summary"]
            assert s["min_distance_median"] == pytest.approx(
                float(np.median([e["min_distance"] for e in eps]))
            )
            assert s["longest_visible_run_mean"] == pytest.approx(
                float(np.mean([e["longest_visible_run"] for e in eps]))
            )
            assert s["violations"] == sum(e["violations"] for e in eps)
            assert sum(s["outcomes"].values()) == len(eps)


# ----------------------------------------------------------------------------
# the record and the printed table
# ----------------------------------------------------------------------------
class TestRecord:
    def test_the_record_reloads_and_carries_its_provenance(self, smoke):
        on_disk = json.loads(pathlib.Path(smoke["_out"]).read_text())
        assert on_disk["n_episodes"] == SMOKE_N
        assert on_disk["seed0"] == M.DIAG_SEED0
        assert on_disk["obstacle"] == dict(M.OBSTACLE)
        p = on_disk["provenance"]
        assert p["spec"] == M.PLAN
        assert len(p["spec_sha256"]) == 64
        assert p["spec_sha256"] == M.provenance()["spec_sha256"]
        assert p["pe2_file"].endswith("pe2/__init__.py")
        assert p["git_commit"]

    def test_the_reading_is_in_the_record_and_quotes_the_plan(self, smoke):
        assert smoke["reading"] == M.READING
        plan = _norm((_ROOT / M.PLAN).read_text())
        for sentence in (
            (
                "capture without contact then comes only from the arena walls "
                "cutting the orbit"
            ),
            "its reading is written before it runs and it moves no bar",
            (
                "If no strategy captures here, any gain in the full round is "
                "wall-cornering and the paper says so"
            ),
        ):
            assert sentence in _norm(M.READING)
            assert sentence in plan

    def test_the_table_prints_the_reading_above_the_three_rows(self, smoke):
        out = M.table(smoke)
        assert out.startswith(M.READING)
        body = out[len(M.READING) :]
        for law in M.LAWS:
            assert re.search(rf"^{re.escape(law)}\s", body, re.MULTILINE)
        assert body.index("PLAN") < body.index("LEAD-0.5") < body.index("LEAD-1.0")
