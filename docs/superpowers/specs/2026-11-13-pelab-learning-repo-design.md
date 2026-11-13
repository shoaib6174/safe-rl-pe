# pelab — a clutter-free pursuit-evasion lab the user runs alone

**Date:** 2026-11-13 · **Status:** approved by the user 2026-11-13 ("review and critique, then go on with implementation, also test"); revised after the critique in section 8 · **Session:** S136

Items marked **(user)** were said by the user. Items marked **(call)** are
Claude's choices, open to correction.

## 1. Intent and why

**(user)** The Claude-driven way of working in `safe-rl-pe` is not working.
The user will now work on their own: run the experiments, visualise them,
implement new models and compare them. The aim is to become an expert who can
do all of this without help. Claude's job changes from doing the work to
teaching and reviewing it.

**(user)** The sequence:

1. Sanitize the current repo.
2. A new repo that provides the environment already built, finished.
3. Teach everything in it, through two documents: one to read, one to learn
   by doing.
4. One or two basic trainings, so the repo has an organised shape.
5. Later approaches (including paper reproductions) are added so that each
   can draw on the earlier ones.

**(user)** First experiments: scripted players against each other (greedy vs
greedy), then greedy against a plain RL model, then reproductions of papers.

**(user)** One environment, with switches for: no obstacles, more obstacles,
partial observation, safety.

**(user)** Paper work moves to the new repo and starts again from the base.
The new repo therefore also carries three more documents: a guideline for
reproducing the important findings of the old repo, a list of suggested
papers to reproduce (recent ones included), and a document explaining the
metrics.

## 2. Goals, non-goals, anti-goals

**Goals** — in the new repo the user can, without Claude:

- G1. Run any two players against each other and read a result with a
  confidence interval.
- G2. Make a GIF of any match-up.
- G3. Drive either robot from the keyboard against any player, including a
  trained model.
- G4. Train a plain RL model against a scripted opponent and score it.
- G5. Add a new approach in its own folder, reuse code from earlier
  approaches, and see it in one comparison table beside all the others.
- G6. Explain how every file in the core library works.
- G7. Re-derive the old repo's important findings by their own hand, each as
  an approach, and know whether the number they get agrees with the archive.
- G8. Read any result table from the old repo or a paper in this area and
  say what each metric measures and what it cannot show.

**Non-goals**

- Carrying over Paper A / Paper B research code (tactics, selectors, recovery
  shields, exploitability rounds, pre-registration machinery). The user
  rebuilds what a finding needs; the archive is the reference answer.
- Carrying over the paper drafts. They stay in the archive; a paper written
  in the new repo is written from results produced there **(call)**.
- Self-play, populations, belief models. Each arrives later as an approach,
  if a reproduction needs it.
- Any change to the archived results.

**Anti-goals** — these would make it a failure even if it runs:

- A framework that hides the mechanics behind configuration.
- Claude writing the approaches while the user watches.
- Clutter returning: loose scripts in the root, results in git, dated notes
  piling up.
- A switch that silently changes behaviour, or a model that loads under a
  config it was not trained for.

## 3. Constraints

- The old repo's results must stay reproducible, so nothing in it is moved or
  deleted.
- The new environment must behave exactly like `pe2` on its default settings
  (same seed, same trajectory), so the knowledge in the old repo still
  applies.
- Dependencies stay small: numpy, gymnasium, matplotlib, imageio, pytest for
  the core; stable-baselines3 + torch only for approaches that train.
- Own virtual environment in the new repo (`./venv/`), never system Python.
- Results (`runs/`) are never committed.

## 4. Approaches considered

| Way | Sketch | Trade-off | Verdict |
|---|---|---|---|
| A. Small library + one folder per approach | `pelab/` holds the world; each `approaches/NN_name/` holds its own training script, notes and results; shared tools work on any player | Some copying between approaches at first; nothing hidden | **Picked (call)** — the user described outcomes and left the how to Claude |
| B. One configurable framework | Algorithms plug into a config system | Quick launches, but the mechanics vanish behind config; this is how the old repo grew a 2,962-line training file | Rejected: works against G6 |
| C. Notebooks per experiment | Everything in cells | Visual, but poor for long trainings and for reuse between approaches | Rejected: works against G5 |

## 5. Design

### 5.1 Step 0 — freeze the old repo **(call)**

"Sanitize" is read as "stop it being the workplace, lose nothing":

- Tag the current commit of `searching-teacher` (the branch all recent work
  is on) as `archive-2026-11-13`. `main` is left alone.
- No files moved, no files deleted, no history rewritten.
- **(user)** Paper work leaves this repo. So Step 0 also commits the
  untracked work (`docs/paper_a/`, the S133–S136 worklogs, the Paper B study
  guide, this design) so the archive is complete, ignores the LaTeX build
  files (`main.bbl`, `main.blg`), and adds an "ARCHIVED" banner to
  `README.md` pointing at the new repo. The commit is made only on the
  user's word at approval.
- Pushing the tag and the niro-2 password that sits in the public
  `CLAUDE.md` are raised with the user; neither is acted on without their
  word.

### 5.2 The new repo

Location `/home/niro-1/Codes/pelab`, package name `pelab` **(call)**. Local
git repository; no remote until the user asks.

```
pelab/
  README.md              one page: install, the five commands, where to read next
  CLAUDE.md              tells Claude it is the teacher/reviewer here (5.7)
  pyproject.toml
  pelab/                 the core library — finished, tested, small
    config.py            EnvConfig + presets
    dynamics.py          how one robot moves one step (unicycle, walls, obstacles)
    env.py               the world: two robots, arena, obstacles, capture
    sensing.py           field of view, line of sight
    obs.py               what each robot is told (partial or full)
    safety.py            the optional action filter (basic CBF shield)
    players.py           the Player contract + scripted players
    match.py             play one episode / a seeded battery between two players
    stats.py             win rates, Wilson intervals, paired tests
    render.py            frames, trails, overlays, GIF
    runs.py              save / load a run with the config it was made under
  tools/                 the commands the user types
    duel.py              player A vs player B over N seeds -> table
    gif.py               one episode -> GIF
    play.py              keyboard vs any player
    compare.py           every registered player vs every other -> leaderboard
  approaches/
    README.md            how to add one; the index of what exists
    _template/           copy this to start a new approach
    01_scripted/         worked example, complete
    02_ppo_vs_greedy/    first training: skeleton, the user fills the core
  docs/
    GUIDE.md             the document to read
    WORKBOOK.md          the document to learn by doing
    METRICS.md           every metric: what it measures, how it is computed, how it misleads
    FINDINGS.md          guideline for reproducing the old repo's important findings
    PAPERS.md            suggested papers to reproduce, with why and how hard
    HISTORY.md           one page: where this came from, the lessons worth keeping
  exercises/             check scripts for the workbook
  tests/                 tests for pelab/
  runs/                  outputs (git-ignored)
```

### 5.3 The environment and its switches

Source: `pe2/{config,env,sensing,obs,scripted,render,eval,rng}.py` and the
base `CBFShield` from `pe2/shield.py` (it needs only `config`,
`env.pursuer_step` and `wrap_angle`; the recovery and predictive shields are
subclasses and stay behind), plus a trimmed `pe2/metrics.py` (`ShieldMetrics`,
which the shielded match and the shield tests use), copied and renamed, with
their tests
(`tests/pe2/test_{config,env,obs,sensing,scripted,eval,rng,shield}.py`
trimmed to what is carried).

Switches, all fields of one frozen `EnvConfig`:

| Switch | Values | Source |
|---|---|---|
| Obstacles | how many are placed (0..N), radius range | exists (`n_physical_obstacles`) |
| Observation | `"partial"` (field of view + range + line of sight) or `"full"` | new field; `full_state_observation` exists in `pe2/obs.py` but the env does not call it yet — small new wiring |
| Safety | which roles are filtered: none, pursuer, evader, both | new field; see "Where safety sits" below |
| Speed ratio, arena size, episode length, start distance | numbers | exist |

Named presets **(call)**: `open` (no obstacles, full observation),
`obstacles`, `partial`, `partial_safe`. `EnvConfig()` with no arguments stays
equal to today's `pe2` default.

**Observation layout stays fixed across presets (call).** The observation
has `7 + 2 * n_obstacles` numbers. `n_obstacles` is treated as "slots
reserved" and stays at 2 in every shipped preset; presets differ in how many
obstacles are actually placed. So `open` places none but keeps the 11-number
layout, and one trained model can be scored on every preset. Placing more
obstacles than there are slots is allowed; which ones fill the slots follows
what `pe2/obs.py` does today, and the guide states it.

**Where safety sits (call).** `env.step` has three visible stages: clip the
requested action, filter it for the roles the config names, run the physics.
The filter code lives in `safety.py`, not in the physics. Consequences:

- Any code that steps the environment gets the configured safety, including
  the user's own training script. There is no path where a "safe" preset runs
  unfiltered.
- The filter applies to whoever plays a filtered role, in training and in
  evaluation alike, the fixed opponent included.
- `info` reports the requested action, the applied action and real contacts,
  so the filter's effect is measured, not assumed.
- With safety off (the default) the step is bit-identical to `pe2`.

**What counts as a mismatch when loading a model (call).** A saved model
stores its role, observation mode and observation length. If any of these
differs from the config it is asked to play under, loading fails with a
message naming the difference. Other differences (speed ratio, arena size,
obstacles placed, safety) are allowed, because testing a model away from its
training conditions is a legitimate experiment; the result table prints the
training config beside the evaluation config so the difference is never
hidden. In `tools/compare.py` an incompatible pairing shows as `n/a` in its
cell; the table still completes.

Carried from the archive on purpose: seeded resets (same seed, same episode)
and scripted players reading the true state while learned players get the
observation.

### 5.4 The Player contract — the one idea everything hangs on

A player is built once per match by its registry function, which hands it
its role (`"pursuer"` or `"evader"`) and the `EnvConfig`. After that it is
anything with:

```python
sees_truth: bool = False                 # class attribute
def reset(self, seed: int) -> None       # start of each episode
def act(self, obs, truth) -> tuple[float, float]     # (speed, turn rate)
```

`truth` is a small read-only snapshot (both poses, the obstacles, the step
count). `match.py` passes it only to players that declare `sees_truth`;
every other player receives `None` there.
The existing scripted players, which read the environment object directly
today, are rewritten to read this snapshot. Learned players never receive
it. Result tables mark truth-seeing players, so it is always visible which
side had the advantage.

Scripted players, a Stable-Baselines3 model, the user's own PyTorch network
and the keyboard are all players. `match.py`, and therefore `duel`, `gif`,
`play` and `compare`, accept any two players. This is what lets a new
approach be compared with every old one without new glue code.

Each approach folder registers its players in a small `players.py`
(name -> function that builds the player), which is how the tools find them:

```
python tools/duel.py greedy_pursuer greedy_evader --preset open --episodes 200
python tools/gif.py  ppo_v1 greedy_evader --seed 7
python tools/play.py --me evader --vs ppo_v1
python tools/compare.py --preset obstacles
```

### 5.5 Approaches — how later work draws on earlier work

Each `approaches/NN_name/` contains:

- `README.md` — the question, the method in plain words, the paper if any,
  the command to run, the result, what was learned.
- `train.py` — the training script, readable top to bottom.
- `players.py` — registers what this approach produced.
- anything specific to it (reward, wrapper, network).

Rules **(call)**:

- An approach may import from `pelab/` and from earlier approaches.
- The first time a second approach needs something an earlier one wrote, it
  imports it. If a third needs it, it moves into `pelab/`. Nothing is
  generalised before then.
- Outputs go to `runs/<approach>/<run-name>/` holding `config.json`, the
  model, a metrics log and the seeds used.
- `tools/compare.py` writes `runs/leaderboard.md`: a table of every
  registered pursuer against every registered evader on a chosen preset.

Shipped approaches:

- **01_scripted** — complete worked example. Greedy and the other scripted
  players against each other on each preset. It shows what a finished
  approach looks like and gives the baseline numbers.
- **02_ppo_vs_greedy** — the first training. The learner is the pursuer,
  the fixed opponent is the greedy evader **(call)**. Measured on the
  existing code over 100 paired seeds: with no obstacles the greedy pursuer
  captures 100/100 and a random pursuer 16/100; on the `pe2` default (two
  obstacles, partial observation) greedy captures 37/100 and random 5/100.
  So `open` is the warm-up (can the learner get from 16 to near 100?) and
  `obstacles` is the real contest (can it beat greedy's 37?). Claude supplies the frame: the
  wrapper that turns the two-robot world into a one-learner problem against a
  fixed opponent player, the run saving, the evaluation hook. The user writes
  the parts that carry the decisions: the reward function, the training
  settings, the stopping rule. A reference answer sits in `exercises/answers/`,
  to look at after trying.

Paper reproductions are not designed here. Each gets its own short design
when the user picks the paper.

### 5.6 The two documents

**`docs/GUIDE.md` — read it.** Ordered so each part needs only the ones
before it:

1. The game in one page, with a picture.
2. How a robot moves (the unicycle, one step of arithmetic by hand).
3. Walls, obstacles, capture, timeout.
4. What a robot is told: full vs partial, the observation vector slot by slot.
5. The safety layer: what it guarantees, what it costs, one worked example.
6. Players and matches.
7. Reading a result: win rate, interval, why seeds are paired.
8. From a match to a training problem (the one-learner wrapper, reward,
   episode).
9. How an approach is organised and how to add one.

Each part names the file and lines it explains, shows a command that makes
the idea visible, and ends with "you understand this when you can…".

**`docs/WORKBOOK.md` — do it.** Graded exercises tied to the guide's parts.
Each has a task, a hint, and a check (`python exercises/check_03.py`) that
says pass or what is wrong. The arc:

- Predict, then run (step a robot by hand, compare with the environment).
- Change a switch and explain the change in the numbers.
- Write a new scripted evader and beat greedy with it.
- Play manually and record what makes the pursuer fail.
- Write the reward for approach 02, train, score it.
- Add approach 03 from the template without help.

The last exercise is the test of the whole project: if the user completes it
alone, the goals are met.

### 5.7 The three further documents

**`docs/METRICS.md`.** One entry per metric, each with: the question it
answers, the formula, the function that computes it, a small worked example
with real numbers, and the way it has misled before. Covered:

- Outcome metrics: capture rate, timeout rate, time to capture.
- Uncertainty: Wilson interval, why evaluations are paired on seeds, McNemar's
  test, how many episodes a claim needs.
- Safety metrics: contacts (violations), how often and how much the filter
  changes the action.
- Stall metrics: immobile-step rate and deadlock rate, and the difference
  between them (in the old repo, counting rotation as immobility made a
  working remedy look like a failure, D129).
- Game metrics: best-response exploitability, the cross-table of players,
  the value of the table as a game, and why worst-case numbers are quoted
  instead of per-round ones (D124).
- Training-curve reading: what a reward curve does and does not say about
  capture rate.

Entries for metrics the core does not compute yet are written when the
approach that needs them arrives; the document says which are live.

**`docs/FINDINGS.md`.** A ladder of the old repo's findings, in the order
they can be rebuilt, each step an approach the user writes. Per finding: the
claim in one sentence, the setup (preset, players, seeds, episodes), what has
to be built, the archived number with its decision id and file, what counts
as agreement, and the known trap. Candidate ladder, to be confirmed against
`docs/decisions.md` and the paper drafts when the document is written:

1. The filter is safe: zero contacts with safety on, contacts without.
2. Safety costs capture: a pursuer loses capture rate when filtered
   (D112: 0.728 to 0.396 against the greedy evader).
3. The cost is mostly a stall, not a restriction: the filtered policy keeps
   asking for a blocked action although an allowed one exists (D125).
4. An escape that offers forward motion removes the stall with zero contacts
   (D131); remedies that only rotate, or that penalise the stall, do not
   (D129, D130).
5. A learned evader can pin a filtered pursuer, and does not need to be
   faster (D114, D115, D123).
6. Which defender is best depends on the metric: the cross-table and its
   worst-case ordering (D124).
7. Two-sided learning locks in: one side wins in most self-play seeds
   (4 of 34 balanced).

Agreement is defined per step, and for most steps it is the direction and
rough size of an effect, not the archived number: the archive's pursuers were
different learners trained for up to 10 M steps, so a plain PPO will not land
on 0.728. The guideline says which steps expect the same number (step 1:
exactly zero contacts) and which expect the same direction.

Steps 1–2 need only the shipped core plus a trained pursuer. Each later step
names what the user must build first. The guideline does not copy code from
the archive; it points to the archived file as the answer to check against.

**`docs/PAPERS.md`.** A short ranked list of papers worth reproducing in this
lab. Per paper: the claim to reproduce, what must be added to the lab, a
difficulty rating, and what reproducing it teaches. Built from two sources:
the old repo's literature review and paper lists (`docs/literature-review/`,
`docs/references/`, `papers/`), and a fresh search for work from 2025–2026.
Every entry is checked to exist and to say what the list claims before it
goes in; a paper that cannot be obtained is listed as "not verified". The
list is ordered easy to hard, starting with papers the old repo already
understands (for example Gonultas and Isler, already reproduced there).

### 5.8 Claude's role in the new repo

The new repo's `CLAUDE.md` states: the user writes approach code; Claude
explains, reviews, answers questions, and writes core-library or tool code
only when asked. No worklog/decision-log protocol from the old repo is
carried over. A single `approaches/README.md` index and each approach's own
README are the record.

## 6. Order of work and how each step is checked

| Step | Done when |
|---|---|
| 0. Freeze old repo | tag `archive-2026-11-13` exists on `searching-teacher` (plus banner and clean status only if paper work has stopped) |
| 1. Core library | carried tests pass; a golden-trajectory test shows `pelab` on default config matches `pe2` step for step on fixed seeds (the golden file is produced once from the old repo and stored in `tests/`, so the new repo never imports the old one); the same for the filtered pursuer against `run_shielded_duel` |
| 2. Switches + presets | tests for each switch; a model/config mismatch raises |
| 3. Tools | `duel`, `gif`, `compare` produce output for scripted players on every preset (smoke-tested as commands); the keyboard player's key-to-action logic is unit-tested and a full episode is driven by a scripted key source; real-time feel in a window is the one thing only the user can confirm |
| 4. Approach 01 | its README numbers are regenerated by its own command |
| 5. GUIDE + WORKBOOK parts 1–7 | every command in them was run; every check script passes on a reference answer and fails on a wrong one |
| 6. Approach 02 frame + GUIDE/WORKBOOK parts 8–9 | on `open`, the reference-answer pursuer captures the greedy evader far more often than the random pursuer's 16/100, with non-overlapping intervals; its `obstacles` number is reported beside greedy's, whatever it is |
| 7. METRICS.md | every live metric's worked example was computed by the named function |
| 8. FINDINGS.md | every archived number in it was read from the archive file it cites; step 1 was run in `pelab` and its outcome recorded; step 2 was run with the reference-answer pursuer and the direction of the effect recorded, agreement or not |
| 9. PAPERS.md | every entry verified as described in 5.7 |
| 10. Hand over | the user does the workbook; Claude reviews |

## 7. Left to the builder

- Keyboard play: reuse the matplotlib renderer if it holds real time (20
  steps per second), otherwise a small pygame view. Measure, then choose.
- Exact names of presets, players and command flags.
- How much of `pe2/scripted.py`'s other scripted players to carry beyond
  greedy (carry those that work without the dropped modules).
- Whether the PPO in approach 02 runs on CPU or GPU on niro-1 (the problem is
  small; CPU is expected to be enough).
- Layout of `runs/<approach>/<run-name>/` beyond the four required items.


## 8. Critique of this design, and what it changed

Done after the user's go-ahead, by reading the code paths the design relies on.

| # | Weak point found | Change |
|---|---|---|
| 1 | Putting the filter inside `env.step` creates a circular import: `pe2/shield.py` imports `pursuer_step` from `pe2/env.py`. | The one-robot kinematics move to `dynamics.py`; `env.py` and `safety.py` both import it. Also better for teaching: motion is its own file. |
| 2 | Two `act` signatures made the "one idea everything hangs on" two ideas. | One signature, `act(obs, truth)`, with `truth=None` for players that do not see it. |
| 3 | A `solutions` branch drifts from `main` and cannot be exercised by the test suite. | Reference answers live in `exercises/answers/`, tested with everything else. |
| 4 | "Reproduce the archived number" is not achievable for findings that used long-trained special learners. | FINDINGS defines agreement per step: same number where that is honest (zero contacts), same direction elsewhere. |
| 5 | The golden-trajectory test would have made the new repo depend on the old one. | Golden files are generated once from the old repo and stored in the new repo's tests. |
| 6 | Keyboard play cannot be fully tested without a person at a screen. | The logic is tested headless; the design says plainly that real-time feel is the user's check. |
| 7 | With no obstacles placed, the obstacle slots are zeros, which also reads as "an obstacle at my own position". Obstacle slots are never hidden by the field of view; only the opponent is. | Not changed (it would break equality with `pe2`); the guide states both facts. |
| 8 | The design is large, and a long build before the user touches anything works against the intent. | Built in three milestones, each usable on its own: (M1) freeze, core, switches, tools, approach 01, GUIDE/WORKBOOK parts 1–7, METRICS; (M2) approach 02 frame and parts 8–9; (M3) FINDINGS and PAPERS. |
| 9 | The freeze commit sweeps in files another session left untracked. | Accepted on the user's statement that paper work leaves this repo; the commit message names them. The tag is not pushed without the user's word. |
