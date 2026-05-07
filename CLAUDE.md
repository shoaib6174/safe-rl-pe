# Project: Safe Deep RL for 1v1 Pursuit-Evasion

## Repository
- **GitHub**: https://github.com/shoaib6174/safe-rl-pe
- **Local**: `/Users/mohammadshoaib/Codes/robotics/claude_pursuit_evasion`

## Structure
- `docs/literature-review/` - Paper summaries and final literature review
- `docs/pathway/` - Pathway A research plan and validation
- `docs/phases/` - Phase 1-5 implementation specifications
- `docs/references/` - BibTeX, method indices, paper lists
- `docs/workflow_tracker.md` - Session index, current blockers, progress overview
- `docs/decisions.md` - Searchable decision log (aggregated from worklogs)
- `docs/worklogs/` - Per-session detailed worklogs (see Session Protocol below)
- `papers/original/` - Papers 01-37 (PDFs gitignored)
- `papers/supplementary/` - Papers N01-N15 (PDFs gitignored)

## Session Protocol
Every Claude Code session MUST maintain a worklog. Follow these steps:

1. **Session start**:
   - Read the **previous session's worklog** (linked in the template) — specifically its "Next Steps" section.
   - Check **Current Blockers** in `docs/workflow_tracker.md` for unresolved issues.
   - Create a new worklog file in `docs/worklogs/` using `YYYY-MM-DD_S##.md` (see `docs/worklogs/TEMPLATE.md`).
   - Fill in session number, date, previous session link, context, and objectives.
2. **During session**:
   - Update "Work Done" and "Files Changed" as tasks are completed.
   - For any non-obvious decision, add it to **both** the worklog's "Decisions Made" table **and** `docs/decisions.md`.
   - If a training run is launched or completed, update `training-runs.md` in auto-memory.
3. **Session end**:
   - Complete all remaining sections (Decisions, Issues & Blockers, Next Steps).
   - Add a one-liner entry to the Session Index in `docs/workflow_tracker.md`.
   - If any blocker is unresolved, add/update it in the **Current Blockers** section of `docs/workflow_tracker.md`.
   - If a blocker was resolved this session, remove it from Current Blockers.

To determine the next session number, check the Session Index in `docs/workflow_tracker.md`.

## Engineering Standards — MANDATORY

**The goal of this project is to produce quality research. Every decision must reflect that.**

- **NO shortcuts, NO hack fixes, NO "make it pass" workarounds.** If something doesn't work, diagnose the root cause properly. Understand WHY it fails before changing anything.
- **Debug systematically.** Read error messages carefully, form hypotheses, test them one at a time. Do not blindly tweak parameters, comment out checks, or add try/except blocks just to suppress errors.
- **Never fake a passing test.** If a test or validation criterion fails, the code is wrong — not the test. If the test itself is genuinely flawed, explain why and fix the test properly with justification.
- **Rework when necessary.** If an approach is fundamentally broken, say so and propose a proper redesign. Wasting time patching a bad foundation is worse than starting that component over.
- **Correctness over speed.** A correct implementation that takes longer is always preferred over a fast but fragile one. We are building research infrastructure that later phases depend on — cutting corners now compounds into major problems later.
- **Ask when uncertain.** If something is ambiguous or you're unsure of the right approach, ask rather than guess. Wrong assumptions waste more time than a quick clarification.

This applies to ALL sessions, ALL phases, ALL code. No exceptions.

## Development Environment — MANDATORY

**Always use the project virtual environment.** Never install packages into or run code with the system Python.

- **Venv path**: `./venv/` (already in `.gitignore`)
- **Python**: `./venv/bin/python`
- **Pip**: `./venv/bin/pip`
- **Pytest**: `./venv/bin/python -m pytest tests/ -v`
- **Run scripts**: `./venv/bin/python scripts/train.py`

If the venv doesn't exist, create it first:
```bash
python -m venv venv
./venv/bin/pip install -r requirements.txt
```

## Training Runs — MANDATORY

**NEVER run training locally.** The local machine (MacBook Air, 8 GB RAM) cannot handle training. Always run training on niro-2.

To launch training:
1. Push latest code to GitHub
2. SSH into niro-2 and pull
3. Run training there (via `nohup` or `tmux` for long runs)
4. Copy results back when done

## Auto Memory Organization

Memory files are organized by topic. Keep MEMORY.md as a concise index (<50 lines).

| File | Contents |
|------|----------|
| `MEMORY.md` | Current state + links to topic files |
| `environment-config.md` | Env state format, obs space, sensing modes, speed ratios, proven configs |
| `self-play-lessons.md` | What works/fails, training dynamics, curriculum strategies |
| `debugging-ops.md` | niro-2 setup, known bugs, diagnostic validity, seed sensitivity, reward pitfalls |
| `training-runs.md` | Full experiment log — update when runs launch or complete |
| `masking-curriculum-research.md` | PO-GRL literature and implementation notes |

Do NOT dump everything into MEMORY.md. Check if a topic file exists first.

## Remote Servers

### niro-2 (Lab PC)
- **Host**: 100.71.2.97
- **User**: niro-2
- **Password**: 123456
- **Access**: `sshpass -p '123456' ssh niro-2@100.71.2.97`
- **Use for**: Downloading paywalled papers, **ALL training runs**
