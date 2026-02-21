# Project: Safe Deep RL for 1v1 Pursuit-Evasion

## Repository
- **GitHub**: https://github.com/shoaib6174/safe-rl-pe
- **Local**: `/Users/mohammadshoaib/Codes/robotics/claude_pursuit_evasion`

## Structure
- `docs/literature-review/` - Paper summaries and final literature review
- `docs/pathway/` - Pathway A research plan and validation
- `docs/phases/` - Phase 1-5 implementation specifications
- `docs/references/` - BibTeX, method indices, paper lists
- `docs/workflow_tracker.md` - Session index and progress overview
- `docs/worklogs/` - Per-session detailed worklogs (see Session Protocol below)
- `papers/original/` - Papers 01-37 (PDFs gitignored)
- `papers/supplementary/` - Papers N01-N15 (PDFs gitignored)

## Session Protocol
Every Claude Code session MUST maintain a worklog. Follow these steps:

1. **Session start**: Create a new worklog file in `docs/worklogs/` using the naming convention `YYYY-MM-DD_S##.md` (see `docs/worklogs/TEMPLATE.md` for the template). Fill in the session number, date, context, and objectives.
2. **During session**: Update the "Work Done" and "Files Changed" sections as tasks are completed.
3. **Session end**: Complete all remaining sections (Decisions Made, Issues & Blockers, Next Steps). Add a one-liner entry to the Session Index table in `docs/workflow_tracker.md`.

To determine the next session number, check the Session Index in `docs/workflow_tracker.md`.

## Remote Servers

### niro-2 (Lab PC)
- **Host**: 100.71.2.97
- **User**: niro-2
- **Password**: 123456
- **Access**: `sshpass -p '123456' ssh niro-2@100.71.2.97`
- **Use for**: Downloading paywalled papers, heavy compute tasks
