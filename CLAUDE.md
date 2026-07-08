# CLAUDE.md

## Git Remote

Every file addition, modification, or deletion must be committed and pushed to GitHub.

- Remote: `git@github.com:linxi-1021/ChemEvoRAG.git`

After making any changes:
1. `git add <files>`
2. `git commit -m "<descriptive message>"`
3. **Before pushing, ask the user for permission.** Do not push unless the user explicitly approves.
4. `git push origin master`

Do not skip any step. If a push fails, diagnose and fix the issue — do not silently move on.

## Change Reporting

After every code change, tell the user **what was modified** before pushing.  Summarize:

- Which files changed
- What was added / removed / changed in each file
- Why the change was made

This applies to every edit — no silent changes.  Report first, then commit and push.

**Commit messages MUST NOT include "Co-Authored-By: Claude" or any reference
to Claude, Anthropic, or AI assistance.**  All commits must appear as the
user's own work.  The author and committer must be `linxi-1021`.

## Communication Discipline

**Answer questions directly. Do NOT modify code, execute commands, or make
changes unless the user explicitly asks you to do so.**  When the user asks
a question, they want an answer — not an immediate code modification.

Before ANY execution (code edits, bash commands, commits), pause and:
1. Consider whether the user actually asked for execution, or just information
2. If unsure, ask the user whether they want you to proceed
3. Use relevant superpowers skills (brainstorming, systematic-debugging, etc.)
   to think through the approach before acting

## Project Discipline

**Do NOT bypass or degrade functionality without the user's explicit permission.**
When a dependency is unavailable (model weights, database, external service), do
NOT quietly replace it with a dummy, stub, or simplified fallback.  Report the
problem and ask the user how to proceed.

**Do NOT modify the project plan, implementation approach, or target results
without the user's explicit permission.**  The Phase 1 goals, architecture, and
success criteria are defined in `docs/phase1_goal_and_stack.md` and the check
report `Phase1目标达成检查报告_2026-05-16.md`.  Changes to these require user
approval first.

**Before any workaround or architecture change**, ask the user.  A bypass that
"works around" a missing component is a design decision — the user makes it,
not Claude.
