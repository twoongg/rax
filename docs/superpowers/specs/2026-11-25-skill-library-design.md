# A growing controller library for weak agents on RoboDojo

Date: 2026-11-25. Branch `skill-library` (base `robodojo-behavior/native`).

## Intent

A weak open-weight agent (DeepSeek through Claude Code or DSH) rarely writes a working RoboDojo
controller from scratch. The RAX authors published four controllers that Claude Opus wrote
(`robot-agent-exploration/robot-session-logs`, folder `robodojo/`: make_kong, plug_in_charger,
pour_by_language, swap_blocks). The experiment gives those controllers to the weak agent as a
library, lets it reuse or imitate them on a task it has not seen, and adds what it builds back
to the library, task after task: a RAX that grows. The early tasks cost time and money; the hope
is that later tasks are solved quickly from what has piled up.

Plot: x = the order in which new tasks are met, y = formal success rate (and exploration
episodes, tokens, wall time). The comparison is the same tasks run without a library.

## Goals

- Seed library: the four Opus controllers, unchanged, each with a card (`SKILL.md`) and the
  task text the agent would get (`TASK.md`).
- A session option that copies the library into the agent's workspace, with a skill and
  numbered `AGENTS.md` steps that route a weak agent through it.
- After each session's formal batch, the session's frozen code, its `TASK.md`, its
  demonstration, its card and its formal score become a new library entry, whatever the score.
- A script that runs the tasks in order: task 1 exploration -> formal -> publish -> task 2
  exploration -> formal -> publish -> ...

## Non-goals

- No refactoring of the Opus code into primitives (it cannot be checked without the simulator).
- No upstream-style merge tool, session branches or evidence validation.
- No change to how exploration, rehearsal or the formal batch work.

## Anti-goals

- Leaking a task's answer: the four seed tasks are not evaluation tasks.
- Instructions a weak model skips: every library step is a numbered step in `AGENTS.md`.

## Decisions (the user's)

- Seed shape: code untouched plus a card per controller (option 1 of three).
- Card format: the upstream `skills-lib` skeleton (frontmatter, section names), relaxed: no
  required images, run records or perception-checkpoint tables; two added sections, reusable
  parts and task-specific constants.
- `TASK.md` lives beside the card as its own file, not inside it.
- The loop is strictly sequential; an entry is added after its formal batch, regardless of the
  score. A minimum-success filter stays open as a later option (implemented as an opt-in flag,
  off by default).
- Task order, nearest tasks first: pour_liquid_into_cup, press_by_number, swap_t,
  pour_balls_into_vase, insert_tubes, insert_key.
- Build approach 1: a small harness option plus an outside script that runs the loop and
  publishes (rejected: porting upstream's `skill_library.py` merge flow; reusing
  `--adaptation-source`, which carries one same-task source and cannot accumulate).

## Library layout

```
<library>/
  README.md              what the library is, where the seed came from (CC-BY-4.0)
  INDEX.md               one row per entry: skill, task, formal, what it does, when to use
  skills/<slug>/
    SKILL.md             the card
    TASK.md              the task text the agent had
    demo/                the demonstration images and manifest, when the session had them
    code/                the controller: controller.py:main(ctx) and its modules, as frozen
```

Slugs are the task name with hyphens (`pour-by-language`); a repeat gets `-2`, `-3`, ...
The library on the server is a git repository; each publish is a commit, and every session
records the commit it was given (`<session>/skill-library.json`).

Card frontmatter: `name` (the slug), `description`, `status` (draft for now), `robot`,
`entry` (`code/controller.py`), `task`, `source`, `formal` (for example `3/10`). Sections:
When to use, Procedure, Code (entry and a table of reusable parts), Task-specific constants,
Failure modes.

## How the agent sees it

- `--skill-library DIR` (configure) stores the path in the session configuration.
- The exploring agent's workspace gets a copy at `skill-library/` (no `.git`); formal copies do
  not (their carried `code/` already holds whatever was copied in).
- Skill `skill-library` (`agent_skills/task/skill-library/`, installed only with a library)
  says how to pick entries, copy code and write the card; `references/card-template.md` is the
  card to fill in.
- `AGENTS.md` adds: step 0 (read `INDEX.md`, compare `TASK.md` files, copy the nearest entry's
  code into `code/<project>/` and run it before writing anything new) and, before the bundle is
  registered (code mode) or before exploration ends (live copies), writing
  `code/<project>/SKILL.md`.
- Bundles only see their own directory, so library code is used by copying it into the
  project, as `point_cloud.py` already is.

## Publishing (`services/skill_library.py`)

`publish(session_root, library, min_success=None)` after the formal batch:

- code: the submitted bundle in code mode (exactly what the formal batch ran); otherwise the
  project folder of the frozen exploration package (`bundles/<checkpoint>/transfer/code/`).
  No frozen code (exploration never closed) -> nothing is published and the reason is returned.
- card: `SKILL.md` from that code, else from the frozen `code/<project>/`; else a stub card
  listing every module's functions and their first docstring lines.
- score: `formal.json` (`bundle_formal` or `agent_formal`: success_count / episode_count); the
  card's `formal` field and the `INDEX.md` row carry it.
- `TASK.md` and `runtime/demonstrations/` come from the session's workspace.
- `min_success`: when given and the score is below it (or missing), nothing is published.

## The sequence script (`scripts/run_skill_library_sequence.py`)

- `--library DIR` (created from `--seed`, default `skill_library_seed/robodojo`, as a git repo),
  `--tasks a,b,c`, `--records FILE` (JSONL, one row per finished task; a rerun skips tasks
  already recorded), `--min-success`, `--no-library` (the comparison run).
- Each task: prepare with `configure_auto_research.py --task T --skill-library DIR` plus
  `--configure-args`, then run the launcher `--non-interactive` plus `--launch-args`, which
  returns after the formal batch; or, with `--runner deepseek`, `run_deepseek_experiment.py`
  with `--runner-args`. Then publish and record: task, session, library commit before and after,
  formal outcome, published slug or the reason it was not.

## Constraints

- This Mac has no GPU or simulator: unit tests and the workspace goldens are the checks here;
  the loop itself runs on GPU servers, which pull this branch from GitHub.
- The workspace validator treats `skill-library/` as task data (like `TASK.md`), so task words
  in cards are allowed there.

## Seed notes

- make_kong imports `robodojo_toolkit` (the authors' planner and IK), which was not published.
  A replacement `robodojo_toolkit.py` was written on the server, and make_kong's formal batch
  passes with it there; the file stays on the server. It is added to a server library with
  `python -m services.skill_library add-file <library> make-kong <path>` (or the sequence
  script's `--add-file make-kong=<path>`), and `check` / the sequence script warn while an
  entry imports a module it lacks. The card covers both cases: copy the toolkit when `code/`
  has it, otherwise replace its calls with the tools it names.
- The Opus sessions' own `TASK.md` files were not published; the seed's are regenerated with
  `--task-rules public`, without demonstration images.
- The Opus controllers' formal scores were not published; their cards say "not published".

## Left to the builder

Exact wording of the skill and steps (within the validator's rules), stub-card details, and
test fixtures.
