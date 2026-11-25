# Card template

Save as `code/<project>/SKILL.md` and replace every `<...>`. Keep it short: a later agent
reads the card to decide whether to open your code.

```markdown
---
name: <your project name, lowercase words with hyphens>
description: <one sentence: what the controller does, start to end>
status: draft
entry: code/controller.py
---

# <your project name>

## When to use

<One sentence on the kind of task this code fits: the motions and the objects (size, shape,
how they are held).> <Anything else it is a good example of, such as a perception method.>
The task text is in [TASK.md](TASK.md).

## Procedure

`controller.py:main(ctx)`:

1. <Phase: what it observes, what it computes, what it moves, what it checks.>
2. <...>

## Code

Entry: `code/controller.py`, `main(ctx)`. Modules: <list>.

| Part | Where | What it does | Reuse notes |
|---|---|---|---|
| `<function or class>` | <file> | <one line> | <general, or what ties it to this task> |

## Task-specific constants

<Every number tuned for this task's objects or layout, with its name, value, unit and
meaning, for example a grasp offset or an approach height.>

## Failure modes

- <What goes wrong, how the code notices, what it does then.>

## Built from

<The library entries you started from, and what you kept, changed or dropped. "None" if you
wrote it from nothing.>
```
