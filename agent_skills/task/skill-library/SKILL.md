---
name: skill-library
description: Start from the controller library in skill-library/. Find the entries nearest to your task, copy their code into your project, change what differs, and write the card that adds your controller to the library.
---

# Controller library

`skill-library/` in your workspace holds robot controllers written for earlier {{ robot.benchmark }} tasks,
one entry per task:

- `skill-library/INDEX.md`: one row per entry, with what it does and when to use it.
- `skills/<slug>/SKILL.md`: the card. What the controller does, its steps, the parts worth
  reusing, the numbers tuned for its own objects, and how it fails.
- `skills/<slug>/TASK.md`: the task it was written for.
- `skills/<slug>/code/`: the controller as it ran, `controller.py:main(ctx)` and its modules.

{% if not agent_formal %}
`formal` in the index and on each card is the share of the formal batch the code completed
(`not published` when it is unknown). Prefer entries that worked; an entry at 0 still shows
what was tried.

{% endif %}
Working code is a much faster start than a controller written from nothing. Do step 1
before your first episode, step 2 in your first episodes, and step 3 before you finish.

## 1. Find the nearest entries

1. Read `skill-library/INDEX.md`.
2. Pick at most three entries whose task is nearest to yours. Compare their `TASK.md` with
   your `TASK.md`: the objects (size, shape, how they are held) and the motions (pick, place,
   push, pour, insert, press, turn, hand over between arms).
3. Read those cards. Open only the code files a card points to; do not read every entry.
4. Write in `memory/PROGRESS.md` which entries you chose, why, and which of their parts you
   plan to keep.

## 2. Reuse the code

Your controller runs from `code/<project>/` and sees only that folder, so copy everything it
uses into it.

- **One close entry:** copy its whole code folder into your project,
  `cp -r skill-library/skills/<slug>/code/. code/<project>/`, and run it once in an
  exploration episode as it is. Read its log and where it stops. Then change what differs:
  object finding, positions, and every number the card lists under "Task-specific
  constants". Keep the parts that work.
- **Parts from several entries:** copy each file you need. If its name clashes with a file
  you already have (two entries may both have `motion.py`), give it a new name and fix the
  imports that use it.
- Every module a copied file imports must be in your project too. A card says when an
  entry imports something that is missing, and what to use instead.
- A number tuned for another object (a grasp offset, a height, a colour threshold) is only a
  first guess: measure it again on your objects.
- Run the copied code before you change much of it: an error you see early is cheaper than
  one you find after a rewrite.

## 3. Write your card

{% if agent_formal %}
Before exploration ends, write `code/<project>/SKILL.md` for your controller and keep it
current as the code changes. When exploration ends, your project and this card join the
library, and later tasks start from them.
{% else %}
Before you register the bundle you will submit, write `code/<project>/SKILL.md` for your
controller, so the bundle carries it. After the formal batch, the submitted bundle and this
card join the library with their result, and later tasks start from them.
{% endif %}

Copy [the card template](references/card-template.md) and fill in every section from what
you know now. Name functions and files exactly; give numbers with units. The library fills
in the task name{% if not agent_formal %}, the result{% endif %} and where the card came from.
