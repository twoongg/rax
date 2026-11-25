# RoboDojo controller library

Robot controllers for the RoboDojo table-top tasks (dual ARX X5), one entry per solved task.
An agent facing a new task reads [INDEX.md](INDEX.md), opens the entries whose task is
nearest to its own, and reuses or imitates their code. After each task's formal batch, the
code that ran there is added as a new entry, whatever its score, with the score on its card.

## Entry layout

```
skills/<slug>/
  SKILL.md   the card: frontmatter (name, description, status, robot, entry, task, source,
             formal) and When to use, Procedure, Code, Task-specific constants, Failure modes
  TASK.md    the task text the agent had
  demo/      the demonstration images, when the task had them
  code/      the controller as it ran: controller.py:main(ctx) and its modules
```

The card follows the skeleton of the RAX skill library
(github.com/robot-agent-exploration/skills-lib) without its evidence rules.

## Seed entries

The first four entries are controllers written by Claude Opus agents in the RAX authors'
sessions, published in the dataset `robot-agent-exploration/robot-session-logs`
(huggingface.co/datasets/robot-agent-exploration/robot-session-logs, folder `robodojo/`,
CC-BY-4.0; paper: arXiv 2610.22496). Their code is unchanged. Their cards and `TASK.md` files
were written here: the `TASK.md` files are regenerated with the public rubric (the sessions'
own were not published), and their formal scores were not published.

`make-kong` does not run as published: it imports a module from the authors' setup that was
not released. Its card lists the tools to use instead.
