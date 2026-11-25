# {{ robot.benchmark }} auto-research

{% if agent_formal %}
Your goal is to learn to complete the configured robot task on layouts you have not
seen: satisfy every condition in `TASK.md`. Solve it by hand first, then build an
automated code controller that does as much of the task as it can, and practise
controlling the robot yourself together with it. Later you continue from what you
keep, in new episodes on new layouts: the controller does what it covers and you
handle the rest.
{% if partial_credit %}
Those later episodes are what counts: each is one live attempt on a new layout, with no
reset or retry, scored with partial credit as `TASK.md` Scoring describes.
{% endif %}
{% else %}
Your goal is an autonomous controller that completes the configured robot task without
live intervention: satisfy every condition in `TASK.md`. Solve the task by hand first to
learn what the controller must do. The final evaluation is the formal batch: your
submitted code run alone on held-out layouts.
{% endif %}

## Start here

Read `TASK.md`, then any existing `memory/PROGRESS.md` and `memory/INDEX.md`, and call
`status` to learn how far the work has got and what budget remains. Do not start a new
episode merely because the conversation resumed. If manual success is not yet
confirmed, begin with step 1 below; if it is, continue development from the saved
evidence with autonomous-control.

The skill links below are entry points. Read the skill for your current step and follow
its references when needed rather than reading the entire workspace first.
{% if skill_library %}

`skill-library/` holds controllers written for earlier tasks on this robot. Before your
first episode, do step 0 below: start from their code instead of from nothing.
{% endif %}

{% if multi_env %}
Exploration runs several parallel environments (`status.exploration`). Read
[multi-env-exploration](.agents/skills/multi-env-exploration/SKILL.md) for how to use
them. Before delegating, read
[multi-agent-exploration](.agents/skills/multi-agent-exploration/SKILL.md): assign
each worker its own environment and a complementary motion or subtask goal.
{% endif %}
{% if shared_env_subagents %}
You may delegate to subagents; before spawning any, read
[multi-agent-exploration](.agents/skills/multi-agent-exploration/SKILL.md).
{% endif %}
{% if handoff and not agent_formal %}
Before exploration ends, also write `handoff/README.md` as
[transfer-handoff](.agents/skills/transfer-handoff/SKILL.md) describes: notes for a
later session on this task with new layouts. It is separate from your submitted
bundle, which runs as it stands.
{% elif handoff and not carries_handoff %}
Before exploration ends, also write `handoff/README.md` as
[transfer-handoff](.agents/skills/transfer-handoff/SKILL.md) describes: notes for a
later session on this task with new layouts.
{% endif %}

## Workflow

{% if skill_library %}
0. **Start from the controller library.** Read
   [skill-library](.agents/skills/skill-library/SKILL.md), then:
   0.1 Read `skill-library/INDEX.md`, pick at most three entries whose task is nearest to
       yours, read their cards, and write your choice in `memory/PROGRESS.md`.
   0.2 Copy the nearest entry's code into `code/<project>/` and run it once in an
       exploration episode before writing new code. Note in `memory/PROGRESS.md` how far
       it got and where it stopped.
   Then do step 1 with what the copied code already does as your starting point.
{% endif %}
{% if demonstrations %}
1. **Understand and solve the task yourself.** Read
   [manual-robot-control](.agents/skills/manual-robot-control/SKILL.md), and
   [in-context-action-learning](.agents/skills/in-context-action-learning/SKILL.md)
   for the supplied {{ 'demonstration image' if one_demonstration_image else 'demonstrations' }}. Match the required final state and every
   condition in `TASK.md` in the current layout, without blindly copying
   demonstrated actions.
{% else %}
1. **Understand and solve the task yourself.** Read
   [manual-robot-control](.agents/skills/manual-robot-control/SKILL.md). Match the
   required final state and every condition in `TASK.md` in the current layout.
{% endif %}
   If details are unclear in the {{ robot.overview }} view, use
   [wrist-camera-inspection](.agents/skills/wrist-camera-inspection/SKILL.md) to move
   an empty arm's wrist camera closer. Control the robot directly through the MCP
   tools, and confirm success with `evaluate` or native `task_complete: true` before
   automating.
{% if robodojo and camera_geometry %}
   Work in this order. Each step names the skill to read before you do it.
   1.1 Write the completion checklist from `TASK.md` in `memory/PROGRESS.md`.
   1.2 Find the objects: call `robodojo_observe`. Its `objects` list gives each object
       an id that stays with it for the episode, with its centre, size and yaw, and
       `object_notes` say what moved, what a gripper holds and what was found again
       ([point-cloud](.agents/skills/point-cloud/SKILL.md)). For overlays, segments and
       positions on an object, use `point_cloud.py` in Python, copied into
       `code/<project>/`. Depth reaches Python only.
   1.3 Plan the grasp, carry and place before moving
       ([grasp-planning](.agents/skills/grasp-planning/SKILL.md)): build each link6 pose
       from where the gripper's centre goes, check every pose of the plan in one
       `robodojo_ik` call, and use only a plan whose poses are all `ok`. Every reply
       gives the gripper's centre as `tcp_positions`; `eef_positions` is link6, about
       15 cm behind it.
   1.4 Move with `robodojo_free_space_move`
       ([free-space-move](.agents/skills/free-space-move/SKILL.md)). After closing the
       gripper, check the grip with `grip` and `verify` of `grasp.py`, lift 1.5 cm, and
       call `robodojo_observe`: `object_notes` should say that gripper holds the object.
       Only then carry. After each release, lift or back the gripper off by at least
       3 cm, call `robodojo_observe`, and compare where the object is and how it is
       turned with what you wanted before you grasp it again.
   1.5 Call `evaluate`. If it is false, observe, find the objects again (1.2) and fix
       the condition that fails.
{% endif %}
{% if agent_formal %}
2. **Automate what you did.** Read
   [autonomous-control](.agents/skills/autonomous-control/SKILL.md) and its section
   "Code you can supervise". Turn your decisions into observation-driven steps that
   check their preconditions, verify their results and report when they fail or meet
   a state they were not built for. Automate as much of the task as you can; you
   handle the rest live.
{% else %}
2. **Reproduce your decisions in an autonomous controller.** Read
   [autonomous-control](.agents/skills/autonomous-control/SKILL.md). Build
   observation-driven control, test functions and full scripts in exploration, inspect
   feedback, and debug.
{% endif %}
   Prefer simple motion functions and heuristics. Use
   [gemini](.agents/skills/gemini/SKILL.md) for visual and spatial perception.
{% if robodojo and camera_geometry %}
   2.1 Copy `grasp.py` next to `point_cloud.py` in `code/<project>/`. The controller
       imports both, and {% if agent_formal %}the code you carry keeps them{% else %}the registered bundle carries them{% endif %}.
   2.2 Do 1.2–1.4 in code with the same helpers, in every episode: observe with
       `robot_spheres=True`, find the objects with `objects()`, check the plan with
       `robodojo_ik`, and check each close with `verify` before lifting.
{% endif %}
{% if agent_formal %}
3. **Practise with your controller.** Put the controller under
   `code/<project>/`. In each new episode, run it on the new layout, watch
   each step, step in where it falls short, and fix the code, as autonomous-control's
   "Pressure-test across layouts" describes. Use every exploration episode this way;
   exploration does not end early.
4. **Keep notes{% if carries_handoff %}, code and handoff{% else %} and code{% endif %} current.** Later you continue from the folders
   `status.carry.carried_folders` lists, at most `carried_limit_bytes` in total, with
   as much of this conversation as `status.carry.conversation` says. Nothing else is
   kept: paths to frames, logs and other files serve you during exploration but will
   not resolve later, so the carried notes{% if carries_handoff %} and `handoff/README.md`{% endif %} state the facts
   themselves (parameters, conditions, outcomes). Copy into those folders any image
   you will need to see again.
{% if carries_handoff %}
   Write `handoff/README.md` as
   [transfer-handoff](.agents/skills/transfer-handoff/SKILL.md) describes.
{% endif %}
{% if skill_library %}
   Keep `code/<project>/SKILL.md` current as
   [skill-library](.agents/skills/skill-library/SKILL.md) step 3 says: when exploration
   ends, your project and its card join the library for later tasks.
{% endif %}

When no exploration episode remains and none is live, `status.next_step` gives your
last steps: update the notes{% if carries_handoff %} and handoff{% endif %}, then end your turn. If something the task
needs cannot be done within the budget, report the blocker; do not bypass it or claim
success.
{% else %}
3. **Pressure-test, freeze and rehearse.** Test the controller across layouts as
   autonomous-control's "Pressure-test across layouts" describes. Put
   `controller.py:main(ctx)`, the modules it imports and its data files under
   `code/<project>/`. Rehearsals and the formal batch run this frozen
   controller alone, against the context they supply, as autonomous-control shows.
{% if skill_library %}
   Before registering the bundle you will submit, write `code/<project>/SKILL.md` as
   [skill-library](.agents/skills/skill-library/SKILL.md) step 3 says, so the bundle
   carries it.
{% endif %}
   `register(source=...)` freezes the directory, and `rehearse(bundle=...)` runs that
   bundle end to end as a fresh process in a fresh exploration episode, without
   you. Inspect the returned evaluation and logs, not just the exit code. Fix failures
   while budget remains; any edit needs a new registration and a new rehearsal.
   `submit` accepts only a bundle whose own rehearsal ran cleanly to its end, with no
   error, timeout or simulator exit; the task need not succeed. Confirm it in
   `status.bundles` (`rehearsal_qualified`) and the `rehearse` return or its run's
   `result.json`, not from your notes. Development tests do not count.
4. **Submit the rehearsed bundle.** Call `submit` once with that exact bundle. It
   closes exploration, freezes the bundle and returns no score. After you end your
   turn, the formal batch runs it from a freeze of your work:
   `status.formal_episode_count` formal episodes, each a fresh process in a fresh
   environment, with no live intervention, edits or retries. You do not see its
   result. Each episode earns the task's own score, with partial credit for partial
   progress, averaged over that fixed count; errors and missing evidence score zero.
   Submit the bundle you expect to score best, even if no rehearsal fully succeeded.
   If a prerequisite cannot be met within the budget, report the blocker; do not
   bypass it or claim success.
{% endif %}

## Episodes and budgets

{{ block("session.contract-pointer") }}
{% if multi_env %}
`start_episode` with the `env_id` of the environment you use starts an exploration
episode on the next layout with the task's step limit; every robot reply reports
`episode_limits` and `episode_ended`. Call `finish` with the same `env_id` to close it
when done.
{% else %}
`start_episode` starts an exploration episode on the next layout with the task's step
limit; every robot reply reports `episode_limits` and `episode_ended`. Call `finish`
to close it when done.
{% endif %}
Layouts differ between episodes and objects move during interaction: reacquire scene
geometry rather than assuming the first layout or earlier object poses persist.

{% if agent_formal %}
Gemini spending has no limit; each call reports its cost (see the gemini skill).
{% else %}
Each rehearsal uses one exploration episode. Gemini spending has no limit; each call
reports its cost (see the gemini skill).
{% endif %}

Ordinary Python uses `from api.runtime import Context`; `ctx.call(tool, **arguments)`
calls an MCP tool. Follow autonomous-control for development scripts and the
controller.

## Workspace and artifacts

Use ordinary Bash, Python and file and image tools in this workspace, your working
directory. It is persistent and the only place you can write; the skills and their
offline references are under `.agents/skills`.
There is no internet or simulator access. Do not seek simulator internals or download
or install simulator code. Disk space is shared (`status` or `df -h`); keep needed
data and remove expendable outputs. Run records are under
`runtime/autonomous_controller/`, read-only.
{% if not agent_formal %}
Your commands pause during registration and rehearsals.
{% endif %}

Inspect selected images, action records and stdout/stderr logs as needed; do not load
whole recordings into context.

## Intermediate notes and compaction

Write intermediate findings to `memory/` as you work, not only at the end. Keep
`memory/PROGRESS.md` current after useful experiments and each workflow step: current
step and episode, confirmed results, failed approaches and lessons, working motion
parameters, code, log and frame paths,
{% if agent_formal %}
remaining budgets, and the next concrete step. Keep reusable robot tips in
`memory/TIPS.md` following
{% else %}
bundle and rehearsal IDs, remaining budgets, and the next concrete step. Keep
reusable robot tips in `memory/TIPS.md` following
{% endif %}
[experience-memory](.agents/skills/experience-memory/SKILL.md). Store evidence on disk
and reference its path instead of copying large traces or image bytes into notes.

Your context is compacted automatically. Maintain these notes continuously so work can
resume after compaction; reread them and check live `status` before continuing. Do not
restart an episode to reduce context. In your final report, give the controller path,
{% if agent_formal %}
what it covers, the layouts it was tested on, unresolved issues and artifact pointers.
{% else %}
the submitted bundle, rehearsal outcomes, unresolved issues and artifact pointers.
{% endif %}
