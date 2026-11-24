---
name: grasp-planning
description: Use before a RoboDojo arm grasps, carries, turns or places an object. Says where the gripper holds (between the finger pads, 15 cm beyond link6), builds a grasp pose from directions, checks every pose of a pick-and-place in one robodojo_ik call before moving, and tells whether a close took the object.
---

# Grasp planning

## The gripper

- Robot poses are `link6` poses. The gripper holds at its tool centre point (TCP), the
  point between the finger pads: 15.0 cm along link6's +x axis
  (`TCP = link6 + R · [0.150, 0, -0.0014]`). Every robot reply gives it as
  `tcp_positions`, beside `eef_positions` (link6), and the last gripper command per arm
  as `gripper_commands`.
- link6 +x is the **approach** axis, the direction the gripper points out between the
  fingers. link6 +y is the **closing** axis, along which the fingers close.
- The fingertips reach 7.5 mm past the TCP along the approach axis.
- The gripper opens 8.8 cm in all and surely closes on objects up to 8.5 cm wide. The
  observation's `states` give each gripper's opening (0 closed, 1 open); the gap is about
  the opening times 8.8 cm.

Check the TCP once in exploration: put an empty gripper's TCP on a point you measured,
such as an object's top edge, and look at a wrist or high view.

## Build a grasp pose

Give `robodojo_pose_math` where the TCP goes and two directions in environment axes:

```json
{"operation": "link6_from_tcp", "tcp_position_m": [0.30, 0.10, 0.03],
 "approach": [0, 0, -1], "closing": [0, 1, 0]}
```

- Straight down: `approach` `[0, 0, -1]`, and `closing` across the object, usually along
  its short side. A vertical approach needs `closing`.
- From the side or tilted: any `approach`. Without `closing`, the fingers close
  horizontally.
- `tcp_from_link6` gives the TCP and both axes of a measured or planned link6 pose.

The returned `pose` is the link6 pose. Format it for a motion tool with `format_target`.

## Check the whole plan before moving

The wrist cannot turn far. `robodojo_ik` returns the joint limits (`joint_limits_rad`).
The last wrist joint turns only about ±100° and the one before it about ±85°. An object
grasped one way can only be turned as far as these joints allow. A grasp must therefore
suit the place as well as the pick.

1. Write out every pose of the motion for one candidate grasp: pre-grasp (backed off
   along the approach), grasp, lift, carry, pre-place, place.
   - While the object is held, it moves with link6. To get the link6 pose that puts the
     object at its place pose, use `relative_pose` from the object's pose to the grasp
     pose, then `compose_pose` of that delta on the object's place pose (`local` both times).
2. Send every pose of several candidates in one `robodojo_ik` call (up to 256 poses),
   with `table_z_m`, the table top's height. Candidates can differ in grasp tilt, finger
   direction, grasp point and arm.
3. Keep a candidate only if every pose has `ok`. A pose is ok when it is reachable, at
   least 0.1 rad from every joint limit, at least 2 cm from the other arm, and the arm
   stays above the table top. `why` names what fails.
4. If no candidate passes, change the plan: use the other arm, set the object down turned
   and grasp it again, or hand it to the other arm.

`robodojo_ik` sees only the robot and the table height, not objects. An ok pose can still
fail to plan: `robodojo_free_space_move` plans its own path and may end in other joint
angles, so preview the move and check the measured pose after each move.
{% if camera_geometry %}

With `robot_spheres=True` (at most 16 poses), each solution carries its body spheres. Ask
for them from Python with `ctx.call`; a direct call shows only their count.
`arm_hits(spheres, points)` counts the depth points the arm would touch. Pass it the
points of the other objects from point-cloud's `objects()`.
{% endif %}

## Close and check

1. Close at the grasp pose and read the opening with `grip(reply, arm, width_m=...)`:
   - about 0 (0.03 or less): the gripper closed on nothing;
   - about 1 (0.97 or more): the fingers did not close; they are on top of something, or
     the object is as wide as the opening;
   - in between: something is held. Compare the gap with the object's width.
2. Lift only 1.5 cm, observe, and check again:
   - the opening is unchanged;
{% if camera_geometry %}
   - `verify(reply, arm, width_m=...)` is ok (it also counts wrist-camera depth points
     between the finger pads);
{% endif %}
   - the object rises with the gripper in the wrist and high views.
3. Lift fully only after that. Otherwise open, back off and grasp again.

While carrying, keep commanding the gripper closed (opening 0). The measured opening of a
held object (say 0.56) is where the fingers stopped, not a command: sent back as
`gripper_opening`, it ends the squeeze and the object slips out. `robodojo_free_space_move`
without `gripper_opening` keeps the last command; `robodojo_step_eef` needs
`gripper_closed: true` on every waypoint, for the holding arm too.

## Python

{% if formal_episode %}
Your carried code imports `grasp.py` from your project folder:
{% else %}
Copy the helper into your project and import it from Python you run in the workspace:

```bash
mkdir -p code/<project> && cp .agents/skills/grasp-planning/grasp.py code/<project>/
```
{% endif %}

```python
import sys
sys.path.insert(0, "code/<project>")
{% if camera_geometry %}
from grasp import hand, link6_target, grip, verify, arm_hits
{% else %}
from grasp import hand, link6_target, grip
{% endif %}
```

{% if agent_formal %}
Keep it in `code/<project>/` so the code you carry keeps it.
{% elif not formal_episode %}
A controller imports it the same way; the registered bundle carries it.
{% endif %}

| Call | Returns |
| --- | --- |
| `hand(reply, arm)` | the measured `tcp`, `approach`, `closing`, `opening` and `gap_m` |
| `link6_target(tcp, approach, closing=None)` | a link6 `{"position", "quaternion_wxyz"}`, the same as `link6_from_tcp` |
| `tcp_of(target)` | `(tcp, approach, closing)` of a link6 pose |
| `grip(reply, arm, width_m=None)` | `state`: `empty`, `stayed_open` or `holding`, with the reason |
{% if camera_geometry %}
| `verify(reply, arm, width_m=None)` | `grip` plus `wrist_points` between the pads: `ok` and the reason |
| `fingers_view(reply, arm)` | the number of wrist-camera depth points between the pads |
| `arm_hits(spheres, points)` | how many points lie inside a solution's spheres |

`verify` and `fingers_view` read the wrist camera's depth through `point_cloud.py` from
[point-cloud](../point-cloud/SKILL.md). Copy it next to `grasp.py`.
{% endif %}
