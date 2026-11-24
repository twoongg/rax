---
name: point-cloud
description: Use when a RoboDojo agent or its controller needs whole objects from depth, such as their centre, footprint, top height or yaw, rather than one surface point. Lists every object on the table with no pixel to choose, grows one object from a pixel, and keeps positions relative to an object across layouts, in Python.
---

# Point cloud

[Object localization](../object-localization/SKILL.md) turns one chosen pixel into one
surface point. `point_cloud.py` in this skill turns every depth pixel into a 3-D point,
lists the objects on the table, and grows one object from a pixel you choose. It reads
only a robot tool's reply (RGB, metric depth, calibration, the measured link6 positions
and the robot's own body spheres), and its positions are in the environment frame that
robot commands use.

## Set up

Copy the helper into your project and import it from Python you run in the workspace:

```bash
mkdir -p code/<project> && cp .agents/skills/point-cloud/point_cloud.py code/<project>/
```

```python
import sys
from api.runtime import Context

sys.path.insert(0, "code/<project>")
from point_cloud import observation, objects, segment, overlay

with Context() as ctx:
{% if multi_env %}
    obs = observation(ctx.env(env_id).call("robodojo_observe", robot_spheres=True))  # The environment you use.
{% else %}
    obs = observation(ctx.call("robodojo_observe", robot_spheres=True))
{% endif %}
```

`robot_spheres=True` adds spheres that cover the robot's own body at its measured joints,
built from the robot's model; the helper then removes the robot from the points. Ask for
them whenever you use this helper.

Depth images and robot spheres reach Python only. When you call `robodojo_observe`
yourself, you see the RGB images, the measured state and a count of the spheres, not
the depth or the sphere numbers; this helper reads both from a reply that Python got
with `ctx.call`.

{% if formal_episode %}
Your carried code imports it the same way (`from point_cloud import ...`).
{% elif agent_formal %}
Your code imports it the same way; keep it in `code/<project>/` so the code you carry
keeps it.
{% else %}
A controller imports it the same way (`from point_cloud import ...`); the registered
bundle carries it.
{% endif %}
A motion tool's reply works as well as an observation: it carries the final frame, but
no robot spheres.

## Calls

| Call | Returns |
| --- | --- |
| `observation(reply)` | `obs.rgb[cam]`, `obs.depth[cam]` (metres, 0 = no depth), `obs.eef` (link6 positions), `obs.spheres` (robot body, `[x, y, z, r]` rows, or None), `obs.cameras` (those with depth), `obs.step_id` |
| `objects(obs, cameras=None, radius_mm=11, color_tol=None, box=None, include_robot=False)` | every object on the table, largest footprint first; see below |
| `segment(obs, cam, u, v, radius_mm=11, color_tol=None, cameras=None, box=None, table_z=None)` | one object grown from RGB pixel (u, v) |
| `describe(points, rgb=None, table_z=None)` | the same summary for points you selected, such as a `cloud(..., box=...)` |
| `cloud(obs, cameras=None, stride=1, max_depth=2.0, box=None, drop_robot=False)` | `(xyz, rgb, cam)`: one point per valid depth pixel, its colour, and its camera's index in `obs.cameras` |
| `robot_mask(obs, xyz)` | True for points within 2 cm of the robot's body spheres |
| `profile(item, bin_m=0.005)` | width and top height in slices along the object's long axis, end to end |
| `local(item, xyz)`, `world(item, offsets)` | points in an object's own frame, and back |
| `flip(item)` | the object's box turned 180 degrees, so that its +x end becomes the other end |
| `pixel_point(obs, cam, u, v)` | the 3-D point under one pixel |
| `project(obs, cam, xyz)` | `(uv, z)`: pixels and optical depths of 3-D points |
| `overlay(obs, cam, xyz)` | that camera's RGB image with the points painted magenta |
| `table_height(xyz)` | `(z, count)` of the largest horizontal surface |

`box` is `((x0, x1), (y0, y1), (z0, z1))` in metres. `cameras` is a name or a list and
defaults to every camera with depth.

Each object is a dict: `label` (such as `"red · 6x4x5 cm"`), `color`, `centroid`, `top`
(centre of the top face of its axis-aligned box), `top_z`, `bottom_z`, `size`, `aabb`,
`obb`, `points`, `rgb`, `n_points` and `table_z`. `obb` is the smallest upright box around
the points: `{"center", "size": [longer side, shorter side, height], "yaw_deg"}`.
`objects` adds `index`, `robot` and `supported`; `segment` adds `seed`.

## Find objects

`objects(obs)` needs no pixel. It drops the table (everything below `table_z` plus 5 mm,
with `table_z` measured in the `cam_high` cloud) and, with robot spheres, every point
within 2 cm of the robot's body. It splits the rest wherever parts are more than
`radius_mm` apart, and leaves out parts that float with nothing under them (`supported`
false: a held object, or a piece of an arm) and leftovers of the arm (`robot`). Pass
`include_robot=True` to see them.

- With robot spheres, an object stays an object while a gripper is near or touches it,
  but its points within 2 cm of the gripper are gone: measure it before the gripper
  comes close.
- Without them (a motion tool's reply, or an observation without `robot_spheres=True`),
  `robot` marks every part within 5 cm of a link6 position, so an object under or in a
  gripper disappears with the arm.

- Touching objects come out as one part. `color_tol=0.08` splits them at colour edges.
- A small fragment close to a much larger part of a similar colour joins it, such as a
  thin tab seen apart from its body.
- Without robot spheres, the robot's base or a resting gripper can still appear.
  Recognise them by size and position.

`segment(obs, cam, u, v)` grows a single object from a pixel you choose, with the same rules.
Use it when you are looking at an image yourself.

Check every new use before relying on it. Save `overlay(obs, cam, item["points"])` as a PNG
under `experiments/` and view it: the magenta must cover the object and nothing else.
Compare `label` and `size` with what the image shows.

## In a controller

A controller has no one to choose pixels, and objects move between layouts. In every
episode it must find each object again:

```python
obs = observation(ctx.call("robodojo_observe", robot_spheres=True))
items = objects(obs)
targets = [o for o in items if 0.05 < o["obb"]["size"][0] < 0.07 and o["top_z"] - o["table_z"] < 0.04]
if len(targets) != 1:
    raise RuntimeError(f"expected one target, found {[o['label'] for o in items]}")
target = targets[0]
```

- Choose by properties you measured in exploration: size, height, colour, rough region.
  Never use an `index` or a position from an earlier layout.
- Stop with a clear error when the match is not exactly one. A wrong object is worse.
- Observe and call `objects` again after every motion that may have moved something.

## Ends, and positions on an object

An `obb` has a direction but no front. To tell the ends apart, compare the first and last
rows of `profile(item)`: an end with a handle, tab, pins or a step is narrower or lower.
Then `flip` the box if needed so that its +x end is the end you mean, and keep using that
box.

Features smaller than a few millimetres, such as holes, slots, buttons or a socket, do not
show up as parts. Keep them relative to a larger object that does:

1. In exploration, find the feature in a close wrist view, get its 3-D point with
   `pixel_point` or `robodojo_pixel_to_position`, and store `local(box, point)` in your code.
2. In every episode, find the object, orient its box the same way, and compute
   `world(box, offset)`.
3. Before contact, confirm the result in a close view. Use relative measurements within
   that one image to make the final correction.

## A held object

The robot spheres cover the fingers over their whole stroke, so most of a held object,
everything between the fingers and 2 cm around them, goes with the gripper; what sticks
out beyond the fingertips stays and floats (`supported` false). View the held object with
the other arm's wrist camera and keep the points in a box beyond the fingertips:
`describe(*cloud(obs, cam, box=..., drop_robot=True)[:2])`. Measure where the fingertips
are once, from the empty gripper. `obs.eef` gives link6, not the fingertips.

## Read the numbers with care

- Only visible surfaces are measured. `centroid` is the mean of the visible points, so it
  leans toward the cameras. When the top is visible, `top` and `obb` describe the footprint
  better. Fusing `cam_high` with a close wrist view adds more sides.
- The bottom 5 mm is cut away with the table, so `bottom_z` and the height are about that
  much off. With robot spheres, so is everything within 2 cm of the robot.
- `cam_high` resolves about 2–3.5 mm per pixel, and a wrist camera 0.15–0.3 m away about
  0.4–0.8 mm. Depth comes in whole millimetres.
- A call takes about a second or less. `stride=2` in `cloud` cuts the points by four.
