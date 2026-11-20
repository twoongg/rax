---
name: point-cloud
description: Use when a RoboDojo agent needs a whole object from depth, such as its centre, footprint, top height or yaw, rather than one surface point. Fuses the depth cameras into a point cloud and grows one object from a chosen pixel, in Python.
---

# Point cloud

[Object localization](../object-localization/SKILL.md) turns one chosen pixel into one
surface point. `point_cloud.py` in this skill turns every depth pixel into a 3-D point and
grows a whole object from one pixel you choose. It reads only a robot tool's reply (RGB,
metric depth and calibration), and its positions are in the environment frame that robot
commands use.

## Set up

Copy the helper into your project and import it from Python you run in the workspace:

```bash
mkdir -p code/<project> && cp .agents/skills/point-cloud/point_cloud.py code/<project>/
```

```python
import sys
from api.runtime import Context

sys.path.insert(0, "code/<project>")
from point_cloud import observation, segment, overlay

with Context() as ctx:
{% if multi_env %}
    obs = observation(ctx.env(env_id).call("robodojo_observe"))  # The environment you use.
{% else %}
    obs = observation(ctx.call("robodojo_observe"))
{% endif %}
```

{% if formal_episode %}
Your carried code imports it the same way (`from point_cloud import ...`).
{% elif agent_formal %}
Your code imports it the same way; keep it in `code/<project>/` so the code you carry
keeps it.
{% else %}
A controller imports it the same way (`from point_cloud import ...`); the registered
bundle carries it.
{% endif %}
A motion tool's reply works as well as an observation: it carries the final frame.

## Calls

| Call | Returns |
| --- | --- |
| `observation(reply)` | `obs.rgb[cam]`, `obs.depth[cam]` (metres, 0 = no depth), `obs.cameras` (those with depth), `obs.step_id` |
| `cloud(obs, cameras=None, stride=1, max_depth=2.0, box=None)` | `(xyz, rgb, cam)`: one point per valid depth pixel, its colour, and its camera's index in `obs.cameras` |
| `segment(obs, cam, u, v, radius_mm=11, color_tol=None, cameras=None, box=None, table_z=None)` | one object grown from RGB pixel (u, v); see below |
| `pixel_point(obs, cam, u, v)` | the 3-D point under one pixel |
| `project(obs, cam, xyz)` | `(uv, z)`: pixels and optical depths of 3-D points |
| `overlay(obs, cam, xyz)` | that camera's RGB image with the points painted magenta |
| `table_height(xyz)` | `(z, count)` of the largest horizontal surface |

`box` is `((x0, x1), (y0, y1), (z0, z1))` in metres. `cameras` is a name or a list and
defaults to every camera with depth.

`segment` returns a dict: `centroid`, `top` (centre of the top face of its axis-aligned
box), `top_z`, `bottom_z`, `size`, `aabb`, `obb`, `points`, `rgb`, `n_points`, `seed`,
`table_z` and `cameras`. `obb` is the smallest upright box around the points:
`{"center", "size": [longer side, shorter side, height], "yaw_deg"}`.

## Segment an object

1. Observe, view the image, and choose a pixel well inside the object, away from its
   edges.
2. `part = segment(obs, cam, u, v)`. It drops the table (everything below `table_z` plus
   5 mm, with `table_z` measured in the `cam_high` cloud), then joins points that lie within
   `radius_mm` of each other.
3. Check the result before using it. Save `overlay(obs, cam, part["points"])` as a PNG
   under `experiments/` and view it: the magenta must cover the object and nothing else.
   Compare `part["size"]` with what the image shows.
4. If it is wrong:
   - It grew into a touching object or the gripper: `color_tol=0.08` stops at colour
     edges, `box` cuts by position, and a smaller `radius_mm` stops at gaps.
   - It split or has few points: use a larger `radius_mm`, or add a view.
   - `table_z` is wrong (a large flat object or the floor won): pass `table_z`.
5. After any motion, observe and segment again. Wrist-camera poses and the scene change.

## Read the numbers with care

- Only visible surfaces are measured. `centroid` is the mean of the visible points, so it
  leans toward the cameras. When the top is visible, `top` and `obb` describe the footprint
  better. Fusing `cam_high` with a close wrist view adds more sides.
- The bottom 5 mm (`table_margin_m`) is cut away with the table, so `bottom_z` and the
  height are about that much off.
- `obb["yaw_deg"]` is the yaw of the longer side, in [-90, 90). A box has no front, so
  decide from the image which end is which.
- The robot is not removed: a held object includes the fingers that touch it.
- `cam_high` resolves about 2–3.5 mm per pixel, and a wrist camera 0.15–0.3 m away about
  0.4–0.8 mm. Depth comes in whole millimetres. Holes, slots and edges smaller than a few
  millimetres do not show up as segments of their own. Find such details in a close wrist
  image, and measure them against a nearby object in that same image.
- A call takes about a second or less. `stride=2` in `cloud` cuts the points by four.
