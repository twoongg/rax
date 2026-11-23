"""Grasping with the dual ARX X5: where the gripper holds and reading a close.

Uses only robot tool replies (the measured link6 poses and gripper openings,
`robodojo_ik` results and, with depth, the wrist camera's depth image). Every position
is in the environment frame (metres) that robot commands use.

    from grasp import hand, link6_target, grip

    reply = ctx.call("robodojo_observe")
    print(hand(reply, "left"))                       # TCP, approach and closing axes, gap
    grasp = link6_target([0.30, 0.10, 0.02], approach=[0, 0, -1], closing=[0, 1, 0])
    plan = ctx.call("robodojo_ik", arm="left", targets=[grasp], table_z_m=0.0)
    # Each of plan's "results" has "ok" and "why".

The gripper's tool centre point (TCP), axes and dimensions are those of the `hand`,
`top_down`/`pose_from_axes` and `verify_grasp`/`fingers_view` calls of AMP's robot SDK
for this robot. Needs numpy and scipy; `fingers_view` and `arm_hits` also need
point_cloud.py (depth only). Copy this file into your project folder so registered
bundles carry it.
"""
import json

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

TCP_IN_LINK6_M = (0.15007, 0.0, -0.00144)  # The point between the finger pads, link6 frame.
TIP_BEYOND_TCP_M = 0.0075   # The fingertips reach this far past the TCP along the approach.
STROKE_M = 0.088            # The full opening, both fingers.
MAX_WIDTH_M = 0.085         # The widest object the fingers surely close on.
EMPTY_OPENING = 0.03        # A close that ends at or below this opening holds nothing.
STAYED_OPEN = 0.97          # A close that ends at or above this did not close at all.
OVERCLOSE_M = 0.022         # Closed this much narrower than expected, with nothing seen: on air.
WIDTH_TOLERANCE_M = 0.013   # A holding gap this far from the object's width is worth a look.
PAD_CLEAR_M = 0.006         # fingers_view counts points this far inside either pad face.
ARMS = {"left": 0, "right": 1}
WRIST = {"left": "cam_left_wrist", "right": "cam_right_wrist"}


def _packet(reply):
    """The JSON packet of a robot tool reply (a dict with a 'content' list), or a packet."""
    if isinstance(reply, dict) and "content" not in reply:
        return reply
    blocks = reply.get("content") if isinstance(reply, dict) else reply
    text = next((b.get("text") for b in blocks or [] if b.get("type") == "text"), None)
    if text is None:
        raise ValueError("Pass the reply of a robot tool: a dict with a 'content' list")
    return json.loads(text)


def _unit(value, label):
    v = np.asarray(value, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(v))
    if not np.isfinite(v).all() or n < 1e-9:
        raise ValueError(f"{label} must be a finite non-zero 3-vector")
    return v / n


def link6_target(tcp, approach, closing=None):
    """The link6 target {"position", "quaternion_wxyz"} that puts the TCP at `tcp` with the
    gripper pointing along `approach` ([0, 0, -1] is straight down) and the fingers
    closing along `closing` (made perpendicular to approach; None: horizontal, which a
    vertical approach cannot use). The same as robodojo_pose_math link6_from_tcp."""
    x = _unit(approach, "approach")
    if closing is None:
        if abs(x[2]) > 0.98:
            raise ValueError("A vertical approach leaves the finger direction open: give closing")
        y = np.cross([0.0, 0.0, 1.0], x)
    else:
        y = _unit(closing, "closing")
        y = y - x * float(x @ y)
        if np.linalg.norm(y) < 1e-6:
            raise ValueError("closing is parallel to approach")
    y /= np.linalg.norm(y)
    rotation = np.column_stack([x, y, np.cross(x, y)])
    position = np.asarray(tcp, dtype=np.float64).reshape(3) - rotation @ TCP_IN_LINK6_M
    q = Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]]
    q = q if q[0] >= 0 else -q
    return {"position": position.tolist(), "quaternion_wxyz": q.tolist()}


def tcp_of(target):
    """(TCP, approach, closing) of a link6 target or pose {"position", "quaternion_wxyz"}."""
    q = np.asarray(target["quaternion_wxyz"], dtype=np.float64)
    rotation = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
    return (np.asarray(target["position"], dtype=np.float64) + rotation @ TCP_IN_LINK6_M,
            rotation[:, 0], rotation[:, 1])


def hand(reply, arm):
    """The arm's measured gripper: {"link6", "quaternion_wxyz", "tcp", "approach",
    "closing", "opening" (0 closed .. 1 open), "gap_m"} from an observation or motion reply.
    gap_m takes the opening as a share of the full stroke; check it once by closing on an
    object of known width."""
    packet = _packet(reply)
    i = ARMS[arm]
    pose = {"position": packet["eef_positions"][i], "quaternion_wxyz": packet["eef_quaternions_wxyz"][i]}
    tcp, approach, closing = tcp_of(pose)
    opening = float(packet["states"][6 + 7 * i])
    return {"link6": pose["position"], "quaternion_wxyz": pose["quaternion_wxyz"],
            "tcp": tcp.round(4).tolist(), "approach": approach.round(4).tolist(),
            "closing": closing.round(4).tolist(), "opening": round(opening, 4),
            "gap_m": round(opening * STROKE_M, 4)}


def grip(reply, arm, *, width_m=None):
    """What a close did, from the measured opening after it: {"state", "opening", "gap_m",
    "why"}. state "empty" (closed on nothing), "stayed_open" (the fingers did not close:
    jammed on top of something, or an object as wide as the opening) or "holding". With
    width_m, the object's width across the fingers, "why" also notes a gap far from it."""
    h = hand(reply, arm)
    opening, gap = h["opening"], h["gap_m"]
    if opening <= EMPTY_OPENING:
        state, why = "empty", f"opening {opening:.2f}: closed on nothing"
    elif opening >= STAYED_OPEN:
        state, why = "stayed_open", f"opening {opening:.2f}: the fingers did not close"
    else:
        state, why = "holding", f"gap {gap * 1000:.0f} mm"
        if width_m is not None and abs(gap - float(width_m)) > WIDTH_TOLERANCE_M:
            why += f", but the object is {float(width_m) * 1000:.0f} mm wide"
    return {"state": state, "opening": opening, "gap_m": gap, "why": why}


def fingers_view(reply, arm, *, camera=None):
    """How many of the arm's wrist-camera depth points lie between the finger pads (in a box
    6 mm inside each pad, 12 mm either side, from 25 mm behind the TCP to 15 mm past it).
    None when the gap is too narrow to tell an object from the pads. Depth only."""
    from point_cloud import cloud, observation

    obs = observation(reply)
    h = hand(_packet(reply), arm)
    half = h["gap_m"] / 2 - PAD_CLEAR_M
    if half < 0.002:
        return None
    xyz = cloud(obs, camera or WRIST[arm], max_depth=0.6)[0].astype(np.float64)
    tcp, approach, closing = (np.asarray(h[k]) for k in ("tcp", "approach", "closing"))
    local = xyz - tcp
    a, c = local @ approach, local @ closing
    w = local @ np.cross(approach, closing)
    return int(np.sum((np.abs(c) < half) & (np.abs(w) < 0.012) & (a > -0.025) & (a < 0.015)))


def verify(reply, arm, *, width_m=None, min_points=15):
    """Is something held? grip() plus, with depth, fingers_view(): {"ok", "why", "state",
    "opening", "gap_m", "wrist_points"}. ok needs "holding" and, when the fingers closed much
    narrower than width_m (or nearly shut without width_m), points seen between the pads."""
    g = grip(reply, arm, width_m=width_m)
    try:
        points = fingers_view(reply, arm)
    except (ImportError, ValueError):
        points = None  # No depth (rgb-only) or no point_cloud.py next to this file.
    seen = points is not None and points >= min_points
    if width_m is not None:
        overclosed = float(width_m) - g["gap_m"] > OVERCLOSE_M and not seen
    else:
        overclosed = g["opening"] <= 0.15 and points == 0
    ok = g["state"] == "holding" and not overclosed
    why = g["why"] + ("; closed past the object with nothing between the pads" if overclosed else "")
    return {"ok": ok, "why": why, "state": g["state"], "opening": g["opening"],
            "gap_m": g["gap_m"], "wrist_points": points}


def arm_hits(spheres, xyz, *, margin_m=0.0):
    """How many points of xyz (N, 3) lie inside a solution's spheres (its "robot_spheres"
    from robodojo_ik(robot_spheres=True)), grown by margin_m. Pass the points of the objects
    the arm must not touch, such as every other object's points from point_cloud.objects()."""
    spheres = np.asarray(spheres, dtype=np.float64).reshape(-1, 4)
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    if not len(spheres) or not len(xyz):
        return 0
    found = cKDTree(xyz).query_ball_point(spheres[:, :3], spheres[:, 3] + margin_m)
    hit = [np.asarray(f, dtype=np.int64) for f in found if f]
    return int(len(np.unique(np.concatenate(hit)))) if hit else 0
