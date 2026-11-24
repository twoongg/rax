"""Where the dual-X5 gripper holds things, relative to link6; pure numpy, no scene state.

The tool centre point (TCP) is the point between the finger pads. It and the gripper's
dimensions are those of AMP's RoboDojo X5 embodiment (``amp.robots.robodojo.embodiment``),
which reads the same RoboDojo X5 description:

- the TCP lies at TCP_IN_LINK6_M in the link6 frame;
- link6 +x is the approach axis, out of the gripper between the fingers;
- link6 +y is the closing axis, along which the fingers close;
- the fingertips reach TIP_BEYOND_TCP_M past the TCP along the approach axis.
"""

from __future__ import annotations

import numpy as np

TCP_IN_LINK6_M = (0.15007, 0.0, -0.00144)
APPROACH_AXIS = 0  # Column of the link6 rotation.
CLOSING_AXIS = 1
TIP_BEYOND_TCP_M = 0.0075
FINGER_STROKE_M = 0.044  # Each finger; the opening is twice this.
STROKE_M = 2 * FINGER_STROKE_M
MAX_WIDTH_M = 0.085  # The widest object the fingers surely close on.
VERTICAL_COS = 0.98  # An approach within about 11 degrees of vertical needs a closing axis.


def _unit(value, label):
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"{label} must be a finite 3-vector")
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        raise ValueError(f"{label} must be non-zero")
    return vector / norm


def link6_rotation(approach, closing=None):
    """The link6 rotation (3x3) whose approach axis is `approach` and closing axis is
    `closing` made orthogonal to it. Without `closing`, the closing axis is horizontal."""
    x = _unit(approach, "approach")
    if closing is None:
        if abs(x[2]) > VERTICAL_COS:
            raise ValueError(
                "A vertical approach leaves the finger direction open: give closing")
        y = np.cross([0.0, 0.0, 1.0], x)
    else:
        y = _unit(closing, "closing")
        y = y - x * float(x @ y)
        if np.linalg.norm(y) < 1e-6:
            raise ValueError("closing is parallel to approach")
    y /= np.linalg.norm(y)
    return np.column_stack([x, y, np.cross(x, y)])


def link6_from_tcp(tcp_position, approach, closing=None):
    """(link6 position (3,), link6 rotation (3, 3)) that put the TCP at tcp_position with
    the given approach and closing axes."""
    rotation = link6_rotation(approach, closing)
    tcp = np.asarray(tcp_position, dtype=np.float64)
    if tcp.shape != (3,) or not np.isfinite(tcp).all():
        raise ValueError("tcp_position_m must be a finite 3-vector")
    return tcp - rotation @ np.asarray(TCP_IN_LINK6_M), rotation


def tcp_from_link6(position, rotation):
    """(TCP position, approach axis, closing axis) of a link6 pose (position, 3x3 rotation)."""
    rotation = np.asarray(rotation, dtype=np.float64)
    position = np.asarray(position, dtype=np.float64)
    return (position + rotation @ np.asarray(TCP_IN_LINK6_M),
            rotation[:, APPROACH_AXIS].copy(), rotation[:, CLOSING_AXIS].copy())


def _rotation_wxyz(quaternion):
    w, x, y, z = np.asarray(quaternion, dtype=np.float64) / np.linalg.norm(quaternion)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def tcp_positions(positions, quaternions_wxyz):
    """Each arm's TCP (the point between the finger pads) from its link6 pose, in metres;
    None for an arm whose pose is not finite or whose quaternion is zero."""
    result = []
    for position, quaternion in zip(np.asarray(positions, dtype=np.float64).reshape(-1, 3),
                                    np.asarray(quaternions_wxyz, dtype=np.float64).reshape(-1, 4)):
        valid = np.isfinite(position).all() and np.isfinite(quaternion).all() and np.linalg.norm(quaternion) > 1e-6
        result.append(np.round(tcp_from_link6(position, _rotation_wxyz(quaternion))[0], 4).tolist()
                      if valid else None)
    return result


def opening_gap_m(opening):
    """The finger gap (m) of a gripper opening reading (0 closed, 1 open)."""
    return float(np.clip(opening, 0.0, 1.0)) * STROKE_M
