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


def opening_gap_m(opening):
    """The finger gap (m) of a gripper opening reading (0 closed, 1 open)."""
    return float(np.clip(opening, 0.0, 1.0)) * STROKE_M
