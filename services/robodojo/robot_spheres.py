"""The robot's own body as spheres, from its URDF meshes and measured joints; no scene objects.

Each link's meshes are covered by spheres in the frame of the nearest arm link at or above
it. Links below the end-effector link (the fingers) are swept over their joint limits, so
their spheres hold every finger position. The cover follows AMP's mesh cover
(``amp.geometry.embodiment.spheres``): ARM_CELL_M cells on the arm, GRIPPER_CELL_M on the
gripper.
"""

from __future__ import annotations

import functools
import itertools
import math
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ARM_CELL_M = 0.04
GRIPPER_CELL_M = 0.01
MAX_SWEEP_POSES = 64
_STL_FACET = np.dtype([("normal", "<f4", 3), ("verts", "<f4", (3, 3)), ("attr", "<u2")])


def _floats(text, default):
    if text is None:
        return np.asarray(default, dtype=np.float64)
    values = np.asarray([float(x) for x in text.split()], dtype=np.float64)
    if values.shape != np.shape(default):
        raise ValueError(f"URDF value {text!r} has the wrong length")
    return values


def _origin(element):
    """A URDF <origin xyz rpy> as a 4x4 (rpy is fixed-axis roll, pitch, yaw)."""
    result = np.eye(4)
    if element is not None:
        result[:3, :3] = Rotation.from_euler(
            "xyz", _floats(element.get("rpy"), (0.0, 0.0, 0.0))
        ).as_matrix()
        result[:3, 3] = _floats(element.get("xyz"), (0.0, 0.0, 0.0))
    return result


def joint_motion(kind, axis, value):
    """The 4x4 a joint adds at a joint value, after its origin."""
    result = np.eye(4)
    if kind in ("revolute", "continuous"):
        result[:3, :3] = Rotation.from_rotvec(np.asarray(axis) * value).as_matrix()
    elif kind == "prismatic":
        result[:3, 3] = np.asarray(axis) * value
    elif kind != "fixed":
        raise ValueError(f"Unsupported URDF joint type {kind!r}")
    return result


def load_stl(path):
    """An STL's (vertices, faces), binary or ASCII; one vertex per facet corner."""
    data = Path(path).read_bytes()
    if len(data) >= 84:
        count = struct.unpack_from("<I", data, 80)[0]
        if count and 84 + _STL_FACET.itemsize * count == len(data):
            facets = np.frombuffer(data, dtype=_STL_FACET, offset=84, count=count)
            vertices = facets["verts"].reshape(-1, 3).astype(np.float64)
            return vertices, np.arange(len(vertices)).reshape(-1, 3)
    rows = [
        [float(x) for x in line.split()[1:4]]
        for line in data.decode("utf-8", "replace").splitlines()
        if line.split()[:1] == ["vertex"]
    ]
    if not rows or len(rows) % 3:
        raise ValueError(f"{path}: not a readable STL")
    vertices = np.asarray(rows, dtype=np.float64)
    return vertices, np.arange(len(vertices)).reshape(-1, 3)


def surface_points(vertices, faces, step):
    """The vertices and a grid on every face, no surface point farther than step from one."""
    vertices = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if not len(faces):
        return vertices
    a, b, c = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    edge = np.max(
        [np.linalg.norm(b - a, axis=1), np.linalg.norm(c - b, axis=1),
         np.linalg.norm(a - c, axis=1)],
        axis=0,
    )
    splits = np.maximum(1, np.ceil(edge / step)).astype(np.int64)
    out = [vertices]
    for k in np.unique(splits[splits > 1]):
        chosen = splits == k
        i, j = np.meshgrid(np.arange(k + 1), np.arange(k + 1), indexing="ij")
        keep = i + j <= k
        u, v = i[keep] / k, j[keep] / k
        w = 1.0 - u - v
        points = (w[None, :, None] * a[chosen, None] + u[None, :, None] * b[chosen, None]
                  + v[None, :, None] * c[chosen, None])
        out.append(points.reshape(-1, 3))
    return np.vstack(out)


def cover_points(points, cell, margin):
    """(n, 4) spheres, one per occupied grid cell, each holding its cell's points plus margin."""
    key = np.floor(points / cell).astype(np.int64)
    _, inverse = np.unique(key, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    count = inverse.max() + 1
    weight = np.bincount(inverse, minlength=count).astype(np.float64)
    centres = np.stack(
        [np.bincount(inverse, points[:, k], minlength=count) for k in range(3)], axis=1
    ) / weight[:, None]
    radii = np.zeros(count)
    np.maximum.at(radii, inverse, np.linalg.norm(points - centres[inverse], axis=1))
    return np.hstack([centres, (radii + margin)[:, None]])


def _box_points(size, step):
    half = np.asarray(size, dtype=np.float64) / 2
    axes = [np.linspace(-h, h, max(2, int(math.ceil(2 * h / step)) + 1)) for h in half]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    return grid[(np.abs(np.abs(grid) - half) < 1e-12).any(axis=1)]


def _cylinder_points(radius, length, step):
    turns = max(8, int(math.ceil(2 * math.pi * radius / step)))
    angle = np.linspace(0, 2 * math.pi, turns, endpoint=False)
    z = np.linspace(-length / 2, length / 2, max(2, int(math.ceil(length / step)) + 1))
    rings = [np.c_[r * np.cos(angle), r * np.sin(angle)]
             for r in np.linspace(0, radius, max(2, int(math.ceil(radius / step)) + 1))]
    side = np.c_[np.tile(rings[-1], (len(z), 1)), np.repeat(z, turns)]
    caps = np.vstack([np.c_[ring, np.full(len(ring), h)]
                      for ring in rings for h in (-length / 2, length / 2)])
    return np.vstack([side, caps])


def _mesh_file(filename, urdf):
    """A URDF mesh filename as a path: relative to the URDF, or package:// under its parents."""
    if filename.startswith("file://"):
        return Path(filename[len("file://"):])
    if filename.startswith("package://"):
        relative = Path(filename[len("package://"):])
        for parent in Path(urdf).resolve().parents:
            for candidate in (parent / relative, parent / Path(*relative.parts[1:])):
                if candidate.is_file():
                    return candidate
        return Path(urdf).parent / Path(*relative.parts[1:])
    path = Path(filename)
    return path if path.is_absolute() else Path(urdf).parent / path


def _geometry_points(link, urdf, step):
    """Surface points of a link's visual geometry (else its collision geometry), link frame."""
    for tag in ("visual", "collision"):
        parts = []
        for element in link.findall(tag):
            geometry = element.find("geometry")
            if geometry is None:
                continue
            shape = next(iter(geometry), None)
            if shape is None:
                continue
            if shape.tag == "mesh" and shape.get("filename"):
                path = _mesh_file(shape.get("filename"), urdf)
                if path.suffix.lower() != ".stl" or not path.is_file():
                    continue
                points = surface_points(*load_stl(path), step)
                points = points * _floats(shape.get("scale"), (1.0, 1.0, 1.0))
            elif shape.tag == "box":
                points = _box_points(_floats(shape.get("size"), (0.0, 0.0, 0.0)), step)
            elif shape.tag == "cylinder":
                points = _cylinder_points(
                    float(shape.get("radius")), float(shape.get("length")), step
                )
            elif shape.tag == "sphere":
                radius = float(shape.get("radius"))
                cube = _box_points((2.0, 2.0, 2.0), step / max(radius, 1e-9))
                points = radius * cube / np.linalg.norm(cube, axis=1, keepdims=True)
            else:
                continue
            frame = _origin(element.find("origin"))
            parts.append(points @ frame[:3, :3].T + frame[:3, 3])
        if parts:
            return np.vstack(parts)
    return np.zeros((0, 3))


def _sweep(chain, points, step):
    """Poses of a link in its frame link's frame over the chain's joint limits, and how far a
    point can lie from its nearest sampled position."""
    moving = [i for i, joint in enumerate(chain) if joint["kind"] != "fixed"]
    reach = float(np.linalg.norm(points, axis=1).max()) if len(points) else 0.0
    per_joint = max(2, int(MAX_SWEEP_POSES ** (1.0 / max(1, len(moving)))))
    samples, slack = {}, 0.0
    for i in moving:
        joint = chain[i]
        low, high = joint["limits"]
        if joint["kind"] == "prismatic":
            lever = 1.0
        else:
            lever = reach + sum(
                float(np.linalg.norm(later["origin"][:3, 3]))
                + (max(abs(v) for v in later["limits"]) if later["kind"] == "prismatic" else 0.0)
                for later in chain[i + 1:]
            )
        travel = max(0.0, high - low) * lever
        k = int(min(per_joint, max(2, math.ceil(travel * len(moving) / step) + 1)))
        samples[i] = np.linspace(low, high, k)
        slack += 0.5 * travel / (k - 1)
    poses = []
    for values in itertools.product(*(samples.get(i, (0.0,)) for i in range(len(chain)))):
        pose = np.eye(4)
        for joint, value in zip(chain, values):
            pose = pose @ joint["origin"] @ joint_motion(joint["kind"], joint["axis"], value)
        poses.append(pose)
    return poses, slack


class ArmSpheres:
    """One arm's body spheres: built once from its URDF, placed from measured joints."""

    def __init__(self, urdf, arm_joints, base_link="base_link", tip="link6"):
        root = ET.parse(urdf).getroot()
        links = {link.get("name"): link for link in root.findall("link")}
        joints = {}
        for joint in root.findall("joint"):
            child = joint.find("child").get("link")
            limit = joint.find("limit")
            kind = joint.get("type")
            low, high = (
                (0.0, 0.0) if limit is None or kind == "fixed"
                else (float(limit.get("lower", 0.0)), float(limit.get("upper", 0.0)))
            )
            if kind == "continuous":
                low, high = -math.pi, math.pi
            axis = joint.find("axis")
            joints[child] = {
                "name": joint.get("name"),
                "kind": kind,
                "parent": joint.find("parent").get("link"),
                "origin": _origin(joint.find("origin")),
                "axis": _floats(axis.get("xyz") if axis is not None else None, (1.0, 0.0, 0.0)),
                "limits": (low, high),
            }
        path, link = [tip], tip
        while link != base_link:
            if link not in joints:
                raise ValueError(f"{tip} does not descend from {base_link} in {urdf}")
            link = joints[link]["parent"]
            path.insert(0, link)
        self.path = path
        self.path_joints = [joints[link] for link in path[1:]]
        moving = [joint["name"] for joint in self.path_joints if joint["kind"] != "fixed"]
        if set(moving) != set(arm_joints):
            raise ValueError(f"URDF chain/arm joint mismatch: {moving}, {list(arm_joints)}")
        self.joint_names = list(arm_joints)

        frames, local, radii = [], [], []
        self.links = []
        for name, link in links.items():
            frame, chain = name, []
            while frame not in path:
                if frame not in joints:
                    break
                chain.insert(0, joints[frame])
                frame = joints[frame]["parent"]
            if frame not in path:
                continue  # Not attached below the arm base, such as a world link.
            cell = GRIPPER_CELL_M if frame == tip and name not in path else ARM_CELL_M
            step = cell / 4.0
            points = _geometry_points(link, urdf, step)
            if not len(points):
                continue
            poses, slack = _sweep(chain, points, step)
            swept = np.vstack([points @ T[:3, :3].T + T[:3, 3] for T in poses])
            spheres = cover_points(swept, cell, step + slack)
            frames.extend([path.index(frame)] * len(spheres))
            local.append(spheres[:, :3])
            radii.append(spheres[:, 3])
            self.links.append(name)
        if not local:
            raise ValueError(f"No STL or primitive geometry found for the arm in {urdf}")
        self.frame = np.asarray(frames, dtype=np.int64)
        self.local = np.vstack(local)
        self.radius = np.concatenate(radii)

    def frames(self, q):
        """(len(path), 4, 4): every arm link's pose in the arm base frame."""
        q = np.asarray(q, dtype=np.float64)
        if q.shape != (len(self.joint_names),) or not np.isfinite(q).all():
            raise ValueError("Invalid arm joint vector")
        values = dict(zip(self.joint_names, q))
        out = [np.eye(4)]
        for joint in self.path_joints:
            out.append(out[-1] @ joint["origin"]
                       @ joint_motion(joint["kind"], joint["axis"], values.get(joint["name"], 0.0)))
        return np.stack(out)

    def world(self, root, q):
        """(centres (S, 3), radii (S,)) in root's frame, root being the arm base's pose."""
        frames = np.asarray(root, dtype=np.float64) @ self.frames(q)
        chosen = frames[self.frame]
        centres = np.einsum("sij,sj->si", chosen[:, :3, :3], self.local) + chosen[:, :3, 3]
        return centres, self.radius.copy()


@functools.lru_cache(maxsize=8)
def arm_spheres(urdf, arm_joints, base_link="base_link", tip="link6"):
    """ArmSpheres built once per process for each arm description."""
    return ArmSpheres(urdf, arm_joints, base_link, tip)
