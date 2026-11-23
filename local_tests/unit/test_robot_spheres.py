"""The robot's body spheres: built from a URDF's meshes, placed from measured joints.

A small six-joint arm with a two-finger gripper is written as a URDF with boxes, a
cylinder, a sphere and binary and ASCII STL meshes. Surface points sampled independently
of the cover, placed with the existing link6 FK, must all lie inside the spheres, for any
joints and any finger opening.
"""
import math
import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from services.robodojo import robot_spheres as rs
from services.robodojo.kinematics import ArmFK
from services.robodojo.session import RoboDojoSession

JOINTS = tuple(f"joint{i}" for i in range(1, 7))
FINGER_STROKE = 0.044


def _box_triangles(size):
    hx, hy, hz = np.asarray(size) / 2
    v = np.array([[x, y, z] for x in (-hx, hx) for y in (-hy, hy) for z in (-hz, hz)])
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    return np.array([[v[a], v[b], v[c]] for a, b, c, d in quads] +
                    [[v[a], v[c], v[d]] for a, b, c, d in quads])


def write_stl(path, size, *, binary):
    triangles = _box_triangles(size)
    if binary:
        data = bytearray(80) + struct.pack("<I", len(triangles))
        for triangle in triangles:
            data += struct.pack("<3f", 0, 0, 0) + struct.pack("<9f", *triangle.ravel()) + b"\0\0"
        path.write_bytes(bytes(data))
    else:
        lines = ["solid box"]
        for triangle in triangles:
            lines += ["facet normal 0 0 0", "outer loop"]
            lines += [f"vertex {x:.6f} {y:.6f} {z:.6f}" for x, y, z in triangle]
            lines += ["endloop", "endfacet"]
        path.write_text("\n".join(lines + ["endsolid box"]) + "\n")


# Link geometry: (kind, parameters, visual origin xyz, visual origin rpy).
GEOMETRY = {
    "world": ("box", (1.0, 1.0, 0.02), (0, 0, -0.5), (0, 0, 0)),     # Above the base: never covered.
    "base_link": ("box", (0.10, 0.10, 0.05), (0, 0, 0.025), (0, 0, 0)),
    "link1": ("cylinder", (0.03, 0.10), (0, 0, 0.05), (0, 0, 0)),
    "link2": ("box", (0.25, 0.04, 0.04), (0.125, 0, 0), (0, 0, 0)),
    "link3": ("stl_ascii", (0.20, 0.035, 0.035), (0.10, 0, 0), (0, 0, 0.3)),
    "link4": ("sphere", (0.02,), (0, 0, 0), (0, 0, 0)),
    "link5": ("box", (0.05, 0.03, 0.03), (0.025, 0, 0), (0, 0, 0)),
    "link6": ("box", (0.02, 0.05, 0.05), (0.01, 0, 0), (0, 0, 0)),
    "gripper_base": ("stl_binary", (0.03, 0.10, 0.04), (0.015, 0, 0), (0, 0, 0)),
    "link7": ("box", (0.05, 0.01, 0.02), (0.025, 0, 0), (0, 0, 0)),
    "link8": ("box", (0.05, 0.01, 0.02), (0.025, 0, 0), (0, 0, 0)),
}
# Joints: (name, type, parent, child, origin xyz, origin rpy, axis, limits).
TREE = [
    ("mount", "fixed", "world", "base_link", (0.1, 0.2, 0.3), (0, 0, 0.5), None, None),
    ("joint1", "revolute", "base_link", "link1", (0, 0, 0.05), (0, 0, 0), (0, 0, 1), (-1.6, 1.6)),
    ("joint2", "revolute", "link1", "link2", (0, 0, 0.10), (0, 0, 0), (0, 1, 0), (-0.1, 3.6)),
    ("joint3", "revolute", "link2", "link3", (0.25, 0, 0), (0, 0, 0), (0, 1, 0), (-0.1, 3.0)),
    ("joint4", "revolute", "link3", "link4", (0.20, 0, 0), (0.2, 0, 0), (1, 0, 0), (-1.3, 1.3)),
    ("joint5", "revolute", "link4", "link5", (0.02, 0, 0), (0, 0, 0), (0, 1, 0), (-1.5, 1.5)),
    ("joint6", "revolute", "link5", "link6", (0.05, 0, 0), (0, 0, 0), (1, 0, 0), (-1.7, 1.7)),
    ("gripper_mount", "fixed", "link6", "gripper_base", (0.02, 0, 0), (0, 0, 0), None, None),
    ("joint7", "prismatic", "gripper_base", "link7", (0.03, 0.005, 0), (0, 0, 0), (0, 1, 0),
     (0.0, FINGER_STROKE)),
    ("joint8", "prismatic", "gripper_base", "link8", (0.03, -0.005, 0), (0, 0, 0), (0, -1, 0),
     (0.0, FINGER_STROKE)),
]


@pytest.fixture
def urdf(tmp_path):
    return write_arm_urdf(tmp_path)


def write_arm_urdf(tmp_path):
    """The test arm's URDF and meshes under tmp_path; returns the URDF path."""
    (tmp_path / "meshes").mkdir()
    links = []
    for name, (kind, params, xyz, rpy) in GEOMETRY.items():
        if kind.startswith("stl"):
            write_stl(tmp_path / "meshes" / f"{name}.stl", params, binary=kind == "stl_binary")
            shape = f'<mesh filename="package://arm/meshes/{name}.stl"/>' if kind == "stl_binary" \
                else f'<mesh filename="meshes/{name}.stl"/>'
        elif kind == "box":
            shape = f'<box size="{" ".join(map(str, params))}"/>'
        elif kind == "cylinder":
            shape = f'<cylinder radius="{params[0]}" length="{params[1]}"/>'
        else:
            shape = f'<sphere radius="{params[0]}"/>'
        links.append(
            f'<link name="{name}"><visual><origin xyz="{" ".join(map(str, xyz))}" '
            f'rpy="{" ".join(map(str, rpy))}"/><geometry>{shape}</geometry></visual></link>')
    joints = []
    for name, kind, parent, child, xyz, rpy, axis, limits in TREE:
        extra = f'<axis xyz="{" ".join(map(str, axis))}"/>' if axis else ""
        if limits:
            extra += f'<limit lower="{limits[0]}" upper="{limits[1]}" effort="1" velocity="1"/>'
        joints.append(
            f'<joint name="{name}" type="{kind}"><parent link="{parent}"/><child link="{child}"/>'
            f'<origin xyz="{" ".join(map(str, xyz))}" rpy="{" ".join(map(str, rpy))}"/>{extra}</joint>')
    path = tmp_path / "arm.urdf"
    path.write_text('<robot name="arm">' + "".join(links + joints) + "</robot>")
    return path


def _pose(xyz, rpy):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    T[:3, 3] = xyz
    return T


def surface_samples(kind, params, rng, n=4000):
    """Random points on a link's geometry surface, in its visual frame."""
    if kind in ("box", "stl_ascii", "stl_binary"):
        half = np.asarray(params) / 2
        points = rng.uniform(-half, half, (n, 3))
        axis = rng.integers(0, 3, n)
        points[np.arange(n), axis] = half[axis] * rng.choice([-1, 1], n)
        corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * half
        return np.vstack([points, corners])
    if kind == "cylinder":
        radius, length = params
        angle = rng.uniform(0, 2 * math.pi, n)
        r = np.where(rng.random(n) < 0.5, radius, radius * np.sqrt(rng.random(n)))
        z = np.where(r == radius, rng.uniform(-length / 2, length / 2, n),
                     rng.choice([-length / 2, length / 2], n))
        return np.c_[r * np.cos(angle), r * np.sin(angle), z]
    direction = rng.normal(size=(n, 3))
    return params[0] * direction / np.linalg.norm(direction, axis=1, keepdims=True)


def link_poses(urdf, q, fingers):
    """Every covered link's pose in the arm base frame, from the existing chain FK."""
    poses = {"base_link": np.eye(4)}
    for i in range(1, 7):
        poses[f"link{i}"] = ArmFK(urdf, JOINTS[:i], "base_link", f"link{i}").matrix(q[:i])
    poses["gripper_base"] = poses["link6"] @ _pose((0.02, 0, 0), (0, 0, 0))
    for name, (y, sign), value in zip(("link7", "link8"), ((0.005, 1), (-0.005, -1)), fingers):
        poses[name] = poses["gripper_base"] @ _pose((0.03, y + sign * value, 0), (0, 0, 0))
    return poses


def test_the_sphere_model_places_link6_as_the_arm_fk_does(urdf):
    model = rs.ArmSpheres(str(urdf), JOINTS, "base_link", "link6")
    fk = ArmFK(urdf, JOINTS, "base_link", "link6")
    rng = np.random.default_rng(0)
    for _ in range(5):
        q = rng.uniform(-1.2, 1.2, 6)
        assert np.allclose(model.frames(q)[-1], fk.matrix(q), atol=1e-12)
    assert model.path == ["base_link"] + [f"link{i}" for i in range(1, 7)]
    assert set(model.links) == set(GEOMETRY) - {"world"}


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_spheres_hold_every_surface_point_at_any_joints_and_finger_opening(urdf, seed):
    model = rs.ArmSpheres(str(urdf), JOINTS, "base_link", "link6")
    rng = np.random.default_rng(seed)
    q = rng.uniform(-1.2, 1.2, 6)
    fingers = rng.uniform(0, FINGER_STROKE, 2)
    root = _pose((0.4, -0.2, 0.05), (0, 0, 0.7))
    centres, radii = model.world(root, q)
    for name, pose in link_poses(urdf, q, fingers).items():
        kind, params, xyz, rpy = GEOMETRY[name]
        points = surface_samples(kind, params, rng)
        placed = (root @ pose @ _pose(xyz, rpy) @ np.c_[points, np.ones(len(points))].T).T[:, :3]
        gap = np.linalg.norm(placed[:, None, :] - centres[None], axis=2) - radii[None]
        assert gap.min(axis=1).max() <= 1e-9, name


def test_the_gripper_gets_finer_spheres_and_the_cover_stays_small(urdf):
    model = rs.ArmSpheres(str(urdf), JOINTS, "base_link", "link6")
    gripper = np.isin(model.frame, [model.path.index("link6")]) & (model.radius < 0.02)
    assert gripper.sum() > 20
    assert model.radius[model.frame < model.path.index("link6")].max() < 0.05
    assert len(model.radius) < 600  # A reply of a few thousand numbers, not a mesh.


def test_stl_reader_takes_binary_and_ascii(tmp_path):
    for binary in (True, False):
        path = tmp_path / f"box_{binary}.stl"
        write_stl(path, (0.2, 0.1, 0.05), binary=binary)
        vertices, faces = rs.load_stl(path)
        assert faces.shape == (12, 3)
        assert np.allclose(vertices.min(axis=0), (-0.1, -0.05, -0.025), atol=1e-6)


def test_a_chain_that_misses_the_arm_joints_is_refused(urdf):
    with pytest.raises(ValueError, match="joint mismatch"):
        rs.ArmSpheres(str(urdf), JOINTS[:5], "base_link", "link6")


def test_session_reports_both_arms_in_the_environment_frame(urdf):
    roots = {"left": _pose((0.0, 0.3, 0.0), (0, 0, 0)), "right": _pose((0.0, -0.3, 0.0), (0, 0, 0))}
    robot = SimpleNamespace(urdf_path=str(urdf), arm_joints_name=list(JOINTS),
                            base_link="base_link", ee_link_name="link6")
    session = RoboDojoSession.__new__(RoboDojoSession)
    session.kinematics = SimpleNamespace(
        robots={"left": robot, "right": robot},
        fk={arm: ArmFK(urdf, JOINTS, "base_link", "link6") for arm in roots},
        root=lambda arm: roots[arm])
    state = np.zeros(14, dtype=np.float32)
    state[:6] = [0.3, 0.5, 0.4, 0.0, 0.2, 0.0]
    state[7:13] = [-0.3, 0.5, 0.4, 0.0, 0.2, 0.0]
    session.obs = {"states": state}
    session.step_id = 12
    reply = session.robot_spheres()
    spheres = np.asarray(reply["spheres_m"])
    left, right = reply["counts"]["left"], reply["counts"]["right"]
    assert reply["step_id"] == 12 and reply["frame"] == "environment_origin"
    assert spheres.shape == (left + right, 4) and left == right
    model = rs.ArmSpheres(str(urdf), JOINTS, "base_link", "link6")
    expected, radii = model.world(roots["right"], state[7:13].astype(np.float64))
    assert np.abs(spheres[left:, :3] - expected).max() <= 0.0005
    assert (spheres[left:, 3] >= radii - 1e-12).all() and (spheres[left:, 3] - radii).max() <= 0.001


def test_observe_adds_robot_spheres_only_when_asked():
    from services.robodojo.mcp_server import RoboDojoMCP

    calls = []
    replies = {"teacher_observation": {"states": np.zeros(14)},
               "robot_spheres": {"step_id": 7, "spheres_m": [[0.1, 0.2, 0.3, 0.01]]}}

    class FakeSim:
        active = True
        episode_id = "episode"
        step_id = 7

        def request(self, op, **kwargs):
            calls.append(op)
            return replies[op]

    server = RoboDojoMCP.__new__(RoboDojoMCP)
    server.sim = FakeSim()
    server._observation_packet = lambda observation: ({"step_id": 7}, [])
    packet, _ = server.call("robodojo_observe", {})
    assert "robot_spheres" not in packet and calls == ["teacher_observation"]
    packet, _ = server.call("robodojo_observe", {"robot_spheres": True})
    assert packet["robot_spheres"]["spheres_m"] == [[0.1, 0.2, 0.3, 0.01]]
    assert calls[1:] == ["teacher_observation", "robot_spheres"]
    with pytest.raises(ValueError, match="Unknown robodojo_observe arguments"):
        server.call("robodojo_observe", {"spheres": True})
    with pytest.raises(ValueError, match="boolean"):
        server.call("robodojo_observe", {"robot_spheres": "yes"})
    server.observation_profile = "rgb-only"
    with pytest.raises(PermissionError, match="depth observations"):
        server.call("robodojo_observe", {"robot_spheres": True})
    packet, _ = server.call("robodojo_observe", {})
    assert "robot_spheres" not in packet


def test_only_depth_sessions_offer_the_robot_spheres_option():
    from services.mcp_contract import Contract

    def observe(profile):
        tools = Contract(observation_profile=profile).robot_definitions()
        return next(tool for tool in tools if tool["name"] == "robodojo_observe")

    assert observe("rgbd")["inputSchema"]["properties"]["robot_spheres"]["type"] == "boolean"
    assert observe("rgb-only")["inputSchema"]["properties"] == {}
    assert "robot_spheres" in observe("rgbd")["inputSchema"]["properties"]  # Not mutated.
