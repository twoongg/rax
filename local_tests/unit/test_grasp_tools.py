"""Grasp tools: the gripper's TCP relative to link6, batch IK with limits and spheres, and
the grasp skill's helper.

The IK runs on the six-joint test arm of test_robot_spheres (its URDF written with boxes,
a cylinder, a sphere and STL meshes). Targets come from the existing link6 FK at random
joints inside the limits, so every one is reachable.
"""
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from local_tests.unit.test_robot_spheres import JOINTS, TREE, _pose, write_arm_urdf
from services.robodojo import gripper
from services.robodojo import robot_spheres as rs
from services.robodojo.arm_ik import ArmIK
from services.robodojo.kinematics import ArmFK, pose as pose_dict
from services.robodojo.session import (
    MAX_IK_SOLUTIONS, MAX_IK_SPHERE_TARGETS, MAX_IK_TARGETS, RoboDojoSession, ik_problems,
)

PROJECT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "grasp", PROJECT / "agent_skills/control/robodojo/grasp-planning/grasp.py")
grasp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(grasp)

LIMITS = np.array([limits for name, *_, limits in TREE if name in JOINTS])
ROOTS = {"left": _pose((0.0, 0.3, 0.0), (0, 0, 0)), "right": _pose((0.0, -0.3, 0.0), (0, 0, 0))}


@pytest.fixture
def urdf(tmp_path):
    return write_arm_urdf(tmp_path)


def _rotation(quaternion_wxyz):
    q = np.asarray(quaternion_wxyz, dtype=float)
    return Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()


# ---------------------------------------------------------------- the TCP

def test_tcp_lies_15_cm_along_link6_x_and_round_trips():
    position, rotation = gripper.link6_from_tcp([0.3, 0.1, 0.05], [0, 0, -1], [0, 1, 0])
    assert np.allclose(rotation[:, 0], [0, 0, -1]) and np.allclose(rotation[:, 1], [0, 1, 0])
    assert np.isclose(np.linalg.det(rotation), 1.0)
    # link6 z = approach x closing = [1, 0, 0], so the TCP's -1.44 mm along it shifts x.
    assert np.allclose(position, [0.3 + 0.00144, 0.1, 0.05 + 0.15007])
    tcp, approach, closing = gripper.tcp_from_link6(position, rotation)
    assert np.allclose(tcp, [0.3, 0.1, 0.05]) and np.allclose(approach, [0, 0, -1])
    assert np.isclose(np.linalg.norm(position - tcp), math.hypot(0.15007, 0.00144))


def test_a_horizontal_closing_axis_is_the_default_and_a_vertical_approach_needs_one():
    _, rotation = gripper.link6_from_tcp([0, 0, 0], [1, 1, 0])
    assert abs(rotation[2, 1]) < 1e-12  # Closing axis horizontal.
    _, tilted = gripper.link6_from_tcp([0, 0, 0], [1, 0, -1], [0, 0, 1])
    assert np.allclose(tilted[:, 1] @ tilted[:, 0], 0.0)  # Made perpendicular.
    with pytest.raises(ValueError, match="give closing"):
        gripper.link6_from_tcp([0, 0, 0], [0, 0, -1])
    with pytest.raises(ValueError, match="parallel"):
        gripper.link6_from_tcp([0, 0, 0], [0, 0, -1], [0, 0, 2])


def test_the_skill_helper_matches_the_server_tcp():
    assert grasp.TCP_IN_LINK6_M == gripper.TCP_IN_LINK6_M
    assert grasp.TIP_BEYOND_TCP_M == gripper.TIP_BEYOND_TCP_M
    assert grasp.STROKE_M == gripper.STROKE_M and grasp.MAX_WIDTH_M == gripper.MAX_WIDTH_M
    rng = np.random.default_rng(0)
    for _ in range(20):
        tcp, approach, closing = rng.normal(size=(3, 3))
        target = grasp.link6_target(tcp, approach, closing)
        position, rotation = gripper.link6_from_tcp(tcp, approach, closing)
        assert np.allclose(target["position"], position)
        assert np.allclose(_rotation(target["quaternion_wxyz"]), rotation)
        back, a, c = grasp.tcp_of(target)
        assert np.allclose(back, tcp) and np.allclose(a, approach / np.linalg.norm(approach))


@pytest.mark.upstream
def test_pose_math_converts_between_tcp_and_link6(isaac_math_root):
    from services.robodojo.pose_math import pose_math

    made = pose_math(isaac_math_root, {
        "operation": "link6_from_tcp", "tcp_position_m": [0.3, -0.1, 0.04],
        "approach": [0.5, 0, -1], "closing": [0, 1, 0]})
    position, rotation = gripper.link6_from_tcp([0.3, -0.1, 0.04], [0.5, 0, -1], [0, 1, 0])
    assert np.allclose(made["pose"]["position_m"], position)
    assert np.allclose(_rotation(made["pose"]["quaternion_wxyz"]), rotation)
    assert made["tcp_in_link6_m"] == list(gripper.TCP_IN_LINK6_M)
    back = pose_math(isaac_math_root, {"operation": "tcp_from_link6", "pose": {
        "position_m": made["pose"]["position_m"], "rotation": made["pose"]["rotation"]}})
    assert np.allclose(back["tcp_position_m"], [0.3, -0.1, 0.04])
    assert np.allclose(back["approach"], np.array([0.5, 0, -1]) / math.hypot(0.5, 1))
    with pytest.raises(ValueError, match="give closing"):
        pose_math(isaac_math_root, {"operation": "link6_from_tcp",
                                    "tcp_position_m": [0, 0, 0], "approach": [0, 0, -1]})


# ---------------------------------------------------------------- batch IK

def _solver(urdf):
    model = rs.ArmSpheres(str(urdf), JOINTS, "base_link", "link6")
    return model, ArmIK(model.path_joints, JOINTS, LIMITS)


def test_ik_fk_matches_the_arm_fk(urdf):
    _, ik = _solver(urdf)
    fk = ArmFK(urdf, JOINTS, "base_link", "link6")
    q = np.random.default_rng(1).uniform(LIMITS[:, 0], LIMITS[:, 1], (10, 6))
    poses, _ = ik.fk(q)
    assert np.abs(poses - np.stack([fk.matrix(row) for row in q])).max() < 1e-12


def test_ik_reaches_targets_made_from_joints_inside_the_limits(urdf):
    _, ik = _solver(urdf)
    fk = ArmFK(urdf, JOINTS, "base_link", "link6")
    q = np.random.default_rng(2).uniform(LIMITS[:, 0] + 0.2, LIMITS[:, 1] - 0.2, (40, 6))
    targets = np.stack([fk.matrix(row) for row in q])
    solutions = ik.solve(targets, np.zeros(6))
    for target, (best,) in zip(targets, solutions):
        assert best["accurate"]
        assert best["position_error_mm"] < 0.5 and best["rotation_error_deg"] < 0.1
        reached = fk.matrix(best["joints"])
        assert np.linalg.norm(reached[:3, 3] - target[:3, 3]) < 5e-4
        assert ((best["joints"] >= LIMITS[:, 0]) & (best["joints"] <= LIMITS[:, 1])).all()
        assert np.isclose(best["limit_margin_rad"],
                          np.minimum(best["joints"] - LIMITS[:, 0], LIMITS[:, 1] - best["joints"]).min())


def test_ik_reports_a_target_out_of_reach_with_its_error(urdf):
    _, ik = _solver(urdf)
    far = _pose((2.0, 0.0, 0.3), (0, 0, 0))
    (best,), = ik.solve(far[None], np.zeros(6))
    assert not best["accurate"] and best["position_error_mm"] > 1000


def test_ik_ranks_distinct_solutions_nearest_the_seed_first(urdf):
    _, ik = _solver(urdf)
    fk = ArmFK(urdf, JOINTS, "base_link", "link6")
    seed = np.array([0.3, 0.8, 1.0, 0.2, 0.4, -0.3])
    (solutions,) = ik.solve(fk.matrix(seed)[None], seed, max_solutions=4)
    assert solutions[0]["accurate"] and solutions[0]["distance_from_seed_rad"] < 1e-3
    accurate = [s for s in solutions if s["accurate"]]
    assert [s["distance_from_seed_rad"] for s in accurate] == sorted(
        s["distance_from_seed_rad"] for s in accurate)
    for i, a in enumerate(solutions):
        for b in solutions[i + 1:]:
            assert np.abs(a["joints"] - b["joints"]).max() >= 0.05


# ---------------------------------------------------------------- the session's ik op

def _session(urdf, left=(0.3, 0.5, 0.4, 0.0, 0.2, 0.0), right=(-0.3, 0.5, 0.4, 0.0, 0.2, 0.0),
             sim_limits=LIMITS):
    robot = SimpleNamespace(urdf_path=str(urdf), arm_joints_name=list(JOINTS),
                            base_link="base_link", ee_link_name="link6")
    session = RoboDojoSession.__new__(RoboDojoSession)
    session.kinematics = SimpleNamespace(
        robots={"left": robot, "right": robot},
        fk={arm: ArmFK(urdf, JOINTS, "base_link", "link6") for arm in ROOTS},
        root=lambda arm: ROOTS[arm], limits=lambda arm: np.array(sim_limits, dtype=float))
    state = np.zeros(14, dtype=np.float32)
    state[:6], state[7:13] = left, right
    session.obs = {"states": state}
    session.step_id = 9
    return session


def _target(urdf, arm, q):
    return pose_dict(ROOTS[arm] @ ArmFK(urdf, JOINTS, "base_link", "link6").matrix(q))


def test_session_ik_answers_in_the_environment_frame_with_limits_and_clearances(urdf):
    session = _session(urdf)
    q = np.array([0.2, 0.9, 0.8, 0.1, 0.5, 0.3])
    reply = session.ik("left", [_target(urdf, "left", q), {
        "position": [3.0, 0.0, 0.0], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]}])
    assert reply["arm"] == "left" and reply["step_id"] == 9 and reply["physical_steps"] == 0
    assert reply["joint_limits_rad"] == np.round(LIMITS, 4).tolist()
    assert reply["reachable_count"] == 1 and reply["ok_count"] == 1
    first, second = reply["results"]
    assert first["reachable"] and not second["reachable"]
    assert first["ok"] and first["why"] == "ok"
    assert not second["ok"] and second["why"].startswith("unreachable")
    best = first["solutions"][0]
    assert best["ok"] and best["why"] == "ok"
    assert "robot_spheres" not in best
    model = rs.ArmSpheres(str(urdf), JOINTS, "base_link", "link6")
    centres, radii = model.world(ROOTS["left"], np.asarray(best["joints"]))
    # The base and the vertical first joint's link never change height: they are left out.
    moving = model.frame >= model.path.index("link2")
    assert math.isclose(best["lowest_z_m"], (centres[moving, 2] - radii[moving]).min(), abs_tol=1e-4)
    other, other_r = model.world(ROOTS["right"], np.array([-0.3, 0.5, 0.4, 0.0, 0.2, 0.0]))
    gap = (np.linalg.norm(centres[:, None] - other[None], axis=2) - radii[:, None] - other_r[None]).min()
    assert math.isclose(best["other_arm_clearance_m"], gap, abs_tol=1e-4)


def test_session_ik_keeps_to_the_urdf_limits_where_the_simulator_allows_more(urdf):
    # RoboDojo's simulator reports about +-10 rad for most X5A joints; the URDF holds the
    # arm's real range, so IK must not plan past it.
    loose = np.array([[-10.0, 10.0]] * 5 + [[-3.14, 3.14]])
    session = _session(urdf, sim_limits=loose)
    q = np.array([0.2, 0.9, 0.8, 0.1, 0.5, 0.3])
    reply = session.ik("left", [_target(urdf, "left", q)], max_solutions=4)
    assert reply["joint_limits_rad"] == np.round(LIMITS, 4).tolist()
    for solution in reply["results"][0]["solutions"]:
        joints = np.asarray(solution["joints"])
        assert ((joints >= LIMITS[:, 0] - 1e-9) & (joints <= LIMITS[:, 1] + 1e-9)).all()
        margin = np.minimum(joints - LIMITS[:, 0], LIMITS[:, 1] - joints).min()
        assert math.isclose(solution["limit_margin_rad"], margin, abs_tol=1e-4)
    # A simulator limit tighter than the URDF's still applies.
    tight = LIMITS.copy()
    tight[5] = (-0.5, 0.5)
    reply = _session(urdf, sim_limits=tight).ik("left", [_target(urdf, "left", q)])
    assert reply["joint_limits_rad"][5] == [-0.5, 0.5]


def test_session_ik_puts_ok_solutions_first_and_checks_the_table(urdf):
    session = _session(urdf)
    # joint5 at zero lines joint4 up with joint6, so the pose has many joint solutions; the
    # seed's own one is 0.05 rad from joint6's limit and is not ok.
    q = np.array([0.2, 0.9, 0.8, 0.0, 0.0, 1.65])
    reply = session.ik("left", [_target(urdf, "left", q)], seed_joints=q.tolist(),
                       max_solutions=4)
    result = reply["results"][0]
    solutions = result["solutions"]
    assert result["reachable"] and result["ok"] and solutions[0]["ok"]
    assert [s["ok"] for s in solutions] == sorted((s["ok"] for s in solutions), reverse=True)
    seed_own = next(s for s in solutions if s["distance_from_seed_rad"] < 1e-3)
    assert not seed_own["ok"] and "joint limit" in seed_own["why"]

    target = _target(urdf, "left", np.array([0.2, 0.9, 0.8, 0.1, 0.5, 0.3]))
    low = session.ik("left", [target], table_z_m=-5.0)
    assert low["results"][0]["ok"] and low["checks"]["table_z_m"] == -5.0
    high = session.ik("left", [target], table_z_m=5.0)
    assert not high["results"][0]["ok"] and "table top" in high["results"][0]["why"]
    assert high["ok_count"] == 0
    for bad in ("0.7", True, float("nan")):
        with pytest.raises(ValueError, match="table_z_m"):
            session.ik("left", [target], table_z_m=bad)


def test_ik_problems_names_what_is_wrong_with_each_solution():
    base = {"position_error_mm": 0.1, "rotation_error_deg": 0.0, "limit_margin_rad": 0.5,
            "lowest_z_m": 0.1, "other_arm_clearance_m": 0.1}
    assert ik_problems(base, True, table_z_m=0.0) == []
    assert "unreachable" in ik_problems(dict(base, position_error_mm=40.0), False)[0]
    assert "joint limit" in ik_problems(dict(base, limit_margin_rad=0.02), True)[0]
    assert "other arm" in ik_problems(dict(base, other_arm_clearance_m=0.005), True)[0]
    assert "table top" in ik_problems(dict(base, lowest_z_m=0.003), True, table_z_m=0.0)[0]
    assert ik_problems(dict(base, lowest_z_m=-0.01), True) == []  # No table_z_m: not judged.


def test_session_ik_returns_spheres_only_when_asked_and_checks_its_arguments(urdf):
    session = _session(urdf)
    target = _target(urdf, "right", np.array([-0.2, 0.7, 0.9, 0.0, 0.3, 0.0]))
    reply = session.ik("right", [target], robot_spheres=True, max_solutions=2)
    spheres = np.asarray(reply["results"][0]["solutions"][0]["robot_spheres"])
    assert spheres.shape[1] == 4 and len(spheres) == len(rs.ArmSpheres(
        str(urdf), JOINTS, "base_link", "link6").radius)
    with pytest.raises(ValueError, match="left or right"):
        session.ik("middle", [target])
    with pytest.raises(ValueError, match="targets"):
        session.ik("left", [])
    with pytest.raises(ValueError, match="at most"):
        session.ik("left", [target] * (MAX_IK_SPHERE_TARGETS + 1), robot_spheres=True)
    with pytest.raises(ValueError, match=r"targets\[0\]"):
        session.ik("left", [{"position": [0, 0, 0]}])
    with pytest.raises(ValueError, match="seed_joints"):
        session.ik("left", [target], seed_joints=[0.0] * 5)
    with pytest.raises(ValueError, match="max_solutions"):
        session.ik("left", [target], max_solutions=MAX_IK_SOLUTIONS + 1)
    with pytest.raises(ValueError, match="times max_solutions"):
        session.ik("left", [target] * (MAX_IK_TARGETS // 2 + 1), max_solutions=2)


def test_the_tool_schema_matches_the_session_limits():
    from services.robodojo.mcp_server import IK_ARGUMENTS, RoboDojoMCP

    tool = next(t for t in RoboDojoMCP.tool_definitions() if t["name"] == "robodojo_ik")
    properties = tool["inputSchema"]["properties"]
    assert set(properties) == IK_ARGUMENTS
    assert properties["targets"]["maxItems"] == MAX_IK_TARGETS
    assert properties["max_solutions"]["maximum"] == MAX_IK_SOLUTIONS
    assert str(MAX_IK_SPHERE_TARGETS) in properties["robot_spheres"]["description"]


def test_the_ik_tool_forwards_to_the_simulator_and_keeps_spheres_to_depth_sessions():
    from services.benchmarks import BENCHMARKS
    from services.mcp_contract import Contract
    from services.robodojo.mcp_server import RoboDojoMCP

    calls = []

    class FakeSim:
        active = True
        episode_id = "episode"
        step_id = 4

        def request(self, op, **kwargs):
            calls.append((op, kwargs))
            return {"step_id": 4, "results": [{"reachable": True, "solutions": [
                {"joints": [0.0] * 6, "robot_spheres": [[0, 0, 0, 0.01]]}]}]}

    server = RoboDojoMCP.__new__(RoboDojoMCP)
    server.sim = FakeSim()
    target = {"position": [0.3, 0.1, 0.2], "quaternion_wxyz": [1, 0, 0, 0]}
    reply, images = server.call("robodojo_ik", {"arm": "left", "targets": [target],
                                                "robot_spheres": True})
    assert images == [] and calls == [("ik", {"arm": "left", "targets": [target],
                                              "robot_spheres": True})]
    assert reply["results"][0]["solutions"][0]["robot_spheres"] == [[0, 0, 0, 0.01]]
    with pytest.raises(ValueError, match="Unknown robodojo_ik arguments"):
        server.call("robodojo_ik", {"arm": "left", "targets": [target], "spheres": True})
    server.observation_profile = "rgb-only"
    with pytest.raises(PermissionError, match="depth observations"):
        server.call("robodojo_ik", {"arm": "left", "targets": [target], "robot_spheres": True})
    reply, _ = server.call("robodojo_ik", {"arm": "left", "targets": [target]})
    assert "robot_spheres" not in reply["results"][0]["solutions"][0]

    assert "robodojo_ik" in BENCHMARKS["robodojo"].robot_tools
    assert "robodojo_ik" not in BENCHMARKS["robodojo"].mutating

    def ik_tool(profile):
        tools = Contract(observation_profile=profile).robot_definitions()
        return next(tool for tool in tools if tool["name"] == "robodojo_ik")

    assert "robot_spheres" in ik_tool("rgbd")["inputSchema"]["properties"]
    assert "robot_spheres" not in ik_tool("rgb-only")["inputSchema"]["properties"]
    assert "targets" in ik_tool("rgb-only")["inputSchema"]["properties"]


# ---------------------------------------------------------------- the skill helper

def _hand_packet(tcp, approach, closing, opening, arm="left"):
    target = grasp.link6_target(tcp, approach, closing)
    states = [0.0] * 14
    states[6 + 7 * grasp.ARMS[arm]] = opening
    positions = [[0.0, 0.3, 0.9], [0.0, -0.3, 0.9]]
    quaternions = [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
    positions[grasp.ARMS[arm]] = target["position"]
    quaternions[grasp.ARMS[arm]] = target["quaternion_wxyz"]
    return {"eef_positions": positions, "eef_quaternions_wxyz": quaternions, "states": states}


def _reply(packet):
    return {"content": [{"type": "text", "text": json.dumps(packet)}]}


def test_hand_and_grip_read_the_measured_gripper():
    packet = _hand_packet([0.3, 0.1, 0.05], [0, 0, -1], [1, 0, 0], 0.4)
    h = grasp.hand(_reply(packet), "left")
    assert np.allclose(h["tcp"], [0.3, 0.1, 0.05], atol=1e-4)
    assert np.allclose(h["approach"], [0, 0, -1]) and np.allclose(h["closing"], [1, 0, 0])
    assert math.isclose(h["gap_m"], 0.4 * 0.088, abs_tol=1e-4)
    assert grasp.grip(packet, "left")["state"] == "holding"
    assert grasp.grip(_hand_packet([0, 0, 0], [1, 0, 0], None, 0.01), "left")["state"] == "empty"
    assert grasp.grip(_hand_packet([0, 0, 0], [1, 0, 0], None, 0.99), "left")["state"] == "stayed_open"
    far = grasp.grip(packet, "left", width_m=0.06)
    assert far["state"] == "holding" and "60 mm wide" in far["why"]
    near = grasp.grip(packet, "left", width_m=0.036)
    assert "wide" not in near["why"]


def test_arm_hits_counts_points_inside_the_spheres():
    spheres = [[0.0, 0.0, 0.0, 0.05], [0.2, 0.0, 0.0, 0.02]]
    points = np.array([[0.01, 0, 0], [0.04, 0, 0], [0.1, 0, 0], [0.21, 0, 0], [0.5, 0, 0]])
    assert grasp.arm_hits(spheres, points) == 3
    assert grasp.arm_hits(spheres, points, margin_m=0.06) == 4
    assert grasp.arm_hits([], points) == 0


def test_fingers_view_counts_depth_points_between_the_pads(monkeypatch):
    import sys

    import local_tests.unit.test_point_cloud as tp

    monkeypatch.setitem(sys.modules, "point_cloud", tp.pc)  # As if copied next to grasp.py.

    grasp_point = np.array([0.0, 0.0, tp.TABLE_Z - tp.ORIGIN[2] + 0.02])  # Environment frame.
    wrist = {"cam_left_wrist": tp.look_at(tp.ORIGIN + [0.02, 0.0, tp.TABLE_Z + 0.25],
                                          tp.ORIGIN + [0.0, 0.0, tp.TABLE_Z])}
    block = tp.box((0.0, 0.0), (0.03, 0.03, 0.04), 0.0, (200, 30, 30))

    def reply(boxes, opening):
        base = tp.reply(boxes, cameras=wrist)
        packet = json.loads(base["content"][0]["text"])
        packet.update(_hand_packet(grasp_point, [0, 0, -1], [0, 1, 0], opening))
        base["content"][0]["text"] = json.dumps(packet)
        return base

    holding = reply([block], 0.5)
    assert grasp.fingers_view(holding, "left") > 50
    assert grasp.fingers_view(reply([], 0.5), "left") == 0
    assert grasp.fingers_view(reply([block], 0.1), "left") is None  # Too narrow to tell.
    assert grasp.verify(holding, "left", width_m=0.03)["ok"]
    air = grasp.verify(reply([], 0.05), "left", width_m=0.03)
    assert not air["ok"] and "nothing between the pads" in air["why"]


def test_verify_without_depth_uses_the_opening_alone():
    packet = _hand_packet([0.3, 0.1, 0.05], [0, 0, -1], [1, 0, 0], 0.4)
    result = grasp.verify(_reply(packet), "left")
    assert result["ok"] and result["wrist_points"] is None
