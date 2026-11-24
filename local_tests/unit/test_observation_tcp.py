"""Every RoboDojo reply says where each TCP is and what each gripper was last commanded."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from services.robodojo.gripper import tcp_positions
from services.robodojo.mcp_server import RoboDojoMCP
from services.robodojo.session import RoboDojoSession

PROJECT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    'grasp_for_tcp', PROJECT/'agent_skills/control/robodojo/grasp-planning/grasp.py')
grasp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(grasp)

POSITIONS = [[0.1, 0.2, 0.3], [-0.2, -0.1, 0.25]]
QUATERNIONS = [Rotation.from_euler('xyz', [10, 80, -30], degrees=True).as_quat()[[3, 0, 1, 2]].tolist(),
               [1.0, 0.0, 0.0, 0.0]]


def test_tcp_positions_match_grasp_tcp_of():
    tcps = tcp_positions(POSITIONS, QUATERNIONS)
    for tcp, position, quaternion in zip(tcps, POSITIONS, QUATERNIONS):
        expected, _, _ = grasp.tcp_of({'position': position, 'quaternion_wxyz': quaternion})
        assert np.allclose(tcp, expected, atol=1e-4)
    assert np.allclose(tcps[1], [-0.2 + 0.15007, -0.1, 0.25 - 0.00144], atol=1e-4)


def session_with_one_camera():
    """A RoboDojoSession whose native environment is a stand-in: one camera, measured grippers 0.25."""
    from services.robodojo import session as module
    session = RoboDojoSession.__new__(RoboDojoSession)
    image = np.zeros((4, 5, 3), np.uint8)
    session.env = SimpleNamespace(
        get_obs=lambda: {'state': {'left_ee_pose': [0, 0, 0, 1, 0, 0, 0], 'right_ee_pose': [0, 0, 0, 1, 0, 0, 0]},
                         'vision': {name: {'color': image} for name in ('cam_high', 'cam_left_wrist', 'cam_right_wrist')},
                         'instruction': 'task'},
        robot_manager=None, camera_manager=None, env_origins=np.zeros((1, 3)))
    session.metadata = {'supports_camera_calibration': False}
    session.step_id = 0
    session._commanded_grippers = None
    measured = np.zeros(14, np.float32)
    measured[[6, 13]] = 0.25
    return session, module, measured


def test_observe_reports_measured_grippers_before_any_command(monkeypatch):
    session, module, measured = session_with_one_camera()
    monkeypatch.setattr(module, 'measured_joint_state', lambda manager: measured)
    session._observe()
    assert np.allclose(session.obs['gripper_commands'], [0.25, 0.25])


def test_observe_reports_the_last_command_after_one(monkeypatch):
    session, module, measured = session_with_one_camera()
    monkeypatch.setattr(module, 'measured_joint_state', lambda manager: measured)
    session._commanded_grippers = np.array([0.0, 1.0], np.float32)
    session._observe()
    assert np.allclose(session.obs['gripper_commands'], [0.0, 1.0])


def test_packet_carries_tcp_and_commands(tmp_path):
    server = RoboDojoMCP.__new__(RoboDojoMCP)
    server.sim = SimpleNamespace(episode_id='episode', step_id=3, task='task', metadata={},
                                 run_id='run')
    server._performance_cost = lambda: {'total_interaction_steps': 3}
    rgb = np.zeros((4, 5, 3), np.uint8)
    observation = {'cam_high': rgb, 'cam_left_wrist': rgb, 'cam_right_wrist': rgb,
                   'states': np.zeros(14, np.float32), 'eef_positions': np.asarray(POSITIONS, np.float32),
                   'eef_quaternions_wxyz': np.asarray(QUATERNIONS, np.float32),
                   'gripper_commands': np.asarray([0.0, 1.0], np.float32),
                   'camera_parameters': {}, 'instruction': 'task'}
    packet, _ = server._observation_packet(observation)
    assert np.allclose(packet['tcp_positions'], tcp_positions(POSITIONS, QUATERNIONS), atol=1e-4)
    assert packet['gripper_commands'] == pytest.approx([0.0, 1.0])


def test_an_arm_without_a_valid_pose_has_no_tcp():
    tcps = tcp_positions([[0.0, 0.0, 0.0], [0.0, 0.0, float('nan')]], [[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    assert tcps == [None, None]
