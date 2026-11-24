"""One no-rollback RoboDojo episode owned by the native simulator process."""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from .camera_calibration import rendered_intrinsics
from services.simulator.io import RejectedBeforeAction, action_execution_scope, reject_unless, require, write_json
from .arm_ik import POSITION_TOLERANCE_MM, ROTATION_TOLERANCE_DEG, ArmIK
from .kinematics import DualKinematics, measured_joint_state, transform, validate_eef_target
from .robot_spheres import arm_spheres
from .trajectory import (
    GOAL_POSITION_TOLERANCE_M,
    GOAL_ROTATION_TOLERANCE_RAD,
    JOINT_TARGET_CONTRACT,
    MAX_TRAJECTORY_ACTIONS,
    policy_actions,
)

CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
ARM_OFFSETS = {"left": 0, "right": 7}
MAX_IK_TARGETS = 256
MAX_IK_SOLUTIONS = 8
MAX_IK_SPHERE_TARGETS = 16
# A robodojo_ik solution is ok when it is accurate and keeps these margins (AMP's grasp checks).
IK_MIN_LIMIT_MARGIN_RAD = 0.1
IK_MIN_OTHER_ARM_CLEARANCE_M = 0.02
IK_ABOVE_TABLE_M = 0.005


def ik_problems(row, accurate, table_z_m=None):
    """What keeps one robodojo_ik solution row from being ok; empty when it is ok."""
    problems = []
    if not accurate:
        problems.append(f"unreachable ({row['position_error_mm']:.0f} mm, "
                        f"{row['rotation_error_deg']:.0f} deg off)")
    if row["limit_margin_rad"] < IK_MIN_LIMIT_MARGIN_RAD:
        problems.append(f"{row['limit_margin_rad']:.2f} rad from a joint limit")
    if row["other_arm_clearance_m"] < IK_MIN_OTHER_ARM_CLEARANCE_M:
        problems.append(f"{row['other_arm_clearance_m'] * 100:.1f} cm from the other arm")
    if table_z_m is not None and row["lowest_z_m"] < table_z_m + IK_ABOVE_TABLE_M:
        problems.append(f"the arm reaches {(row['lowest_z_m'] - table_z_m) * 100:+.1f} cm "
                        "from the table top")
    return problems


def register_native_evaluation(env):
    """Run the same task-check registration used by ``EvalEnv.run_eval``."""
    env.run_reward()
    if hasattr(env, "get_score"):
        env.get_score()
    names = ("check_list", "final_check_list", "trigger_check_list")
    counts = {
        name: [len(group) for group in getattr(env.reward_manager, name)]
        for name in names
    }
    require(
        all(
            sum(counts[name][index] for name in names) > 0
            for index in range(env.num_envs)
        ),
        "Native task completion conditions are empty; refusing vacuous success",
    )
    support_envs = []
    if getattr(env, "interact", False) and hasattr(env, "query_support_arm_traj"):
        support_envs = list(env.get_running_env_idx_list())
        for env_idx in support_envs:
            env.query_support_arm_traj(env_idx=env_idx)
    return {
        "registered": True,
        "source": "native EvalEnv.run_eval preamble",
        "condition_group_counts": counts,
        "initial_support_arm_query_envs": support_envs,
    }


class RoboDojoSession:
    """Simulator-side controller, planner cache, recorder, and RPC surface."""

    def __init__(self, env, output, task, *, enforce_step_limit=True):
        self.env, self.output = env, Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.episode_id, self.step_id = None, 0
        self.terminated = self.truncated = self.success = False
        self.invalid_reason, self.finished = None, False
        self.video_writer, self.video_frames = None, 0
        self.finish_reason = None
        self._motion_plans = OrderedDict()
        self._motion_plan_limit = 16
        self._commanded_grippers = None  # The last executed (left, right) gripper commands.
        self.native_step_limit = int(env.step_lim)
        require(self.native_step_limit > 0, "Native task step limit must be positive")
        # The legacy setup flag selects episode mode, not whether time is bounded.
        self.episode_mode = "formal" if enforce_step_limit else "exploration"
        self.episode_step_limit = self.native_step_limit
        env.enforce_step_limit = True
        physics_dt = float(env.dt)
        ticks_per_observation = int(env.obs_manager.collect_interval)
        control_dt = physics_dt * ticks_per_observation
        require(
            np.isclose(
                control_dt,
                1.0 / float(env.obs_manager.collect_freq),
                atol=1e-9,
                rtol=0.0,
            ),
            "RoboDojo observation frequency and collect interval disagree",
        )
        self.metadata = {
            "task": task,
            "instruction": task,
            "simulator": "RoboDojo",
            "robot_adapter": "dual_arx_x5",
            "control_dt": control_dt,
            "control_frequency_hz": 1.0 / control_dt,
            "physics_dt": physics_dt,
            "physics_frequency_hz": 1.0 / physics_dt,
            "inner_control_ticks_per_observation": ticks_per_observation,
            "action_layer": "25Hz absolute dual-arm joint targets",
            "curobo_execution_layer": "25Hz absolute dual-arm joint targets via native take_action",
            "torque_layer": "PhysX implicit-actuator PD",
            "step_limit_enforced": True,
            "native_step_limit_enforced": True,
            "episode_mode": self.episode_mode,
            "native_reference_step_limit": self.native_step_limit,
            "episode_step_limit": self.episode_step_limit,
            "interaction_step_guidance": (
                "The original native task step limit is enforced in every episode; "
                "conserve interaction steps and batch predictable motion."
            ),
            "action_dim": 14,
            "action_horizon": 50,
            "frame": "environment_origin",
            "eef_link": "link6",
            "no_rollback": True,
            "supports_fk_preview": True,
            "supports_temporal_observations": True,
            "temporal_observation_frequency_hz": 1.0 / control_dt,
            "cameras": list(CAMERAS),
            "gripper_semantics": "continuous_0_closed_1_open",
            "free_space_world_model": {
                "privileged_task_objects": False,
                "depth_reconstruction": False,
                "static_table": True,
                "robot_self_collision": True,
                "other_arm_collision": False,
                "dynamic_object_collision": False,
                "planner_gripper_configuration": "fixed cuRobo retract/default state",
            },
        }
        vision_cfg = getattr(env.obs_manager, "obs_config", {}).get("vision", {})
        self.metadata.update(
            supports_depth=bool(vision_cfg.get("depth", False)),
            supports_camera_calibration=bool(
                vision_cfg.get("intrinsic_matrix", False)
                or vision_cfg.get("extrinsic_matrix", False)
            ),
            depth_representation="distance_to_image_plane_meters",
            camera_extrinsic_convention="camera_to_world_usd",
            camera_optical_convention=(
                "USD +Y up/-Z forward; RGB/depth intrinsics use CV +Y down/+Z forward"
            ),
        )
        # The session records one frame per actual 25 Hz action/control block.
        # Disable only EvalEnv's duplicate stream, not task checks or rendering.
        env._stream_vision = lambda *args, **kwargs: None

    def _operator_reward_state(self):
        """Return native evaluation state for the trusted operator only.

        This payload is persisted beside the simulator run, outside the agent
        workspace.  It must never be returned by the MCP bridge.
        """

        manager = self.env.reward_manager
        gated_scores = getattr(manager, "_gated_score_lst", None)
        score_percent = None
        if gated_scores is not None and len(gated_scores) > 0:
            score_percent = float(gated_scores[0])
        groups = {}
        for name in ("check_list", "final_check_list", "trigger_check_list"):
            values = getattr(manager, name, None)
            groups[name] = len(values[0]) if values and len(values) > 0 else 0
        return {
            "native_reward": 1.0 if self.success else 0.0,
            "native_success": bool(self.success),
            "native_score_percent": score_percent,
            "terminated": bool(self.terminated),
            "truncated": bool(self.truncated),
            "native_end_flag": bool(self.env.end_flag[0]),
            "valid_for_success_rate": bool(
                (self.terminated or self.truncated)
                and 0 not in getattr(self.env, "unstable_envs", set())
                and not self.invalid_reason
            ),
            "unstable_native_layout": bool(
                0 in getattr(self.env, "unstable_envs", set())
            ),
            "condition_group_counts": groups,
            "score_completed_count": int(
                getattr(manager, "score_completed_count", [0])[0]
            ),
            "final_score_completed_count": int(
                getattr(manager, "final_score_completed_count", [0])[0]
            ),
        }

    def _write_operator_state(self, reason=None):
        """Publish an atomic, mode-0600 snapshot for the operator dashboard."""

        if self.episode_id is None:
            return
        # Latest RGB frames for the operator's live view (overwritten each step).
        observation = f"{self.episode_id}/live.npz" if hasattr(self, "obs") else None
        state = {
            "schema": "robodojo_operator_state_v1",
            "simulator_pid": os.getpid(),
            "updated_at_unix_s": time.time(),
            "task": self.metadata["task"],
            "instruction": self.metadata.get("instruction"),
            "episode_id": self.episode_id,
            "step_id": int(self.step_id),
            "native_reference_step_limit": self.native_step_limit,
            "step_limit_enforced": True,
            "native_step_limit_enforced": bool(self.env.enforce_step_limit),
            "episode_step_limit": self.episode_step_limit,
            "episode_mode": self.episode_mode,
            "exploration_limit_reached": bool(self.episode_mode == "exploration" and self.truncated),
            "observation_file": observation,
            "cameras": list(CAMERAS),
            "reason": reason,
            "reward": self._operator_reward_state(),
        }
        path = self.output / "operator_state.json"
        write_json(path, state)
        path.chmod(0o600)

    def _check_identity(self, episode_id, step_id):
        require(
            episode_id == self.episode_id and step_id == self.step_id,
            f"Stale request for episode {episode_id} step {step_id}; the simulator "
            f"is at episode {self.episode_id} step {self.step_id}",
        )

    def _observe(self, record=False):
        raw = self.env.get_obs()
        state = raw["state"]
        # Upstream puts last commanded grippers in raw["state"]; use physical
        # proprioception so state and action labels remain distinct.
        states = measured_joint_state(self.env.robot_manager)
        images, depth_fields, camera_parameters = {}, {}, {}
        aliases = {
            "cam_high": ("cam_high", "cam_head", "head_camera", "top_camera"),
            "cam_left_wrist": ("cam_left_wrist", "left_camera"),
            "cam_right_wrist": ("cam_right_wrist", "right_camera"),
        }
        for key, names in aliases.items():
            source = next((name for name in names if name in raw["vision"]), None)
            require(
                source is not None,
                f"Missing {key}; native cameras={list(raw['vision'])}",
            )
            camera_data = raw["vision"][source]
            images[key] = np.asarray(camera_data["color"], dtype=np.uint8)[..., :3]
            if "depth" in camera_data:
                depth = np.asarray(camera_data["depth"], dtype=np.float32)
                if depth.ndim >= 1 and depth.shape[-1] == 1:
                    depth = depth.squeeze(-1)
                require(
                    depth.shape == images[key].shape[:2],
                    f"Invalid {key} depth shape: {depth.shape}",
                )
                depth_fields[f"{key}_depth_m"] = depth

            camera_manager = getattr(self.env, "camera_manager", None)
            camera_names = (
                list(getattr(camera_manager, "camera_names", [[]])[0])
                if camera_manager
                else []
            )
            if source in camera_names and self.metadata["supports_camera_calibration"]:
                camera_id = camera_names.index(source)
                camera_parameters[key] = {
                    "camera_name": source,
                    "resolution": [
                        int(images[key].shape[1]),
                        int(images[key].shape[0]),
                    ],
                    "intrinsic_matrix": rendered_intrinsics(
                        camera_manager.get_camera_intrinsics(camera_id, 0),
                    ).tolist(),
                    "camera_to_world_usd": np.asarray(
                        camera_manager.get_camera_extrinsics(camera_id, 0),
                        dtype=np.float64,
                    ).tolist(),
                    "extrinsic_convention": "camera_to_world_usd",
                    "optical_convention": (
                        "USD +Y up/-Z forward; convert to CV +Y down/+Z forward for pinhole projection"
                    ),
                    "depth": {
                        "available": f"{key}_depth_m" in depth_fields,
                        "type": "distance_to_image_plane",
                        "unit": "meters",
                        "invalid_values": "non-finite or <= 0",
                    },
                }
        poses = np.asarray(
            [state[f"{arm}_ee_pose"] for arm in ("left", "right")], dtype=np.float32
        )
        environment_origins = getattr(self.env, "env_origins", None)
        require(environment_origins is not None, "Missing native environment origins")
        environment_origin = environment_origins[0]
        if hasattr(environment_origin, "detach"):
            environment_origin = environment_origin.detach()
        if hasattr(environment_origin, "cpu"):
            environment_origin = environment_origin.cpu()
        environment_origin = np.asarray(environment_origin, dtype=np.float64)
        require(
            environment_origin.shape == (3,) and np.isfinite(environment_origin).all(),
            "Invalid native environment origin",
        )
        self.obs = dict(
            **images,
            **depth_fields,
            states=states,
            eef_positions=poses[:, :3],
            eef_quaternions_wxyz=poses[:, 3:],
            environment_origin_world_m=environment_origin,
            instruction=raw["instruction"],
            total_interaction_steps=self.step_id,
            camera_parameters=camera_parameters,
        )
        if record:
            directory = self.episode_dir / "observations"
            directory.mkdir(exist_ok=True)
            # Images are kept in the published frame sequences and sensors.mp4;
            # compressing full RGB-D here too cost ~0.15 s per step and 1-2 GB per
            # episode, so only the small fields are recorded.
            recorded = {
                key: value
                for key, value in self.obs.items()
                if key != "camera_parameters"
                and key not in images
                and key not in depth_fields
            }
            np.savez(directory / f"{self.step_id:06d}.npz", **recorded)
            live = self.episode_dir / "live.npz"
            with open(live.with_suffix(".tmp"), "wb") as stream:
                np.savez(stream, **images)
            live.with_suffix(".tmp").replace(live)
            if camera_parameters:
                write_json(
                    directory / f"{self.step_id:06d}_camera_parameters.json",
                    camera_parameters,
                )
            if self.video_writer is None:
                import imageio.v2 as imageio

                self.video_writer = imageio.get_writer(
                    str(self.output / "sensors.mp4"),
                    fps=1.0 / self.metadata["control_dt"],
                    codec="libx264",
                    pixelformat="yuv420p",
                    macro_block_size=2,
                    output_params=["-movflags", "+faststart"],
                )
            frames = [
                np.asarray(Image.fromarray(images[key]).resize((640, 360)))
                for key in CAMERAS
            ]
            self.video_writer.append_data(np.concatenate(frames, axis=1))
            self.video_frames += 1
            self._write_operator_state()
        return self.obs

    def _motion_state(self):
        state = np.asarray(self.obs["states"], dtype=np.float32).reshape(14)
        require(np.isfinite(state).all(), "Invalid current state for motion planning")
        return state.copy()

    def _hold_state(self, measured):
        """The 14D command that holds still: the measured arm joints with each gripper at its
        last command. Fingers closed on an object stop part-way, so commanding back their
        measured opening (say 0.56) would end the squeeze and drop the object. Before any
        command, the measured openings."""
        hold = np.array(measured, dtype=np.float32).reshape(14)
        if self._commanded_grippers is not None:
            hold[[6, 13]] = self._commanded_grippers
        return hold

    def _ensure_curobo_planner(self, arm):
        require(arm in ("left", "right"), f"Invalid motion-planning arm: {arm}")
        robot = self.kinematics.robots[arm]
        manager = self.env.robot_manager
        planner = manager.planner.get(robot.robot_name)
        if planner is None:
            manager._setup_planner(robot)
            planner = manager.planner.get(robot.robot_name)
        require(planner is not None, f"No cuRobo planner available for {arm}")
        return robot, planner

    def _arm_model(self, arm, joints):
        """The arm's sphere model, checked against the arm FK at joints."""
        robot = self.kinematics.robots[arm]
        model = arm_spheres(
            str(robot.urdf_path), tuple(robot.arm_joints_name),
            robot.base_link, robot.ee_link_name,
        )
        require(
            np.allclose(model.frames(joints)[-1], self.kinematics.fk[arm].matrix(joints),
                        atol=1e-6),
            f"Sphere model and arm FK disagree for {arm}",
        )
        return model

    @staticmethod
    def _sphere_rows(centres, radii):
        # Millimetres, radii rounded up so that every sphere still holds its points.
        return np.c_[np.round(centres, 3), np.ceil(radii * 1000.0) / 1000.0]

    def robot_spheres(self):
        """The robot's body spheres at the observed joints, in the environment frame."""
        state = self._motion_state()
        spheres, counts = [], {}
        for arm, offset in ARM_OFFSETS.items():
            joints = state[offset:offset + 6].astype(np.float64)
            model = self._arm_model(arm, joints)
            centres, radii = model.world(self.kinematics.root(arm), joints)
            spheres.append(self._sphere_rows(centres, radii))
            counts[arm] = len(radii)
        return {
            "step_id": int(self.step_id),
            "frame": "environment_origin",
            "spheres_m": np.vstack(spheres).tolist(),
            "counts": counts,
            "source": "robot URDF meshes at the measured joints; fingers cover their full stroke",
        }

    @staticmethod
    def _height_fixed_frames(model, root):
        """Path frames below the first joint that can change the height of what it carries."""
        frames = root @ model.frames(np.zeros(len(model.joint_names)))
        for index, joint in enumerate(model.path_joints):
            if joint["kind"] == "fixed":
                continue
            axis = (frames[index] @ joint["origin"])[:3, :3] @ joint["axis"]
            if abs(axis[2]) < 0.99 * np.linalg.norm(axis):
                return index + 1
        return len(model.path)

    def _ik_limits(self, arm, model):
        """(6, 2) joint limits for IK: the arm URDF's, tightened by the simulator's.

        The simulator's soft limits can be far wider than the arm turns (about ±10 rad on
        RoboDojo's X5A), while the URDF holds its real range (AMP's X5 limits), so IK keeps
        to both."""
        limits = np.array(self.kinematics.limits(arm), dtype=np.float64).reshape(-1, 2)
        urdf = {joint["name"]: joint["limits"] for joint in model.path_joints
                if joint["kind"] != "fixed"}
        for index, name in enumerate(model.joint_names):
            low, high = urdf.get(name, (0.0, 0.0))
            if low < high:
                limits[index] = max(limits[index, 0], low), min(limits[index, 1], high)
        return limits

    def ik(self, arm, targets, seed_joints=None, max_solutions=1, robot_spheres=False,
           table_z_m=None):
        """Link6 IK for many targets of one arm; robot-only, no step. Each solution says how
        accurate it is, how far its joints keep from their limits, how low the moving arm
        reaches, how close it comes to the other arm at its measured joints, and whether
        that is ok (``ik_problems``). Solutions that are ok come first."""
        require(arm in ARM_OFFSETS, "arm must be left or right")
        require(table_z_m is None or (
            isinstance(table_z_m, (int, float)) and not isinstance(table_z_m, bool)
            and np.isfinite(table_z_m)), "table_z_m must be a finite number")
        require(isinstance(targets, (list, tuple)) and 1 <= len(targets) <= MAX_IK_TARGETS,
                f"targets must be a list of 1..{MAX_IK_TARGETS} link6 poses")
        require(type(max_solutions) is int and 1 <= max_solutions <= MAX_IK_SOLUTIONS,
                f"max_solutions must be an integer in 1..{MAX_IK_SOLUTIONS}")
        require(type(robot_spheres) is bool, "robot_spheres must be a boolean")
        require(len(targets) * max_solutions <= MAX_IK_TARGETS,
                f"targets times max_solutions must be at most {MAX_IK_TARGETS}")
        require(not robot_spheres or len(targets) <= MAX_IK_SPHERE_TARGETS,
                f"robot_spheres needs at most {MAX_IK_SPHERE_TARGETS} targets")
        state = self._motion_state()
        offset = ARM_OFFSETS[arm]
        measured = state[offset:offset + 6].astype(np.float64)
        if seed_joints is None:
            seed = measured
        else:
            seed = np.asarray(seed_joints, dtype=np.float64)
            require(seed.shape == (6,) and np.isfinite(seed).all(),
                    "seed_joints must be six finite joint angles")
        root = self.kinematics.root(arm)
        to_base = np.linalg.inv(root)
        goals = []
        for index, target in enumerate(targets):
            try:
                validate_eef_target(target)
            except ValueError as exc:
                raise ValueError(f"targets[{index}]: {exc}") from None
            goals.append(to_base @ transform(target["position"], target["quaternion_wxyz"]))
        model = self._arm_model(arm, measured)
        limits = self._ik_limits(arm, model)
        solver = ArmIK(model.path_joints, model.joint_names, limits)
        found = solver.solve(np.stack(goals), seed, max_solutions)

        other = "right" if arm == "left" else "left"
        other_joints = state[ARM_OFFSETS[other]:ARM_OFFSETS[other] + 6].astype(np.float64)
        other_centres, other_radii = self._arm_model(other, other_joints).world(
            self.kinematics.root(other), other_joints)
        moving = model.frame >= self._height_fixed_frames(model, root)
        if not moving.any():
            moving[:] = True
        results = []
        for solutions in found:
            rows = []
            for solution in solutions:
                centres, radii = model.world(root, solution["joints"])
                gaps = (np.linalg.norm(centres[:, None] - other_centres[None], axis=2)
                        - radii[:, None] - other_radii[None])
                row = {
                    "joints": np.round(solution["joints"], 5).tolist(),
                    "position_error_mm": round(solution["position_error_mm"], 3),
                    "rotation_error_deg": round(solution["rotation_error_deg"], 3),
                    "limit_margin_rad": round(solution["limit_margin_rad"], 4),
                    "distance_from_seed_rad": round(solution["distance_from_seed_rad"], 4),
                    "lowest_z_m": round(float((centres[moving, 2] - radii[moving]).min()), 4),
                    "other_arm_clearance_m": round(float(gaps.min()), 4),
                }
                problems = ik_problems(row, solution["accurate"], table_z_m)
                row = {"ok": not problems, "why": "; ".join(problems) or "ok", **row}
                if robot_spheres:
                    row["robot_spheres"] = self._sphere_rows(centres, radii).tolist()
                rows.append(row)
            rows.sort(key=lambda item: not item["ok"])
            results.append({"reachable": bool(solutions and solutions[0]["accurate"]),
                            "ok": bool(rows and rows[0]["ok"]),
                            "why": rows[0]["why"] if rows else "no solution",
                            "solutions": rows})
        return {
            "arm": arm,
            "step_id": int(self.step_id),
            "frame": "environment_origin",
            "measured_joints": np.round(measured, 5).tolist(),
            "seed_joints": np.round(seed, 5).tolist(),
            "joint_limits_rad": np.round(limits, 4).tolist(),
            "reachable_count": sum(item["reachable"] for item in results),
            "ok_count": sum(item["ok"] for item in results),
            "results": results,
            "tolerances": {"position_mm": POSITION_TOLERANCE_MM,
                           "rotation_deg": ROTATION_TOLERANCE_DEG},
            "checks": {"min_limit_margin_rad": IK_MIN_LIMIT_MARGIN_RAD,
                       "min_other_arm_clearance_m": IK_MIN_OTHER_ARM_CLEARANCE_M,
                       "table_z_m": table_z_m, "above_table_m": IK_ABOVE_TABLE_M},
            "notes": (
                "Robot-only: no task objects. Solutions are link6 joint angles: ok ones first, "
                "then accurate ones nearest the seed. ok means reachable, at least "
                f"{IK_MIN_LIMIT_MARGIN_RAD} rad from every joint limit, at least "
                f"{IK_MIN_OTHER_ARM_CLEARANCE_M * 100:.0f} cm from the other arm and, with "
                "table_z_m, above the table top; why names what fails. Joint limits are the "
                "arm URDF's, tightened by the simulator's. lowest_z_m is the lowest point of "
                "the arm's moving links and fingers; other_arm_clearance_m is against the "
                "other arm at its measured joints. Finger spheres cover their whole stroke, "
                "so both are conservative near the gripper by up to about 2 cm. An ok pose "
                "can still fail to plan: robodojo_free_space_move plans its own path."
            ),
            "physical_steps": 0,
        }

    def _motion_plan_diagnostics(self, arm, joints, target):
        actual = self.kinematics.root(arm) @ self.kinematics.fk[arm].matrix(joints)
        goal = transform(target["position"], target["quaternion_wxyz"])
        return {
            "final_position_error_m": float(
                np.linalg.norm(actual[:3, 3] - goal[:3, 3])
            ),
            "final_rotation_error_rad": float(
                Rotation.from_matrix(goal[:3, :3] @ actual[:3, :3].T).magnitude()
            ),
            "final_joint_state": np.asarray(joints, dtype=np.float32).tolist(),
            "physical_tracking_verified": False,
        }

    def _store_motion_plan(self, entry):
        plan_id = uuid.uuid4().hex
        entry = dict(entry, motion_plan_id=plan_id)
        self._motion_plans[plan_id] = entry
        self._motion_plans.move_to_end(plan_id)
        while len(self._motion_plans) > self._motion_plan_limit:
            self._motion_plans.popitem(last=False)
        return entry

    def _update_terminal_state(self):
        ended = bool(self.env.end_flag[0])
        limit_reached = self.step_id >= self.episode_step_limit
        self.success = bool(ended and self.env.success[0])
        self.truncated = bool(limit_reached and not self.success)
        self.terminated = bool(ended and not self.truncated)
        return ended or limit_reached

    def _execute_motion_plan(self, plan_id):
        plan = self._motion_plans.get(str(plan_id))
        if plan is None:
            return {
                "status": "Plan_Not_Found",
                "motion_plan_id": str(plan_id),
                "executed": False,
                "reason": "Motion plan is missing or expired from the bounded cache.",
            }
        current = self._motion_state()
        if self.step_id != plan["step_id"] or not np.allclose(
            current, plan["start_state"], atol=2e-3, rtol=0.0
        ):
            return {
                "status": "Plan_Stale",
                "motion_plan_id": plan["motion_plan_id"],
                "executed": False,
                "reason": "An action ran or measured state changed since preview; re-plan before execution.",
                "planned_start_state": plan["start_state"].tolist(),
                "current_state": current.tolist(),
            }

        frames, steps = [], []
        actions = plan["actions"]
        for index, action in enumerate(actions):
            if self.terminated or self.truncated or self.finished:
                break
            result = self.chunk_step(
                [action],
                source="curobo_free_space",
                motion_plan_id=plan["motion_plan_id"],
                control_block_index=index,
            )
            frames.extend(result["frames"])
            steps.extend(result["steps"])

        self._motion_plans.pop(plan["motion_plan_id"], None)
        complete = len(steps) == len(actions)
        measured = self._motion_state()
        offset = 0 if plan["arm"] == "left" else 7
        tracking = self._motion_plan_diagnostics(
            plan["arm"], measured[offset:offset + 6], plan["target"]
        )
        tracking["physical_tracking_verified"] = True
        goal_reached = (
            tracking["final_position_error_m"] <= GOAL_POSITION_TOLERANCE_M
            and tracking["final_rotation_error_rad"] <= GOAL_ROTATION_TOLERANCE_RAD
        )
        return {
            # Success means all commands ran, not that the robot reached the goal.
            "status": "Success" if complete else "Partial",
            "motion_plan_id": plan["motion_plan_id"],
            "arm": plan["arm"],
            "executed": bool(steps),
            "executed_control_blocks": len(steps),
            "planned_control_blocks": len(actions),
            "goal_reached": bool(goal_reached),
            "measured_goal_error": tracking,
            "goal_tolerances": {
                "position_m": GOAL_POSITION_TOLERANCE_M,
                "rotation_rad": GOAL_ROTATION_TOLERANCE_RAD,
            },
            "observation_frequency_hz": self.metadata["control_frequency_hz"],
            "frame_count": len(frames),
            "frames": frames,
            "steps": steps,
            "joint_target_contract": JOINT_TARGET_CONTRACT,
            "step_id": self.step_id,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "success": self.success,
        }

    def free_space_move(
        self,
        *,
        arm,
        target,
        preview_only=False,
        include_trajectory=False,
    ):
        """Plan internally with cuRobo; execute ordinary 25Hz joint actions."""
        arm = str(arm).strip().lower()
        require(
            type(preview_only) is bool and type(include_trajectory) is bool,
            "Mode flags must be boolean",
        )
        target = validate_eef_target(target)
        robot, planner = self._ensure_curobo_planner(arm)
        start_state = self._motion_state()
        hold_state = self._hold_state(start_state)
        offset = 0 if arm == "left" else 7
        current_joints = start_state[offset : offset + 6]
        start_opening = float(hold_state[offset + 6])
        target_opening = (
            start_opening
            if target.get("gripper_opening") is None
            else float(target["gripper_opening"])
        )
        result = planner.plan_path(
            current_joints,
            target["position"] + target["quaternion_wxyz"],
            real_robot_pose=robot.entity_origin_pose,
        )
        if result.get("status") != "Success":
            return {
                "status": "Planning_Failed",
                "planner": "curobo",
                "arm": arm,
                "executed": False,
                "planner_result": result.get("status"),
                "failure_stage": result.get("failure_stage", "planning"),
                "reason": result.get("reason", "Planner returned no feasible trajectory"),
                "diagnostics": result.get("diagnostics", {}),
            }
        positions = np.asarray(result.get("position"), dtype=np.float32)
        actions = policy_actions(
            positions, hold_state, arm, target_opening,
            planner_dt=float(result.get("interpolation_dt")),
            control_dt=self.metadata["control_dt"],
        )
        diagnostics = self._motion_plan_diagnostics(arm, positions[-1], target)
        if (
            diagnostics["final_position_error_m"] > GOAL_POSITION_TOLERANCE_M
            or diagnostics["final_rotation_error_rad"] > GOAL_ROTATION_TOLERANCE_RAD
        ):
            return {
                "status": "Planning_Failed", "planner": "curobo", "arm": arm,
                "executed": False, "reason": "Planned endpoint exceeds goal tolerances",
                "diagnostics": diagnostics,
            }
        if len(actions) > MAX_TRAJECTORY_ACTIONS:
            return {
                "status": "Planning_Failed", "executed": False, "arm": arm,
                "failure_stage": "action_budget",
                "reason": f"Trajectory exceeds the {MAX_TRAJECTORY_ACTIONS}-step planning/preview limit; choose a nearer waypoint",
                "action_count": len(actions), "max_actions": MAX_TRAJECTORY_ACTIONS,
            }
        entry = self._store_motion_plan(
            {
                "arm": arm,
                "target": target,
                "start_state": start_state,
                "actions": actions,
                "step_id": int(self.step_id),
            }
        )
        response = {
            "status": "Success",
            "planner": "curobo",
            "arm": arm,
            "motion_plan_id": entry["motion_plan_id"],
            "executed": False,
            "preview_only": True,
            "start_state": start_state.tolist(),
            "final_arm_joint_state": positions[-1].tolist(),
            "target": target,
            "diagnostics": diagnostics,
            "timing": {
                "control_blocks": len(actions),
                "remaining_episode_steps": max(0, self.episode_step_limit - self.step_id),
                "fits_remaining_budget": len(actions) <= self.episode_step_limit - self.step_id,
                "control_frequency_hz": self.metadata["control_frequency_hz"],
                "simulator_advance_duration_s": len(actions) * self.metadata["control_dt"],
            },
            "joint_target_contract": JOINT_TARGET_CONTRACT,
            "goal_tolerances": {
                "position_m": GOAL_POSITION_TOLERANCE_M,
                "rotation_rad": GOAL_ROTATION_TOLERANCE_RAD,
            },
            "execution_contract": {
                "environment_action_rows": True,
                "executor": "same native take_action as robodojo_step and robodojo_step_eef",
                "benchmark_step_accounting": "one 14D action row per 25Hz step",
                "planned_velocities_executed": False,
                "success_means": "commands completed; inspect transition.goal_reached separately",
            },
            "collision_model": self.metadata["free_space_world_model"],
            "execute_with": {"motion_plan_id": entry["motion_plan_id"]},
        }
        if include_trajectory:
            response["trajectory_preview"] = {
                "actions": actions.tolist(),
                "frequency_hz": self.metadata["control_frequency_hz"],
                "action_count": len(actions),
                "includes_initial_state": False,
                "note": f"Complete executable targets, without display subsampling. Pass actions to robot_preview_run or replay in robodojo_step batches of at most {MAX_TRAJECTORY_ACTIONS}.",
            }
        if not preview_only:
            response["execution"] = self._execute_motion_plan(entry["motion_plan_id"])
            response["executed"] = response["execution"]["executed"]
            response["preview_only"] = False
        return response

    def reset(self, seed, source, policy_version):
        require(self.episode_id is None, "One fresh episode only; no rollback/reset")
        self.env.reset(seed=[seed])
        registration = register_native_evaluation(self.env)
        write_json(self.output / "native_evaluation.json", registration)
        self.episode_id = uuid.uuid4().hex
        self.episode_dir = self.output / self.episode_id
        self.episode_dir.mkdir()
        self.invalid_reason = None
        self.kinematics = DualKinematics(self.env)
        write_json(self.output / "fk_validation.json", self.kinematics.check())
        self._observe(record=True)
        fingerprints = {}
        for key, value in self.obs.items():
            if key == "camera_parameters":
                fingerprints[key] = {
                    "schema": "camera_parameters_v1",
                    "sha256": hashlib.sha256(
                        json.dumps(value, sort_keys=True).encode()
                    ).hexdigest(),
                }
                continue
            array = np.asarray(value)
            fingerprints[key] = {
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
            }
        write_json(
            self.output / "initial_observation_fingerprint.json",
            {
                "fields": fingerprints,
                "scope": "RGB, depth when enabled, proprio, EEF, and instruction; not complete physics state",
                "bitwise_physics_reproducibility_guaranteed": False,
            },
        )
        self.metadata["instruction"] = self.obs["instruction"]
        identity = hashlib.sha256(self.obs["states"].tobytes()).hexdigest()
        result = {
            "episode_id": self.episode_id,
            "step_id": 0,
            "initial_state_hash": identity,
            "instruction": self.obs["instruction"],
            "metadata": dict(
                self.metadata,
                layout_id=seed,
                eval_seed=getattr(self.env, "eval_seed", None),
                initial_state_hash_scope="robot_proprio_only_not_complete_scene",
                policy_version=policy_version,
                controller_source=source,
                native_reset_settling_not_counted_as_policy_controls=True,
            ),
        }
        write_json(self.output / "reset.json", result)
        return result

    def chunk_step(self, actions, **kwargs):
        if self.finished or self.terminated or self.truncated:
            # Refused before any action.
            raise RejectedBeforeAction("Episode finished")
        actions = np.asarray(actions, dtype=np.float32)
        reject_unless(
            actions.ndim == 2
            and actions.shape[1] == 14
            and 1 <= len(actions) <= MAX_TRAJECTORY_ACTIONS
            and np.isfinite(actions).all(),
            f"Expected 1..{MAX_TRAJECTORY_ACTIONS} finite 14D absolute actions",
        )
        reject_unless(
            np.all((actions[:, [6, 13]] >= 0) & (actions[:, [6, 13]] <= 1)),
            "Invalid gripper opening",
        )
        rows, frames = [], []
        for action in actions:
            command = {
                key: value
                for arm, offset in (("left", 0), ("right", 7))
                for key, value in (
                    (f"{arm}_arm_joint_state", action[offset : offset + 6]),
                    (f"{arm}_ee_joint_state", action[offset + 6 : offset + 7]),
                )
            }
            before = int(self.env.take_action_cnt[0])
            self.env.take_action(command)
            require(
                int(self.env.take_action_cnt[0]) == before + 1,
                "Native action was not executed exactly once",
            )
            self._commanded_grippers = action[[6, 13]].copy()
            self.step_id += 1
            ended = self._update_terminal_state()
            observation = self._observe(record=True)
            frames.append(observation)
            row = {
                "valid": True,
                "executed_action": action.copy(),
                "execution_mode": "policy_joint_target",
                "observation_step_id": self.step_id - 1,
                "measured_state": observation["states"].copy(),
                "step_id": self.step_id,
                "terminated": self.terminated,
                "truncated": self.truncated,
                "success": self.success,
                "source": "mcp_joint_action",
            }
            row.update(kwargs)
            rows.append(row)
            write_json(self.episode_dir / f"action_{self.step_id - 1:06d}.json", row)
            if ended:
                self._write_summary("terminal")
                break
        return {
            "episode_id": self.episode_id,
            "step_id": self.step_id,
            "observation_frequency_hz": self.metadata["control_frequency_hz"],
            "frame_count": len(frames),
            "frames": frames,
            "steps": rows,
            "joint_target_contract": JOINT_TARGET_CONTRACT,
        }

    def _write_summary(self, reason):
        if self.episode_id is None:
            return
        if self.video_writer is not None:
            self.video_writer.close()
            self.video_writer = None
        if self.finish_reason is not None:
            reason = self.finish_reason
        complete = self.terminated or self.truncated
        invalid = 0 in getattr(self.env, "unstable_envs", set())
        eligible = complete and not invalid and not self.invalid_reason
        score = None
        # Native run_eval awards success=1, otherwise the task's gated process
        # score. Early controller exit/timeout does not erase earned partial
        # credit; binary completion eligibility remains a separate decision.
        if not invalid:
            score = (
                1.0
                if self.success
                else (
                    float(self.env.reward_manager.get_score()[0]) / 100
                    if hasattr(self.env, "get_score")
                    else 0.0
                )
            )
        write_json(
            self.output / "evaluation_outcome.json",
            {
                "complete": complete,
                "valid_for_success_rate": eligible,
                "native_success": self.success if eligible else None,
                "native_score": score,
                "valid_for_score": score is not None,
                "score_source": "native_eval" if score is not None else "invalid_native_layout",
                "invalid_reason": self.invalid_reason,
                "status": (
                    "invalid_native_layout"
                    if invalid
                    else "native_completed"
                    if eligible
                    else "incomplete"
                ),
                "reason": reason,
                "native_control_steps": self.step_id,
                "native_reference_step_limit": self.native_step_limit,
                "step_limit_enforced": True,
                "native_step_limit_enforced": bool(self.env.enforce_step_limit),
                "episode_step_limit": self.episode_step_limit,
            },
        )
        write_json(
            self.output / "summary.json",
            {
                "episode_id": self.episode_id,
                "step_id": self.step_id,
                "success": self.success,
                "terminated": self.terminated,
                "truncated": self.truncated,
                "complete": complete,
                "reason": reason,
                "video_frames": self.video_frames,
                "control_dt": self.metadata["control_dt"],
                "no_rollback": True,
            },
        )
        self._write_operator_state(reason=reason)

    def dispatch(self, op, args):
        # Cached plans execute multiple chunks inside one RPC.
        with action_execution_scope(self, op):
            return self._dispatch(op, args)

    def _dispatch(self, op, args):
        if op == "metadata":
            return self.metadata
        if op == "reset":
            return self.reset(**args)
        if op == "finish_pilot":
            # Closing stays possible after any error; the outcome records why it ended.
            require(args["episode_id"] == self.episode_id, "Stale episode")
        else:
            self._check_identity(args["episode_id"], args["step_id"])
        data = {
            key: value
            for key, value in args.items()
            if key not in ("episode_id", "step_id")
        }
        if op == "teacher_observation":
            return self.obs
        if op in ("robot_spheres", "ik"):
            # Robot-only geometry; it never touches the simulator's scene.
            try:
                return self.robot_spheres() if op == "robot_spheres" else self.ik(**data)
            except (TypeError, ValueError) as exc:
                raise RejectedBeforeAction(str(exc)) from exc
        if op in ("fk_preview", "eef_joint_target"):
            # Pure kinematics on requested targets; invalid requests leave the simulator untouched.
            try:
                if op == "fk_preview":
                    return self.kinematics.preview(data["actions"])
                return self.kinematics.target(data["targets"])
            except ValueError as exc:
                raise RejectedBeforeAction(str(exc)) from exc
        if op == "free_space_move":
            return self.free_space_move(**data)
        if op == "execute_motion_plan":
            return self._execute_motion_plan(data["motion_plan_id"])
        if op == "chunk_step":
            return self.chunk_step(**data)
        if op == "finish_pilot":
            self.finished = True
            self.finish_reason = data["reason"]
            self._write_summary(data["reason"])
            return {
                "episode_id": self.episode_id,
                "step_id": self.step_id,
                "success": self.success,
                "terminated": self.terminated,
                "truncated": self.truncated,
                "video_frames": self.video_frames,
                "control_dt": self.metadata["control_dt"],
                "reason": data["reason"],
            }
        raise ValueError("Unsupported no-rollback operation: " + op)
