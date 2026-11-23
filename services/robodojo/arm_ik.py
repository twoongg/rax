"""Batch link6 IK for one arm: many targets, several seeds each, robot-only and pure numpy.

Damped least squares from the seed, the arm's zero pose and a few fixed pseudo-random
configurations, iterated together for every target and clamped to the joint limits. A
target that none of them reaches gets fresh random starts, up to RESTARTS more rounds. The
seeds, accuracy tolerances and ranking follow AMP's agent IK (``amp.planning.ik``):
accurate solutions nearest the seed first, near-duplicates dropped.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial.transform import Rotation

POSITION_TOLERANCE_MM = 5.0
ROTATION_TOLERANCE_DEG = 3.0
RANDOM_SEEDS = 8  # AMP's batch IK uses 4.
RESTARTS = 3      # Rounds of fresh random starts for targets still unsolved.
ITERATIONS = 200
MAX_STEP_RAD = 0.2
DAMPING = 0.05
SAME_RAD = 0.05


def _rotations(axes, angles):
    """(B, 3, 3) rotations about unit axes (3,) by angles (B,) (Rodrigues)."""
    k = np.array([[0.0, -axes[2], axes[1]], [axes[2], 0.0, -axes[0]], [-axes[1], axes[0], 0.0]])
    s, c = np.sin(angles)[:, None, None], np.cos(angles)[:, None, None]
    return np.eye(3) + s * k + (1.0 - c) * (k @ k)


class ArmIK:
    """IK over a serial chain of fixed and revolute joints from the arm base to link6.

    ``chain`` is a list of dicts with ``name``, ``kind``, ``origin`` (4x4) and ``axis`` (3,),
    as ``robot_spheres.ArmSpheres.path_joints`` holds them; ``limits`` is (n, 2) radians in
    ``joint_names`` order.
    """

    def __init__(self, chain, joint_names, limits):
        self.joint_names = list(joint_names)
        self.limits = np.asarray(limits, dtype=np.float64).reshape(len(self.joint_names), 2)
        if not (np.isfinite(self.limits).all() and (self.limits[:, 0] < self.limits[:, 1]).all()):
            raise ValueError("Joint limits must be finite with lower < upper")
        self.chain = []
        for joint in chain:
            if joint["kind"] not in ("fixed", "revolute"):
                raise ValueError(f"Unsupported arm joint type {joint['kind']!r}")
            index = self.joint_names.index(joint["name"]) if joint["kind"] == "revolute" else None
            axis = np.asarray(joint["axis"], dtype=np.float64)
            self.chain.append((np.asarray(joint["origin"], dtype=np.float64), index,
                               axis / np.linalg.norm(axis)))
        if sorted(i for _, i, _ in self.chain if i is not None) != list(range(len(self.joint_names))):
            raise ValueError("The chain's revolute joints must be exactly the arm joints")

    def fk(self, q):
        """(B, 4, 4) link6 poses in the base frame and (B, n, 2, 3) each joint's world axis
        and origin, for joint vectors q (B, n)."""
        q = np.asarray(q, dtype=np.float64).reshape(-1, len(self.joint_names))
        pose = np.broadcast_to(np.eye(4), (len(q), 4, 4)).copy()
        axes = np.zeros((len(q), len(self.joint_names), 2, 3))
        for origin, index, axis in self.chain:
            pose = pose @ origin
            if index is None:
                continue
            axes[:, index, 0] = pose[:, :3, :3] @ axis
            axes[:, index, 1] = pose[:, :3, 3]
            motion = np.broadcast_to(np.eye(4), (len(q), 4, 4)).copy()
            motion[:, :3, :3] = _rotations(axis, q[:, index])
            pose = pose @ motion
        return pose, axes

    def margin(self, q):
        """Distance (rad) from each joint vector to its nearest limit; negative outside."""
        q = np.asarray(q, dtype=np.float64).reshape(-1, len(self.joint_names))
        return np.minimum(q - self.limits[:, 0], self.limits[:, 1] - q).min(axis=1)

    def _random(self, round_index):
        low, high = self.limits[:, 0], self.limits[:, 1]
        rng = np.random.default_rng(round_index)
        return rng.uniform(low + 0.1, high - 0.1, size=(RANDOM_SEEDS, len(low)))

    def seeds(self, seed):
        """The first round's starts: the seed, the zero pose and fixed random ones."""
        low, high = self.limits[:, 0], self.limits[:, 1]
        return np.vstack([np.clip(seed, low, high), np.clip(0.0, low, high), self._random(0)])

    def _iterate(self, targets, q):
        """Run DLS on every (target, start) pair; the best q, position and rotation errors."""
        low, high = self.limits[:, 0], self.limits[:, 1]
        best_q, best_cost = q.copy(), np.full(len(q), np.inf)
        best_pos, best_rot = np.full(len(q), np.inf), np.full(len(q), np.inf)
        active = np.ones(len(q), dtype=bool)
        for _ in range(ITERATIONS):
            pose, axes = self.fk(q[active])
            goal = targets[active]
            e_pos = goal[:, :3, 3] - pose[:, :3, 3]
            e_rot = Rotation.from_matrix(
                goal[:, :3, :3] @ np.swapaxes(pose[:, :3, :3], 1, 2)).as_rotvec()
            pos, rot = np.linalg.norm(e_pos, axis=1), np.linalg.norm(e_rot, axis=1)
            cost = pos + 0.1 * rot
            index = np.flatnonzero(active)
            better = cost < best_cost[index]
            chosen = index[better]
            best_q[chosen], best_cost[chosen] = q[chosen], cost[better]
            best_pos[chosen], best_rot[chosen] = pos[better], rot[better]
            done = (pos < 1e-5) & (rot < 1e-4)
            active[index[done]] = False
            keep = ~done
            if not keep.any():
                break
            axis, origin = axes[keep, :, 0], axes[keep, :, 1]
            jacobian = np.concatenate(
                [np.cross(axis, pose[keep, None, :3, 3] - origin), axis], axis=2)
            jacobian = np.swapaxes(jacobian, 1, 2)  # (B, 6, n)
            error = np.concatenate([e_pos[keep], e_rot[keep]], axis=1)
            scale = np.minimum(1.0, np.linalg.norm(error, axis=1) / 0.05)
            system = jacobian @ np.swapaxes(jacobian, 1, 2) + (
                (DAMPING ** 2) * scale)[:, None, None] * np.eye(6)
            step = np.einsum("bji,bj->bi", jacobian, np.linalg.solve(system, error[..., None])[..., 0])
            largest = np.abs(step).max(axis=1, keepdims=True)
            step *= np.minimum(1.0, MAX_STEP_RAD / np.maximum(largest, 1e-12))
            moving = index[keep]
            q[moving] = np.clip(q[moving] + step, low, high)
        return best_q, best_pos, best_rot

    def solve(self, targets, seed, max_solutions=1):
        """For each target link6 pose (B, 4, 4) in the base frame, up to max_solutions
        solutions: accurate ones nearest the seed first, then the rest by error."""
        targets = np.asarray(targets, dtype=np.float64).reshape(-1, 4, 4)
        seed = np.asarray(seed, dtype=np.float64).reshape(len(self.joint_names))
        owners, joints, pos, rot = [], [], [], []
        pending = np.arange(len(targets))
        for round_index in range(RESTARTS + 1):
            starts = self.seeds(seed) if round_index == 0 else self._random(round_index)
            owner = np.repeat(pending, len(starts))
            q, p, r = self._iterate(targets[owner], np.tile(starts, (len(pending), 1)))
            owners.append(owner)
            joints.append(q)
            pos.append(p)
            rot.append(r)
            accurate = (p * 1000.0 <= POSITION_TOLERANCE_MM) & (
                np.degrees(r) <= ROTATION_TOLERANCE_DEG)
            pending = np.setdiff1d(pending, owner[accurate])
            if not len(pending):
                break
        owners, joints = np.concatenate(owners), np.concatenate(joints)
        pos, rot = np.concatenate(pos), np.concatenate(rot)
        margins = self.margin(joints)
        out = []
        for t in range(len(targets)):
            found = []
            for row in np.flatnonzero(owners == t):
                pos_mm, rot_deg = float(pos[row] * 1000.0), math.degrees(float(rot[row]))
                accurate = pos_mm <= POSITION_TOLERANCE_MM and rot_deg <= ROTATION_TOLERANCE_DEG
                distance = float(np.linalg.norm(joints[row] - seed))
                found.append((not accurate, distance if accurate else pos_mm + rot_deg, row,
                              accurate, pos_mm, rot_deg, distance))
            found.sort(key=lambda item: (item[0], item[1]))
            kept = []
            for _, _, row, accurate, pos_mm, rot_deg, distance in found:
                if any(np.abs(joints[row] - item["joints"]).max() < SAME_RAD for item in kept):
                    continue
                kept.append({"joints": joints[row].copy(), "accurate": accurate,
                             "position_error_mm": pos_mm, "rotation_error_deg": rot_deg,
                             "limit_margin_rad": float(margins[row]),
                             "distance_from_seed_rad": distance})
                if len(kept) >= max_solutions:
                    break
            out.append(kept)
        return out
