"""
Offline sanity check (no ROS / no hardware needed): exercises the pure
numpy modules with synthetic data to confirm the shared-control math
runs end-to-end and returns sane values before deploying on the FR3.

Run with:  python3 test_performance_offline.py
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dh_utils import human_arm_jacobian, human_arm_wrist_position, cartesian_to_human_joint_velocity
from performance import (smoothness_factor, directness_factor, joint_safety_factor,
                          manipulability_index, manipulability_factor, total_performance,
                          DEFAULT_JOINT_LIMITS)
from path_follower import CirclePath, ReactivePathFollower


def test_human_arm_kinematics():
    q = np.array([0.3, 0.5, 0.0, 1.2])
    l1, l2 = 0.30, 0.27
    p = human_arm_wrist_position(q, l1, l2)
    J = human_arm_jacobian(q, l1, l2)
    assert p.shape == (3,)
    assert J.shape == (3, 4)
    assert np.all(np.isfinite(J))
    print("wrist position:", p)
    print("arm Jacobian:\n", J)


def test_joint_safety_factor():
    l1, l2 = 0.30, 0.27
    # Case A: joint well within range -> high performance expected.
    q_mid = np.mean(DEFAULT_JOINT_LIMITS, axis=1)
    v = np.array([0.05, 0.0, 0.0])
    eta_mid = joint_safety_factor(q_mid, v, l1, l2)

    # Case B: q4 (elbow) pinned near its upper limit -> lower performance
    # expected when the candidate command drives it further towards it.
    q_near_limit = q_mid.copy()
    q_near_limit[3] = DEFAULT_JOINT_LIMITS[3, 1] - 0.05
    eta_limit = joint_safety_factor(q_near_limit, v, l1, l2)

    print("eta_safety (mid-range):", eta_mid)
    print("eta_safety (near elbow limit):", eta_limit)
    assert 0.0 <= eta_mid <= 1.0
    assert 0.0 <= eta_limit <= 1.0


def test_manipulability_factor():
    rng = np.random.default_rng(0)
    J_now = rng.normal(size=(6, 7))
    J_worse = J_now * 0.2   # artificially "closer to singular" (lower rank energy)
    J_better = J_now * 1.5
    w_now = manipulability_index(J_now)
    eta_worse = manipulability_factor(J_now, J_worse)
    eta_better = manipulability_factor(J_now, J_better)
    print("w(q):", w_now, " eta(worse):", eta_worse, " eta(better):", eta_better)
    assert eta_worse < eta_better


def test_path_follower():
    path = CirclePath(center=[0.5, 0.0, 0.4], radius=0.075)
    follower = ReactivePathFollower(path, Ka=2.0)
    x = np.array([0.55, 0.02, 0.4])
    v_r, tangent = follower.robot_command(x)
    print("v_r:", v_r, " tangent:", tangent)
    assert v_r.shape == (3,)
    assert abs(np.linalg.norm(tangent) - 1.0) < 1e-6


def test_full_blend():
    v_h = np.array([0.02, 0.01, 0.0])
    v_r = np.array([0.03, -0.01, 0.0])
    v_prev = np.array([0.02, 0.0, 0.0])
    tangent = np.array([0.0, 1.0, 0.0])

    factors_h = {
        "smoothness": smoothness_factor(v_h, v_prev),
        "directness": directness_factor(v_h, tangent),
    }
    factors_r = {
        "smoothness": smoothness_factor(v_r, v_prev),
        "directness": directness_factor(v_r, tangent),
    }
    eta_h = total_performance(factors_h)
    eta_r = total_performance(factors_r)
    v_hat_s = eta_r * v_r + eta_h * v_h
    print("eta_h:", eta_h, " eta_r:", eta_r, " v_hat_s:", v_hat_s)
    assert np.all(np.isfinite(v_hat_s))


if __name__ == "__main__":
    test_human_arm_kinematics()
    test_joint_safety_factor()
    test_manipulability_factor()
    test_path_follower()
    test_full_blend()
    print("\nAll offline sanity checks passed.")
