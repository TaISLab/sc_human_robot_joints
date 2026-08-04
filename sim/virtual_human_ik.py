"""
virtual_human_ik.py
--------------------
Simple damped-least-squares numerical IK for the 4-DoF human-arm model
(dh_utils.py), used ONLY to drive a synthetic "virtual patient" towards
a target wrist position during simulation. Not used by the real
shared_control_node.py (which consumes the real, vision/tactile
estimated q1..q4 directly).

Redundancy resolution: the arm has one more joint (4) than Cartesian
position constraints (3), so a given wrist target is reachable by an
entire 1-parameter family of configurations (the "elbow swivel"). Without
a secondary objective, which point of that family the solver lands on is
an arbitrary artifact of the initial guess q0 and the numerics of the
minimum-norm step -- meaning, in particular, that whether q3 (shoulder
internal/external rotation) ends up near its limit for a given wrist
target is essentially accidental, not a deliberate feature of any
"risky"/adversarial synthetic user. This was flagged during review and is
fixed here via a null-space secondary task (Liegeois' gradient-projection
method, see dh_utils.cartesian_to_human_joint_velocity) that biases the
redundant degree of freedom towards the center of each joint's range at
every IK iteration.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dh_utils import human_arm_wrist_position, human_arm_jacobian  # noqa: E402


def solve_ik(x_target, q0, l1, l2, joint_limits, n_iter=25, damping=1e-2, step=0.7,
             center_gain=0.3):
    """Damped least-squares IK, clipped to joint_limits after each step.

    `center_gain` (>=0) controls the strength of the null-space,
    joint-centering secondary task; 0 recovers the previous pure
    minimum-norm behavior.
    """
    q = np.array(q0, dtype=float)
    q_mid = 0.5 * (joint_limits[:, 0] + joint_limits[:, 1])
    q_half_range = 0.5 * (joint_limits[:, 1] - joint_limits[:, 0])
    for _ in range(n_iter):
        p = human_arm_wrist_position(q, l1, l2)
        err = np.asarray(x_target, dtype=float) - p
        J = human_arm_jacobian(q, l1, l2)
        JJt = J @ J.T
        damped_pinv = J.T @ np.linalg.solve(JJt + damping ** 2 * np.eye(3), np.eye(3))
        dq_primary = damped_pinv @ err

        dq_secondary = np.zeros(4)
        if center_gain > 0.0:
            rho = (q - q_mid) / q_half_range
            N = np.eye(4) - damped_pinv @ J
            dq_secondary = N @ (-center_gain * rho)

        q = q + step * dq_primary + step * dq_secondary
        q = np.clip(q, joint_limits[:, 0], joint_limits[:, 1])

        if np.linalg.norm(err) < 1e-4:
            break
    return q
