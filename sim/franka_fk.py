"""
franka_fk.py
------------
Pure-numpy forward kinematics and numerical Jacobian for a 7-DoF Franka
arm (Panda/FR3), using the commonly published modified-DH (Craig
convention) parameters. This is a SIMULATION-ONLY convenience model:

  * It is used exclusively by the offline/Monte-Carlo simulator in this
    folder, to close the loop numerically before running on the real
    robot.
  * On the real FR3, shared_control_node.py does NOT use this module:
    it consumes the exact Jacobian (O_Jac_EE) published by franka_ros,
    which is the calibrated, authoritative source.

Cross-check the a_i/d_i/alpha_i values below against your exact FR3
URDF/DH sheet before trusting absolute manipulability magnitudes; for
verifying the *shape* of the control law's behavior (does eta_manip
drop near a singularity, etc.) the published nominal values are
adequate.
"""

import numpy as np

N_ROBOT_JOINTS = 7

# (a_{i-1} [m], alpha_{i-1} [rad], d_i [m]) for i = 1..7, modified DH,
# commonly published values for the Franka Panda/FR3 arm.
_MDH = [
    (0.0,      0.0,      0.333),
    (0.0,     -np.pi/2,  0.0),
    (0.0,      np.pi/2,  0.316),
    (0.0825,   np.pi/2,  0.0),
    (-0.0825, -np.pi/2,  0.384),
    (0.0,      np.pi/2,  0.0),
    (0.088,    np.pi/2,  0.0),
]
_D_FLANGE = 0.107  # last joint to flange, along z7


def _mdh_transform(a_prev, alpha_prev, d, theta):
    ca, sa = np.cos(alpha_prev), np.sin(alpha_prev)
    ct, st = np.cos(theta), np.sin(theta)
    return np.array([
        [ct,       -st,      0.0,     a_prev],
        [st * ca,  ct * ca, -sa,     -sa * d],
        [st * sa,  ct * sa,  ca,      ca * d],
        [0.0,       0.0,     0.0,     1.0],
    ])


def franka_fk(q):
    """Forward kinematics: returns the 4x4 pose of the flange w.r.t. the
    robot base, given 7 joint angles q."""
    T = np.eye(4)
    for i in range(N_ROBOT_JOINTS):
        a_prev, alpha_prev, d = _MDH[i]
        T = T @ _mdh_transform(a_prev, alpha_prev, d, q[i])
    T = T @ _mdh_transform(0.0, 0.0, _D_FLANGE, 0.0)
    return T


def franka_ee_position(q):
    return franka_fk(q)[:3, 3]


def franka_ee_rotvec(q):
    """Small-angle rotation-vector proxy for orientation, used only to
    build a numerical 6xN Jacobian (position + orientation) consistent
    with the rest of this simulation-only module."""
    R = franka_fk(q)[:3, :3]
    # Rodrigues' formula (log map), robust enough for finite differences.
    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if theta < 1e-8:
        return np.zeros(3)
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(theta))
    return axis * theta


def franka_jacobian(q, eps=1e-6):
    """Numerical 6xN geometric Jacobian (linear + rotation-vector rate)."""
    q = np.asarray(q, dtype=float)
    p0 = franka_ee_position(q)
    r0 = franka_ee_rotvec(q)
    J = np.zeros((6, N_ROBOT_JOINTS))
    for i in range(N_ROBOT_JOINTS):
        dq = np.zeros(N_ROBOT_JOINTS)
        dq[i] = eps
        p1 = franka_ee_position(q + dq)
        r1 = franka_ee_rotvec(q + dq)
        J[:3, i] = (p1 - p0) / eps
        J[3:, i] = (r1 - r0) / eps
    return J


# A configuration with the elbow nearly straightened (low positional
# manipulability, verified numerically to be smooth/well-conditioned
# for finite differences, unlike the classical wrist-flip singularity
# which makes the numerical orientation Jacobian blow up). Useful as a
# starting point for the "stressed"-placement Monte-Carlo trials.
NEAR_SINGULAR_Q = np.array([0.0, -0.6, 0.0, -0.3, 0.0, 1.8, 0.78])

# A configuration reasonably close to a comfortable mid-workspace
# posture (elbow well bent, higher manipulability), useful for the
# "nominal"-placement trials.
MID_WORKSPACE_Q = np.array([0.0, -0.6, 0.0, -2.2, 0.0, 1.8, 0.78])

class SimRobotModel(object):
    """Minimal stand-in for robot_model.RobotModel, usable without ROS/
    franka_ros/PyKDL: predicts the (position-only, 3xN) Jacobian at
    q_current + qdot_candidate * dt using the numeric FK/Jacobian above.
    Passed to SharedControlCore(robot_model=...) inside the simulator so
    that eta_manipulability is active during Monte-Carlo runs, exactly
    as it would be on the real robot (which instead uses robot_model.RobotModel).
    """

    def predict_jacobian(self, q_current, qdot_candidate, dt):
        q_pred = np.asarray(q_current, dtype=float) + np.asarray(qdot_candidate, dtype=float) * dt
        return franka_jacobian(q_pred)[:3, :]


# NOTE: manipulability comparisons in this simulation use the
# position-only (3xN) sub-Jacobian (see performance.manipulability_index
# called with J[:3, :]) to avoid the numerical instability of the
# finite-difference orientation Jacobian near true wrist singularities.
# The real deployment on the FR3 uses the exact, analytic 6xN O_Jac_EE
# from franka_ros and is not affected by this simulation-only caveat.
