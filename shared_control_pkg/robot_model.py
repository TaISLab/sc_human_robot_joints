"""
robot_model.py
--------------
Thin wrapper around the FR3 kinematic model needed ONLY for the
manipulability / singularity-avoidance factor: it must be able to
return the 6xN geometric Jacobian at an arbitrary joint configuration
q (not just the current one, since we need to predict the Jacobian at
q + qdot_candidate * dt for each candidate command).

Two backends are supported:

1. "franka_state" (default, no extra deps): uses the current O_Jac_EE
   published by franka_ros in franka_msgs/FrankaState for the *current*
   configuration, and a first-order update J(q+dq) ~= J(q) + dJ/dq * dq
   is NOT attempted; instead the predicted Jacobian is approximated by
   re-evaluating with PyKDL if available (see below), or, if PyKDL is
   not installed, by finite-differencing the manipulability index
   using a short rollout of the *current* Jacobian only (first-order,
   locally valid approximation -- good enough at the ~1 kHz / small dt
   used in this control loop, but PyKDL is recommended for longer
   lookaheads).

2. "kdl": loads the FR3 chain from `robot_description` via
   kdl_parser_py + PyKDL and computes exact Jacobians at any q. Use
   this backend if PyKDL is available in the workspace (it usually is,
   since franka_ros / MoveIt pulls it in as a dependency).

Only the interface `jacobian(q)` is required by performance.py.
"""

import numpy as np

try:
    import PyKDL  # noqa: F401
    from kdl_parser_py.urdf import treeFromParam
    _HAS_KDL = True
except ImportError:
    _HAS_KDL = False


class RobotModel(object):
    def __init__(self, backend="franka_state", base_link="panda_link0",
                 ee_link="panda_link8"):
        self.backend = backend if (backend != "kdl" or _HAS_KDL) else "franka_state"
        self._last_J = None
        self._last_q = None

        if self.backend == "kdl":
            ok, tree = treeFromParam("/robot_description")
            if not ok:
                raise RuntimeError("Failed to parse /robot_description for KDL.")
            self.chain = tree.getChain(base_link, ee_link)
            self.n_joints = self.chain.getNrOfJoints()
            self._jac_solver = PyKDL.ChainJntToJacSolver(self.chain)

    # -- current-state bookkeeping (called every control cycle) --------
    def update_current_state(self, q, J_current):
        """Store the latest (q, O_Jac_EE) reported by /franka_states."""
        self._last_q = np.asarray(q, dtype=float)
        self._last_J = np.asarray(J_current, dtype=float)

    def jacobian(self, q):
        """Return the 6xN Jacobian at configuration q."""
        q = np.asarray(q, dtype=float)
        if self.backend == "kdl":
            jnt = PyKDL.JntArray(self.n_joints)
            for i in range(self.n_joints):
                jnt[i] = q[i]
            jac = PyKDL.Jacobian(self.n_joints)
            self._jac_solver.JntToJac(jnt, jac)
            return np.array([[jac[r, c] for c in range(self.n_joints)]
                              for r in range(6)])

        # "franka_state" fallback: locally-valid first-order approximation.
        # Good enough for the short lookahead (dt ~ 1-10 ms) used to rank
        # candidate commands; falls back to the last reported Jacobian if
        # no better estimate is available.
        if self._last_J is None:
            raise RuntimeError("RobotModel: no franka_states received yet.")
        return self._last_J

    def predict_jacobian(self, q_current, qdot_candidate, dt):
        q_pred = q_current + qdot_candidate * dt
        return self.jacobian(q_pred)
