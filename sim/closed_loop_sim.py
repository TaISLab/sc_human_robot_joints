"""
closed_loop_sim.py
--------------------
Fast, ROS-free, numpy-only closed-loop simulation of the shared-control
law, used to:

  1. Verify the control law's behavior (does eta_joint_safety/eta_manip
     actually kick in where expected?) before spending real robot/
     participant time.
  2. Tune the free parameters (C_s, C_m, performance weights, joint
     margin threshold tau, manipulability lookahead dt) offline.
  3. Run a Monte-Carlo battery of synthetic "virtual patients" (varying
     anthropometry, motor noise/skill, and a fraction that deliberately
     drifts towards a joint limit) under the "nominal" and "stressed"
     workspace placements defined in the experimental design, producing
     illustrative time series and summary statistics.

IMPORTANT: this is a software/parameter-verification tool, not a
substitute for the human-subject validation described in the paper's
experimental design (Section 5). It uses a synthetic human model (a
simple noisy admittance-like controller in Cartesian space, integrated
through the human-arm Jacobian), not real interaction forces or real
human motor behavior.

Usage:
    python3 sim/closed_loop_sim.py --placement nominal --n_users 30
    python3 sim/closed_loop_sim.py --placement stressed --n_users 30
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dh_utils import human_arm_wrist_position, cartesian_to_human_joint_velocity  # noqa: E402
from performance import DEFAULT_JOINT_LIMITS, manipulability_index  # noqa: E402
from path_follower import CirclePath, ReactivePathFollower  # noqa: E402
from shared_control_core import SharedControlCore  # noqa: E402
from sim.virtual_human_ik import solve_ik  # noqa: E402
from sim.franka_fk import franka_jacobian, NEAR_SINGULAR_Q, MID_WORKSPACE_Q, SimRobotModel  # noqa: E402


# Shoulder position in the robot base frame (simplifying assumption:
# frame {0} axes parallel to the robot base frame, only translated).
# Adjust to your actual chair/workstation geometry before trusting
# absolute placement numbers; the qualitative behavior of the control
# law does not depend on this offset.
SHOULDER_OFFSET = np.array([0.15, -0.35, 0.35])

# Null-space secondary-task gain used to resolve the human arm's 1-DOF
# kinematic redundancy by biasing it towards the center of each joint's
# range, instead of leaving it as an arbitrary artifact of the IK seed
# (see dh_utils.cartesian_to_human_joint_velocity / sim/virtual_human_ik.py).
HUMAN_CENTER_GAIN = 0.3


def make_virtual_user(rng, risky=False):
    """Randomize one synthetic participant's anthropometry and 'skill'."""
    l1 = rng.uniform(0.27, 0.34)
    l2 = rng.uniform(0.24, 0.31)
    noise_std = rng.uniform(0.005, 0.02)       # m/s, Cartesian command noise
    skill = rng.uniform(0.5, 1.0)              # 1 = perfectly directed command
    return {"l1": l1, "l2": l2, "noise_std": noise_std, "skill": skill,
            "risky": risky}


def placement_path(placement):
    # Direction from the shoulder, scaled to a fraction of a typical
    # combined upper-arm+forearm reach (~0.57 m for the anthropometry
    # ranges sampled in make_virtual_user), so the target is always
    # kinematically reachable by the 4-DoF human-arm model.
    direction = np.array([0.45, 0.55, 0.15])
    direction = direction / np.linalg.norm(direction)
    typical_reach = 0.57

    if placement == "nominal":
        center = SHOULDER_OFFSET + direction * (0.55 * typical_reach)
        radius = 0.05
        q_robot0 = MID_WORKSPACE_Q.copy()
    elif placement == "stressed":
        center = SHOULDER_OFFSET + direction * (0.85 * typical_reach)
        radius = 0.05
        q_robot0 = NEAR_SINGULAR_Q.copy()
    else:
        raise ValueError(placement)
    return CirclePath(center=center, radius=radius), q_robot0


DEFAULT_WEIGHTS = {"smoothness": 1.0, "directness": 1.0,
                    "joint_safety": 1.0, "manipulability": 1.0}


def run_trial(user, placement, active_factors, dt=0.02, duration=12.0, seed=0,
              weights=None, C1=1.0, C2=1.0, Cs=1.5, Cm=1.5, dt_lookahead=0.05):
    rng = np.random.default_rng(seed)
    path, q_robot = placement_path(placement)
    follower = ReactivePathFollower(path, Ka=2.0)
    core = SharedControlCore(
        weights=weights or DEFAULT_WEIGHTS,
        C1=C1, C2=C2, Cs=Cs, Cm=Cm, dt_lookahead=dt_lookahead,
        robot_model=SimRobotModel())

    l1, l2 = user["l1"], user["l2"]
    joint_limits = DEFAULT_JOINT_LIMITS

    # start the virtual human at a comfortable posture near the path start
    x0 = path.point(0.0)
    q_human = solve_ik(x0 - SHOULDER_OFFSET, np.array([0.5, 1.5, 0.0, 1.2]),
                        l1, l2, joint_limits)

    log = {"t": [], "err": [], "eta_h": [], "eta_r": [], "eta_s": [],
           "min_margin": [], "manip": [], "eta_joint_safety_h": [],
           "eta_manip_r": [], "qdot_robot_norm": [], "x": [], "x_d": [],
           "v_h": [], "v_s": []}

    n_steps = int(duration / dt)
    for step in range(n_steps):
        t = step * dt
        x = SHOULDER_OFFSET + human_arm_wrist_position(q_human, l1, l2)

        x_d, tangent, _ = follower.next_goal(x)
        v_r = follower.Ka * (x_d - x)

        # --- synthetic human command: admittance-like, noisy, and, for
        # "risky" virtual users, biased towards whichever joint is
        # already closest to its limit (to stress-test eta_joint_safety).
        v_h_intent = user["skill"] * 2.0 * (x_d - x)
        if user["risky"]:
            rho = (q_human - 0.5 * (joint_limits[:, 0] + joint_limits[:, 1])) \
                / (0.5 * (joint_limits[:, 1] - joint_limits[:, 0]))
            worst = np.argmax(np.abs(rho))
            bias_dir = np.sign(rho[worst])
            # small extra push expressed in Cartesian space via the arm Jacobian
            from dh_utils import human_arm_jacobian
            J = human_arm_jacobian(q_human, l1, l2)
            v_h_intent = v_h_intent + 0.4 * bias_dir * J[:, worst]
        v_h = v_h_intent + rng.normal(scale=user["noise_std"], size=3)

        # --- robot state: integrate q_robot assuming perfect tracking of v_s
        # Position-only (3x7) Jacobian throughout the simulator: avoids the
        # numerical instability of the finite-difference orientation
        # Jacobian near true wrist singularities (see sim/franka_fk.py).
        J_robot = franka_jacobian(q_robot)[:3, :]

        v_s, info = core.step(v_h, v_r, tangent, q_human=q_human, l1=l1, l2=l2,
                               q_robot=q_robot, J_robot=J_robot,
                               active_factors=active_factors)

        qdot_robot = np.linalg.pinv(J_robot[:3, :]) @ v_s
        q_robot = q_robot + qdot_robot * dt

        # Null-space, joint-centering secondary task (center_gain>0): the
        # arm's 1-DOF redundancy (3 Cartesian constraints, 4 joints) is no
        # longer resolved arbitrarily -- see dh_utils.py and
        # sim/virtual_human_ik.py docstrings. This only affects the
        # component of qdot_human that does NOT change the resulting
        # wrist velocity; the "risky" bias above (lines ~124-132), being
        # part of v_s's Cartesian direction, is unaffected by it.
        qdot_human = cartesian_to_human_joint_velocity(
            q_human, l1, l2, v_s, joint_limits=joint_limits,
            center_gain=HUMAN_CENTER_GAIN)
        q_human = np.clip(q_human + qdot_human * dt,
                           joint_limits[:, 0], joint_limits[:, 1])

        margins = 1.0 - np.abs((q_human - 0.5 * (joint_limits[:, 0] + joint_limits[:, 1]))
                                / (0.5 * (joint_limits[:, 1] - joint_limits[:, 0])))

        # Cross-track error to the NEAREST point on the path (not to the
        # lookahead goal x_d, which is deliberately ~rho ahead by
        # construction of the reactive pure-pursuit-style follower).
        s_near = path.nearest_s(x)
        cross_track_err = np.linalg.norm(path.point(s_near) - x)

        log["t"].append(t)
        log["err"].append(cross_track_err)
        log["eta_h"].append(info["eta_h"])
        log["eta_r"].append(info["eta_r"])
        log["eta_s"].append(info["eta_s"])
        log["min_margin"].append(float(np.min(margins)))
        log["manip"].append(manipulability_index(J_robot[:3, :]))
        log["eta_joint_safety_h"].append(info["factors_h"].get("joint_safety", np.nan))
        log["eta_manip_r"].append(info["factors_r"].get("manipulability", np.nan))
        log["qdot_robot_norm"].append(float(np.linalg.norm(qdot_robot)))
        log["x"].append(x.copy())
        log["x_d"].append(x_d.copy())
        log["v_h"].append(v_h.copy())
        log["v_s"].append(v_s.copy())

    return {k: np.array(v) for k, v in log.items()}


def monte_carlo(placement, n_users, active_factors, seed=0, out_csv=None):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_users):
        risky = (i % 3 == 0)  # ~1/3 of virtual users deliberately stress joint limits
        user = make_virtual_user(rng, risky=risky)
        result = run_trial(user, placement, active_factors, seed=seed + i)
        rows.append({
            "user": i, "risky": risky, "placement": placement,
            "mean_tracking_err": float(np.mean(result["err"])),
            "min_joint_margin": float(np.min(result["min_margin"])),
            "pct_time_margin_below_0.2": float(np.mean(result["min_margin"] < 0.2)),
            "min_manipulability": float(np.min(result["manip"])),
            "pct_time_manip_below_0.02": float(np.mean(result["manip"] < 0.02)),
        })

    if out_csv:
        import csv
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--placement", choices=["nominal", "stressed"], default="stressed")
    ap.add_argument("--n_users", type=int, default=20)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    for label, factors in [
        ("baseline_m2", ("smoothness", "directness")),
        ("extended_m4", ("smoothness", "directness", "joint_safety", "manipulability")),
    ]:
        out_csv = args.out or f"sim_{args.placement}_{label}.csv"
        rows = monte_carlo(args.placement, args.n_users, factors, out_csv=out_csv)
        margins = [r["min_joint_margin"] for r in rows]
        manips = [r["min_manipulability"] for r in rows]
        print(f"[{args.placement} | {label}] "
              f"min_joint_margin: mean={np.mean(margins):.3f} "
              f"min_manipulability: mean={np.mean(manips):.4f} "
              f"-> {out_csv}")
