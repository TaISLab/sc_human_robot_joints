#!/usr/bin/env python3
"""
shared_control_node.py
-----------------------
ROS (Noetic) node that computes the emergent shared Cartesian velocity
command v_s for the FR3, given:

  INPUTS (already produced by existing nodes, NOT reimplemented here):
    * /admittance_control/human_velocity   (geometry_msgs/TwistStamped)
        v_h, computed by the existing admittance/impedance controller
        from the gripper's interaction-force estimate.
    * /human_arm/joint_state               (sensor_msgs/JointState)
        q1..q4 (rad) + optionally l1, l2 (upper-arm / forearm length,
        m) published by the visuo-tactile perception pipeline.
    * /franka_state_controller/franka_states (franka_msgs/FrankaState)
        current robot joint configuration q, and the 6x7 zero-Jacobian
        O_Jac_EE, both published natively by franka_ros.

  OUTPUT:
    * /shared_control/cartesian_velocity_command (geometry_msgs/TwistStamped)
        v_s, to be consumed directly by Franka's Cartesian velocity
        controller (franka_example_controllers/CartesianVelocityController
        or an equivalent custom controller).

This node implements ONLY the missing piece requested: combination of
the human command and the reactive path-following robot command,
weighted by FOUR local performance factors (smoothness, directness,
and the two novel factors: human joint-limit safety and robot
manipulability / singularity avoidance).
"""

import numpy as np
import rospy
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import JointState
from franka_msgs.msg import FrankaState

from path_follower import CirclePath, ReactivePathFollower
from robot_model import RobotModel
from shared_control_core import SharedControlCore


def _vec3(twist):
    return np.array([twist.twist.linear.x, twist.twist.linear.y, twist.twist.linear.z])


def _twist_msg(v, frame_id="panda_link0"):
    msg = TwistStamped()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = frame_id
    msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z = v
    return msg


class SharedControlNode(object):
    def __init__(self):
        rospy.init_node("shared_control_node")

        # ---- parameters -------------------------------------------------
        p = rospy.get_param
        self.dt_lookahead = p("~dt_lookahead", 0.05)      # s, for J prediction
        self.v_max = p("~v_max", 0.15)                     # m/s, EE speed cap
        self.alpha_lpf = p("~lpf_alpha", 0.2)               # low-pass filter
        self.weights = p("~performance_weights", {
            "smoothness": 1.0, "directness": 1.0,
            "joint_safety": 1.0, "manipulability": 1.0,
        })
        self.C1 = p("~C_smoothness", 1.0)
        self.C2 = p("~C_directness", 1.0)
        self.Cs = p("~C_joint_safety", 1.0)
        self.Cm = p("~C_manipulability", 1.0)
        self.default_l1 = p("~upper_arm_length", 0.30)
        self.default_l2 = p("~forearm_length", 0.27)

        center = p("~path/center", [0.5, 0.0, 0.4])
        radius = p("~path/radius", 0.075)
        Ka = p("~path/Ka", 2.0)
        path = CirclePath(center=center, radius=radius)
        self.follower = ReactivePathFollower(path, Ka=Ka)

        backend = p("~robot_model_backend", "franka_state")
        self.robot_model = RobotModel(backend=backend)

        self.core = SharedControlCore(
            dt_lookahead=self.dt_lookahead, v_max=self.v_max,
            lpf_alpha=self.alpha_lpf, weights=self.weights,
            C1=self.C1, C2=self.C2, Cs=self.Cs, Cm=self.Cm,
            robot_model=self.robot_model)

        # ---- state --------------------------------------------------
        self.q_human = None
        self.l1, self.l2 = self.default_l1, self.default_l2
        self.q_robot = None
        self.J_robot = None
        self.x_robot = None

        # ---- ROS I/O --------------------------------------------------
        rospy.Subscriber("/admittance_control/human_velocity", TwistStamped,
                          self._on_human_velocity, queue_size=1)
        rospy.Subscriber("/human_arm/joint_state", JointState,
                          self._on_human_joint_state, queue_size=1)
        rospy.Subscriber("/franka_state_controller/franka_states", FrankaState,
                          self._on_franka_state, queue_size=1)
        self.pub = rospy.Publisher(
            "/shared_control/cartesian_velocity_command", TwistStamped,
            queue_size=1)

        self.v_h = np.zeros(3)
        self._ready = False

    # -- callbacks ------------------------------------------------------
    def _on_human_velocity(self, msg):
        self.v_h = _vec3(msg)

    def _on_human_joint_state(self, msg):
        name_to_pos = dict(zip(msg.name, msg.position))
        try:
            self.q_human = np.array([name_to_pos["q1"], name_to_pos["q2"],
                                      name_to_pos["q3"], name_to_pos["q4"]])
        except KeyError:
            rospy.logwarn_throttle(5.0, "human_arm/joint_state missing q1..q4")
            return
        self.l1 = name_to_pos.get("l1", self.default_l1)
        self.l2 = name_to_pos.get("l2", self.default_l2)

    def _on_franka_state(self, msg):
        # O_Jac_EE is column-major, 6x7, as published by franka_ros.
        J = np.array(msg.O_Jac_EE).reshape(6, 7, order="F")
        q = np.array(msg.q)
        x = np.array(msg.O_T_EE).reshape(4, 4, order="F")[:3, 3]
        self.J_robot, self.q_robot, self.x_robot = J, q, x
        self.robot_model.update_current_state(q, J)
        self._ready = True

    # -- core computation -------------------------------------------------
    def step(self):
        if not self._ready or self.q_robot is None:
            return

        v_r, tangent = self.follower.robot_command(self.x_robot)

        v_out, _info = self.core.step(
            self.v_h, v_r, tangent,
            q_human=self.q_human, l1=self.l1, l2=self.l2,
            q_robot=self.q_robot, J_robot=self.J_robot)

        self.pub.publish(_twist_msg(v_out))

    def run(self, rate_hz=1000.0):
        rate = rospy.Rate(rate_hz)
        while not rospy.is_shutdown():
            self.step()
            rate.sleep()


if __name__ == "__main__":
    node = SharedControlNode()
    node.run()
