"""
benchmark.py
------------
Reproducible timing benchmark for SharedControlCore.step(), used to
argue real-time feasibility at the FR3's ~1 kHz Cartesian velocity
control rate. Reports, per control cycle:

  * baseline (m=2, smoothness+directness only)
  * extended (m=4) with the analytic, O(1) robot Jacobian access that
    the real deployment uses (RobotModel backend "franka_state", which
    simply reads the last O_Jac_EE reported by franka_ros)
  * extended (m=4) WORST CASE, where the robot Jacobian is instead
    recomputed from scratch via finite differences for every candidate
    (as it would be if no analytic/KDL Jacobian were available) --
    included specifically to show that this naive alternative would
    VIOLATE the 1 kHz budget, motivating the use of franka_ros' exact
    O_Jac_EE (or PyKDL) rather than any from-scratch numerical Jacobian.

Usage:
    python3 sim/benchmark.py
"""
import sys
import os
import platform
import timeit

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared_control_core import SharedControlCore  # noqa: E402
from robot_model import RobotModel  # noqa: E402
from sim.franka_fk import franka_jacobian, MID_WORKSPACE_Q, SimRobotModel  # noqa: E402

TUNED_WEIGHTS = {"smoothness": 1.0, "directness": 1.0,
                 "joint_safety": 16.0, "manipulability": 24.0}
N = 3000
CONTROL_PERIOD_US = 1000.0  # 1 kHz -> 1000 us budget per cycle


def bench(core, **kwargs):
    def one_step():
        core.step(v_h, v_r, tangent, **kwargs)
    t = timeit.timeit(one_step, number=N) / N
    return t * 1e6  # microseconds


def print_environment():
    """Prints the exact hardware/software environment these numbers were
    measured on. IMPORTANT: this is the development/CI sandbox used to
    write and validate this code, NOT necessarily the real-time control
    PC that will run shared_control_node.py on the FR3. It is an ARM64
    (aarch64) Linux VM (uname reports 'Vendor ID: Apple', i.e. a Linux
    guest virtualized on Apple-Silicon host hardware), running a
    generic (non-PREEMPT_RT) kernel. The target deployment machine
    described in the sensor-system paper is an x86_64 PC with a
    real-time-patched Linux kernel. Absolute timings measured here are
    NOT expected to transfer 1:1 to that machine (ARM vs x86, no RT
    kernel, virtualization overhead, different core/cache layout);
    they should be treated as order-of-magnitude / relative-comparison
    evidence (e.g. extended vs baseline, cached vs from-scratch
    Jacobian) pending re-benchmarking on the actual control PC.
    """
    u = platform.uname()
    print("=" * 70)
    print("Benchmark environment (report alongside all timings below):")
    print(f"  system/kernel : {u.system} {u.release} ({u.version.strip()})")
    print(f"  machine/arch  : {u.machine}")
    print(f"  processor     : {u.processor or 'n/a (not exposed by this platform)'}")
    print(f"  python        : {platform.python_version()}")
    try:
        print(f"  numpy         : {np.__version__}")
    except Exception:
        pass
    print("  NOTE: development/CI sandbox (ARM64 Linux VM on Apple-Silicon")
    print("        host, non-real-time kernel) -- NOT the target x86_64")
    print("        real-time control PC. Treat absolute figures as")
    print("        provisional; re-benchmark on deployment hardware.")
    print("=" * 70)


if __name__ == "__main__":
    print_environment()
    q_h = np.array([1.0, 1.5, 0.0, 1.2])
    l1, l2 = 0.30, 0.27
    v_h = np.array([0.02, 0.01, 0.0])
    v_r = np.array([0.015, 0.02, 0.0])
    tangent = np.array([0.0, 1.0, 0.0])
    J_robot = franka_jacobian(MID_WORKSPACE_Q)[:3, :]

    baseline = SharedControlCore(weights={"smoothness": 1.0, "directness": 1.0},
                                  C1=1.0, C2=1.0, robot_model=None)
    t_base = bench(baseline, active_factors=("smoothness", "directness"))

    rm_real = RobotModel(backend="franka_state")
    rm_real.update_current_state(MID_WORKSPACE_Q, J_robot)
    extended_real = SharedControlCore(weights=TUNED_WEIGHTS, C1=1.0, C2=1.0,
                                       Cs=12.0, Cm=24.0, dt_lookahead=0.2,
                                       robot_model=rm_real)
    t_ext_real = bench(extended_real, q_human=q_h, l1=l1, l2=l2,
                        q_robot=MID_WORKSPACE_Q, J_robot=J_robot)

    extended_worst = SharedControlCore(weights=TUNED_WEIGHTS, C1=1.0, C2=1.0,
                                        Cs=12.0, Cm=24.0, dt_lookahead=0.2,
                                        robot_model=SimRobotModel())
    t_ext_worst = bench(extended_worst, q_human=q_h, l1=l1, l2=l2,
                         q_robot=MID_WORKSPACE_Q, J_robot=J_robot)

    for label, t in [
        ("baseline (m=2)", t_base),
        ("extended (m=4), analytic O_Jac_EE [real deployment]", t_ext_real),
        ("extended (m=4), from-scratch numeric robot Jacobian [worst case]", t_ext_worst),
    ]:
        margin = 100.0 * (1.0 - t / CONTROL_PERIOD_US)
        print(f"{label:60s}: {t:8.2f} us/cycle  "
              f"(max ~{1e6/t:7.0f} Hz, {margin:5.1f}% margin at 1 kHz)")
