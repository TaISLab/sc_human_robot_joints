"""
passivity_observer.py
----------------------
Time-domain passivity observer (Hannaford & Ryu, "Time-domain passivity
control of haptic interfaces", IEEE T-RO 2002) applied to the
human-facing port of the shared-control law, over the offline simulator.

WHY: the shared command is v_s = eta_s*(eta_r*v_r + eta_h*v_h), i.e. a
CLASSICAL admittance blend whose gains (eta_s, eta_h, eta_r) are not
constant but STATE-dependent, computed online from the current candidate
commands and kinematic margins. Multiplying the output of a passive
admittance block by a state-dependent (not merely time-varying) gain does
NOT automatically preserve passivity -- this is the well-documented
variable-impedance/-admittance passivity pitfall (Ferraguti, Secchi &
Fantuzzi, "A tank-based approach to impedance control with variable
stiffness", ICRA 2013; extended to variable admittance in Ferraguti et
al. 2015). This script does not prove passivity (that would require the
energy-tank redesign discussed in paper.tex Sec. 4.8, left as future
work); it MONITORS it empirically over simulated trials, using the
classical passivity-observer construction:

    PO(t) = E0 + integral_0^t F_h(tau)^T v_s(tau) dtau

If PO(t) stays >= 0 for all t, the tested trajectory is consistent with
passivity of the map from the human's effort F_h to the actually
EXECUTED command v_s (the controller never gave back more energy at the
port than it received). If PO(t) goes negative, that instant is a
passivity violation -- evidence (not proof, since only specific
trajectories are tested) that the variable-gain blend can act as a net
energy source.

CAVEAT (important, stated explicitly rather than glossed over): the
offline simulator does not model human FORCE at all -- the "virtual
patient" is synthesized directly in Cartesian VELOCITY space (v_h), by
design, since the real admittance block is treated as an unmodified,
external, black-box module (see closed_loop_sim.py). To evaluate a
force-based passivity observer, F_h is RECONSTRUCTED here by inverting an
ASSUMED first-order virtual admittance, F_h = M*dv_h/dt + D*v_h, with
illustrative M, D (not the real deployed admittance's actual parameters,
which are internal to that separate, already-implemented module and were
never exposed to this paper's code). Results here should therefore be
read as a proof-of-concept demonstration of the METHOD, and an honest
empirical check under one plausible admittance model -- not a certified
passivity result for the real hardware's actual admittance parameters.

Usage:
    python3 sim/passivity_observer.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from closed_loop_sim import make_virtual_user, run_trial  # noqa: E402

# Illustrative virtual admittance parameters used ONLY to reconstruct F_h
# from the synthetic v_h(t) for this observer -- see module docstring.
M_VIRTUAL = 1.0     # kg, illustrative virtual mass
D_VIRTUAL = 8.0      # N.s/m, illustrative virtual damping

TUNED_WEIGHTS = {"smoothness": 1.0, "directness": 1.0,
                 "joint_safety": 16.0, "manipulability": 16.0}


def reconstruct_force(v_h, dt):
    """F_h = M dv_h/dt + D v_h, via simple finite differences."""
    dv = np.gradient(v_h, dt, axis=0)
    return M_VIRTUAL * dv + D_VIRTUAL * v_h


def passivity_observer(F_h, v_s, dt, E0=0.0):
    """PO(t) = E0 + cumulative integral of F_h . v_s (trapezoidal)."""
    power = np.sum(F_h * v_s, axis=1)  # (N,) instantaneous port power
    energy_increment = 0.5 * (power[:-1] + power[1:]) * dt
    po = np.empty(len(power))
    po[0] = E0
    po[1:] = E0 + np.cumsum(energy_increment)
    return po, power


def run_check(placement, active_factors, weights, label, n_users=4, duration=16.0,
              dt=0.02, Cs=12.0, Cm=16.0, dt_lookahead=0.2):
    violations = 0
    min_po_overall = np.inf
    for seed_u in range(n_users):
        rng = np.random.default_rng(seed_u)
        user = make_virtual_user(rng, risky=(seed_u % 2 == 0))
        res = run_trial(user, placement, active_factors, duration=duration, dt=dt,
                         seed=100 + seed_u, weights=weights, Cs=Cs, Cm=Cm,
                         dt_lookahead=dt_lookahead)
        F_h = reconstruct_force(res["v_h"], dt)
        po, power = passivity_observer(F_h, res["v_s"], dt)
        min_po = po.min()
        min_po_overall = min(min_po_overall, min_po)
        if min_po < -1e-6:
            violations += 1
        print(f"  [{label}] user={seed_u} risky={user['risky']:<5} "
              f"min(PO)={min_po:9.5f} J  final(PO)={po[-1]:9.5f} J  "
              f"{'VIOLATION' if min_po < -1e-6 else 'ok'}")
    print(f"  -> {label}: {violations}/{n_users} trials with a passivity-observer "
          f"violation, worst-case min(PO)={min_po_overall:.5f} J\n")
    return violations, min_po_overall


if __name__ == "__main__":
    print(f"Virtual admittance used to reconstruct F_h: M={M_VIRTUAL} kg, "
          f"D={D_VIRTUAL} N.s/m (illustrative -- see module docstring)\n")

    print("=== baseline (m=2), stressed placement ===")
    run_check("stressed", ("smoothness", "directness"), None, "baseline")

    print("=== extended (m=4), stressed placement, provisional gains ===")
    run_check("stressed", ("smoothness", "directness", "joint_safety", "manipulability"),
              TUNED_WEIGHTS, "extended")

    print("=== baseline (m=2), nominal placement ===")
    run_check("nominal", ("smoothness", "directness"), None, "baseline")

    print("=== extended (m=4), nominal placement, provisional gains ===")
    run_check("nominal", ("smoothness", "directness", "joint_safety", "manipulability"),
              TUNED_WEIGHTS, "extended")
