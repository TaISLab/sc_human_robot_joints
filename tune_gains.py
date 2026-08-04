"""
tune_gains.py
--------------
Paired parameter sweep for the manipulability factor (C_m, its
performance weight, and the prediction lookahead dt_lookahead):
for each candidate parameter set, runs the SAME set of synthetic
virtual users/seeds under the baseline (m=2) and the extended (m=4)
controller in the "stressed" placement, and reports the paired
improvement (extended - baseline) in:

  * min_manipulability (want: less negative / more positive, i.e. higher)
  * pct_time_manip_below_thresh (want: negative, i.e. less time unsafe)
  * peak joint speed while manipulability is low (want: negative, i.e. gentler)
  * mean_tracking_err (want: not much worse, sanity check)

Usage:
    python3 sim/tune_gains.py
"""
import sys
import os
import itertools

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sim.closed_loop_sim import run_trial, make_virtual_user, DEFAULT_WEIGHTS  # noqa: E402

MANIP_THRESH = 0.02
N_SEEDS = 2
DURATION = 4.0
DT = 0.04


def evaluate(weight_manip, Cm, dt_lookahead, placement="stressed"):
    rng = np.random.default_rng(42)
    users = [make_virtual_user(rng, risky=(i % 2 == 0)) for i in range(N_SEEDS)]

    weights_ext = dict(DEFAULT_WEIGHTS)
    weights_ext["manipulability"] = weight_manip

    diffs = {"d_min_manip": [], "d_pct_below": [], "d_peak_qdot_low_manip": [],
             "d_mean_err": []}

    for i, user in enumerate(users):
        seed = 100 + i
        base = run_trial(user, placement, ("smoothness", "directness"),
                          duration=DURATION, seed=seed, dt=DT)
        ext = run_trial(user, placement,
                         ("smoothness", "directness", "joint_safety", "manipulability"),
                         duration=DURATION, seed=seed, dt=DT,
                         weights=weights_ext, Cm=Cm, dt_lookahead=dt_lookahead)

        def peak_qdot_low_manip(res):
            mask = res["manip"] < MANIP_THRESH
            return float(np.max(res["qdot_robot_norm"][mask])) if mask.any() else 0.0

        diffs["d_min_manip"].append(np.min(ext["manip"]) - np.min(base["manip"]))
        diffs["d_pct_below"].append(np.mean(ext["manip"] < MANIP_THRESH)
                                     - np.mean(base["manip"] < MANIP_THRESH))
        diffs["d_peak_qdot_low_manip"].append(
            peak_qdot_low_manip(ext) - peak_qdot_low_manip(base))
        diffs["d_mean_err"].append(np.mean(ext["err"]) - np.mean(base["err"]))

    return {k: float(np.mean(v)) for k, v in diffs.items()}


if __name__ == "__main__":
    grid_weight_manip = [1.0, 4.0, 8.0]
    grid_Cm = [1.5, 4.0]
    grid_dt = [0.05, 0.2]

    print(f"{'w_manip':>8} {'Cm':>5} {'dt_la':>6} | "
          f"{'d_min_manip':>12} {'d_pct_below':>12} {'d_peak_qdot':>12} {'d_mean_err':>10}")
    results = []
    for w, cm, dt in itertools.product(grid_weight_manip, grid_Cm, grid_dt):
        r = evaluate(w, cm, dt)
        results.append((w, cm, dt, r))
        print(f"{w:8.1f} {cm:5.1f} {dt:6.2f} | "
              f"{r['d_min_manip']:12.5f} {r['d_pct_below']:12.4f} "
              f"{r['d_peak_qdot_low_manip']:12.4f} {r['d_mean_err']:10.4f}")

    # Rank by: higher d_min_manip, more negative d_pct_below and
    # d_peak_qdot_low_manip, without a large increase in tracking error.
    def score(r):
        return (r["d_min_manip"] - r["d_pct_below"] - 0.5 * r["d_peak_qdot_low_manip"]
                - 2.0 * max(0.0, r["d_mean_err"]))

    best = max(results, key=lambda x: score(x[3]))
    print("\nBest combo (w_manip, Cm, dt_lookahead):", best[0], best[1], best[2])
    print("Metrics:", best[3])
