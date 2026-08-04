"""
make_figures.py
----------------
Generates the paper figures that do NOT require real human-subject data,
i.e. everything supportable by the software already validated in
simulation (Section 4.6 of paper.tex). Saves vector PDFs directly to
../../paper/figures/, following IEEE's own graphics guidelines (vector
EPS/PDF for line art; https://journals.ieeeauthorcenter.ieee.org).

This script produces:
  fig_trajectory.pdf   -- traced circle, baseline (m=2) vs extended (m=4),
                           stressed placement, adversarial ("risky") user.
  fig_timeseries.pdf   -- eta_h/eta_r/eta_s, min joint margin, and
                           manipulability over time, same trial, extended
                           controller.

It intentionally reuses run_trial()/make_virtual_user()/placement_path()
unmodified from closed_loop_sim.py -- the same code path already used for
the gain tuning and the validation findings reported in the paper -- so
the figures are guaranteed to be consistent with the numbers quoted in
the text.

A companion script, make_figures_real.py (to be written once real trial
CSVs are available, ~3 weeks per the project timeline), will produce the
Results-section figures (Section 6) from real data using the same visual
style defined here (see PLOT_STYLE below) so that simulation and
real-data figures look consistent throughout the paper.

Usage:
    python3 sim/make_figures.py
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from closed_loop_sim import make_virtual_user, placement_path, run_trial  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "paper", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# PROVISIONAL post-kinematics-fix values (see sim/README.md and paper.tex
# Sec. 4.6): the manipulability gain is re-confirmed at this order after
# fixing the elbow-flexion DH bug; the joint_safety gain is NOT yet
# re-confirmed and is carried over from the pre-fix tuning pending a full
# Monte-Carlo re-run.
TUNED_WEIGHTS = {"smoothness": 1.0, "directness": 1.0,
                 "joint_safety": 16.0, "manipulability": 16.0}

# Shared visual style, reused by the future real-data figure script so
# simulation and experimental figures look consistent.
PLOT_STYLE = dict(
    baseline_color="#7f7f7f", baseline_ls="--",
    extended_color="#1f5aa6", extended_ls="-",
    ref_color="#c0392b", ref_ls=":",
    figsize_single=(3.4, 2.6),   # ~ IEEE single-column width, inches
    dpi=300,
)


def fig_trajectory(placement="nominal", duration=16.0):
    # NOTE: "stressed" deliberately places part of the reference circle
    # near/at the edge of a typical synthetic user's reach (by design, to
    # stress robot manipulability and human joint limits together -- see
    # paper.tex Sec. 5.1); some anthropometry draws under "stressed" may
    # therefore not complete a full geometric lap within the plotted
    # window. "nominal" is used here for a clean illustrative trace;
    # "stressed" is exercised quantitatively (not visually) in
    # fig_timeseries() below and in the Monte-Carlo statistics.
    rng = np.random.default_rng(7)
    user = make_virtual_user(rng, risky=False)
    path, _ = placement_path(placement)

    res_base = run_trial(user, placement, ("smoothness", "directness"),
                          seed=42, duration=duration, Cs=12.0, Cm=16.0, dt_lookahead=0.2)
    res_ext = run_trial(user, placement,
                         ("smoothness", "directness", "joint_safety", "manipulability"),
                         seed=42, duration=duration, weights=TUNED_WEIGHTS, Cs=12.0, Cm=16.0,
                         dt_lookahead=0.2)

    theta = np.linspace(0, 2 * np.pi, 200)
    ref_x = path.center[0] + path.radius * np.cos(theta)
    ref_y = path.center[1] + path.radius * np.sin(theta)

    fig, ax = plt.subplots(figsize=PLOT_STYLE["figsize_single"])
    ax.plot(ref_x, ref_y, PLOT_STYLE["ref_ls"], color=PLOT_STYLE["ref_color"],
            lw=1.2, label="Reference path")
    ax.plot(res_base["x"][:, 0], res_base["x"][:, 1],
            PLOT_STYLE["baseline_ls"], color=PLOT_STYLE["baseline_color"],
            lw=1.3, label="Baseline ($m$=2)")
    ax.plot(res_ext["x"][:, 0], res_ext["x"][:, 1],
            PLOT_STYLE["extended_ls"], color=PLOT_STYLE["extended_color"],
            lw=1.3, label="Extended ($m$=4, proposed)")
    ax.set_xlabel("$x$ (m)")
    ax.set_ylabel("$y$ (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(fontsize=7, loc="best", frameon=False)
    ax.set_title(f"Well-behaved user, {placement} placement", fontsize=9)
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "fig_trajectory.pdf")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


def fig_timeseries():
    # Adversarial ("risky") user, stressed placement: chosen deliberately
    # to stress-test the safety factors (Sec. 4.6), not to produce a
    # visually clean lap -- see fig_trajectory() for that.
    rng = np.random.default_rng(7)
    user = make_virtual_user(rng, risky=True)
    res = run_trial(user, "stressed",
                     ("smoothness", "directness", "joint_safety", "manipulability"),
                     seed=42, duration=16.0, weights=TUNED_WEIGHTS, Cs=12.0, Cm=16.0,
                     dt_lookahead=0.2)

    fig, axes = plt.subplots(3, 1, figsize=(3.4, 5.4), sharex=True)

    axes[0].plot(res["t"], res["eta_h"], label=r"$\eta_h$", lw=1.1)
    axes[0].plot(res["t"], res["eta_r"], label=r"$\eta_r$", lw=1.1)
    axes[0].plot(res["t"], res["eta_s"], label=r"$\eta_s$", lw=1.1, color="k", ls="--")
    axes[0].set_ylabel(r"$\eta$")
    axes[0].legend(fontsize=7, ncol=3, frameon=False, loc="upper right")

    axes[1].plot(res["t"], res["min_margin"], color="#1f5aa6", lw=1.1)
    axes[1].axhline(0.2, color=PLOT_STYLE["ref_color"], ls=":", lw=1.0,
                     label=r"pre-registered $\tau$")
    axes[1].set_ylabel("min joint\nmargin $m_i$")
    axes[1].legend(fontsize=7, frameon=False, loc="upper right")

    axes[2].plot(res["t"], res["manip"], color="#1f5aa6", lw=1.1)
    axes[2].axhline(0.02, color=PLOT_STYLE["ref_color"], ls=":", lw=1.0,
                     label="min. acceptable $w$")
    axes[2].set_ylabel(r"manipulability $w(q_r)$")
    axes[2].set_xlabel("time (s)")
    axes[2].legend(fontsize=7, frameon=False, loc="upper right")

    fig.align_ylabels(axes)
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "fig_timeseries.pdf")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    fig_trajectory()
    fig_timeseries()
