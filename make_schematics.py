"""
make_schematics.py
-------------------
Generates the paper's *schematic* figures -- the ones that illustrate the
control architecture, the human-arm kinematic model, and the experimental
workspace layout. Unlike make_figures.py, these do NOT depend on
simulation output or human-subject data: they are drawn directly from the
system's known geometry/architecture, so they can be finalized now, ahead
of the real experiments.

Produces (vector PDFs, saved to ../../paper/figures/):
  fig_architecture.pdf -- block diagram of the shared-control law (Sec. 4).
  fig_kinematics.pdf   -- 4-DoF human-arm model + joint-margin concept (Sec. 4.2).
  fig_workspace.pdf    -- top-down nominal vs. stressed placement (Sec. 5.1).

Visual style intentionally reuses make_figures.py's PLOT_STYLE color coding
so all figures in the paper read consistently:
  gray, dashed   = existing/baseline element (Ruiz-Ruiz et al. framework)
  blue, solid    = new/proposed element (this paper's contribution)
  red, dotted    = limit / reference quantity

Usage:
    python3 sim/make_schematics.py
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Arc, Wedge
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from closed_loop_sim import SHOULDER_OFFSET, placement_path  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "paper", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

GRAY = "#7f7f7f"     # existing/baseline
BLUE = "#1f5aa6"     # new/proposed (this paper)
RED = "#c0392b"      # limit/reference
LIGHTBLUE = "#dce8f5"
LIGHTGRAY = "#eaeaea"


# --------------------------------------------------------------------------
def _box(ax, xy, w, h, text, fc="white", ec="black", lw=1.0, fontsize=7.2,
          zorder=3, style="round,pad=0.02,rounding_size=0.02"):
    x, y = xy
    patch = FancyBboxPatch((x, y), w, h, boxstyle=style,
                            fc=fc, ec=ec, lw=lw, zorder=zorder)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
             fontsize=fontsize, zorder=zorder + 1, linespacing=1.25)
    return patch


def _arrow(ax, p0, p1, color="black", lw=1.1, style="-|>", mutation=8,
            connectionstyle=None, zorder=2):
    arr = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=mutation,
                           color=color, lw=lw, connectionstyle=connectionstyle,
                           zorder=zorder, shrinkA=0, shrinkB=0)
    ax.add_patch(arr)


# --------------------------------------------------------------------------
def fig_architecture():
    fig, ax = plt.subplots(figsize=(7.0, 3.55))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 52)
    ax.axis("off")

    # -- existing (gray) input blocks ------------------------------------
    _box(ax, (1, 36), 17, 8, "Interaction\nforce (gripper)", fc=LIGHTGRAY, ec=GRAY)
    _box(ax, (1, 24), 17, 8, "Path follower\n(reactive, virtual sphere)", fc=LIGHTGRAY, ec=GRAY)
    _box(ax, (1, 12), 17, 8, "Visuo-tactile\nperception: $q_{1..4}$, $l_1,l_2$", fc=LIGHTGRAY, ec=GRAY)
    _box(ax, (1, 0), 17, 8, "Robot state:\n$q_r$, $J_r(q_r)$", fc=LIGHTGRAY, ec=GRAY)

    _box(ax, (22, 36), 15, 8, "Admittance law\n(unmodified)", fc=LIGHTGRAY, ec=GRAY)
    ax.text(38.6, 40, r"$v_h$", fontsize=8, va="center")

    _box(ax, (22, 24), 15, 8, "Path velocity\n$v_r=K_a(x_d-x)$", fc=LIGHTGRAY, ec=GRAY)
    ax.text(38.6, 28, r"$v_r$", fontsize=8, va="center")

    # -- performance factors (baseline gray + new blue) -------------------
    _box(ax, (22, 12), 15, 8, "Smoothness $+$\ndirectness", fc=LIGHTGRAY, ec=GRAY, fontsize=6.8)
    _box(ax, (39, 12), 16, 8, "Joint-limit safety\n$\\eta_{\\mathrm{js}}(q_{1..4})$", fc=LIGHTBLUE, ec=BLUE, fontsize=6.8)
    _box(ax, (39, 0), 16, 8, "Manipulability\n$\\eta_{\\mathrm{man}}(q_r)$", fc=LIGHTBLUE, ec=BLUE, fontsize=6.8)

    # -- combination block --------------------------------------------------
    _box(ax, (58, 18), 20, 20, "Performance-weighted\ncombination\n$\\eta_h,\\ \\eta_r,\\ \\eta_s$\n(existing 2 factors +\n2 new factors)",
         fc="white", ec="black", lw=1.3, fontsize=7.0)

    _box(ax, (81, 20), 17, 12,
         "$v_s=\\eta_s(\\eta_r v_r+\\eta_h v_h)$", fc="white", ec="black", lw=1.3, fontsize=7.4)

    # output
    _box(ax, (81, 4), 17, 10, "Admittance-controlled\nFR3 end-effector", fc=LIGHTGRAY, ec=GRAY, fontsize=6.8)

    # -- arrows --------------------------------------------------------------
    _arrow(ax, (18, 40), (22, 40))
    _arrow(ax, (37, 40), (58, 30), connectionstyle="arc3,rad=-0.15")
    _arrow(ax, (18, 28), (22, 28))
    _arrow(ax, (37, 28), (58, 27), connectionstyle="arc3,rad=-0.05")

    _arrow(ax, (18, 16), (22, 16))
    _arrow(ax, (37, 16), (58, 26), connectionstyle="arc3,rad=0.15")

    _arrow(ax, (18, 4), (39, 4), connectionstyle="arc3,rad=0.0")
    _arrow(ax, (18, 16), (39, 16), connectionstyle="arc3,rad=0.35")
    _arrow(ax, (47, 12), (58, 24), connectionstyle="arc3,rad=-0.15", color=BLUE)
    _arrow(ax, (47, 4), (58, 20), connectionstyle="arc3,rad=-0.1", color=BLUE)

    _arrow(ax, (78, 28), (81, 27))
    _arrow(ax, (89.5, 20), (89.5, 14))

    # legend swatches, aligned with their labels (avoid raw \cite{} in
    # matplotlib text -- it is not run through LaTeX here)
    ax.add_patch(FancyBboxPatch((58, 46.2), 3.2, 2.6, boxstyle="round,pad=0.02",
                                 fc=LIGHTGRAY, ec=GRAY, lw=1.0))
    ax.text(62.2, 47.5, "Existing (Ruiz-Ruiz et al., 2023)",
            fontsize=7, color=GRAY, va="center")
    ax.add_patch(FancyBboxPatch((58, 43.0), 3.2, 2.6, boxstyle="round,pad=0.02",
                                 fc=LIGHTBLUE, ec=BLUE, lw=1.0))
    ax.text(62.2, 44.3, "New in this paper", fontsize=7, color=BLUE, va="center")

    fig.tight_layout(pad=0.3)
    out_path = os.path.join(OUT_DIR, "fig_architecture.pdf")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


# --------------------------------------------------------------------------
def fig_kinematics():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))

    # --- (a) planar sketch of the 4-DoF arm model -------------------------
    ax = axes[0]
    l1, l2 = 0.30, 0.27
    q1, q4 = np.deg2rad(35), np.deg2rad(75)  # illustrative shoulder flex, elbow flex
    shoulder = np.array([0.0, 0.0])
    elbow = shoulder + l1 * np.array([np.cos(q1), np.sin(q1)])
    wrist = elbow + l2 * np.array([np.cos(q1 - q4), np.sin(q1 - q4)])

    ax.plot(*zip(shoulder, elbow), color=BLUE, lw=2.4, solid_capstyle="round")
    ax.plot(*zip(elbow, wrist), color=BLUE, lw=2.4, solid_capstyle="round")
    for pt, lbl, off in [(shoulder, "shoulder\n($q_1,q_2,q_3$)", (-0.02, -0.075)),
                          (elbow, "elbow\n($q_4$)", (0.015, 0.01)),
                          (wrist, "wrist", (0.01, -0.02))]:
        ax.scatter(*pt, s=28, color="black", zorder=5)
        ax.annotate(lbl, pt, xytext=(pt[0] + off[0], pt[1] + off[1]), fontsize=6.6)

    ax.annotate("", xy=elbow, xytext=shoulder,
                arrowprops=dict(arrowstyle="-", color="none"))
    ax.text(*(0.5 * (shoulder + elbow) + np.array([-0.03, 0.02])), "$l_1$", fontsize=7.5)
    ax.text(*(0.5 * (elbow + wrist) + np.array([0.0, 0.02])), "$l_2$", fontsize=7.5)

    # elbow flexion angle arc between upper-arm and forearm directions
    ang_upper = np.degrees(q1 + np.pi)
    ang_fore = np.degrees(q1 - q4 + np.pi)
    arc = Arc(elbow, 0.10, 0.10, angle=0, theta1=min(ang_upper, ang_fore),
              theta2=max(ang_upper, ang_fore), color=RED, lw=1.2)
    ax.add_patch(arc)

    # shoulder reference axis + q1 arc
    ax.plot([shoulder[0], shoulder[0] + 0.18], [shoulder[1], shoulder[1]],
            color="gray", lw=0.8, ls=":")
    arc1 = Arc(shoulder, 0.14, 0.14, angle=0, theta1=0, theta2=np.degrees(q1),
               color=RED, lw=1.2)
    ax.add_patch(arc1)
    ax.text(0.10, 0.045, "$q_1$", fontsize=7, color=RED)
    ax.text(elbow[0] + 0.02, elbow[1] - 0.075, "$q_4$", fontsize=7, color=RED)

    ax.text(0.02, -0.16,
            "$q_2,q_3$: shoulder ab/adduction $+$\naxial rotation (out of plane, not shown)",
            fontsize=6.2, style="italic", color="0.35")

    ax.set_xlim(-0.10, 0.42)
    ax.set_ylim(-0.22, 0.32)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("(a) 4-DoF human-arm model", fontsize=8.2)

    # --- (b) joint-margin concept -----------------------------------------
    ax2 = axes[1]
    lo, hi = -10, 110
    q = 82  # current joint value, close to its upper limit
    mid = 0.5 * (lo + hi)
    half_range = 0.5 * (hi - lo)
    margin = 1.0 - abs(q - mid) / half_range

    ax2.add_patch(plt.Rectangle((lo, 0.4), hi - lo, 0.5, fc=LIGHTGRAY, ec="none"))
    tau = 0.2
    band_lo = mid - (1 - tau) * half_range
    band_hi = mid + (1 - tau) * half_range
    ax2.add_patch(plt.Rectangle((lo, 0.4), band_lo - lo, 0.5, fc="#f6d0cb", ec="none"))
    ax2.add_patch(plt.Rectangle((band_hi, 0.4), hi - band_hi, 0.5, fc="#f6d0cb", ec="none"))
    ax2.axvline(lo, color="black", lw=1.2)
    ax2.axvline(hi, color="black", lw=1.2)
    ax2.axvline(mid, color="gray", lw=0.8, ls=":")
    ax2.scatter([q], [0.65], color=BLUE, s=45, zorder=5)
    ax2.annotate(f"$q_i$", (q, 0.65), xytext=(q + 2, 1.0), fontsize=8, color=BLUE)
    ax2.annotate("", xy=(hi, 0.65), xytext=(q, 0.65),
                 arrowprops=dict(arrowstyle="<->", color=RED, lw=1.1))
    ax2.text((q + hi) / 2, 0.15, "margin\n$m_i$", fontsize=6.8, color=RED, ha="center")
    ax2.text(lo, -0.35, "$q_i^{\\min}$", fontsize=7, ha="center")
    ax2.text(hi, -0.35, "$q_i^{\\max}$", fontsize=7, ha="center")
    ax2.text((lo + band_lo) / 2, -0.35, "", fontsize=6)
    ax2.text(band_hi + (hi - band_hi) / 2, -0.6, "$m_i<\\tau$\n(penalized)",
              fontsize=6.4, color="#a33", ha="center")

    ax2.set_xlim(lo - 5, hi + 5)
    ax2.set_ylim(-0.9, 1.3)
    ax2.axis("off")
    ax2.set_title("(b) joint-margin factor $\\eta_{\\mathrm{js}}$", fontsize=8.2)

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "fig_kinematics.pdf")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


# --------------------------------------------------------------------------
def fig_workspace():
    fig, ax = plt.subplots(figsize=(3.4, 3.0))

    direction = np.array([0.45, 0.55, 0.15])
    direction = direction / np.linalg.norm(direction)
    typical_reach = 0.57
    shoulder_xy = SHOULDER_OFFSET[:2]
    dir_xy = direction[:2] / np.linalg.norm(direction[:2])

    # reach boundary (arc of the typical combined-arm-length workspace)
    reach_circle = plt.Circle(shoulder_xy, typical_reach, fill=False,
                               color="gray", lw=1.0, ls=":")
    ax.add_patch(reach_circle)
    reach_label_xy = shoulder_xy + typical_reach * np.array([np.cos(np.deg2rad(20)),
                                                                np.sin(np.deg2rad(20))])
    ax.annotate("typical max. reach", reach_label_xy,
                xytext=(reach_label_xy[0] - 0.02, reach_label_xy[1] + 0.02),
                fontsize=6.2, color="0.4", ha="right")

    for placement, frac, color, ls, label in [
            ("nominal", 0.55, GRAY, "--", "Nominal"),
            ("stressed", 0.85, BLUE, "-", "Stressed")]:
        path, _ = placement_path(placement)
        center = path.center[:2]
        circ = plt.Circle(center, path.radius, fill=False, color=color, lw=1.6, ls=ls)
        ax.add_patch(circ)
        ax.plot([shoulder_xy[0], center[0]], [shoulder_xy[1], center[1]],
                color=color, lw=0.7, ls=":")
        ax.annotate(f"{label} ({int(frac*100)}% reach)", center,
                    xytext=(center[0] + 0.035, center[1] + 0.05), fontsize=6.8, color=color)

    ax.scatter(*shoulder_xy, color="black", s=35, zorder=5)
    ax.annotate("shoulder", shoulder_xy, xytext=(shoulder_xy[0] - 0.10, shoulder_xy[1] - 0.05),
                fontsize=7)

    ax.set_xlabel("$x$ (m)")
    ax.set_ylabel("$y$ (m)")
    margin = 0.08
    ax.set_xlim(shoulder_xy[0] - margin, shoulder_xy[0] + typical_reach + margin)
    ax.set_ylim(shoulder_xy[1] - margin, shoulder_xy[1] + typical_reach + margin)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "fig_workspace.pdf")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    fig_architecture()
    fig_kinematics()
    fig_workspace()
