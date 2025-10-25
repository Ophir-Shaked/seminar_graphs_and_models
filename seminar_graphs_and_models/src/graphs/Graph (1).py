# -*- coding: utf-8 -*-
# Graph (1) — Python code for seminar figure/model.
"""
Graph (1): e^h vs. Euler and RK4 (single-step values)
-----------------------------------------------------
Context
-------
For the scalar ODE y' = y with y(0)=1:
- Exact one-step value at step size h is e^h.
- Explicit Euler one-step gives 1 + h (1st-order).
- Classical RK4 one-step for y' = y equals the 4th-order Taylor truncation:
  1 + h + h^2/2! + h^3/3! + h^4/4!  (local error O(h^5)).

This script plots the three single-step values as functions of h on a log scale
and marks a specific reference step size H_MARK for comparison.

All numeric and string literals are centralized in the CONSTANTS section.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

# =========================
# CONSTANTS
# =========================
# Domain of h
H_MIN: float = 1e-6
H_MAX: float = 1e-1
N_H: int = 200

# Reference/annotation
H_MARK: float = 1e-3
N_MARK_POINTS: int = 3
ANNOT_FONTSIZE: int = 9
ANNOT_VALIGN: str = "bottom"
ANNOT_X_OFFSET_FACTOR: float = 1.05  # place text a bit to the right of H_MARK

# Figure & styling
FIGSIZE: tuple[float, float] = (6.0, 4.0)
X_SCALE: str = "log"
GRID_ENABLED: bool = True
GRID_WHICH: str = "both"
GRID_ALPHA: float = 0.7
VLINE_STYLE: str = ":"
VLINE_ALPHA: float = 0.5
POINT_ZORDER: int = 5
LEGEND_LOC: str = "best"

# Labels & title
TITLE: str = "e^h vs Euler and RK4 (single-step values)"
XLABEL: str = "Step size h (log scale)"
YLABEL: str = "Value after ONE step"
LABEL_EXACT: str = "Exact: e^h"
LABEL_EULER: str = "Euler: 1 + h"
LABEL_RK4: str = "RK4 (Taylor up to h^4)"

# Footnote
FOOTNOTE_X: float = 0.02
FOOTNOTE_Y: float = 0.02
EXP_FORMAT: str = ".0e"
FOOTNOTE_TEMPLATE: str = "h ∈ [{hmin}, {hmax}] (log-spaced). Example shown: h = {hmark}."

# Math constants (avoid inline numerics)
ONE: float = 1.0
FACT_2: float = 2.0
FACT_3: float = 6.0     # 3!
FACT_4: float = 24.0    # 4!

# =========================
# HELPERS
# =========================
def fmt_sci(x: float) -> str:
    """Scientific format using EXP_FORMAT (e.g., '.0e')."""
    return format(x, EXP_FORMAT)

# =========================
# MODEL / FORMULAS
# =========================
def exact(h: np.ndarray | float) -> np.ndarray | float:
    """Exact single-step value for y' = y, y(0)=ONE at step size h: e^h."""
    return np.exp(h)

def euler(h: np.ndarray | float) -> np.ndarray | float:
    """Explicit Euler single-step approximation for y' = y: ONE + h."""
    return ONE + h

def rk4_taylor(h: np.ndarray | float) -> np.ndarray | float:
    """
    Classical RK4 single-step for y' = y equals the 4th-order Taylor series:
      ONE + h + h^2/FACT_2 + h^3/FACT_3 + h^4/FACT_4
    This identity holds specifically for the linear ODE y' = y.
    """
    return ONE + h + (h**2)/FACT_2 + (h**3)/FACT_3 + (h**4)/FACT_4

# =========================
# PLOTTING
# =========================
def main() -> None:
    # Create log-spaced step sizes
    h = np.logspace(np.log10(H_MIN), np.log10(H_MAX), N_H)

    # Compute the three curves
    y_exact = exact(h)
    y_euler = euler(h)
    y_rk4   = rk4_taylor(h)

    # Prepare figure
    plt.figure(figsize=FIGSIZE)

    # Plot curves
    plt.plot(h, y_exact, label=LABEL_EXACT)
    plt.plot(h, y_euler, label=LABEL_EULER)
    plt.plot(h, y_rk4,   label=LABEL_RK4)

    # Axis scale
    plt.xscale(X_SCALE)

    # Annotation values at H_MARK
    y_mark_exact = exact(H_MARK)
    y_mark_euler = euler(H_MARK)
    y_mark_rk4   = rk4_taylor(H_MARK)

    plt.axvline(H_MARK, linestyle=VLINE_STYLE, alpha=VLINE_ALPHA)
    plt.scatter([H_MARK]*N_MARK_POINTS, [y_mark_exact, y_mark_euler, y_mark_rk4], zorder=POINT_ZORDER)

    # Label near the exact point
    plt.text(H_MARK*ANNOT_X_OFFSET_FACTOR, y_mark_exact,
             f"h = {fmt_sci(H_MARK)}", va=ANNOT_VALIGN, fontsize=ANNOT_FONTSIZE)

    # Labels, title, legend, grid
    plt.xlabel(XLABEL)
    plt.ylabel(YLABEL)
    plt.title(TITLE)
    plt.legend(loc=LEGEND_LOC)
    if GRID_ENABLED:
        plt.grid(True, which=GRID_WHICH, alpha=GRID_ALPHA)

    # Footnote
    footnote = FOOTNOTE_TEMPLATE.format(
        hmin=fmt_sci(H_MIN), hmax=fmt_sci(H_MAX), hmark=fmt_sci(H_MARK)
    )
    plt.gcf().text(FOOTNOTE_X, FOOTNOTE_Y, footnote, fontsize=ANNOT_FONTSIZE)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
