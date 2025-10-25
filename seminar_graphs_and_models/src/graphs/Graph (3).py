# -*- coding: utf-8 -*-
# Graph (3) — Python code for seminar figure/model.
"""
Graph (3): Stability functions R(z) for Euler and RK4 on the negative real axis
-------------------------------------------------------------------------------
Context
-------
For a one-step method applied to y' = λ y, the amplification (stability) factor is R(z),
where z = h * λ. Absolute stability requires |R(z)| <= 1.

- Explicit Euler: R_E(z) = 1 + z
- Classical RK4:  R_RK4(z) = 1 + z + z^2/2! + z^3/3! + z^4/4!

This script plots R(z) along the negative real axis and draws horizontal lines at ±1
to mark the boundary of absolute stability on the real line.

All numeric and string literals are centralized in the CONSTANTS section.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

# =========================
# CONSTANTS
# =========================
# Domain for z on the real axis (negative to zero)
Z_MIN: float = -4.0
Z_MAX: float = 0.0
N_Z: int = 400

# Factorials for RK4 polynomial terms
FACT_2: float = 2.0
FACT_3: float = 6.0    # 3!
FACT_4: float = 24.0   # 4!

# Stability boundary lines (±1)
BOUNDARY_POS: float = 1.0
BOUNDARY_NEG: float = -1.0

# Plot appearance
FIGSIZE: tuple[float, float] = (6.0, 4.0)
XLABEL: str = "z = hλ (real negative axis)"
YLABEL: str = "R(z)"
TITLE: str = "Stability functions R(z) for Euler and RK4"
LEGEND_LOC: str = "best"
GRID_ENABLED: bool = True
GRID_WHICH: str = "both"
GRID_ALPHA: float = 0.8

# Line styles (colors left to matplotlib defaults)
STYLE_EULER: str = "-"     # line for Euler curve
STYLE_RK4: str = "-"       # line for RK4 curve
STYLE_BOUNDARY: str = "--" # dashed for ±1 boundaries

# Labels
LABEL_EULER: str = "Euler R(z) = 1 + z"
LABEL_RK4: str = "RK4 R(z)"
LABEL_BOUNDARY: str = "Stability boundary ±1"

# =========================
# STABILITY FUNCTIONS
# =========================
def R_euler(z: np.ndarray | float) -> np.ndarray | float:
    """Explicit Euler stability function: R(z) = 1 + z."""
    return 1.0 + z

def R_rk4(z: np.ndarray | float) -> np.ndarray | float:
    """Classical RK4 stability polynomial up to z^4/4!."""
    return 1.0 + z + (z**2)/FACT_2 + (z**3)/FACT_3 + (z**4)/FACT_4

# =========================
# MAIN / PLOT
# =========================
def main() -> None:
    # z along the negative real axis
    z = np.linspace(Z_MIN, Z_MAX, N_Z)

    # Values of stability functions
    R_e = R_euler(z)
    R_r = R_rk4(z)

    # Plot
    plt.figure(figsize=FIGSIZE)
    plt.plot(z, R_e, STYLE_EULER, label=LABEL_EULER)
    plt.plot(z, R_r, STYLE_RK4, label=LABEL_RK4)

    # Stability boundaries at ±1
    plt.axhline(BOUNDARY_POS, linestyle=STYLE_BOUNDARY, label=LABEL_BOUNDARY)
    plt.axhline(BOUNDARY_NEG, linestyle=STYLE_BOUNDARY)

    # Labels & cosmetics
    plt.xlabel(XLABEL)
    plt.ylabel(YLABEL)
    plt.title(TITLE)
    plt.legend(loc=LEGEND_LOC)
    if GRID_ENABLED:
        plt.grid(True, which=GRID_WHICH, alpha=GRID_ALPHA)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
