# -*- coding: utf-8 -*-
# Graph (2) — Python code for seminar figure/model.
"""
Graph (2): Exact vs. Euler vs. RK4 for y' = y, y(0) = 1
-------------------------------------------------------
Context
-------
We solve the IVP y'(t) = y(t), y(0) = 1 on [T0, T_END].
- Exact solution: y(t) = exp(t).
- Euler method (1st-order).
- Classical RK4 (4th-order).

This script compares the trajectories for a single fixed step size H.

All numeric and string literals are centralized in the CONSTANTS section.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

# =========================
# CONSTANTS
# =========================
# Problem definition
Y0: float = 1.0
T0: float = 0.0
T_END: float = 5.0

# Discretization
H: float = 0.25                       # primary step size for Euler/RK4
N_EXACT_SAMPLES: int = 1000           # resolution for the exact curve

# RK4 coefficients / helper numbers
HALF: float = 0.5
SIXTH: float = 1.0 / 6.0
TWO: float = 2.0

# Plotting parameters
FIGSIZE: tuple[float, float] = (6.0, 4.0)
XLABEL: str = "t"
YLABEL: str = "y(t)"
TITLE_TEMPLATE: str = "Exact vs Euler vs RK4 — h={h}"
LABEL_EXACT: str = "Exact y(t)=e^t"
LABEL_EULER_TEMPLATE: str = "Euler (h={h})"
LABEL_RK4_TEMPLATE: str = "RK4 (h={h})"
GRID_ENABLED: bool = True
GRID_WHICH: str = "both"
GRID_ALPHA: float = 0.8
LEGEND_LOC: str = "best"

# Marker/line styles (kept as constants; colors left to matplotlib defaults)
EULER_STYLE: str = "o--"
RK4_STYLE: str = "s--"
EXACT_STYLE: str = "-"   # solid line

# =========================
# MODEL
# =========================
def f(t: float | np.ndarray, y: float | np.ndarray) -> float | np.ndarray:
    """RHS of the ODE: y' = y."""
    return y

def exact(t: np.ndarray) -> np.ndarray:
    """Exact solution y(t) = exp(t)."""
    return np.exp(t)

# =========================
# INTEGRATORS
# =========================
def euler(
    f_handle, y0: float, t0: float, t_end: float, h: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Explicit Euler method:
      y_{n+1} = y_n + h * f(t_n, y_n)
    """
    t = np.arange(t0, t_end + h, h)
    y = np.zeros(t.shape, dtype=float)
    y[0] = y0
    for n in range(t.size - 1):
        y[n + 1] = y[n] + h * f_handle(t[n], y[n])
    return t, y

def rk4(
    f_handle, y0: float, t0: float, t_end: float, h: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Classical 4th-order Runge–Kutta (RK4):
      k1 = f(t_n,          y_n)
      k2 = f(t_n + h/2,    y_n + h*k1/2)
      k3 = f(t_n + h/2,    y_n + h*k2/2)
      k4 = f(t_n + h,      y_n + h*k3)
      y_{n+1} = y_n + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
    """
    t = np.arange(t0, t_end + h, h)
    y = np.zeros(t.shape, dtype=float)
    y[0] = y0
    for n in range(t.size - 1):
        t_n = t[n]
        y_n = y[n]
        k1 = f_handle(t_n,                   y_n)
        k2 = f_handle(t_n + h * HALF,        y_n + h * k1 * HALF)
        k3 = f_handle(t_n + h * HALF,        y_n + h * k2 * HALF)
        k4 = f_handle(t_n + h,               y_n + h * k3)
        y[n + 1] = y_n + h * SIXTH * (k1 + TWO * k2 + TWO * k3 + k4)
    return t, y

# =========================
# PLOTTING
# =========================
def main() -> None:
    # Exact solution on a fine grid
    t_exact = np.linspace(T0, T_END, N_EXACT_SAMPLES)
    y_exact = exact(t_exact)

    # Numerical solutions with step H
    t_euler, y_euler = euler(f, Y0, T0, T_END, H)
    t_rk4, y_rk4 = rk4(f, Y0, T0, T_END, H)

    # Plot
    plt.figure(figsize=FIGSIZE)
    plt.plot(t_exact, y_exact, EXACT_STYLE, label=LABEL_EXACT)
    plt.plot(t_euler, y_euler, EULER_STYLE, label=LABEL_EULER_TEMPLATE.format(h=H))
    plt.plot(t_rk4, y_rk4, RK4_STYLE, label=LABEL_RK4_TEMPLATE.format(h=H))
    plt.xlabel(XLABEL)
    plt.ylabel(YLABEL)
    plt.title(TITLE_TEMPLATE.format(h=H))
    plt.legend(loc=LEGEND_LOC)
    if GRID_ENABLED:
        plt.grid(True, which=GRID_WHICH, alpha=GRID_ALPHA)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
