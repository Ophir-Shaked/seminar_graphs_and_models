# -*- coding: utf-8 -*-
# Graph (5) — Python code for seminar figure/model.
"""
Graph (5): Simple Pendulum — Euler vs RK4 vs Small-Angle Analytic Solution
------------------------------------------------------------------------
Model
-----
θ' = ω
ω' = -(g/L) * sin(θ)

Initial conditions: θ(0) = THETA0, ω(0) = OMEGA0
Small-angle analytic solution (sin θ ≈ θ):
  θ_exact(t) = θ0 * cos(ω_n t) + (ω0/ω_n) * sin(ω_n t),  ω_n = sqrt(g/L)

All numeric and string literals are centralized in the CONSTANTS section.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

# =========================
# CONSTANTS
# =========================
# Physical parameters
G: float = 9.81                 # gravity [m/s^2]
L: float = 1.0                  # pendulum length [m]

# Initial conditions
THETA0: float = 0.2             # initial angle [rad]
OMEGA0: float = 0.0             # initial angular velocity [rad/s]
Y0: np.ndarray = np.array([THETA0, OMEGA0])

# Time domain
T0: float = 0.0                 # start time [s]
T_END: float = 10.0             # end time [s]
H: float = 0.05                 # step size for integrators [s]
N_EXACT_SAMPLES: int = 1000     # resolution for exact curve

# RK constants
HALF: float = 0.5
SIXTH: float = 1.0 / 6.0
TWO: float = 2.0

# Plot appearance
FIGSIZE: tuple[float, float] = (8.0, 4.0)
XLABEL: str = "Time t [s]"
YLABEL: str = "Angle θ(t) [rad]"
TITLE: str = "Pendulum Simulation: Euler vs RK4 vs Exact (small-angle)"
LABEL_EULER: str = "Euler"
LABEL_RK4: str = "RK4"
LABEL_EXACT: str = "Exact (analytic, small-angle)"
LEGEND_LOC: str = "best"
GRID_ENABLED: bool = True
GRID_WHICH: str = "both"
GRID_ALPHA: float = 0.8

# Line/marker styles (colors left to matplotlib defaults)
STYLE_EULER: str = "-"
STYLE_RK4: str = "-"
STYLE_EXACT: str = "--"
EXACT_LINEWIDTH: float = 1.5

# =========================
# MODEL
# =========================
def f(t: float, y: np.ndarray) -> np.ndarray:
    """
    Pendulum ODE system:
      y = [θ, ω]
      θ' = ω
      ω' = -(G/L) * sin(θ)
    """
    theta, omega = y
    return np.array([omega, -(G / L) * np.sin(theta)])

def theta_exact_small_angle(t: np.ndarray, theta0: float, omega0: float) -> np.ndarray:
    """
    Small-angle analytic approximation:
      θ_exact(t) = θ0 cos(ω_n t) + (ω0/ω_n) sin(ω_n t),  ω_n = sqrt(G/L)
    """
    omega_n = np.sqrt(G / L)
    return theta0 * np.cos(omega_n * t) + (omega0 / omega_n) * np.sin(omega_n * t)

# =========================
# INTEGRATORS
# =========================
def euler(
    f_handle, y0: np.ndarray, t0: float, t_end: float, h: float
) -> tuple[np.ndarray, np.ndarray]:
    """Explicit Euler for systems: y_{n+1} = y_n + h f(t_n, y_n)."""
    t = np.arange(t0, t_end + h, h)
    y = np.zeros((t.size, y0.size), dtype=float)
    y[0] = y0
    for n in range(t.size - 1):
        y[n + 1] = y[n] + h * f_handle(t[n], y[n])
    return t, y

def rk4(
    f_handle, y0: np.ndarray, t0: float, t_end: float, h: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Classical RK4 for systems:
      k1 = f(t_n, y_n)
      k2 = f(t_n + h/2, y_n + h k1/2)
      k3 = f(t_n + h/2, y_n + h k2/2)
      k4 = f(t_n + h,   y_n + h k3)
      y_{n+1} = y_n + (h/6)(k1 + 2k2 + 2k3 + k4)
    """
    t = np.arange(t0, t_end + h, h)
    y = np.zeros((t.size, y0.size), dtype=float)
    y[0] = y0
    for n in range(t.size - 1):
        t_n = t[n]
        y_n = y[n]
        k1 = f_handle(t_n,                 y_n)
        k2 = f_handle(t_n + h * HALF,      y_n + h * k1 * HALF)
        k3 = f_handle(t_n + h * HALF,      y_n + h * k2 * HALF)
        k4 = f_handle(t_n + h,             y_n + h * k3)
        y[n + 1] = y_n + h * SIXTH * (k1 + TWO * k2 + TWO * k3 + k4)
    return t, y

# =========================
# MAIN / PLOT
# =========================
def main() -> None:
    # Numeric solutions
    t_eu, y_eu = euler(f, Y0, T0, T_END, H)
    t_rk, y_rk = rk4(f, Y0, T0, T_END, H)

    # Small-angle exact reference
    t_exact = np.linspace(T0, T_END, N_EXACT_SAMPLES)
    theta_exact = theta_exact_small_angle(t_exact, THETA0, OMEGA0)

    # Plot θ(t)
    plt.figure(figsize=FIGSIZE)
    plt.plot(t_eu, y_eu[:, 0], STYLE_EULER, label=LABEL_EULER)
    plt.plot(t_rk, y_rk[:, 0], STYLE_RK4, label=LABEL_RK4)
    plt.plot(t_exact, theta_exact, STYLE_EXACT, linewidth=EXACT_LINEWIDTH, label=LABEL_EXACT)

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
