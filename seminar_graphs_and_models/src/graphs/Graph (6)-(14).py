# -*- coding: utf-8 -*-
# Graphs (6)-(14) — Python code for seminar figure/model.
"""
Graphs (6)-(14): Van der Pol — Robust demos with feathered stiff-region shading
-------------------------------------------------------------------------------
What this script shows
- Van der Pol oscillator z' = [x', y'] = [y, μ(1 - x^2)y - x]
- Reference (Radau) vs. explicit Euler and RK4 for multiple μ, two step sizes
- Feathered gray band for the "stiff region" |x| ≤ 1 in phase and x(t) plots
- Overflow guards for explicit solvers (truncate gracefully if values blow up)

Design notes
- All numeric and string literals live in CONSTANTS to eliminate magic numbers.
- Avoids mathtext parsing pitfalls by using plain Unicode "≤" in annotations.
- Colors: Reference=gray dashed, RK4=orange, Euler=green (configurable below).
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# =========================
# CONSTANTS
# =========================
# Generic scalars
ZERO: float = 0.0
ONE: float = 1.0
TWO: float = 2.0
HALF: float = 0.5
SIXTH: float = 1.0 / 6.0
THREE: float = 3.0

# Overflow / sanity limits
BOUNDS_INF_NORM: float = 1e6    # truncate trajectory if exceeded

# Reference solver configuration
REF_METHOD: str = "Radau"
REF_RTOL: float = 1e-10
REF_ATOL: float = 1e-12
REF_N_PLOT: int = 4001

# Default demo/problem parameters
DEFAULT_MU: float = 40.0
DEFAULT_T_START: float = 0.0
DEFAULT_T_END_DEMO: float = 14.0     # shorter window for big μ to reduce overflow risk
DEFAULT_T_END_PANELS: float = 40.0
DEFAULT_Z0_X: float = 2.0
DEFAULT_Z0_Y: float = 0.0
DEFAULT_Z0: tuple[float, float] = (DEFAULT_Z0_X, DEFAULT_Z0_Y)

# Step sizes for the "stiffness demo" (single x(t) plot)
DEMO_H_LARGE: float = 0.03
DEMO_H_SMALL: float = 0.003

# Phase/x(t) panel configurations (μ, h_large, h_small)
PANEL_CONFIGS: tuple[tuple[float, float, float], ...] = (
    (40.0, 0.01, 0.001),
    (7.0,  0.05, 0.005),
    (3.0,  0.05, 0.005),
    (0.5,  0.10, 0.01),
)

# Plot appearance: colors, styles, sizes
COLOR_REF: str = "0.35"     # gray for reference
COLOR_RK4: str = "orange"
COLOR_EULER: str = "green"
LINESTYLE_REF: str = "--"
LINEWIDTH_REF: float = 1.3
FIGSIZE_DEMO: tuple[float, float] = (12.0, 4.0)
FIGSIZE_PANELS: tuple[float, float] = (12.0, 4.0)
LEGEND_LOC: str = "best"
XLABEL_TIME: str = "t"
YLABEL_XT: str = "x(t)"
XLABEL_PHASE: str = "x"
YLABEL_PHASE: str = "y = x'(t)"

# Feathered shading configuration (stiff region |x| ≤ 1)
STIFF_X0: float = -1.0
STIFF_X1: float = 1.0
STIFF_Y0: float = -1.0
STIFF_Y1: float = 1.0
SHADE_BASE_ALPHA: float = 0.28
SHADE_BLUR: float = 0.22
SHADE_COLOR_RGB: tuple[float, float, float] = (0.85, 0.85, 0.85)  # neutral gray
SHADE_NX: int = 800
SHADE_NY: int = 800
ANNOT_FONTSIZE: int = 9
ANNOT_TXT: str = "stiff region: |x| ≤ 1"
ANNOT_POS_PHASE: tuple[float, float] = (0.03, 0.93)  # axes fraction
ANNOT_POS_TIME: tuple[float, float] = (0.02, 0.04)   # axes fraction

# Titles
TITLE_DEMO_TEMPLATE: str = "Stiffness demo — μ = {mu}"
TITLE_PHASE_LARGE_TEMPLATE: str = "μ = {mu} — Phase portrait (large h)"
TITLE_PHASE_SMALL_TEMPLATE: str = "μ = {mu} — Phase portrait (small h)"
TITLE_XT_LARGE_TEMPLATE: str = "μ = {mu} — x(t) (large h)"
TITLE_XT_SMALL_TEMPLATE: str = "μ = {mu} — x(t) (small h)"

# Labels
LABEL_REF: str = "Reference (solve_ivp)"
LABEL_RK4_TEMPLATE: str = "RK4 (h={h})"
LABEL_EULER_TEMPLATE: str = "Euler (h={h})"

# =========================
# MODEL: Van der Pol
# =========================
def vdp_rhs(t: float, z: np.ndarray, mu: float) -> np.ndarray:
    """
    Van der Pol oscillator:
      x' = y
      y' = μ(1 - x^2) y - x
    """
    x, y = z
    return np.array([y, mu * (ONE - x * x) * y - x], dtype=float)

# =========================
# EXPLICIT METHODS (safe)
# =========================
def _finite_and_bounded(arr: np.ndarray, bound: float) -> bool:
    """Check that arr is finite and its ∞-norm is below 'bound'."""
    return np.all(np.isfinite(arr)) and np.linalg.norm(arr, ord=np.inf) < bound

def euler(
    f_handle, z0: np.ndarray, t0: float, t_end: float, h: float, *f_args,
    bound: float = BOUNDS_INF_NORM
) -> tuple[np.ndarray, np.ndarray]:
    """
    Explicit Euler for systems:
      z_{n+1} = z_n + h f(t_n, z_n, ...)
    Truncates and returns partial arrays if overflow or non-finite values occur.
    """
    N = int(np.ceil((t_end - t0) / h))
    t = t0 + np.arange(N + 1) * h
    z = np.empty((N + 1, len(z0)), dtype=float)
    z[0] = z0
    last = 0
    for n in range(N):
        z[n + 1] = z[n] + h * f_handle(t[n], z[n], *f_args)
        last = n + 1
        if not _finite_and_bounded(z[n + 1], bound):
            t = t[: last + 1]
            z = z[: last + 1]
            break
    return t, z

def rk4(
    f_handle, z0: np.ndarray, t0: float, t_end: float, h: float, *f_args,
    bound: float = BOUNDS_INF_NORM
) -> tuple[np.ndarray, np.ndarray]:
    """
    Classical RK4 for systems:
      k1 = f(t_n, z_n)
      k2 = f(t_n + h/2, z_n + h k1/2)
      k3 = f(t_n + h/2, z_n + h k2/2)
      k4 = f(t_n + h,   z_n + h k3)
      z_{n+1} = z_n + (h/6)(k1 + 2k2 + 2k3 + k4)
    Truncates and returns partial arrays if overflow or non-finite values occur.
    """
    N = int(np.ceil((t_end - t0) / h))
    t = t0 + np.arange(N + 1) * h
    z = np.empty((N + 1, len(z0)), dtype=float)
    z[0] = z0
    last = 0
    for n in range(N):
        tn, zn = t[n], z[n]
        k1 = f_handle(tn,             zn,             *f_args)
        k2 = f_handle(tn + h * HALF,  zn + h * k1 * HALF, *f_args)
        k3 = f_handle(tn + h * HALF,  zn + h * k2 * HALF, *f_args)
        k4 = f_handle(tn + h,         zn + h * k3,    *f_args)
        z[n + 1] = zn + (h * SIXTH) * (k1 + TWO * k2 + TWO * k3 + k4)
        last = n + 1
        if not _finite_and_bounded(z[n + 1], bound):
            t = t[: last + 1]
            z = z[: last + 1]
            break
    return t, z

# =========================
# REFERENCE SOLUTION
# =========================
def reference_dense(mu: float, z0: np.ndarray, T: float, n_plot: int = REF_N_PLOT
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dense high-accuracy reference using SciPy solve_ivp (Radau)."""
    t_ref = np.linspace(ZERO, T, n_plot)
    sol = solve_ivp(
        lambda t, z: vdp_rhs(t, z, mu),
        (ZERO, T),
        z0,
        t_eval=t_ref,
        method=REF_METHOD,
        rtol=REF_RTOL,
        atol=REF_ATOL,
    )
    x_ref, y_ref = sol.y
    return t_ref, x_ref, y_ref

# =========================
# FEATHERED SHADING HELPERS
# =========================
def _smoothstep01(x: np.ndarray) -> np.ndarray:
    """C^1 smoothstep on [0,1]."""
    x_clamped = np.clip(x, ZERO, ONE)
    return x_clamped * x_clamped * (THREE - TWO * x_clamped)

def shade_vertical_blur(
    ax,
    x0: float = STIFF_X0,
    x1: float = STIFF_X1,
    base_alpha: float = SHADE_BASE_ALPHA,
    blur: float = SHADE_BLUR,
    color_rgb: tuple[float, float, float] = SHADE_COLOR_RGB,
    nx: int = SHADE_NX,
) -> None:
    """Vertical feathered band between x0..x1 using an RGBA image."""
    ymin, ymax = ax.get_ylim()
    xmin, xmax = ax.get_xlim()
    xs = np.linspace(x0 - blur, x1 + blur, nx)
    left =  (xs - (x0 - blur)) / (blur if blur > ZERO else ONE)
    right = ((x1 + blur) - xs) / (blur if blur > ZERO else ONE)
    alpha = _smoothstep01(left) * _smoothstep01(right) * base_alpha
    strip = np.zeros((2, nx, 4), dtype=float)
    strip[..., :3] = color_rgb
    strip[..., 3]  = alpha
    ax.imshow(
        strip,
        origin="lower",
        aspect="auto",
        extent=(xs[0], xs[-1], ymin, ymax),
        zorder=0,
        interpolation="bicubic",
    )
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

def shade_horizontal_blur(
    ax,
    y0: float = STIFF_Y0,
    y1: float = STIFF_Y1,
    base_alpha: float = SHADE_BASE_ALPHA,
    blur: float = SHADE_BLUR,
    color_rgb: tuple[float, float, float] = SHADE_COLOR_RGB,
    ny: int = SHADE_NY,
) -> None:
    """Horizontal feathered band between y0..y1 using an RGBA image."""
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    ys = np.linspace(y0 - blur, y1 + blur, ny)
    bot = (ys - (y0 - blur)) / (blur if blur > ZERO else ONE)
    top = ((y1 + blur) - ys) / (blur if blur > ZERO else ONE)
    alpha = _smoothstep01(bot) * _smoothstep01(top) * base_alpha
    strip = np.zeros((ny, 2, 4), dtype=float)
    strip[..., :3] = color_rgb
    strip[..., 3]  = alpha[:, None]
    ax.imshow(
        strip,
        origin="lower",
        aspect="auto",
        extent=(xmin, xmax, ys[0], ys[-1]),
        zorder=0,
        interpolation="bicubic",
    )
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

def label_stiff_region(ax, where: str = "phase") -> None:
    """Place a plaintext label for the stiff region band."""
    if where == "phase":
        ax.text(ANNOT_POS_PHASE[0], ANNOT_POS_PHASE[1], ANNOT_TXT, transform=ax.transAxes, fontsize=ANNOT_FONTSIZE)
    else:
        ax.text(ANNOT_POS_TIME[0], ANNOT_POS_TIME[1], ANNOT_TXT, transform=ax.transAxes, fontsize=ANNOT_FONTSIZE)

# =========================
# FIGURE BUILDERS
# =========================
def stiffness_demo(
    mu: float = DEFAULT_MU,
    h_large: float = DEMO_H_LARGE,
    h_small: float = DEMO_H_SMALL,
    T: float = DEFAULT_T_END_DEMO,
    z0: tuple[float, float] = DEFAULT_Z0,
) -> None:
    """
    Single x(t) plot comparing Euler/RK4 at two step sizes against a dense reference.
    Uses a shorter horizon for large μ to reduce explicit-solver blow-ups.
    """
    z0_arr = np.array(z0, dtype=float)
    t_ref, x_ref, _ = reference_dense(mu, z0_arr, T)
    t_rL, z_rL = rk4(  vdp_rhs, z0_arr, DEFAULT_T_START, T, h_large, mu)
    t_eL, z_eL = euler(vdp_rhs, z0_arr, DEFAULT_T_START, T, h_large, mu)
    t_rS, z_rS = rk4(  vdp_rhs, z0_arr, DEFAULT_T_START, T, h_small, mu)
    t_eS, z_eS = euler(vdp_rhs, z0_arr, DEFAULT_T_START, T, h_small, mu)

    fig = plt.figure(figsize=FIGSIZE_DEMO)
    ax = plt.gca()

    ax.plot(t_ref, x_ref, linestyle=LINESTYLE_REF, color=COLOR_REF, linewidth=LINEWIDTH_REF, label=LABEL_REF)
    ax.plot(t_rL, z_rL[:, 0], color=COLOR_RK4,   label=LABEL_RK4_TEMPLATE.format(h=h_large))
    ax.plot(t_eL, z_eL[:, 0], color=COLOR_EULER, label=LABEL_EULER_TEMPLATE.format(h=h_large))
    ax.plot(t_rS, z_rS[:, 0], color=COLOR_RK4,   label=LABEL_RK4_TEMPLATE.format(h=h_small))
    ax.plot(t_eS, z_eS[:, 0], color=COLOR_EULER, label=LABEL_EULER_TEMPLATE.format(h=h_small))

    # Shading after lines so limits are known
    shade_horizontal_blur(ax, y0=STIFF_Y0, y1=STIFF_Y1, base_alpha=SHADE_BASE_ALPHA, blur=SHADE_BLUR, color_rgb=SHADE_COLOR_RGB)
    label_stiff_region(ax, where="time")

    ax.set_title(TITLE_DEMO_TEMPLATE.format(mu=mu))
    ax.set_xlabel(XLABEL_TIME)
    ax.set_ylabel(YLABEL_XT)
    ax.legend(loc=LEGEND_LOC)
    fig.tight_layout()
    plt.show()

def two_panels_for_mu(
    mu: float,
    h_large: float,
    h_small: float,
    T: float = DEFAULT_T_END_PANELS,
    z0: tuple[float, float] = DEFAULT_Z0,
) -> None:
    """
    For a given μ, produce:
      - Phase portraits: reference + (Euler, RK4) at large/small h (two subplots)
      - x(t):            reference + (Euler, RK4) at large/small h (two subplots)
    """
    z0_arr = np.array(z0, dtype=float)
    t_ref, x_ref, y_ref = reference_dense(mu, z0_arr, T)

    t_eL, z_eL = euler(vdp_rhs, z0_arr, DEFAULT_T_START, T, h_large, mu)
    t_rL, z_rL = rk4(  vdp_rhs, z0_arr, DEFAULT_T_START, T, h_large, mu)
    t_eS, z_eS = euler(vdp_rhs, z0_arr, DEFAULT_T_START, T, h_small, mu)
    t_rS, z_rS = rk4(  vdp_rhs, z0_arr, DEFAULT_T_START, T, h_small, mu)

    # ----- Phase portraits -----
    fig1, axs1 = plt.subplots(1, 2, figsize=FIGSIZE_PANELS)

    # Large h
    ax = axs1[0]
    ax.plot(x_ref, y_ref, linestyle=LINESTYLE_REF, color=COLOR_REF, linewidth=LINEWIDTH_REF, label=LABEL_REF)
    ax.plot(z_rL[:, 0], z_rL[:, 1], color=COLOR_RK4,   label=LABEL_RK4_TEMPLATE.format(h=h_large))
    ax.plot(z_eL[:, 0], z_eL[:, 1], color=COLOR_EULER, label=LABEL_EULER_TEMPLATE.format(h=h_large))
    ax.set_title(TITLE_PHASE_LARGE_TEMPLATE.format(mu=mu))
    ax.set_xlabel(XLABEL_PHASE)
    ax.set_ylabel(YLABEL_PHASE)
    ax.legend(loc=LEGEND_LOC)
    shade_vertical_blur(ax, x0=STIFF_X0, x1=STIFF_X1, base_alpha=SHADE_BASE_ALPHA, blur=SHADE_BLUR, color_rgb=SHADE_COLOR_RGB)
    label_stiff_region(ax, where="phase")

    # Small h
    ax = axs1[1]
    ax.plot(x_ref, y_ref, linestyle=LINESTYLE_REF, color=COLOR_REF, linewidth=LINEWIDTH_REF, label=LABEL_REF)
    ax.plot(z_rS[:, 0], z_rS[:, 1], color=COLOR_RK4,   label=LABEL_RK4_TEMPLATE.format(h=h_small))
    ax.plot(z_eS[:, 0], z_eS[:, 1], color=COLOR_EULER, label=LABEL_EULER_TEMPLATE.format(h=h_small))
    ax.set_title(TITLE_PHASE_SMALL_TEMPLATE.format(mu=mu))
    ax.set_xlabel(XLABEL_PHASE)
    ax.set_ylabel(YLABEL_PHASE)
    ax.legend(loc=LEGEND_LOC)
    shade_vertical_blur(ax, x0=STIFF_X0, x1=STIFF_X1, base_alpha=SHADE_BASE_ALPHA, blur=SHADE_BLUR, color_rgb=SHADE_COLOR_RGB)
    label_stiff_region(ax, where="phase")

    fig1.tight_layout()
    plt.show()

    # ----- x(t) -----
    fig2, axs2 = plt.subplots(1, 2, figsize=FIGSIZE_PANELS)

    # Large h
    ax = axs2[0]
    ax.plot(t_ref, x_ref, linestyle=LINESTYLE_REF, color=COLOR_REF, linewidth=LINEWIDTH_REF, label=LABEL_REF)
    ax.plot(t_rL, z_rL[:, 0], color=COLOR_RK4,   label=LABEL_RK4_TEMPLATE.format(h=h_large))
    ax.plot(t_eL, z_eL[:, 0], color=COLOR_EULER, label=LABEL_EULER_TEMPLATE.format(h=h_large))
    ax.set_title(TITLE_XT_LARGE_TEMPLATE.format(mu=mu))
    ax.set_xlabel(XLABEL_TIME)
    ax.set_ylabel(YLABEL_XT)
    ax.legend(loc=LEGEND_LOC)
    shade_horizontal_blur(ax, y0=STIFF_Y0, y1=STIFF_Y1, base_alpha=SHADE_BASE_ALPHA, blur=SHADE_BLUR, color_rgb=SHADE_COLOR_RGB)
    label_stiff_region(ax, where="time")

    # Small h
    ax = axs2[1]
    ax.plot(t_ref, x_ref, linestyle=LINESTYLE_REF, color=COLOR_REF, linewidth=LINEWIDTH_REF, label=LABEL_REF)
    ax.plot(t_rS, z_rS[:, 0], color=COLOR_RK4,   label=LABEL_RK4_TEMPLATE.format(h=h_small))
    ax.plot(t_eS, z_eS[:, 0], color=COLOR_EULER, label=LABEL_EULER_TEMPLATE.format(h=h_small))
    ax.set_title(TITLE_XT_SMALL_TEMPLATE.format(mu=mu))
    ax.set_xlabel(XLABEL_TIME)
    ax.set_ylabel(YLABEL_XT)
    ax.legend(loc=LEGEND_LOC)
    shade_horizontal_blur(ax, y0=STIFF_Y0, y1=STIFF_Y1, base_alpha=SHADE_BASE_ALPHA, blur=SHADE_BLUR, color_rgb=SHADE_COLOR_RGB)
    label_stiff_region(ax, where="time")

    fig2.tight_layout()
    plt.show()

# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    # Safer single-figure stiffness demo
    stiffness_demo(
        mu=DEFAULT_MU,
        h_large=DEMO_H_LARGE,
        h_small=DEMO_H_SMALL,
        T=DEFAULT_T_END_DEMO,
        z0=DEFAULT_Z0,
    )

    # Two-panel figures across various μ and step sizes
    for mu_val, hL, hS in PANEL_CONFIGS:
        two_panels_for_mu(
            mu=mu_val,
            h_large=hL,
            h_small=hS,
            T=DEFAULT_T_END_PANELS,
            z0=DEFAULT_Z0,
        )
