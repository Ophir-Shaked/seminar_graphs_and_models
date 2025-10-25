# -*- coding: utf-8 -*-
# Graph (6)-(14) — Python code for seminar figure/model.
"""
Hodgkin–Huxley: Euler vs RK4 vs Reference (Radau/fine-RK4)
=========================================================

What this script does
---------------------
- Implements the classic HH model with numerically stable gating-rate formulas.
- Provides explicit Euler and RK4 steppers.
- Computes a high-accuracy reference with SciPy's Radau (if available)
  or falls back to a very fine-step RK4.
- Generates comparison plots for each variable (V, m, h, n) with exactly
  three curves: Euler, RK4, Reference.
- Saves figures to the chosen output directory.

All numeric and string literals live in the CONSTANTS section (no magic numbers).
"""

from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt

# ---------------- Optional SciPy reference solver (stiff-friendly) ----------------
try:
    from scipy.integrate import solve_ivp
    _SCIPY_OK = True
except Exception:
    _SCIPY_OK = False

# =========================
# CONSTANTS
# =========================
# Physical params (classic HH, squid giant axon) — units in comments
C_M_UF_CM2: float = 1.0          # µF/cm^2
G_NA_MS_CM2: float = 120.0       # mS/cm^2
G_K_MS_CM2:  float = 36.0        # mS/cm^2
G_L_MS_CM2:  float = 0.3         # mS/cm^2
E_NA_MV:     float = 50.0        # mV
E_K_MV:      float = -77.0       # mV
E_L_MV:      float = -54.387     # mV

# External stimulus (square pulse)
I_STEP_UA_CM2: float = 10.0      # µA/cm^2 amplitude
I_ON_MS:        float = 5.0      # start time (ms)
I_OFF_MS:       float = 6.0      # end time (ms)

# Numerical safety / gating-rate stability
EXP_CLIP_MIN: float   = -50.0    # clip for exp() input
EXP_CLIP_MAX: float   = 50.0
SINGULARITY_EPS: float = 1e-6    # threshold for l'Hôpital near removable singularities

# State clamping (optional) to keep numerics sane in explicit solvers
CLAMP_V_MIN_MV: float = -120.0
CLAMP_V_MAX_MV: float = 80.0
GATE_MIN: float        = 0.0
GATE_MAX: float        = 1.0

# RK4 helper constants
HALF: float  = 0.5
SIXTH: float = 1.0 / 6.0
TWO: float   = 2.0

# Reference solver config
REF_METHOD: str  = "Radau"
REF_RTOL: float  = 1e-9
REF_ATOL: float  = 1e-9
DT_REF_MS: float = 0.001          # uniform output grid for reference (ms)

# Default simulation configuration
T_FINAL_MS: float = 30.0
DTS_MS: list[float] = [0.01, 0.1]  # Euler/RK4 step sizes to compare (ms)
OUTDIR_PATH: str = "./hh_outputs"

# Initial conditions (rest)
V_REST_MV: float = -65.0

# Plot styling
FIGSIZE: tuple[float, float] = (8.0, 4.5)
LINEWIDTH_REF: float = 2.2
LINEWIDTH_NUM: float = 1.6
GRID_ALPHA: float     = 0.6
LABEL_REF: str        = "Reference"
LABEL_EULER_TMPL: str = "Euler Δt={dt:g} ms"
LABEL_RK4_TMPL: str   = "RK4 Δt={dt:g} ms"
VAR_INFO: list[tuple[int, str, str]] = [
    (0, "Membrane potential V(t)", "V [mV]"),
    (1, "Sodium activation m(t)",  "m"),
    (2, "Sodium inactivation h(t)","h"),
    (3, "Potassium activation n(t)","n"),
]

# =========================
# EXTERNAL CURRENT
# =========================
def I_ext(t_ms: float) -> float:
    """Square pulse current in µA/cm^2."""
    return I_STEP_UA_CM2 if (I_ON_MS <= t_ms <= I_OFF_MS) else 0.0

# =========================
# GATING KINETICS (stable)
# =========================
def _safe_exp(x: np.ndarray | float) -> np.ndarray | float:
    """Exponent with clipping to avoid overflow in extreme values."""
    return np.exp(np.clip(x, EXP_CLIP_MIN, EXP_CLIP_MAX))

def alpha_m(V: np.ndarray | float) -> np.ndarray | float:
    x = (25.0 - V) / 10.0
    return np.where(
        np.abs(x) < SINGULARITY_EPS,
        1.0,  # limit of 0.1*(25-V)/(e^{(25-V)/10}-1) as x->0 equals 0.1*10
        0.1 * (25.0 - V) / (_safe_exp(x) - 1.0),
    )

def beta_m(V: np.ndarray | float) -> np.ndarray | float:
    return 4.0 * _safe_exp(-V / 18.0)

def alpha_h(V: np.ndarray | float) -> np.ndarray | float:
    return 0.07 * _safe_exp(-V / 20.0)

def beta_h(V: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (_safe_exp((30.0 - V) / 10.0) + 1.0)

def alpha_n(V: np.ndarray | float) -> np.ndarray | float:
    x = (10.0 - V) / 10.0
    return np.where(
        np.abs(x) < SINGULARITY_EPS,
        0.1,  # limit of 0.01*(10-V)/(e^{(10-V)/10}-1) as x->0 equals 0.01*10
        0.01 * (10.0 - V) / (_safe_exp(x) - 1.0),
    )

def beta_n(V: np.ndarray | float) -> np.ndarray | float:
    return 0.125 * _safe_exp(-V / 80.0)

def dm_dt(V, m): return alpha_m(V) * (1.0 - m) - beta_m(V) * m
def dh_dt(V, h): return alpha_h(V) * (1.0 - h) - beta_h(V) * h
def dn_dt(V, n): return alpha_n(V) * (1.0 - n) - beta_n(V) * n

# =========================
# MEMBRANE EQUATION
# =========================
def dV_dt(t_ms, V, m, h, n):
    I_Na = G_NA_MS_CM2 * (m**3) * h * (V - E_NA_MV)
    I_K  = G_K_MS_CM2  * (n**4) * (V - E_K_MV)
    I_L  = G_L_MS_CM2  * (V - E_L_MV)
    return (I_ext(t_ms) - I_Na - I_K - I_L) / C_M_UF_CM2

# =========================
# RHS in vector form
# =========================
def hh_rhs(t_ms: float, y: np.ndarray) -> np.ndarray:
    V, m, h_gate, n = y
    return np.array(
        [
            dV_dt(t_ms, V, m, h_gate, n),
            dm_dt(V, m),
            dh_dt(V, h_gate),
            dn_dt(V, n),
        ],
        dtype=float,
    )

# =========================
# INTEGRATORS
# =========================
def euler_step(t_ms: float, y: np.ndarray, dt_ms: float) -> np.ndarray:
    V, m, h_gate, n = y
    dV = dV_dt(t_ms, V, m, h_gate, n)
    dm = dm_dt(V, m)
    dh = dh_dt(V, h_gate)
    dn = dn_dt(V, n)
    return np.array(
        [V + dt_ms * dV, m + dt_ms * dm, h_gate + dt_ms * dh, n + dt_ms * dn],
        dtype=float,
    )

def rk4_step(t_ms: float, y: np.ndarray, dt_ms: float) -> np.ndarray:
    k1 = hh_rhs(t_ms, y)
    k2 = hh_rhs(t_ms + HALF * dt_ms, y + HALF * dt_ms * k1)
    k3 = hh_rhs(t_ms + HALF * dt_ms, y + HALF * dt_ms * k2)
    k4 = hh_rhs(t_ms + dt_ms,        y + dt_ms * k3)
    return y + (dt_ms * SIXTH) * (k1 + TWO * k2 + TWO * k3 + k4)

def simulate(method: str, T_ms: float, dt_ms: float, y0: np.ndarray, clamp: bool = True):
    """Simulate HH using Euler or RK4 with step dt_ms up to time T_ms."""
    N = int(np.ceil(T_ms / dt_ms))
    t = np.linspace(0.0, N * dt_ms, N + 1)
    Y = np.zeros((N + 1, 4), dtype=float)
    Y[0] = y0

    stepper = euler_step if method.lower() == "euler" else rk4_step

    for k in range(N):
        y_next = stepper(t[k], Y[k], dt_ms)
        if clamp:
            # Clamp gates to [0,1] and bound V for numerical safety
            y_next[1] = np.clip(y_next[1], GATE_MIN, GATE_MAX)  # m
            y_next[2] = np.clip(y_next[2], GATE_MIN, GATE_MAX)  # h
            y_next[3] = np.clip(y_next[3], GATE_MIN, GATE_MAX)  # n
            y_next[0] = np.clip(y_next[0], CLAMP_V_MIN_MV, CLAMP_V_MAX_MV)
        if not np.all(np.isfinite(y_next)):
            # Early stop if things blow up
            t = t[: k + 1]
            Y = Y[: k + 1]
            break
        Y[k + 1] = y_next

    return t, Y

# =========================
# REFERENCE TRAJECTORY
# =========================
def compute_reference(T_ms: float, y0: np.ndarray, dt_ref_ms: float = DT_REF_MS):
    """Dense, high-accuracy reference trajectory on uniform grid (dt_ref_ms)."""
    t_ref = np.arange(0.0, T_ms + 1e-12, dt_ref_ms)
    if _SCIPY_OK:
        sol = solve_ivp(
            fun=hh_rhs,
            t_span=(0.0, T_ms),
            y0=y0,
            method=REF_METHOD,
            atol=REF_ATOL,
            rtol=REF_RTOL,
            t_eval=t_ref,
        )
        if sol.success:
            return t_ref, sol.y.T  # shape: (len(t_ref), 4)
        # else: fall back to fine-step RK4
    return integrate_fine_rk4(T_ms, y0, dt_ref_ms)

def integrate_fine_rk4(T_ms: float, y0: np.ndarray, dt_ref_ms: float):
    N = int(np.ceil(T_ms / dt_ref_ms))
    t = np.linspace(0.0, N * dt_ref_ms, N + 1)
    Y = np.zeros((N + 1, 4), dtype=float)
    Y[0] = y0
    for k in range(N):
        y = Y[k]
        k1 = hh_rhs(t[k], y)
        k2 = hh_rhs(t[k] + HALF * dt_ref_ms, y + HALF * dt_ref_ms * k1)
        k3 = hh_rhs(t[k] + HALF * dt_ref_ms, y + HALF * dt_ref_ms * k2)
        k4 = hh_rhs(t[k] + dt_ref_ms,        y + dt_ref_ms * k3)
        y_next = y + (dt_ref_ms * SIXTH) * (k1 + TWO * k2 + TWO * k3 + k4)
        # Clamp for safety
        y_next[1] = np.clip(y_next[1], GATE_MIN, GATE_MAX)
        y_next[2] = np.clip(y_next[2], GATE_MIN, GATE_MAX)
        y_next[3] = np.clip(y_next[3], GATE_MIN, GATE_MAX)
        y_next[0] = np.clip(y_next[0], CLAMP_V_MIN_MV, CLAMP_V_MAX_MV)
        Y[k + 1] = y_next
    return t, Y

# =========================
# PLOTTING
# =========================
def format_axes(ax, title: str, ylabel: str = ""):
    ax.set_title(title)
    ax.set_xlabel("t [ms]")
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=GRID_ALPHA)

def save_three_function_overlays(dts_ms: list[float], T_ms: float, y0: np.ndarray, outdir: str):
    """
    For each dt, plot V,m,h,n with Euler, RK4, and Reference in one figure per variable.
    Returns a list of saved file paths.
    """
    os.makedirs(outdir, exist_ok=True)
    saved: list[str] = []

    # Precompute dense reference once
    t_ref, Y_ref = compute_reference(T_ms, y0, dt_ref_ms=DT_REF_MS)

    for dt in dts_ms:
        t_eu, Y_eu = simulate(method="euler", T_ms=T_ms, dt_ms=dt, y0=y0, clamp=True)
        t_rk, Y_rk = simulate(method="rk4",   T_ms=T_ms, dt_ms=dt, y0=y0, clamp=True)

        for idx, title, ylabel in VAR_INFO:
            plt.figure(figsize=FIGSIZE)
            plt.plot(t_ref, Y_ref[:, idx], lw=LINEWIDTH_REF, label=LABEL_REF)
            plt.plot(t_eu,  Y_eu[:,  idx], lw=LINEWIDTH_NUM, label=LABEL_EULER_TMPL.format(dt=dt))
            plt.plot(t_rk,  Y_rk[:,  idx], lw=LINEWIDTH_NUM, label=LABEL_RK4_TMPL.format(dt=dt))
            format_axes(plt.gca(), title + f" — 3 functions (Δt={dt:g})", ylabel)
            plt.legend()
            fname = os.path.join(outdir, f"HH_three_dt{dt:g}_var{idx}_overlay.png").replace(" ", "_")
            plt.tight_layout()
            plt.savefig(fname, dpi=150)
            plt.close()
            saved.append(fname)

    return saved

# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    # Initial conditions at rest (consistent with gating steady states at V_REST_MV)
    m0 = float(alpha_m(V_REST_MV) / (alpha_m(V_REST_MV) + beta_m(V_REST_MV)))
    h0 = float(alpha_h(V_REST_MV) / (alpha_h(V_REST_MV) + beta_h(V_REST_MV)))
    n0 = float(alpha_n(V_REST_MV) / (alpha_n(V_REST_MV) + beta_n(V_REST_MV)))
    y0 = np.array([V_REST_MV, m0, h0, n0], dtype=float)

    saved_files = save_three_function_overlays(DTS_MS, T_FINAL_MS, y0, OUTDIR_PATH)

    print("Saved figures:")
    for fpath in saved_files:
        print(" -", os.path.abspath(fpath))
    if not _SCIPY_OK:
        print("[Notice] SciPy not found — reference used fine-step RK4.")
