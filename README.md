# Seminar Graphs & Models — Source Code

This repository contains **stand‑alone Python scripts** that generate figures and model outputs used in a numerical‑methods seminar. Each script lives in `src/graphs/` and can be run independently.

## Algorithms & Topics
- **ODE solvers**: explicit Euler, classical **Runge–Kutta 4 (RK4)**, and general Runge–Kutta stepping utilities.
- **Error analysis**: local/global truncation error checks, step‑size sensitivity.
- **Dynamical systems demos**: example right‑hand sides (e.g., oscillators) with phase‑plane/trajectory plots.
- **Figure generation**: reproducible scripts that write PNG/SVG outputs for inclusion in reports.

## Libraries
- **Python 3.9+**
- **NumPy** — vectorized numerics
- **SciPy** — ODE helpers (for stiff/reference solvers)
- **Matplotlib** — plotting
- **Pandas** (optional) — light data handling

```bash
pip install numpy scipy matplotlib pandas
```

## Layout
```
.
├─ src/
│  └─ graphs/
│     ├─ Graph (1).py
│     ├─ Graph (2).py
│     ├─ Graph (3).py
│     ├─ Graph (5).py
│     ├─ Graph (6)-(14).py
│     ├─ Hodgkin–Huxley Euler vs RK4 vs Reference (Radaufine-RK4).py
│     └─ Shows all .png images saved by the HH plotting script..py
├─ docs/
│  └─ notes_*.md
└─ README.md
```

## Usage
Run any script directly:
```bash
python src/graphs/"Graph (1).py"
```
Many scripts save figures next to the script or into a `figures/` or `hh_outputs/` subfolder.

## File Index
| Script | Description |
|---------|--------------|
| `Graph (1).py` | e^h vs Euler and RK4 single-step comparison |
| `Graph (2).py` | Exact vs Euler vs RK4 trajectory comparison |
| `Graph (3).py` | Stability functions R(z) for Euler and RK4 |
| `Graph (5).py` | Simple pendulum — Euler vs RK4 vs analytic small-angle |
| `Graph (6)-(14).py` | Van der Pol oscillator demos with stiff-region shading |
| `Hodgkin–Huxley Euler vs RK4 vs Reference (Radaufine-RK4).py` | Hodgkin–Huxley neuron model comparison |
| `Shows all .png images saved by the HH plotting script..py` | Utility to display saved Hodgkin–Huxley images |
