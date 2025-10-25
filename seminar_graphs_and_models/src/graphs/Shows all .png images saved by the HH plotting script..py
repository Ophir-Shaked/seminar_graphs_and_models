# -*- coding: utf-8 -*-
# Graph (7) — Python code for seminar figure/model.
# Last updated: 2025-10-25 21:59:15
"""
Display saved Hodgkin–Huxley figures
------------------------------------
Shows all .png images saved by the HH plotting script.

All literals are centralized in CONSTANTS (no magic numbers).
"""

from __future__ import annotations
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# =========================
# CONSTANTS
# =========================
# Candidate output directories (first one that exists will be used)
OUTDIR_LOCAL: str = "./hh_outputs"
OUTDIR_COLAB: str = "/content/hh_outputs"
SEARCH_DIRS: tuple[str, ...] = (OUTDIR_COLAB, OUTDIR_LOCAL)

# File matching
IMG_EXT: str = ".png"
GLOB_PATTERN_TMPL: str = "*{ext}"

# Figure appearance
FIGSIZE: tuple[float, float] = (8.0, 4.5)
AXIS_OFF: bool = True

# Printing
PRINT_PREFIX: str = "Showing: "

# Sorting
SORT_REVERSE: bool = False  # set True to show newest-last if filenames encode time

# =========================
# MAIN
# =========================
def main() -> None:
    # Pick the first existing directory from SEARCH_DIRS
    outdir = next((d for d in SEARCH_DIRS if os.path.isdir(d)), None)
    if outdir is None:
        print("No output directory found. Checked:", ", ".join(SEARCH_DIRS))
        return

    # Build glob pattern and list files
    pattern = os.path.join(outdir, GLOB_PATTERN_TMPL.format(ext=IMG_EXT))
    files = sorted(glob.glob(pattern), reverse=SORT_REVERSE)

    if not files:
        print(f"No images found in: {outdir} (pattern: {pattern})")
        return

    for fname in files:
        print(PRINT_PREFIX + fname)
        img = mpimg.imread(fname)
        plt.figure(figsize=FIGSIZE)
        plt.imshow(img)
        if AXIS_OFF:
            plt.axis("off")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
