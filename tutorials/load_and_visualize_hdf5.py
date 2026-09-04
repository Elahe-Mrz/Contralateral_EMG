#!/usr/bin/env python3
"""Load and visualize a synchronized excerpt from one released HDF5 take."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from load_release import load_synced_hdf5  # noqa: E402


ANGLES = ["index_q1", "middle_q1", "ring_q1", "pinky_q1", "thumb_q2", "thumb_q1"]
FORCES = ["middle_N", "index_N", "thumb_N"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hdf5_file", type=Path)
    parser.add_argument("--start-s", type=float, default=0.0)
    parser.add_argument("--duration-s", type=float, default=15.0)
    parser.add_argument("--output", type=Path, default=Path("hdf5_example.png"))
    args = parser.parse_args()

    frame = load_synced_hdf5(args.hdf5_file)
    if "timestamp" not in frame:
        raise KeyError("The synchronized table has no timestamp column")
    time = frame["timestamp"].to_numpy(float)
    finite = np.flatnonzero(np.isfinite(time))
    if not len(finite):
        raise ValueError("The synchronized table has no finite timestamps")
    time = time - time[finite[0]]
    keep = (time >= args.start_s) & (time <= args.start_s + args.duration_s)
    if keep.sum() < 2:
        raise ValueError("The requested interval contains fewer than two rows")
    shown_time = time[keep] - args.start_s

    emg = [name for name in (f"EMG {i}" for i in range(1, 9)) if name in frame][:3]
    angles = [name for name in ANGLES if name in frame]
    forces = [name for name in FORCES if name in frame]
    if not emg or not angles or not forces:
        raise KeyError("Expected sEMG, angle, or force columns are missing")

    figure, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True, constrained_layout=True)
    for name in emg:
        axes[0].plot(shown_time, frame.loc[keep, name].to_numpy(float) / 1000.0,
                     linewidth=0.4, alpha=0.8, label=name)
    axes[0].set_ylabel("sEMG (mV)")
    for name in angles:
        axes[1].plot(shown_time, frame.loc[keep, name].to_numpy(float), linewidth=1.1, label=name)
    axes[1].set_ylabel("Joint angle (deg)")
    for name in forces:
        axes[2].plot(shown_time, frame.loc[keep, name].to_numpy(float), linewidth=1.2, label=name)
    axes[2].set_ylabel("Force (N)")
    axes[2].set_xlabel("Time within excerpt (s)")
    for axis in axes:
        axis.legend(frameon=False, ncol=3, fontsize=8)
        axis.grid(alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
        axis.margins(x=0)
    figure.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"Loaded {args.hdf5_file}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
