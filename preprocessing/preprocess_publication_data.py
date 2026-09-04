#!/usr/bin/env python3
"""
Publication preprocessing pipeline for EMG, force, IMU, and retargeted angles.

This is a copy adapted from Preprocessing_Codes/preprocess_all.py. It never
modifies input recordings: every derived CSV, plot, and manifest is written to
an explicitly separate output root. Event-window trimming and lag/resync are
deliberately disabled by default so the data-paper derivatives retain their
recorded timestamps and full duration.

Usage:
    python3 preprocess_all.py "/path/to/root_folder"

The script will:
1. Find all subfolders with raw data files
2. Preprocess EMG (bandpass, notch, detrend)
3. Preprocess angles and forces (median + lowpass)
4. Sync all data to the EMG time grid, including IMU
5. Save preprocessed files and plots in each folder
"""

import os
import json
import re
import argparse
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch
from scipy.ndimage import median_filter


# ================= CONFIG =================

# EMG preprocessing
EMG_FS = 2000.0
EMG_BP_LO = 20.0
EMG_BP_HI = 450.0
EMG_USE_60HZ_NOTCH = True
EMG_USE_50HZ_NOTCH = False
EMG_NOTCH_Q = 30.0

# Force / retargeted-angle preprocessing. These publication defaults reduce
# isolated noise without the 0.8 s median + 0.5 s moving average used by the
# legacy script, which can obscure onset timing and individual repetitions.
FORCE_FS = 100.0
ANGLES_FS = 15.0
FORCE_MEDIAN_MS = 100
FORCE_LOWPASS_HZ = 5.0
FORCE_BASELINE_MODE = "first_seconds"  # "first_seconds" or "percentile"
FORCE_BASELINE_FIRST_SECONDS = 10.0
FORCE_BASELINE_PERCENTILE = 25.0
FORCE_BASELINE_MAX_FRACTION_OF_P95 = 0.15
FORCE_ZERO_DEADBAND_N_MAX = 0.25
FORCE_ZERO_DEADBAND_FRACTION_OF_P95 = 0.05
ANGLES_MEDIAN_MS = 200
ANGLES_LOWPASS_HZ = 5.0
FORCE_MEAN_MS = 0.0
ANGLES_MEAN_MS = 0.0

# IMU preprocessing. Accelerometer and gyroscope retain their physical DC
# components (including gravity); no per-recording normalization or offset
# subtraction is performed. Magnetometer is filtered more strongly because it
# is particularly susceptible to high-frequency environmental noise.
IMU_FS = 200.0
IMU_MEDIAN_MS = 0.0
IMU_ACCEL_GYRO_LOWPASS_HZ = 20.0
IMU_MAG_LOWPASS_HZ = 10.0
IMU_MAX_GAP_S = 0.05


def configure_preprocessing_profile(profile: str) -> None:
    """Select the documented legacy or publication force/angle settings."""
    global FORCE_MEDIAN_MS, FORCE_LOWPASS_HZ, FORCE_MEAN_MS
    global ANGLES_MEDIAN_MS, ANGLES_LOWPASS_HZ, ANGLES_MEAN_MS
    if profile == "legacy":
        # Exact effective behavior of the former preprocess_all.py: its force
        # low-pass declaration was commented out, therefore not applied.
        FORCE_MEDIAN_MS, FORCE_LOWPASS_HZ, FORCE_MEAN_MS = 800, 0.0, 100.0
        ANGLES_MEDIAN_MS, ANGLES_LOWPASS_HZ, ANGLES_MEAN_MS = 800, 1.0, 500.0
    elif profile == "publication":
        FORCE_MEDIAN_MS, FORCE_LOWPASS_HZ, FORCE_MEAN_MS = 100, 5.0, 0.0
        ANGLES_MEDIAN_MS, ANGLES_LOWPASS_HZ, ANGLES_MEAN_MS = 200, 5.0, 0.0
    else:
        raise ValueError(f"Unknown preprocessing profile: {profile}")

# Spike removal for EMG. Conservative Hampel settings: catch isolated pops
# without flattening real activation bursts.
EMG_HAMPEL_WINDOW_MS = 25.0
EMG_HAMPEL_N_SIGMAS = 10.0
EMG_HAMPEL_MAX_SPIKE_MS = 10.0
EMG_HAMPEL_GLOBAL_FLOOR_SIGMAS = 4.0
EMG_ADC_RAIL_UV = 24000.0
EMG_ADC_RAIL_TOLERANCE_UV = 0.0
EMG_ADC_MERGE_GAP_MS = 5.0

# Skip if output exists
SKIP_IF_OUTPUT_EXISTS = True

preprocessed_folder = "preprocessed"

# Choose which preprocessed angles file to use for sync:
#   "calibrated" -> *_calibrated_ANGLES_preprocessed.csv
#   "raw"        -> raw/non-calibrated online *_ANGLES_preprocessed.csv
#   "offline"    -> offline_retargeted_angles_*_ANGLES_preprocessed.csv
SYNC_ANGLES = "offline"  # "raw", "calibrated", "offline", or "all"

# P007 was re-triangulated and retargeted with the corrected June 01 stereo
# calibration.  Those *_calibfixed_raw files are the approved canonical angle
# source for P007 synchronization; the generic offline files remain canonical
# for all other participants.
CORRECTED_OFFLINE_PARTICIPANTS = {"P007"}

# Choose which angle file should create Angles_preprocessed_plot.png.
# Options: "raw", "calibrated", "read", "offline", or "all".
PLOT_ANGLES = "offline"
ANGLE_TIME_COLUMN = "timestamp"
ANGLE_INTERPOLATION_MAX_GAP_S = 2.0
# "frame_capture_t"
# "timestamp"

# ================= FILE FINDING =================

def find_files(folder: Path, patterns: List[str], exclude_patterns: List[str] = None) -> List[Path]:
    """Find files matching any of the patterns."""
    if exclude_patterns is None:
        exclude_patterns = ["_preprocessed", "_synced","_onsets"]
    
    found = []
    for f in folder.iterdir():
        if not f.is_file():
            continue
        # Only consider CSV files (include .csv, exclude everything else)
        if f.suffix.lower() != '.csv':
            continue
        name_lower = f.name.lower()
        # Exclude lock files and already processed files
        if name_lower.startswith('.~lock'):
            continue
        if any(ex in name_lower for ex in exclude_patterns):
            continue
        # Check patterns
        if any(p.lower() in name_lower for p in patterns):
            found.append(f)
    return found


def find_data_folders(root: Path, recursive: bool = True) -> List[Path]:
    """Find all folders containing raw data files."""
    folders = set()
    
    # Patterns for raw data files
    emg_patterns = ["noraxon", "emg_2000", "emg_"]
    fsr_patterns = ["fsr_"]
    angles_patterns = ["retargeting_angles_", "offline_retargeted_angles_", "_calibrated"]
    
    if recursive:
        for f in root.rglob("*"):
            if f.is_file() and f.suffix.lower() == '.csv':
                name_lower = f.name.lower()
                # Skip if in preprocessed folder
                if "preprocessed" in f.parts:
                    continue
                if any(p in name_lower for p in emg_patterns + fsr_patterns + angles_patterns):
                    if "_preprocessed" not in name_lower and "_synced" not in name_lower:
                        folders.add(f.parent)
    else:
        for f in root.iterdir():
            if f.is_dir():
                files = [fi for fi in f.iterdir() if fi.is_file() and fi.suffix.lower() == '.csv']
                has_data = any(
                    any(p in fn.name.lower() for p in emg_patterns + fsr_patterns + angles_patterns)
                    for fn in files
                )
                if has_data:
                    folders.add(f)
    
    return sorted(folders)


# ================= EMG PREPROCESSING =================

def find_emg_cols(df: pd.DataFrame) -> List[str]:
    """Detect EMG columns."""
    candidates = []
    for c in df.columns:
        cu = c.strip()
        if re.match(r"^(CH|Ch|Channel)[ _\-]*\d+$", cu):
            candidates.append(c)
            continue
        if re.match(r"^EMG[ _\-]*\d+", cu, flags=re.IGNORECASE):
            candidates.append(c)
            continue
        if re.search(r"\bEMG\b", cu, flags=re.IGNORECASE) and re.search(r"\d+", cu):
            candidates.append(c)
    if not candidates:
        candidates = [c for c in df.columns if c.upper().startswith("CH")]
    return candidates


def infer_sampling_frequency(df: pd.DataFrame, candidates: List[str], fallback_hz: float) -> float:
    """Infer sampling frequency from positive timestamp differences when available."""
    time_col = _pick_time_column(df, candidates)
    if not time_col:
        return fallback_hz
    t = time_values_seconds(df, time_col)
    dt = np.diff(t[np.isfinite(t)])
    dt = dt[dt > 0]
    if len(dt) == 0:
        return fallback_hz
    fs = 1.0 / float(np.median(dt))
    return fs if np.isfinite(fs) and fs > 0 else fallback_hz


def _scipy_bandpass(sig, fs, lo, hi, order=2):
    nyq = 0.5 * fs
    lo_n = lo / nyq
    hi_n = hi / nyq
    lo_n = max(lo_n, 1e-6)
    hi_n = min(hi_n, 0.999999)
    if lo_n >= hi_n:
        return sig
    b, a = butter(order, [lo_n, hi_n], btype='bandpass')
    return filtfilt(b, a, sig, method="gust")


def _scipy_notch(sig, fs, f0, q=50.0):
    nyq = 0.5 * fs
    if f0 >= nyq:
        return sig
    w0 = f0 / nyq
    b, a = iirnotch(w0, q)
    return filtfilt(b, a, sig, method="pad")


def _scipy_detrend_dc(sig):
    m = np.nanmean(sig)
    return sig - (0.0 if np.isnan(m) else m)


def repair_emg_adc_saturation(df: pd.DataFrame, emg_cols: List[str], fs: float) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Interpolate short clusters containing raw ADC-rail samples before filtering.

    Rail samples are invalid clipped observations.  Neighbouring rail runs
    separated by <=5 ms are treated as one transient so a clipped burst is not
    fragmented into several partial repairs.  Raw source files are never edited.
    """
    result, repairs = df.copy(), []
    max_gap = max(0, int(round(EMG_ADC_MERGE_GAP_MS * fs / 1000.0)))
    threshold = EMG_ADC_RAIL_UV - EMG_ADC_RAIL_TOLERANCE_UV
    for channel in emg_cols:
        values = pd.to_numeric(result[channel], errors="coerce").to_numpy(float)
        mask = np.isfinite(values) & (np.abs(values) >= threshold)
        edges = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(np.int8))).reshape(-1, 2)
        runs = [(int(a), int(b)) for a, b in edges]
        merged: list[list[int]] = []
        for start, end in runs:
            if merged and start - merged[-1][1] <= max_gap:
                merged[-1][1] = end
            else:
                merged.append([start, end])
        for start, end in merged:
            if start == 0 or end >= len(values):
                continue
            left, right = values[start - 1], values[end]
            if not (np.isfinite(left) and np.isfinite(right)):
                continue
            values[start:end] = np.linspace(left, right, end - start + 2)[1:-1]
            repairs.append({"channel": channel, "start_sample": start, "end_sample": end - 1,
                            "sample_count": end - start, "duration_s": (end - start) / fs})
        result[channel] = values
    return result, repairs


def preprocess_emg(df: pd.DataFrame, emg_cols: List[str], fs: float) -> pd.DataFrame:
    """Preprocess EMG: detrend, bandpass, notch."""
    df = df.copy()
    for ch in emg_cols:
        sig = df[ch].to_numpy(dtype=np.float64, copy=True)
        
        # Interpolate small NaN runs
        if np.isnan(sig).any():
            sig = pd.Series(sig).interpolate(limit_direction="both").to_numpy(np.float64)
        
        # Detrend DC
        sig = _scipy_detrend_dc(sig)
        
        # Bandpass
        sig = _scipy_bandpass(sig, fs, EMG_BP_LO, EMG_BP_HI, order=2)
        
        # Notch 60Hz
        if EMG_USE_60HZ_NOTCH:
            sig = _scipy_notch(sig, fs, 60.0, EMG_NOTCH_Q)
        
        # Notch 50Hz
        if EMG_USE_50HZ_NOTCH:
            sig = _scipy_notch(sig, fs, 50.0, EMG_NOTCH_Q)
        
        df[ch] = sig
    
    return df


def _short_true_runs(mask: np.ndarray, max_len: int) -> np.ndarray:
    """Keep only True runs no longer than max_len samples."""
    if not mask.any():
        return mask

    keep = np.zeros_like(mask, dtype=bool)
    idx = np.flatnonzero(mask)
    splits = np.where(np.diff(idx) > 1)[0] + 1
    for run in np.split(idx, splits):
        if len(run) <= max_len:
            keep[run] = True
    return keep


def hampel_filter_emg_spikes(
    sig: np.ndarray,
    fs: float,
    window_ms: float = EMG_HAMPEL_WINDOW_MS,
    n_sigmas: float = EMG_HAMPEL_N_SIGMAS,
    max_spike_ms: float = EMG_HAMPEL_MAX_SPIKE_MS,
    global_floor_sigmas: float = EMG_HAMPEL_GLOBAL_FLOOR_SIGMAS,
) -> Tuple[np.ndarray, int]:
    """Replace obvious short EMG spikes with the local median."""
    y = np.asarray(sig, dtype=np.float64).copy()
    finite = np.isfinite(y)
    if finite.sum() < 3:
        return y, 0

    if not finite.all():
        y = pd.Series(y).interpolate(limit_direction="both").to_numpy(np.float64)

    window = _median_window_samples(window_ms, fs)
    local_median = median_filter(y, size=window, mode="nearest")
    local_mad = median_filter(np.abs(y - local_median), size=window, mode="nearest")
    local_sigma = 1.4826 * local_mad

    center = np.nanmedian(y)
    global_sigma = 1.4826 * np.nanmedian(np.abs(y - center))
    if not np.isfinite(global_sigma) or global_sigma <= 0:
        positive_local = local_sigma[local_sigma > 0]
        global_sigma = np.nanmedian(positive_local) if len(positive_local) else 0.0

    eps = np.finfo(np.float64).eps
    threshold = np.maximum(n_sigmas * local_sigma, global_floor_sigmas * max(global_sigma, eps))
    spike_mask = np.abs(y - local_median) > threshold
    spike_mask &= finite
    spike_mask = _short_true_runs(spike_mask, max(1, int(round((max_spike_ms / 1000.0) * fs))))

    n_spikes = int(spike_mask.sum())
    if n_spikes:
        y[spike_mask] = local_median[spike_mask]
    return y, n_spikes


def replace_emg_spikes_hampel(df: pd.DataFrame, emg_cols: List[str], fs: float) -> pd.DataFrame:
    """Conservatively replace obvious isolated EMG spikes via Hampel filtering."""
    df = df.copy()
    for col in emg_cols:
        sig = df[col].to_numpy(dtype=np.float64, copy=True)
        sig, n_spikes = hampel_filter_emg_spikes(sig, fs)
        if n_spikes > 0:
            print(f"    {col}: {n_spikes} isolated spikes replaced by Hampel median")
            df[col] = sig
    return df


# ================= ANGLES/FORCES PREPROCESSING =================

def ensure_odd(n: int) -> int:
    return n if n % 2 == 1 else n + 1


def _median_window_samples(ms: float, fs: float) -> int:
    k = int(round((ms / 1000.0) * fs))
    return max(3, ensure_odd(k))


def butter_lowpass(y: np.ndarray, fs: float, cutoff: float, order: int = 2) -> np.ndarray:
    if cutoff <= 0:
        return y
    b, a = butter(order, cutoff / (0.5 * fs), btype="low")
    return filtfilt(b, a, y, axis=0)


def mean_filter(y: np.ndarray, window: int) -> np.ndarray:
    """Apply moving-average (mean) filter without zero-padding edge artifacts."""
    if window <= 1:
        return y
    w = int(window)
    if w % 2 == 0:
        w += 1
    y = np.asarray(y, dtype=float)
    pad = w // 2
    kernel = np.ones(w) / w
    if y.ndim == 1:
        y_pad = np.pad(y, (pad, pad), mode="edge")
        return np.convolve(y_pad, kernel, mode="valid")
    out = np.zeros_like(y)
    for i in range(y.shape[1]):
        y_pad = np.pad(y[:, i], (pad, pad), mode="edge")
        out[:, i] = np.convolve(y_pad, kernel, mode="valid")
    return out


def hysteresis_zero(x, t_on=0.25, t_off=0.15):
    y = x.copy()
    contact = False
    for i in range(len(y)):
        v = y[i]
        if not np.isfinite(v):
            continue
        if contact:
            if v <= t_off:
                contact = False
                y[i] = 0.0
        else:
            if v >= t_on:
                contact = True
            else:
                y[i] = 0.0
    return y


def estimate_thresholds(x):
    x0 = x[x < np.percentile(x, 20)]
    mu = np.median(x0)
    mad = np.median(np.abs(x0 - mu))
    return mu + 5 * mad, mu + 3 * mad


def force_baseline_and_deadband(sig: np.ndarray, t: np.ndarray = None) -> Tuple[float, float]:
    """Estimate force baseline and deadband for one force channel."""
    finite = np.asarray(sig, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 0.0

    p95 = max(float(np.nanpercentile(finite, 95)), 0.0)
    use_first_seconds = False
    baseline_values = finite
    if FORCE_BASELINE_MODE == "first_seconds" and t is not None:
        t_arr = np.asarray(t, dtype=float)
        y_arr = np.asarray(sig, dtype=float)
        valid_t = np.isfinite(t_arr)
        if valid_t.any():
            t0 = float(t_arr[valid_t][0])
            baseline_mask = valid_t & (t_arr <= t0 + FORCE_BASELINE_FIRST_SECONDS) & np.isfinite(y_arr)
            if baseline_mask.any():
                baseline_values = y_arr[baseline_mask]
                use_first_seconds = True

    if use_first_seconds:
        baseline = float(np.nanmedian(baseline_values))
    else:
        baseline = float(np.nanpercentile(finite, FORCE_BASELINE_PERCENTILE))
        if p95 > 0:
            baseline = min(baseline, FORCE_BASELINE_MAX_FRACTION_OF_P95 * p95)

    if p95 > 0:
        deadband = min(FORCE_ZERO_DEADBAND_N_MAX, FORCE_ZERO_DEADBAND_FRACTION_OF_P95 * p95)
    else:
        baseline = max(baseline, 0.0)
        deadband = 0.0
    return baseline, deadband


def preprocess_forces(df: pd.DataFrame, force_cols: List[str]) -> pd.DataFrame:
    """Preprocess forces: median filter + lowpass."""
    df = df.copy()
    if not force_cols:
        return df

    force_fs = FORCE_FS
    if "t" in df.columns and len(df) > 1:
        dt = df["t"].diff().median()
        if pd.notna(dt) and dt > 0:
            force_fs = 1.0 / dt
            print(f"    Using real force frequency: {force_fs:.2f} Hz")

    # Baseline removal BEFORE filtering
    t = df["t"].to_numpy() if "t" in df.columns else None
    for col in force_cols:
        sig = df[col].to_numpy()
        baseline, _ = force_baseline_and_deadband(sig, t)
        df[col] = sig - baseline
        if FORCE_BASELINE_MODE == "first_seconds" and t is not None:
            print(f"    {col}: first {FORCE_BASELINE_FIRST_SECONDS:g}s baseline removed = {baseline:.3f} N")
        # (No clipping here)

    F = df[force_cols].to_numpy()

    # Median filter
    if FORCE_MEDIAN_MS > 0:
        k = _median_window_samples(FORCE_MEDIAN_MS, force_fs)
        F = median_filter(F, size=(k, 1), mode="nearest")

    # Optional moving average. Disabled by default because the low-pass stage
    # below already supplies explicit, documented smoothing.
    if FORCE_MEAN_MS > 0:
        mean_win = max(1, int(round((FORCE_MEAN_MS / 1000.0) * force_fs)))
        F = mean_filter(F, mean_win)

    # Lowpass filter
    if FORCE_LOWPASS_HZ > 0:
        F = butter_lowpass(F, force_fs, FORCE_LOWPASS_HZ, order=4)

    # Clip to zero and suppress small idle residuals after baseline subtraction.
    for i, col in enumerate(force_cols):
        corrected = np.clip(F[:, i], 0, None)
        _, deadband = force_baseline_and_deadband(corrected)
        df[col] = np.where(corrected < deadband, 0.0, corrected)

    return df


def preprocess_angles(df: pd.DataFrame, angle_cols: List[str]) -> pd.DataFrame:
    """Preprocess angles: median filter + lowpass."""
    df = df.copy()
    if not angle_cols:
        return df
    
    angles_fs = ANGLES_FS
    if "t" in df.columns and len(df) > 1:
        dt = df["t"].diff().median()
        if pd.notna(dt) and dt > 0:
            angles_fs = 1.0 / dt
            print(f"    Using real angles frequency: {angles_fs:.2f} Hz")
    
    A = df[angle_cols].to_numpy()
    
    # Median filter
    if ANGLES_MEDIAN_MS > 0:
        k = _median_window_samples(ANGLES_MEDIAN_MS, angles_fs)
        A = median_filter(A, size=(k, 1), mode="nearest")
        
    
    # Lowpass
    if ANGLES_LOWPASS_HZ > 0:
        A = butter_lowpass(A, angles_fs, ANGLES_LOWPASS_HZ, order=4)


    # Optional moving average; disabled by default to avoid a second,
    # undocumented temporal smoothing stage after the low-pass filter.
    if ANGLES_MEAN_MS > 0:
        mean_win = max(1, int(round((ANGLES_MEAN_MS / 1000.0) * angles_fs)))
        A = mean_filter(A, mean_win)

    
    for i, col in enumerate(angle_cols):
        df[col] = A[:, i]
    
    return df


# ================= IMU PREPROCESSING =================

def find_imu_cols(df: pd.DataFrame) -> List[str]:
    """Return tri-axial accelerometer, gyro, and magnetometer channels."""
    return [
        col for col in df.columns
        if re.match(r"^IMU[ _-]*\d+_(?:a|g|m)[xyz]$", col.strip(), flags=re.IGNORECASE)
    ]


def _filter_imu_segments(y: np.ndarray, t: np.ndarray, cutoff_hz: float, fs: float) -> np.ndarray:
    """Low-pass finite, contiguous IMU sections without filtering across gaps."""
    out = np.asarray(y, dtype=float).copy()
    valid = np.isfinite(out) & np.isfinite(t)
    idx = np.flatnonzero(valid)
    if len(idx) == 0 or cutoff_hz <= 0:
        return out
    breaks = np.where((np.diff(idx) != 1) | (np.diff(t[idx]) > IMU_MAX_GAP_S))[0] + 1
    for segment in np.split(idx, breaks):
        # filtfilt's default padding needs more than 3 * filter order samples.
        if len(segment) >= 16:
            out[segment] = butter_lowpass(out[segment], fs, cutoff_hz, order=4)
    return out


def preprocess_imu(df: pd.DataFrame, imu_cols: List[str]) -> pd.DataFrame:
    """Lightly filter IMU channels while retaining their calibrated sensor units."""
    df = df.copy()
    if not imu_cols:
        return df
    if "t" not in df.columns:
        raise ValueError("IMU preprocessing requires a timestamp column converted to 't'")
    t = pd.to_numeric(df["t"], errors="coerce").to_numpy(dtype=float)
    fs = IMU_FS
    finite_t = t[np.isfinite(t)]
    if len(finite_t) > 1:
        dt = float(np.nanmedian(np.diff(finite_t)))
        if np.isfinite(dt) and dt > 0:
            fs = 1.0 / dt
    print(f"    Using IMU sampling frequency: {fs:.2f} Hz")
    for col in imu_cols:
        y = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        if IMU_MEDIAN_MS > 0:
            y = median_filter(y, size=_median_window_samples(IMU_MEDIAN_MS, fs), mode="nearest")
        suffix = col.strip().lower().rsplit("_", 1)[-1]
        cutoff = IMU_MAG_LOWPASS_HZ if suffix.startswith("m") else IMU_ACCEL_GYRO_LOWPASS_HZ
        df[col] = _filter_imu_segments(y, t, cutoff, fs)
    return df


# ================= SYNC =================

def _pick_time_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = list(df.columns)
    for c in candidates:
        if c in cols:
            print (f"    Found time column: {c}")
            return c
    lowmap = {c.lower(): c for c in cols}
    for c in candidates:
        if c.lower() in lowmap:
            print (f"    Found time column: {lowmap[c.lower()]}")
            return lowmap[c.lower()]
    return None


def angle_file_kind(path: Path) -> str:
    """Return raw/calibrated/read/offline for an angle file when possible."""
    name = path.stem.lower()
    if name.startswith("offline_retargeted_angles_"):
        return "offline"
    for kind in ("calibrated", "raw", "read"):
        if f"_{kind}" in name:
            return kind
    return "unknown"


def set_time_seconds(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Set df['t'] in seconds from the first available time column."""
    tcol = _pick_time_column(df, candidates)
    if not tcol:
        return None

    t = pd.to_numeric(df[tcol], errors="coerce")
    if tcol.lower() == "timestamp_fused_ms":
        t = t / 1000.0
        print("    Converted timestamp_fused_ms from ms to seconds")
    df["t"] = t
    return tcol


def set_angle_time_seconds(df: pd.DataFrame) -> str:
    """Set angle df['t'] in seconds from the configured angle time column."""
    tcol = _pick_time_column(df, [ANGLE_TIME_COLUMN])
    if not tcol:
        raise ValueError(f"Angle data must contain '{ANGLE_TIME_COLUMN}' for timing")

    df["t"] = time_values_seconds(df, tcol)
    print(f"    Using {tcol} for angle time")
    return tcol


def time_values_seconds(df: pd.DataFrame, time_col: str) -> np.ndarray:
    """Return a time column as seconds."""
    t = pd.to_numeric(df[time_col], errors="coerce").to_numpy()
    time_col_lower = time_col.lower()
    if time_col_lower.endswith("_ms") or time_col_lower in {"timestamp_ms", "timestamp_fused_ms"}:
        t = t / 1000.0
    return t


def make_time_monotonic(df: pd.DataFrame, time_col: str = "t", label: str = "data") -> pd.DataFrame:
    """Sort by time and drop duplicate/non-finite timestamps."""
    if time_col not in df.columns:
        return df

    before = len(df)
    clean = df.copy()
    clean[time_col] = pd.to_numeric(clean[time_col], errors="coerce")
    clean = clean[np.isfinite(clean[time_col])].copy()
    clean = clean.sort_values(time_col, kind="mergesort")
    clean = clean.drop_duplicates(subset=[time_col], keep="first")
    clean = clean.reset_index(drop=True)

    removed = before - len(clean)
    if removed:
        print(f"    {label}: removed {removed} duplicate/non-finite timestamp rows")
    return clean


def sorted_unique_xy(t: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return finite x/y pairs sorted by time with duplicate x-values removed."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(t) & np.isfinite(y)
    t = t[valid]
    y = y[valid]
    if len(t) == 0:
        return t, y

    order = np.argsort(t, kind="mergesort")
    t = t[order]
    y = y[order]
    keep = np.concatenate(([True], np.diff(t) > 0))
    return t[keep], y[keep]


def interpolate_to_grid(t_grid, t_src, y_src, max_gap_s=1.0):
    """Interpolate source data to target time grid."""
    y_out = np.full_like(t_grid, np.nan, dtype=float)
    
    t_src, y_src = sorted_unique_xy(t_src, y_src)
    
    if len(t_src) < 2:
        return y_out
    
    y_interp = np.interp(t_grid, t_src, y_src)
    
    # Mask gaps
    dt = np.diff(t_src)
    gap_idx = np.where(dt > max_gap_s)[0]
    
    mask = np.ones_like(t_grid, dtype=bool)
    for g in gap_idx:
        t_lo = t_src[g]
        t_hi = t_src[g + 1]
        mask &= ~((t_grid > t_lo) & (t_grid < t_hi))
    
    y_out[mask] = y_interp[mask]
    return y_out


def remove_force_baseline(df, force_cols: List[str]) -> pd.DataFrame:
    """Remove baseline from force signals."""
    df = df.copy()
    t = df["t"].to_numpy() if "t" in df.columns else None
    for c in force_cols:
        sig = df[c].to_numpy()
        baseline, deadband = force_baseline_and_deadband(sig, t)
        corrected = np.clip(sig - baseline, 0, None)
        df[c] = np.where(corrected < deadband, 0.0, corrected)
        print(
            f"    {c}: baseline removed = {baseline:.3f} N "
            f"(p{FORCE_BASELINE_PERCENTILE:g}, deadband {deadband:.2f} N)"
        )
    return df


# def retime_to_uniform(df, time_col="timestamp", fs=2000.0) -> pd.DataFrame:
#     """Retime to uniform sampling."""
#     dt = 1.0 / fs
#     df = df.copy()
#     df[time_col] = np.arange(len(df)) * dt
#     return df


def sync_data(emg_df: pd.DataFrame, fsr_df: pd.DataFrame,
              angles_df: pd.DataFrame, imu_df: Optional[pd.DataFrame] = None,
              fs: float = 2000.0,
              t_start: float = None, t_end: float = None) -> pd.DataFrame:
    """Sync all data to EMG time grid with optional time window trimming."""
    
    # Trim to time window if provided
    if t_start is not None and t_end is not None:
        if emg_df is not None and "t" in emg_df.columns:
            emg_df = emg_df[(emg_df["t"] >= t_start) & (emg_df["t"] <= t_end)].copy()
        if fsr_df is not None and "t" in fsr_df.columns:
            fsr_df = fsr_df[(fsr_df["t"] >= t_start) & (fsr_df["t"] <= t_end)].copy()
        if angles_df is not None and "t" in angles_df.columns:
            angles_df = angles_df[(angles_df["t"] >= t_start) & (angles_df["t"] <= t_end)].copy()
        if imu_df is not None and "t" in imu_df.columns:
            imu_df = imu_df[(imu_df["t"] >= t_start) & (imu_df["t"] <= t_end)].copy()
    
    if emg_df is None or len(emg_df) == 0:
        return None
    
    t_emg = emg_df["t"].to_numpy()
    synced = emg_df.copy()
    
    # Interpolate forces
    if fsr_df is not None and len(fsr_df) > 0:
        force_cols = [c for c in fsr_df.columns if c != "t" and c.endswith("_N")]
        for c in force_cols:
            if c in fsr_df.columns:
                synced[c] = interpolate_to_grid(t_emg, fsr_df["t"], fsr_df[c], 1.0)
    
    # Interpolate angles
    if angles_df is not None and len(angles_df) > 0:
        angle_cols = [c for c in angles_df.columns if c != "t" and ("_q1" in c or "_q2" in c)]
        for c in angle_cols:
            if c in angles_df.columns:
                synced[c] = interpolate_to_grid(
                    t_emg, angles_df["t"], angles_df[c], ANGLE_INTERPOLATION_MAX_GAP_S
                )

    # Interpolate all IMU channels to the EMG grid. Keep gaps as NaN instead
    # of bridging missing IMU packets with artificial values.
    if imu_df is not None and len(imu_df) > 0:
        imu_cols = find_imu_cols(imu_df)
        for c in imu_cols:
            synced[c] = interpolate_to_grid(t_emg, imu_df["t"], imu_df[c], IMU_MAX_GAP_S)
    
    # The preprocessed force file already removes the first-seconds baseline.
    # Avoid applying that mode again after event trimming, where the first
    # synced seconds may no longer be rest.
    force_cols = [c for c in synced.columns if c.endswith("_N")]
    if force_cols and FORCE_BASELINE_MODE != "first_seconds":
        synced = remove_force_baseline(synced, force_cols)
    
    # Keep a single canonical time column in the synced output.
    # EMG is the master sync grid, so this timestamp is the original EMG time.
    synced["timestamp"] = synced["t"]
    source_time_cols = ["t", "unix_ts", "host_t_s", "device_t_s", "time", "frame_capture_t"]
    synced.drop(columns=[c for c in source_time_cols if c in synced.columns], inplace=True)
    synced = synced[["timestamp"] + [c for c in synced.columns if c != "timestamp"]]
    # synced = retime_to_uniform(synced, "timestamp", fs)
    
    return synced


# ================= EVENT-BASED TRIMMING =================

def _contains(s: str, *keys) -> bool:
    """Check if string contains any of the keys."""
    s = (s or "").lower()
    return any(k in s for k in keys)


def load_events(events_csv: str) -> pd.DataFrame:
    """Load events CSV file."""
    df = pd.read_csv(events_csv)
    # Ensure numeric columns
    for c in ("t_start_host_s", "t_end_host_s", "duration_s", "t_host_s"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("grasp", "force", "event"):
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    return df


def detect_event_type(events_df: pd.DataFrame) -> str:
    """Detect if this is guided or unguided experiment."""
    cols = events_df.columns.tolist()
    
    # Guided has trial_index, grasp, force columns
    if all(c in cols for c in ["trial_index", "grasp", "force"]):
        return "guided"
    
    # Unguided has event column with session_start/session_stop
    if "event" in cols:
        return "unguided"
    
    return "unknown"


def get_time_window_guided(events_df: pd.DataFrame) -> Tuple[float, float]:
    """Get time window from guided experiment events.
    
    Uses the same logic as original script:
    - Find first "Open Hand" / "Relaxed" event to get t_start
    - Use last event's t_end as t_end
    """
    # Find relaxed/open events for start time
    relaxed_mask = events_df.apply(
        lambda r: (_contains(r.get("grasp", ""), "open", "rest", "release")
                   or _contains(r.get("force", "",), "relax", "rest")),
        axis=1
    )
    relaxed_rows = events_df.loc[relaxed_mask]
    
    if not relaxed_rows.empty:
        t_start = float(relaxed_rows.iloc[0]["t_end_host_s"])
    else:
        t_start = float(events_df["t_start_host_s"].min())
    
    # End time = last event's t_end
    t_end = float(events_df["t_end_host_s"].max())
    
    return t_start, t_end


def get_time_window_unguided(events_df: pd.DataFrame) -> Tuple[float, float]:
    """Get time window from unguided experiment events.
    
    Uses session_start and session_stop events.
    """
    # Find session_start and session_stop
    start_rows = events_df[events_df["event"].str.lower() == "session_start"]
    stop_rows = events_df[events_df["event"].str.lower() == "session_stop"]
    
    if start_rows.empty or stop_rows.empty:
        # Fallback: use min/max of t_host_s
        t_start = float(events_df["t_host_s"].min())
        t_end = float(events_df["t_host_s"].max())
        return t_start, t_end
    
    t_start = float(start_rows.iloc[0]["t_host_s"])
    t_end = float(stop_rows.iloc[0]["t_host_s"])
    
    return t_start, t_end


def get_time_window(events_csv: str) -> Tuple[Optional[float], Optional[float]]:
    """Get time window from events file.
    
    Returns (t_start, t_end) or (None, None) if no valid events found.
    """
    try:
        events_df = load_events(events_csv)
        event_type = detect_event_type(events_df)
        
        if event_type == "guided":
            return get_time_window_guided(events_df)
        elif event_type == "unguided":
            return get_time_window_unguided(events_df)
        else:
            print(f"    Unknown event format, skipping trim")
            return None, None
    except Exception as e:
        print(f"    Error reading events: {e}")
        return None, None


# ================= PLOTTING =================

def plot_synced(csv_path: str, save_path: str):
    """Plot synced data."""
    df = pd.read_csv(csv_path)
    
    if "timestamp" not in df.columns:
        print(f"    No timestamp column in {csv_path}")
        return
    
    t = df["timestamp"].to_numpy()
    t_rel = t - t[0]
    
    emg_cols = [c for c in df.columns if "EMG" in c.upper() or re.match(r"^ch\d+$", c, re.I)]
    angle_cols = [c for c in df.columns if "_q" in c]
    force_cols = [c for c in df.columns if c.endswith("_N")]
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # EMG
    if emg_cols:
        offset = 0
        spacing = 50
        for c in emg_cols[:8]:
            axes[0].plot(t_rel, df[c] + offset, lw=0.8, label=c)
            offset += spacing
        axes[0].set_ylabel("EMG (stacked)")
    else:
        axes[0].text(0.5, 0.5, "No EMG", transform=axes[0].transAxes, ha="center")
    
    # Forces
    if force_cols:
        for c in force_cols:
            axes[1].plot(t_rel, df[c], lw=1.5, label=c)
        axes[1].set_ylabel("Force (N)")
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, "No forces", transform=axes[1].transAxes, ha="center")
    
    # Angles
    if angle_cols:
        for c in angle_cols:
            axes[2].plot(t_rel, df[c], lw=1.2, label=c)
        axes[2].set_ylabel("Angle (deg)")
        axes[2].legend(ncol=3)
    else:
        axes[2].text(0.5, 0.5, "No angles", transform=axes[2].transAxes, ha="center")
    
    axes[2].set_xlabel("Time (s)")
    plt.suptitle("Synced EMG + Forces + Angles")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"    Plot saved: {save_path}")


def plot_preproc_comparison(preprocessed_dir: Path, raw_df: pd.DataFrame, proc_df: pd.DataFrame,
                            data_type: str, cols: List[str], time_col: str = None):
    """Plot raw vs preprocessed comparison."""
    print(f"  Plotting {data_type} comparison...")
    if not cols or raw_df.empty or proc_df.empty:
        return
    
    # Auto-detect time column if not provided - use same logic as preprocessing
    if time_col is None:
        if data_type == "Angles":
            if "t" not in raw_df.columns and ANGLE_TIME_COLUMN in raw_df.columns:
                raw_df = raw_df.copy()
                raw_df["t"] = time_values_seconds(raw_df, ANGLE_TIME_COLUMN)
            time_col = "t" if "t" in proc_df.columns else ANGLE_TIME_COLUMN
        # First try processed df (has "t" added during preprocessing)
        elif "t" in proc_df.columns:
            time_col = "t"
        # Then try same candidates as preprocessing (include unix_ts for EMG)
        else:
            time_col = _pick_time_column(proc_df, ["timestamp", "host_t_s", "unix_ts", "time", "t", "frame_capture_t"])
    
    # Check if found column exists in both dataframes
    if time_col is None or time_col not in raw_df.columns or time_col not in proc_df.columns:
        if data_type == "Angles":
            print(f"    Warning: Angle plot requires '{ANGLE_TIME_COLUMN}' in raw and preprocessed data")
            return
        # Fallback: try raw_df directly with preprocessing candidates
        time_col = _pick_time_column(raw_df, ["timestamp", "host_t_s", "unix_ts", "time", "t", "frame_capture_t", "device_t_s"])
        if time_col is None or time_col not in raw_df.columns:
            print(f"    Warning: Could not find time column for {data_type} plot")
            return
    
    # Ensure both dataframes have the time column
    if time_col not in proc_df.columns:
        # Try to find alternative in proc_df
        alt_col = _pick_time_column(proc_df, ["timestamp", "host_t_s", "unix_ts", "time", "t", "frame_capture_t"])
        if alt_col:
            time_col = alt_col
        else:
            print(f"    Warning: Time column '{time_col}' not in processed data for {data_type} plot")
            return
    
    t_raw = time_values_seconds(raw_df, time_col)
    t_proc = time_values_seconds(proc_df, time_col)
    raw_finite = np.isfinite(t_raw)
    proc_finite = np.isfinite(t_proc)
    if not raw_finite.any() or not proc_finite.any():
        print(f"    Warning: No valid time values for {data_type} plot")
        return
    t_raw_rel = t_raw - t_raw[raw_finite][0]
    t_proc_rel = t_proc - t_proc[proc_finite][0]
    
    n_axes = len(cols) * 2 if data_type == "EMG" else len(cols)
    fig, axes = plt.subplots(n_axes, 1, figsize=(14, 2*n_axes), sharex=True)
    if n_axes == 1:
        axes = [axes]
    
    for i, col in enumerate(cols):
        raw_y = pd.to_numeric(raw_df[col], errors="coerce").to_numpy()
        proc_y = pd.to_numeric(proc_df[col], errors="coerce").to_numpy()
        raw_x, raw_y = sorted_unique_xy(t_raw_rel, raw_y)
        proc_x, proc_y = sorted_unique_xy(t_proc_rel, proc_y)
        if data_type == "EMG":
            raw_ax = axes[2*i]
            proc_ax = axes[2*i + 1]
            raw_ax.plot(raw_x, raw_y, lw=0.8, alpha=0.7, label="Raw")
            proc_ax.plot(proc_x, proc_y, lw=0.9, alpha=0.9, label="Preprocessed")
            raw_ax.set_ylabel(f"{col} raw")
            proc_ax.set_ylabel(f"{col} proc")
            raw_ax.legend()
            proc_ax.legend()
            raw_ax.grid(True, alpha=0.3)
            proc_ax.grid(True, alpha=0.3)
        else:
            axes[i].plot(raw_x, raw_y, lw=0.8, alpha=0.6, label="Raw")
            axes[i].plot(proc_x, proc_y, lw=1.0, alpha=0.9, label="Preprocessed")
            axes[i].set_ylabel(col)
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
    
    axes[-1].set_xlabel("Time (s)")
    plt.suptitle(f"{data_type} Raw vs Preprocessed")
    plt.tight_layout()
    
    save_path = preprocessed_dir / f"{data_type}_preprocessed_plot.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"    {data_type} plot saved: {save_path}")


# ================= MAIN =================

def process_folder(folder: Path, root: Path, output_root: Path, use_events_window: bool, sync_enabled: bool):
    """Process a single folder: preprocess EMG, forces, angles, then sync."""
    if SYNC_ANGLES not in {"calibrated", "raw", "offline"}:
        raise ValueError('SYNC_ANGLES must be "calibrated", "raw", or "offline"')

    print(f"\n{'='*60}")
    print(f"Processing: {folder}")
    print('='*60)
    
    # Create a separate derived-data folder while preserving the raw hierarchy.
    rel_path = folder.relative_to(root)
    preprocessed_dir = output_root / rel_path
    preprocessed_dir.mkdir(parents=True, exist_ok=True)
    
    # Find raw files
    emg_files = find_files(folder, ["noraxon", "emg_2000", "emg_"])
    fsr_files = find_files(folder, ["fsr_"])
    imu_files = find_files(folder, ["imu_"])
    # Preprocess online and offline angle files; sync choice is controlled
    # separately by SYNC_ANGLES in the config section.
    angles_files = sorted(find_files(folder, ["retargeting_angles_", "offline_retargeted_angles_", "_calibrated"]))
    
    print(f"  Found EMG files: {[f.name for f in emg_files]}")
    print(f"  Found FSR files: {[f.name for f in fsr_files]}")
    print(f"  Found IMU files: {[f.name for f in imu_files]}")
    print(f"  Found Angles files: {[f.name for f in angles_files]}")
    
    # === EMG Preprocessing ===
    emg_df = None
    emg_cols = []
    if emg_files:
        for emg_file in emg_files:
            out_file = preprocessed_dir / f"{emg_file.stem}_preprocessed.csv"
            if SKIP_IF_OUTPUT_EXISTS and out_file.exists():
                print(f"  EMG: skipping (exists): {out_file.name}")
                emg_df = pd.read_csv(out_file)
                emg_cols = find_emg_cols(emg_df)
                if emg_cols:
                    raw_df = pd.read_csv(emg_file)
                    plot_preproc_comparison(preprocessed_dir, raw_df, emg_df, "EMG", emg_cols)
                break
            
            print(f"  Preprocessing EMG: {emg_file.name}")
            df = pd.read_csv(emg_file)
            emg_cols = find_emg_cols(df)
            
            if emg_cols:
                emg_fs = infer_sampling_frequency(df, ["unix_ts", "timestamp", "host_t_s", "t"], EMG_FS)
                print(f"    Using EMG sampling frequency: {emg_fs:.2f} Hz")
                df, rail_repairs = repair_emg_adc_saturation(df, emg_cols, emg_fs)
                if rail_repairs:
                    print(f"    Repaired {len(rail_repairs)} ADC-saturation cluster(s) before filtering")
                df = replace_emg_spikes_hampel(df, emg_cols, emg_fs)
                df = preprocess_emg(df, emg_cols, emg_fs)
                df.to_csv(out_file, index=False)
                print(f"    Saved: {out_file.name}")
                emg_df = df
                
                # Plot comparison
                raw_df = pd.read_csv(emg_file)
                plot_preproc_comparison(preprocessed_dir, raw_df, df, "EMG", emg_cols)
            else:
                print(f"    No EMG columns found!")
    
    # === Forces Preprocessing ===
    fsr_df = None
    force_cols = []
    if fsr_files:
        for fsr_file in fsr_files:
            out_file = preprocessed_dir / f"{fsr_file.stem}_FORCES_preprocessed.csv"
            if SKIP_IF_OUTPUT_EXISTS and out_file.exists():
                print(f"  Forces: skipping (exists): {out_file.name}")
                fsr_df = pd.read_csv(out_file)
                force_cols = [c for c in fsr_df.columns if c.endswith("_N")]
                break
            
            print(f"  Preprocessing Forces: {fsr_file.name}")
            df = pd.read_csv(fsr_file)
            tcol = _pick_time_column(df, ["host_t_s", "timestamp", "device_t_s", "t"])
            if tcol:
                df["t"] = pd.to_numeric(df[tcol], errors="coerce")
            force_cols = [c for c in df.columns if c.endswith("_N")]
            
            if force_cols:
                df = preprocess_forces(df, force_cols)
                df.to_csv(out_file, index=False)
                print(f"    Saved: {out_file.name}")
                fsr_df = df
                
                # Plot comparison
                raw_df = pd.read_csv(fsr_file)
                plot_preproc_comparison(preprocessed_dir, raw_df, df, "Forces", force_cols)

    # === IMU Preprocessing ===
    imu_df = None
    imu_cols = []
    if imu_files:
        for imu_file in imu_files:
            out_file = preprocessed_dir / f"{imu_file.stem}_IMU_preprocessed.csv"
            if SKIP_IF_OUTPUT_EXISTS and out_file.exists():
                print(f"  IMU: skipping (exists): {out_file.name}")
                imu_df = pd.read_csv(out_file)
                imu_cols = find_imu_cols(imu_df)
                break
            print(f"  Preprocessing IMU: {imu_file.name}")
            df = pd.read_csv(imu_file)
            tcol = _pick_time_column(df, ["unix_ts", "timestamp", "host_t_s", "device_t_s", "t"])
            if not tcol:
                raise ValueError(f"No supported IMU timestamp in {imu_file.name}")
            df["t"] = time_values_seconds(df, tcol)
            df = make_time_monotonic(df, "t", imu_file.name)
            imu_cols = find_imu_cols(df)
            if not imu_cols:
                raise ValueError(f"No IMU channels found in {imu_file.name}")
            df = preprocess_imu(df, imu_cols)
            df.to_csv(out_file, index=False)
            print(f"    Saved: {out_file.name}")
            imu_df = df
            raw_df = pd.read_csv(imu_file)
            raw_df["t"] = time_values_seconds(raw_df, tcol)
            plot_preproc_comparison(preprocessed_dir, raw_df, df, "IMU", imu_cols[:12])
    
    # === Angles Preprocessing ===
    angles_df = None
    angle_cols = []
    if angles_files:
        for angles_file in angles_files:
            angles_kind = angle_file_kind(angles_file)
            out_file = preprocessed_dir / f"{angles_file.stem}_ANGLES_preprocessed.csv"
            if SKIP_IF_OUTPUT_EXISTS and out_file.exists():
                print(f"  Angles: skipping (exists): {out_file.name}")
                angles_df = pd.read_csv(out_file)
                set_angle_time_seconds(angles_df)
                angles_df = make_time_monotonic(angles_df, "t", out_file.name)
                angle_cols = [c for c in angles_df.columns if "_q1" in c or "_q2" in c]
                if angle_cols and (PLOT_ANGLES == "all" or angles_kind == PLOT_ANGLES):
                    raw_df = pd.read_csv(angles_file)
                    plot_preproc_comparison(preprocessed_dir, raw_df, angles_df, "Angles", angle_cols[:6])
                continue
            
            print(f"  Preprocessing Angles: {angles_file.name}")
            df = pd.read_csv(angles_file)
            set_angle_time_seconds(df)
            df = make_time_monotonic(df, "t", angles_file.name)
            angle_cols = [c for c in df.columns if "_q1" in c or "_q2" in c]
            
            if angle_cols:
                df = preprocess_angles(df, angle_cols)
                df.to_csv(out_file, index=False)
                print(f"    Saved: {out_file.name}")
                angles_df = df
                
                # Plot comparison
                if PLOT_ANGLES == "all" or angles_kind == PLOT_ANGLES:
                    raw_df = pd.read_csv(angles_file)
                    plot_preproc_comparison(preprocessed_dir, raw_df, df, "Angles", angle_cols[:6])  # Limit to 6 for readability
    
    # === Sync ===
    if not sync_enabled:
        print("  Sync skipped by --skip-sync (native-rate comparison mode).")
        return
    if emg_df is not None:
        # Find preprocessed files for sync
        # EMG: _preprocessed.csv but NOT _FORCES_preprocessed or _ANGLES_preprocessed
        emg_pre = [
            f for f in preprocessed_dir.glob("*_preprocessed.csv")
            if "emg" in f.name.lower()
            and "_FORCES_" not in f.name
            and "_ANGLES_" not in f.name
            and "_IMU_" not in f.name
        ]
        fsr_pre = list(preprocessed_dir.glob("*_FORCES_preprocessed.csv"))
        imu_pre = list(preprocessed_dir.glob("*_IMU_preprocessed.csv"))
        
        # Find angle files in preprocessed folder
        angles_calibrated = sorted(preprocessed_dir.glob("*_calibrated_ANGLES_preprocessed.csv"))
        angles_offline = sorted(preprocessed_dir.glob("offline_retargeted_angles_*_ANGLES_preprocessed.csv"))
        curated_offline = sorted(preprocessed_dir.glob("offline_retargeted_angles_*_stereo_interpolation_patch_ANGLES_preprocessed.csv"))
        angles_raw = sorted(
            f for f in preprocessed_dir.glob("*_ANGLES_preprocessed.csv")
            if "_calibrated" not in f.name.lower()
            and not f.name.lower().startswith("offline_retargeted_angles_")
        )
        participant_id = next((part for part in preprocessed_dir.parts if part in CORRECTED_OFFLINE_PARTICIPANTS), None)
        corrected_offline = sorted(preprocessed_dir.glob("retargeting_angles_*_calibfixed_raw_ANGLES_preprocessed.csv"))
        if SYNC_ANGLES == "calibrated":
            selected_angles = angles_calibrated
        elif SYNC_ANGLES == "offline":
            selected_angles = (corrected_offline if participant_id and corrected_offline
                               else curated_offline if curated_offline else angles_offline)
        else:
            selected_angles = angles_raw
        
        # Event windows are opt-in. Defaulting to full recordings keeps this
        # preprocessing stage separate from any later inclusion/trimming rule.
        events_files = list(folder.glob("*events*.csv"))
        t_start, t_end = None, None
        if use_events_window and events_files:
            print(f"  Loading events: {events_files[0].name}")
            t_start, t_end = get_time_window(str(events_files[0]))
            if t_start is not None:
                print(f"    Time window: {t_start:.3f} → {t_end:.3f}")
            else:
                print(f"    No valid time window found, using full data")
        
        if emg_pre and (fsr_pre or selected_angles or imu_pre):
            emg_file = emg_pre[0]
            print(f"  EMG master file: {emg_file.name}")
            fsr_file = fsr_pre[0] if fsr_pre else None
            
            # Load EMG and FSR once
            emg_sync = pd.read_csv(emg_file)
            tcol = _pick_time_column(emg_sync, ["timestamp", "host_t_s", "unix_ts"])
            if tcol:
                emg_sync["t"] = pd.to_numeric(emg_sync[tcol], errors="coerce")
            print(f"    EMG master grid: {len(emg_sync)} rows")
            
            fsr_sync = pd.read_csv(fsr_file) if fsr_file else None
            if fsr_sync is not None:
                tcol = _pick_time_column(fsr_sync, ["host_t_s", "timestamp", "device_t_s"])
                if tcol:
                    fsr_sync["t"] = pd.to_numeric(fsr_sync[tcol], errors="coerce")

            imu_sync = pd.read_csv(imu_pre[0]) if imu_pre else None
            if imu_sync is not None:
                tcol = _pick_time_column(imu_sync, ["t", "unix_ts", "timestamp", "host_t_s"])
                if not tcol:
                    raise ValueError(f"No supported IMU timestamp in {imu_pre[0].name}")
                imu_sync["t"] = time_values_seconds(imu_sync, tcol)
                imu_sync = make_time_monotonic(imu_sync, "t", imu_pre[0].name)
            
            angles_sync = None
            if selected_angles:
                print(f"  Syncing with {SYNC_ANGLES} angles...")
                angles_file = selected_angles[0]
                print(f"    Angle file: {angles_file.name}")
                angles_sync = pd.read_csv(angles_file)
                set_angle_time_seconds(angles_sync)
                angles_sync = make_time_monotonic(angles_sync, "t", angles_file.name)
            else:
                print(f"  No {SYNC_ANGLES} preprocessed angles found; syncing without angles.")

            synced = sync_data(emg_sync.copy(), fsr_sync, angles_sync, imu_sync, EMG_FS, t_start, t_end)
            print(f"    Synced master grid: {len(synced)} rows")
            synced_file = preprocessed_dir / "synced_data.csv"
            synced.to_csv(synced_file, index=False)
            print(f"    Saved: {synced_file.name}")

            plot_path = preprocessed_dir / "synced_data_plot.png"
            plot_synced(str(synced_file), str(plot_path))
        else:
            print(f"  Skipping sync: need EMG + (FSR or Angles)")
    else:
        print(f"  Skipping: no EMG data")


def write_processing_manifest(output_root: Path, input_root: Path, use_events_window: bool, profile: str, sync_enabled: bool) -> None:
    """Write the exact publication-pipeline settings beside derived data."""
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "pipeline": Path(__file__).name,
        "input_root": str(input_root),
        "output_root": str(output_root),
        "event_window_trimming": bool(use_events_window),
        "preprocessing_profile": profile,
        "synced_csv_created": bool(sync_enabled),
        "sync_master_grid": "EMG unix_ts (nominal 2000 Hz)",
        "modalities": {
            "emg": {
                "bandpass_hz": [EMG_BP_LO, EMG_BP_HI], "bandpass_order": 2,
                "notch_hz": 60 if EMG_USE_60HZ_NOTCH else None, "notch_q": EMG_NOTCH_Q,
                "hampel": {"window_ms": EMG_HAMPEL_WINDOW_MS, "n_sigmas": EMG_HAMPEL_N_SIGMAS,
                           "max_spike_ms": EMG_HAMPEL_MAX_SPIKE_MS},
            },
            "force": {
                "baseline_mode": FORCE_BASELINE_MODE, "baseline_first_seconds": FORCE_BASELINE_FIRST_SECONDS,
                "median_ms": FORCE_MEDIAN_MS, "lowpass_hz": FORCE_LOWPASS_HZ,
                "mean_ms": FORCE_MEAN_MS,
            },
            "angles": {"median_ms": ANGLES_MEDIAN_MS, "lowpass_hz": ANGLES_LOWPASS_HZ,
                       "mean_ms": ANGLES_MEAN_MS, "source": SYNC_ANGLES},
            "imu": {"median_ms": IMU_MEDIAN_MS, "accel_gyro_lowpass_hz": IMU_ACCEL_GYRO_LOWPASS_HZ,
                    "mag_lowpass_hz": IMU_MAG_LOWPASS_HZ, "offset_removed": False,
                    "interpolation_max_gap_s": IMU_MAX_GAP_S},
        },
        "lag_or_resync_applied": False,
    }
    (output_root / "processing_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    global SKIP_IF_OUTPUT_EXISTS, FORCE_BASELINE_MODE, FORCE_BASELINE_FIRST_SECONDS

    parser = argparse.ArgumentParser(description='Preprocess and sync all data in a folder.')
    parser.add_argument('folder', type=str, nargs='?', default=None,
                        help='Root folder containing session subfolders')
    parser.add_argument('--output-root', type=str, default=None,
                        help='Separate root for derived data (default: sibling preprocessed_publication directory)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Allow replacement of existing derived outputs; raw inputs are never changed')
    parser.add_argument('--use-events-window', action='store_true',
                        help='Restrict synced output to event timestamps (off by default)')
    parser.add_argument('--skip-sync', action='store_true',
                        help='Save only native-rate modality outputs; useful for filter comparisons')
    parser.add_argument('--profile', choices=['publication', 'legacy'], default='publication',
                        help='publication uses moderate force/angle filtering; legacy reproduces former effective settings')
    parser.add_argument('--force-baseline', choices=["first_seconds", "percentile"], default=FORCE_BASELINE_MODE,
                        help='Force baseline method. first_seconds uses the first 10 seconds of each force file.')
    parser.add_argument('--force-baseline-seconds', type=float, default=FORCE_BASELINE_FIRST_SECONDS,
                        help='Seconds from the start of each force file to use when --force-baseline first_seconds.')
    parser.add_argument('--take', action='append', default=[],
                        help='Relative take folder to process; may be specified more than once.')
    args = parser.parse_args()
    
    configure_preprocessing_profile(args.profile)
    if args.overwrite:
        SKIP_IF_OUTPUT_EXISTS = False
    FORCE_BASELINE_MODE = args.force_baseline
    FORCE_BASELINE_FIRST_SECONDS = args.force_baseline_seconds
    
    if args.folder is None:
        parser.error("folder is required; provide the root of the raw release")
    
    root = Path(args.folder)
    if not root.exists():
        print(f"Error: folder does not exist: {root}")
        return
    
    root = root.resolve()
    output_root = Path(args.output_root).resolve() if args.output_root else root.parent / preprocessed_folder
    if output_root == root or root in output_root.parents:
        raise ValueError("--output-root must be separate from, and outside, the raw input root")
    write_processing_manifest(output_root, root, args.use_events_window, args.profile, not args.skip_sync)

    print(f"Scanning: {root}")
    print(f"Derived-data output: {output_root}")
    print(f"Event-window trimming: {'enabled' if args.use_events_window else 'disabled'}")
    print(f"Preprocessing profile: {args.profile}")
    folders = find_data_folders(root)
    if args.take:
        requested = {Path(item).as_posix() for item in args.take}
        folders = [folder for folder in folders if folder.relative_to(root).as_posix() in requested]
        missing = requested - {folder.relative_to(root).as_posix() for folder in folders}
        if missing:
            raise FileNotFoundError(f"Requested take(s) not found below {root}: {sorted(missing)}")
    
    if not folders:
        print("No folders with data found!")
        return
    
    print(f"Found {len(folders)} folder(s) with data:")
    for f in folders:
        print(f"  - {f}")
    
    # Process each folder
    for folder in folders:
        try:
            process_folder(folder, root, output_root, args.use_events_window, not args.skip_sync)
        except Exception as e:
            print(f"Error processing {folder}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print("Done!")
    print('='*60)


if __name__ == '__main__':
    main()
