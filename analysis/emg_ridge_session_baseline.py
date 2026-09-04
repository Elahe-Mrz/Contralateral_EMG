#!/usr/bin/env python3
"""Participant-specific causal sEMG ridge baseline with gesture-stratified take holdout.

The baseline answers whether the released timestamp-aligned sEMG contains
information about continuous retargeted angles and fingertip forces. It uses
only causal sEMG features from each prediction time and never splits windows
randomly: each fold holds out one complete take of each gesture for one
participant.

With ``--include-imu``, the same model adds causal summary features from the
released, timestamp-synchronized IMU channels. The participant-specific split
and all target, window, and evaluation settings remain unchanged.

Features per available EMG channel (default 500-ms causal window): RMS, mean
absolute value, waveform length, and zero crossings.  The multi-output ridge
targets six robot-joint angles and three non-negative fingertip forces in N.
Each participant uses the EMG channels present in all of their takes (P025,
for example, is evaluated with its seven recorded EMG channels). Documented
participant-level exclusions can be supplied with ``--exclude-emg-channel``;
by default P007 EMG7 is excluded from every P007 take because it is unusable
in two Pinch recordings. The five
ordered takes of Pinch, Power, and Spherical form five gesture-balanced folds:
each test fold contains one take from every gesture, and training uses the
other twelve takes from that participant.

Example
-------
python emg_ridge_session_baseline.py \
  --data-root Preprocessed \
  --output baseline_emg_ridge_session_20260826
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


EMG_COLUMNS = tuple(f"EMG {index}" for index in range(1, 9))
ANGLE_COLUMNS = ("index_q1", "middle_q1", "ring_q1", "pinky_q1", "thumb_q2", "thumb_q1")
FORCE_COLUMNS = ("middle_N", "index_N", "thumb_N")
@dataclass(frozen=True)
class TakeInfo:
    participant: str
    gesture: str
    take: str
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-ms", type=float, default=500.0)
    parser.add_argument("--stride-ms", type=float, default=50.0,
                        help="Prediction spacing; windows remain causal at every selected endpoint.")
    parser.add_argument("--alpha", type=float, default=1.0, help="Ridge penalty after feature standardization.")
    parser.add_argument("--force-transform", choices=("log1p", "none"), default="none",
                        help="Force target transform. The default direct-N targets avoid log inverse extrapolation.")
    parser.add_argument("--include-imu", action="store_true",
                        help="Append causal IMU mean and standard-deviation features from synced_data.csv.")
    parser.add_argument("--participant", action="append", default=[],
                        help="Restrict to participant ID; repeatable, useful for smoke tests.")
    parser.add_argument("--take", action="append", default=[],
                        help="Restrict to relative take folder; repeatable, useful for smoke tests.")
    parser.add_argument("--exclude-emg-channel", action="append", default=["P007:EMG 7"],
                        metavar="PARTICIPANT:CHANNEL",
                        help=("Exclude an EMG channel from every take for a participant; repeatable. "
                              "Default: P007:EMG 7."))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def discover_takes(root: Path, participants: set[str], selected_takes: set[str]) -> list[TakeInfo]:
    takes: list[TakeInfo] = []
    for synced in sorted(root.rglob("synced_data.csv")):
        relative = synced.relative_to(root)
        if len(relative.parts) < 4:
            continue
        participant, gesture, take = relative.parts[:3]
        if participants and participant not in participants:
            continue
        if selected_takes and relative.parent.as_posix() not in selected_takes:
            continue
        takes.append(TakeInfo(participant, gesture, relative.parent.as_posix(), synced))
    if not takes:
        raise FileNotFoundError(f"No synchronized takes found below {root}")
    return takes


def prefix_sum(values: np.ndarray) -> np.ndarray:
    return np.vstack((np.zeros((1, values.shape[1]), dtype=np.float64), np.cumsum(values, axis=0, dtype=np.float64)))


def causal_features(emg: np.ndarray, window: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    """Return causal features and endpoint row indices without crossing takes."""
    if len(emg) < window:
        return np.empty((0, emg.shape[1] * 4)), np.empty(0, dtype=int)
    endpoints = np.arange(window - 1, len(emg), stride, dtype=int)
    starts = endpoints - window + 1
    abs_values = np.abs(emg)
    squared_prefix = prefix_sum(emg * emg)
    absolute_prefix = prefix_sum(abs_values)
    rms = np.sqrt((squared_prefix[endpoints + 1] - squared_prefix[starts]) / window)
    mav = (absolute_prefix[endpoints + 1] - absolute_prefix[starts]) / window
    differences = np.abs(np.diff(emg, axis=0))
    waveform_length = prefix_sum(differences)[endpoints] - prefix_sum(differences)[starts]
    crossings = ((emg[:-1] * emg[1:]) < 0).astype(np.float64)
    zero_crossings = prefix_sum(crossings)[endpoints] - prefix_sum(crossings)[starts]
    return np.hstack((rms, mav, waveform_length, zero_crossings)), endpoints


def causal_imu_features(imu: np.ndarray, window: int, endpoints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Causal per-channel IMU mean/SD and a finite-window mask."""
    if not len(endpoints):
        return np.empty((0, imu.shape[1] * 2)), np.empty(0, dtype=bool)
    starts = endpoints - window + 1
    finite = np.isfinite(imu)
    clean = np.where(finite, imu, 0.0)
    sums = prefix_sum(clean)
    squared_sums = prefix_sum(clean * clean)
    mean = (sums[endpoints + 1] - sums[starts]) / window
    variance = (squared_sums[endpoints + 1] - squared_sums[starts]) / window - mean * mean
    invalid_counts = prefix_sum((~finite).astype(np.float64))
    valid = (invalid_counts[endpoints + 1] - invalid_counts[starts]).sum(axis=1) == 0
    return np.hstack((mean, np.sqrt(np.maximum(variance, 0.0)))), valid


def target_transform(force: np.ndarray, kind: str) -> np.ndarray:
    return np.log1p(np.clip(force, 0, None)) if kind == "log1p" else force


def inverse_force(force: np.ndarray, kind: str) -> np.ndarray:
    """Return physically valid force predictions in N.

    The baseline's default direct-force targets do not require an inverse
    transform.  The non-negativity constraint is applied only at evaluation,
    so models cannot receive credit or penalty for impossible negative force.
    """
    values = np.expm1(force) if kind == "log1p" else force
    return np.maximum(values, 0.0)


def load_take(info: TakeInfo, window_ms: float, stride_ms: float, force_transform: str,
              include_imu: bool, excluded_emg: set[str]) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], tuple[str, ...], dict[str, object]]:
    available = set(pd.read_csv(info.path, nrows=0).columns)
    emg_columns = tuple(column for column in EMG_COLUMNS if column in available and column not in excluded_emg)
    imu_columns = tuple(sorted(column for column in available if column.startswith("IMU "))) if include_imu else ()
    if not emg_columns:
        raise ValueError("no EMG columns found")
    if include_imu and not imu_columns:
        raise ValueError("--include-imu requested but no IMU columns found")
    required = {"timestamp", *ANGLE_COLUMNS, *FORCE_COLUMNS}
    missing = sorted(required - available)
    if missing:
        raise ValueError(f"required columns missing: {missing}")
    columns = ["timestamp", *emg_columns, *imu_columns, *ANGLE_COLUMNS, *FORCE_COLUMNS]
    frame = pd.read_csv(info.path, usecols=columns)
    time = pd.to_numeric(frame["timestamp"], errors="coerce").to_numpy(np.float64)
    finite_time = time[np.isfinite(time)]
    if len(finite_time) < 2:
        raise ValueError("fewer than two finite timestamps")
    dt = float(np.median(np.diff(finite_time)))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("invalid sampling interval")
    fs_hz = 1.0 / dt
    window = max(2, int(round(window_ms * fs_hz / 1000.0)))
    stride = max(1, int(round(stride_ms * fs_hz / 1000.0)))
    emg = frame.loc[:, emg_columns].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
    features, endpoints = causal_features(emg, window, stride)
    imu_valid = np.ones(len(endpoints), dtype=bool)
    if include_imu:
        imu = frame.loc[:, imu_columns].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
        imu_features, imu_valid = causal_imu_features(imu, window, endpoints)
        features = np.hstack((features, imu_features))
    targets = frame.loc[:, [*ANGLE_COLUMNS, *FORCE_COLUMNS]].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)[endpoints]
    targets[:, len(ANGLE_COLUMNS):] = target_transform(targets[:, len(ANGLE_COLUMNS):], force_transform)
    valid = imu_valid & np.isfinite(features).all(axis=1) & np.isfinite(targets).all(axis=1)
    inventory = {
        "participant": info.participant, "gesture": info.gesture, "take_folder": info.take,
        "file": str(info.path), "emg_channels": ";".join(emg_columns), "emg_channel_count": len(emg_columns),
        "imu_channels": ";".join(imu_columns), "imu_channel_count": len(imu_columns),
        "sampling_hz": fs_hz, "window_samples": window,
        "stride_samples": stride, "candidate_windows": len(endpoints), "retained_windows": int(valid.sum()),
        "excluded_windows": int((~valid).sum()),
    }
    return features[valid], targets[valid], emg_columns, imu_columns, inventory


def parse_emg_exclusions(specifications: list[str]) -> dict[str, set[str]]:
    """Parse repeatable ``PARTICIPANT:CHANNEL`` CLI values."""
    exclusions: dict[str, set[str]] = defaultdict(set)
    for specification in specifications:
        if ":" not in specification:
            raise ValueError(
                f"Invalid --exclude-emg-channel value {specification!r}; use PARTICIPANT:CHANNEL"
            )
        participant, channel = (part.strip() for part in specification.split(":", 1))
        if not participant or channel not in EMG_COLUMNS:
            raise ValueError(
                f"Invalid --exclude-emg-channel value {specification!r}; "
                f"channel must be one of {list(EMG_COLUMNS)}"
            )
        exclusions[participant].add(channel)
    return dict(exclusions)


def metric_row(participant: str, fold: int, output: str, domain: str,
               actual: np.ndarray, predicted: np.ndarray,
               held_out_takes: str, n_train_windows: int,
               n_test_windows: int) -> dict[str, object]:
    return {
        "participant": participant,
        "held_out_fold": fold,
        "held_out_takes": held_out_takes,
        "n_train_windows": n_train_windows,
        "n_test_windows": n_test_windows,
        "output": output,
        "domain": domain,
        "mae": float(np.mean(np.abs(predicted - actual))),
        "pearson_r": pearson(actual, predicted),
    }


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def main() -> None:
    args = parse_args()
    if args.window_ms <= 0 or args.stride_ms <= 0 or args.alpha < 0:
        raise ValueError("window/stride must be positive and alpha non-negative")
    root, output = args.data_root.resolve(), args.output.resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {output}; use --overwrite to replace it")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    emg_exclusions = parse_emg_exclusions(args.exclude_emg_channel)
    takes = discover_takes(root, set(args.participant), set(args.take))
    data: dict[str, dict[str, list[tuple[str, np.ndarray, np.ndarray]]]] = defaultdict(lambda: defaultdict(list))
    participant_emg_channels: dict[str, tuple[str, ...]] = {}
    participant_imu_channels: dict[str, tuple[str, ...]] = {}
    inventory: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    for info in takes:
        try:
            print(f"Loading {info.take}", flush=True)
            features, targets, emg_columns, imu_columns, row = load_take(
                info, args.window_ms, args.stride_ms, args.force_transform,
                args.include_imu, emg_exclusions.get(info.participant, set())
            )
            established = participant_emg_channels.setdefault(info.participant, emg_columns)
            if emg_columns != established:
                raise ValueError(
                    f"inconsistent EMG channel layout for {info.participant}: "
                    f"expected {established}, found {emg_columns}"
                )
            if args.include_imu:
                established_imu = participant_imu_channels.setdefault(info.participant, imu_columns)
                if imu_columns != established_imu:
                    raise ValueError(
                        f"inconsistent IMU channel layout for {info.participant}: "
                        f"expected {established_imu}, found {imu_columns}"
                    )
            inventory.append(row)
            print(f"  retained {len(features)} windows", flush=True)
            if len(features):
                data[info.participant][info.gesture].append((info.take, features, targets))
        except Exception as exc:
            failures.append({"participant": info.participant, "gesture": info.gesture, "take_folder": info.take, "file": str(info.path), "reason": str(exc)})
    with (output / "take_window_inventory.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(inventory[0]) if inventory else ["participant"])
        writer.writeheader(); writer.writerows(inventory)
    with (output / "load_failures.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["participant", "gesture", "take_folder", "file", "reason"])
        writer.writeheader(); writer.writerows(failures)

    fold_rows: list[dict[str, object]] = []
    held_out_take_rows: list[dict[str, object]] = []
    eligibility: list[dict[str, object]] = []
    target_names = [*ANGLE_COLUMNS, *FORCE_COLUMNS]
    required_gestures = ("Pinch", "Power", "Spherical")
    for participant, gesture_takes in sorted(data.items()):
        ordered = {gesture: sorted(gesture_takes.get(gesture, []), key=lambda item: item[0]) for gesture in required_gestures}
        gesture_counts = {gesture: len(values) for gesture, values in ordered.items()}
        fold_count = min(gesture_counts.values()) if gesture_counts else 0
        eligible = fold_count >= 2
        eligibility.append({"participant": participant, "pinch_take_count": gesture_counts["Pinch"], "power_take_count": gesture_counts["Power"], "spherical_take_count": gesture_counts["Spherical"], "available_stratified_folds": fold_count, "eligible_for_leave_one_take_per_gesture_out": eligible})
        if not eligible:
            continue
        for fold_index in range(fold_count):
            test_values = [ordered[gesture][fold_index] for gesture in required_gestures]
            train_values = [value for gesture in required_gestures for index, value in enumerate(ordered[gesture]) if index != fold_index]
            x_train = np.vstack([value[1] for value in train_values]); y_train = np.vstack([value[2] for value in train_values])
            x_test = np.vstack([value[1] for value in test_values]); y_test = np.vstack([value[2] for value in test_values])
            model = make_pipeline(StandardScaler(), Ridge(alpha=args.alpha))
            model.fit(x_train, y_train)
            prediction = model.predict(x_test)
            held_out_takes = ";".join(value[0] for value in test_values)
            take_start = 0
            for take_name, take_features, _ in test_values:
                take_end = take_start + len(take_features)
                gesture = take_name.split("/", 2)[1]
                for index, name in enumerate(target_names):
                    domain = "angle_deg" if index < len(ANGLE_COLUMNS) else "force_N"
                    actual = y_test[take_start:take_end, index]
                    predicted = prediction[take_start:take_end, index]
                    if domain == "force_N":
                        actual, predicted = inverse_force(actual, args.force_transform), inverse_force(predicted, args.force_transform)
                    held_out_take_rows.append({
                        **metric_row(participant, fold_index + 1, name, domain, actual, predicted,
                                     take_name, len(x_train), len(take_features)),
                        "gesture": gesture,
                        "held_out_take": take_name,
                    })
                take_start = take_end
            for index, name in enumerate(target_names):
                domain = "angle_deg" if index < len(ANGLE_COLUMNS) else "force_N"
                actual = y_test[:, index]
                predicted = prediction[:, index]
                if domain == "force_N":
                    actual, predicted = inverse_force(actual, args.force_transform), inverse_force(predicted, args.force_transform)
                fold_rows.append(metric_row(participant, fold_index + 1, name, domain, actual, predicted,
                                            held_out_takes, len(x_train), len(x_test)))
    with (output / "take_eligibility.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["participant", "pinch_take_count", "power_take_count", "spherical_take_count", "available_stratified_folds", "eligible_for_leave_one_take_per_gesture_out"])
        writer.writeheader(); writer.writerows(eligibility)
    with (output / "fold_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["participant", "held_out_fold", "held_out_takes", "n_train_windows", "n_test_windows", "output", "domain", "mae", "pearson_r"])
        writer.writeheader(); writer.writerows(fold_rows)
    with (output / "held_out_take_metrics.csv").open("w", newline="") as handle:
        fieldnames = ["participant", "held_out_fold", "gesture", "held_out_take", "n_train_windows", "n_test_windows", "output", "domain", "mae", "pearson_r"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in held_out_take_rows:
            writer.writerow({field: row[field] for field in fieldnames})
    metrics = pd.DataFrame(fold_rows)
    if len(metrics):
        summary = metrics.groupby(["domain", "output"], as_index=False).agg(folds=("mae", "count"), mae_mean=("mae", "mean"), mae_median=("mae", "median"), pearson_r_mean=("pearson_r", "mean"), pearson_r_median=("pearson_r", "median"))
        summary.to_csv(output / "summary_by_output.csv", index=False)
        participant_summary = metrics.groupby(["participant", "domain", "output"], as_index=False).agg(
            folds=("mae", "count"), mae_mean=("mae", "mean"), mae_median=("mae", "median"),
            pearson_r_mean=("pearson_r", "mean"), pearson_r_median=("pearson_r", "median"),
        )
        participant_summary.to_csv(output / "participant_summary_by_output.csv", index=False)
        participant_domain = metrics.groupby(["participant", "domain"], as_index=False).agg(
            outputs=("output", "nunique"), fold_output_scores=("mae", "count"),
            mae_mean=("mae", "mean"), mae_median=("mae", "median"),
            pearson_r_mean=("pearson_r", "mean"), pearson_r_median=("pearson_r", "median"),
        )
        participant_domain.to_csv(output / "participant_domain_summary.csv", index=False)
    take_metrics = pd.DataFrame(held_out_take_rows)
    if len(take_metrics):
        participant_take_summary = take_metrics.groupby(
            ["participant", "domain", "output"], as_index=False
        ).agg(
            held_out_takes=("mae", "count"),
            mae_mean=("mae", "mean"), mae_median=("mae", "median"),
            pearson_r_mean=("pearson_r", "mean"), pearson_r_median=("pearson_r", "median"),
        )
        participant_take_summary.to_csv(output / "participant_held_out_take_summary_by_output.csv", index=False)
        participant_equal_summary = participant_take_summary.groupby(
            ["domain", "output"], as_index=False
        ).agg(
            participants=("participant", "nunique"),
            held_out_takes_per_participant=("held_out_takes", "median"),
            mae_mean=("mae_mean", "mean"), mae_median=("mae_median", "median"),
            pearson_r_mean=("pearson_r_mean", "mean"), pearson_r_median=("pearson_r_median", "median"),
        )
        participant_equal_summary.to_csv(output / "held_out_take_summary_by_output_equal_participant.csv", index=False)
    config = {"data_root": str(root), "window_ms": args.window_ms, "stride_ms": args.stride_ms, "alpha": args.alpha, "force_transform": args.force_transform, "force_prediction_constraint": "predictions clipped to >= 0 N for evaluation", "emg_feature_order": ["RMS", "MAV", "waveform_length", "zero_crossings"], "include_imu": args.include_imu, "imu_feature_order": ["mean", "standard_deviation"] if args.include_imu else [], "emg_channel_policy": "participant-specific channels present consistently across that participant's takes after documented exclusions", "excluded_emg_channels_by_participant": {participant: sorted(channels) for participant, channels in sorted(emg_exclusions.items())}, "emg_channels_by_participant": {participant: list(channels) for participant, channels in sorted(participant_emg_channels.items())}, "imu_channel_policy": "participant-specific channels present consistently across that participant's takes" if args.include_imu else "not used", "imu_channels_by_participant": {participant: list(channels) for participant, channels in sorted(participant_imu_channels.items())}, "evaluation_protocol": "participant-specific leave-one-take-per-gesture-out; each fold holds one Pinch, one Power, and one Spherical take out together", "primary_metric_scope": "concatenated three-take held-out fold; reported in fold_metrics.csv and summary_by_output.csv", "secondary_metric_scope": "each held-out take evaluated separately, then averaged equally within participant; reported in held_out_take_metrics.csv and held_out_take_summary_by_output_equal_participant.csv", "angle_targets": list(ANGLE_COLUMNS), "force_targets": list(FORCE_COLUMNS), "take_count_discovered": len(takes), "takes_loaded": len(inventory), "load_failures": len(failures), "participants_with_evaluable_take_folds": int(sum(row["eligible_for_leave_one_take_per_gesture_out"] for row in eligibility)), "fold_count": len({(row["participant"], row["held_out_fold"]) for row in fold_rows})}
    (output / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
