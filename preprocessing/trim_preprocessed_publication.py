#!/usr/bin/env python3
"""Create a separate, timestamp-synchronized trimmed publication derivative.

Inputs are never changed. The output contains canonical processed EMG, force,
IMU, offline-retargeted angles, and synced CSV for each take, plus detailed
per-take and per-file provenance reports. The data-release curation excludes
the non-canonical online-angle provenance files after trimming.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path


TARGET_SECONDS = 120.0
LONG_OVERLAP_SECONDS = 135.0
DEFAULT_INITIAL_REST_SECONDS = 15.0


@dataclass(frozen=True)
class ExceptionRule:
    kind: str  # "last" or "start_offset"
    value_s: float
    note: str


# Confirmed manual exceptions from the recording notes.
EXCEPTIONS: dict[str, ExceptionRule] = {
    # Corrected P007 angles first become observed at 1780338385.686262 s;
    # retaining the final observed span avoids endpoint extrapolation across
    # the initial tracking failure rather than padding it to 120 s.
    "P007/Power/20260601_145645_power_grasp_unguided_take11": ExceptionRule("last", 117.34987473487854, "Corrected P007 angles begin 2.650 s after the nominal 120-s window; retain final fully observed angle span"),
    "P009/Power/20260602_114342_power_grasp_unguided_take08": ExceptionRule("start_offset", 30.0, "Power session 3: first 30 s are rest"),
    "P013/Power/20260604_141611_power_grasp_unguided_take08": ExceptionRule("last", TARGET_SECONDS, "Power session 2: retain last 120 s"),
    "P019/Power/20260617_183201_power_grasp_unguided_take06": ExceptionRule("last", TARGET_SECONDS, "Power session 1: first repetition omitted; retain last 120 s"),
    "P019/Power/20260617_183201_power_grasp_unguided_take07": ExceptionRule("last", TARGET_SECONDS, "Power session 2: first repetition omitted; retain last 120 s"),
    "P021/Power/20260619_113916_power_grasp_unguided_take03": ExceptionRule("last", TARGET_SECONDS, "Power session 1: retain last 120 s"),
    "P025/Power/20260702_083734_power_grasp_unguided_take10": ExceptionRule("start_offset", 60.0, "Power session 2: first 60 s are rest"),
}

TIME_COLUMNS = ("timestamp", "t", "unix_ts", "host_t_s", "device_t_s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--apply", action="store_true", help="write the new trimmed derivative; otherwise validate and report only")
    parser.add_argument("--take", action="append", default=[], help="Relative take folder to re-trim; may be specified more than once.")
    parser.add_argument("--participant", action="append", default=[],
                        help="Participant ID to re-trim; may be specified more than once.")
    parser.add_argument("--overwrite-existing", action="store_true", help="Allow replacement of selected existing output files; requires --take.")
    parser.add_argument("--report-suffix", default="selected_retrim", help="Suffix for selected-take report files.")
    parser.add_argument("--update-master-manifest", action="store_true", help="Merge selected applied take rows into the dataset-wide trim manifests.")
    parser.add_argument("--exclude-online-output", action="store_true",
                        help="Use online-angle coverage when selecting the shared window but do not copy that non-canonical derivative to the output.")
    return parser.parse_args()


def as_number(value: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def final_csv_row(path: Path) -> list[str] | None:
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - 1024 * 1024))
        lines = [line for line in handle.read().splitlines() if line.strip()]
    if not lines:
        return None
    return next(csv.reader([lines[-1].decode("utf-8", errors="replace")]), None)


def merge_master_report(path: Path, rows: list[dict[str, object]], key: str) -> None:
    """Replace selected rows by key while preserving all unrelated master rows."""
    existing: list[dict[str, str]] = []
    if path.exists():
        with path.open(newline="") as handle:
            existing = list(csv.DictReader(handle))
    replacements = {str(row[key]): row for row in rows}
    merged = [row for row in existing if str(row.get(key, "")) not in replacements]
    merged.extend(replacements.values())
    merged.sort(key=lambda row: (str(row.get("take_folder", "")), str(row.get("kind", ""))))
    fieldnames = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged)


def csv_bounds(path: Path) -> tuple[str, float, float]:
    with path.open("r", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
        first = next(reader, [])
    time_col = next((name for name in TIME_COLUMNS if name in header), None)
    if not time_col or not first:
        raise ValueError(f"No usable timestamp in {path}")
    index = header.index(time_col)
    last = final_csv_row(path)
    if not last or len(first) <= index or len(last) <= index:
        raise ValueError(f"Cannot read first/last timestamp in {path}")
    start, end = as_number(first[index]), as_number(last[index])
    if not (math.isfinite(start) and math.isfinite(end) and end >= start):
        raise ValueError(f"Invalid timestamp bounds in {path}: {start}, {end}")
    return time_col, start, end


def canonical_file(folder: Path, kind: str) -> Path:
    if kind == "emg":
        candidates = sorted(folder.glob("emg_*_preprocessed.csv"))
    elif kind == "force":
        candidates = sorted(folder.glob("FSR_*_FORCES_preprocessed.csv"))
    elif kind == "imu":
        candidates = sorted(folder.glob("imu_*_IMU_preprocessed.csv"))
    elif kind == "offline_angles":
        if "P007" in folder.parts:
            candidates = sorted(folder.glob("retargeting_angles_*_calibfixed_raw_ANGLES_preprocessed.csv"))
        else:
            curated = sorted(folder.glob("offline_retargeted_angles_*_stereo_interpolation_patch_ANGLES_preprocessed.csv"))
            candidates = curated if curated else [p for p in sorted(folder.glob("offline_retargeted_angles_*_ANGLES_preprocessed.csv")) if "calibfixed" not in p.name.lower() and "_fixed" not in p.name.lower()]
    elif kind == "online_angles":
        candidates = [p for p in sorted(folder.glob("retargeting_angles_*_raw_ANGLES_preprocessed.csv")) if all(token not in p.name.lower() for token in ("offline", "calibfixed", "calibrated", "_fixed"))]
    elif kind == "synced":
        candidates = [folder / "synced_data.csv"]
    else:
        raise ValueError(kind)
    if len(candidates) != 1 or not candidates[0].is_file():
        raise ValueError(f"Expected exactly one canonical {kind} file in {folder}; found {len(candidates)}")
    return candidates[0]


def select_window(take_key: str, start: float, end: float) -> tuple[float, float, str, str]:
    duration = end - start
    exception = EXCEPTIONS.get(take_key)
    if duration < TARGET_SECONDS:
        return start, end, "short_untrimmed", f"Shared overlap {duration:.3f} s is shorter than {TARGET_SECONDS:g} s"
    if exception:
        if exception.kind == "last":
            return end - exception.value_s, end, f"manual_last_{exception.value_s:g}s", exception.note
        if exception.kind == "start_offset":
            selected_start = start + exception.value_s
            selected_end = selected_start + TARGET_SECONDS
            if selected_end > end:
                raise ValueError(f"Manual start-offset window exceeds shared overlap for {take_key}")
            return selected_start, selected_end, "manual_start_offset", exception.note
        raise ValueError(f"Unknown exception rule {exception.kind}")
    if duration >= LONG_OVERLAP_SECONDS:
        return start + DEFAULT_INITIAL_REST_SECONDS, start + DEFAULT_INITIAL_REST_SECONDS + TARGET_SECONDS, "default_remove_first_15_then_tail", "Shared overlap >=135 s"
    return end - TARGET_SECONDS, end, "default_last_120", "Shared overlap is 120--135 s"


def trim_csv(source: Path, destination: Path, time_column: str, start: float, end: float, stable_sort: bool = False) -> dict[str, object]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    input_rows = output_rows = 0
    first_out = last_out = math.nan
    try:
        with source.open("r", newline="") as reader_handle, os.fdopen(fd, "w", newline="") as writer_handle:
            reader = csv.DictReader(reader_handle)
            if not reader.fieldnames or time_column not in reader.fieldnames:
                raise ValueError(f"Missing {time_column} in {source}")
            writer = csv.DictWriter(writer_handle, fieldnames=reader.fieldnames, extrasaction="raise")
            writer.writeheader()
            rows = list(reader) if stable_sort else reader
            if stable_sort:
                rows = [row for _, row in sorted(enumerate(rows), key=lambda item: (not math.isfinite(as_number(item[1].get(time_column, ""))), as_number(item[1].get(time_column, "")), item[0]))]
            for row in rows:
                input_rows += 1
                value = as_number(row.get(time_column, ""))
                if math.isfinite(value) and start <= value <= end:
                    writer.writerow(row)
                    output_rows += 1
                    first_out = value if not math.isfinite(first_out) else first_out
                    last_out = value
        if output_rows == 0:
            raise ValueError(f"No rows retained from {source}")
        os.replace(temp_name, destination)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
    return {"input_rows": input_rows, "output_rows": output_rows, "output_start_s": first_out, "output_end_s": last_out, "output_duration_s": last_out - first_out}


def main() -> int:
    args = parse_args()
    input_root, output_root = args.input_root.resolve(), args.output_root.resolve()
    report_root = output_root if args.apply else output_root.with_name(output_root.name + "_dry_run")
    if not input_root.is_dir():
        raise FileNotFoundError(input_root)
    if args.overwrite_existing and not (args.take or args.participant):
        raise ValueError("--overwrite-existing requires at least one --take or --participant")
    if args.update_master_manifest and (not args.apply or not (args.take or args.participant)):
        raise ValueError("--update-master-manifest requires --apply and at least one --take or --participant")
    if args.apply and not args.overwrite_existing and output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_root}")
    take_dirs = sorted(path.parent for path in input_root.rglob("synced_data.csv"))
    if args.take:
        requested = {Path(item).as_posix() for item in args.take}
        take_dirs = [path for path in take_dirs if path.relative_to(input_root).as_posix() in requested]
        missing = requested - {path.relative_to(input_root).as_posix() for path in take_dirs}
        if missing:
            raise FileNotFoundError(f"Requested take(s) not found below input root: {sorted(missing)}")
    if args.participant:
        requested_participants = set(args.participant)
        take_dirs = [path for path in take_dirs if path.relative_to(input_root).parts[0] in requested_participants]
        found_participants = {path.relative_to(input_root).parts[0] for path in take_dirs}
        missing_participants = requested_participants - found_participants
        if missing_participants:
            raise FileNotFoundError(f"Requested participant(s) not found below input root: {sorted(missing_participants)}")
    if not take_dirs:
        raise RuntimeError(f"No synced_data.csv files below {input_root}")
    file_reports: list[dict[str, object]] = []
    take_reports: list[dict[str, object]] = []
    kinds = ("emg", "force", "imu", "offline_angles", "online_angles", "synced")
    for take_dir in take_dirs:
        relative = take_dir.relative_to(input_root)
        take_key = str(relative)
        files = {kind: canonical_file(take_dir, kind) for kind in kinds}
        bounds = {kind: csv_bounds(path) for kind, path in files.items()}
        shared_start = max(value[1] for value in bounds.values())
        shared_end = min(value[2] for value in bounds.values())
        if shared_end < shared_start:
            raise ValueError(f"No common timestamp overlap for {take_key}")
        keep_start, keep_end, rule, note = select_window(take_key, shared_start, shared_end)
        take_reports.append({"take_folder": take_key, "shared_start_s": shared_start, "shared_end_s": shared_end, "shared_duration_s": shared_end - shared_start, "keep_start_s": keep_start, "keep_end_s": keep_end, "target_duration_s": keep_end - keep_start, "rule": rule, "note": note})
        for kind, source in files.items():
            if kind == "online_angles" and args.exclude_online_output:
                continue
            time_col, input_start, input_end = bounds[kind]
            destination = output_root / relative / source.name
            report: dict[str, object] = {"take_folder": take_key, "kind": kind, "input_file": str(source), "output_file": str(destination), "time_column": time_col, "input_start_s": input_start, "input_end_s": input_end, "input_duration_s": input_end - input_start, "keep_start_s": keep_start, "keep_end_s": keep_end, "rule": rule, "note": note}
            if args.apply:
                report.update(trim_csv(source, destination, time_col, keep_start, keep_end, stable_sort=kind in {"offline_angles", "online_angles"}))
            file_reports.append(report)
        print(f"{'Trimmed' if args.apply else 'Validated'} {take_key}: {rule}", flush=True)
    if args.apply and not args.take:
        shutil.copy2(input_root / "processing_manifest.json", output_root / "source_processing_manifest.json")
    report_root.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.report_suffix}" if args.take else ""
    with (report_root / f"trim_manifest{suffix}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(take_reports[0]))
        writer.writeheader(); writer.writerows(take_reports)
    with (report_root / f"trim_file_report{suffix}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(file_reports[0]))
        writer.writeheader(); writer.writerows(file_reports)
    metadata = {"input_root": str(input_root), "output_root": str(output_root), "target_seconds": TARGET_SECONDS, "long_overlap_seconds": LONG_OVERLAP_SECONDS, "default_initial_rest_seconds": DEFAULT_INITIAL_REST_SECONDS, "manual_exceptions": {key: {"kind": value.kind, "value_s": value.value_s, "note": value.note} for key, value in EXCEPTIONS.items()}, "online_angle_coverage_used": True, "online_angle_output_included": not args.exclude_online_output, "apply": bool(args.apply), "take_count": len(take_reports)}
    (report_root / f"trimming_provenance{suffix}.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if args.update_master_manifest:
        merge_master_report(output_root / "trim_manifest.csv", take_reports, "take_folder")
        merge_master_report(output_root / "trim_file_report.csv", file_reports, "output_file")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
