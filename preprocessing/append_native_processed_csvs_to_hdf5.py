#!/usr/bin/env python3
"""Append lossless native-rate processed CSV tables to per-take HDF5 files.

This complements ``convert_synced_csv_to_hdf5.py``. For every per-take CSV
other than ``synced_data.csv``, it creates a named group under
``native_processed/<source filename without .csv>/`` in the corresponding HDF5
file. Numeric values are stored as float64 with lossless gzip/shuffle
compression. Root-level CSV/JSON provenance files are intentionally left as
small standalone release documentation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


CHUNK_ROWS = 16_384


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--hdf5-root", type=Path, required=True)
    parser.add_argument("--take", action="append", default=[],
                        help="Relative take folder; repeatable. Defaults to all takes.")
    parser.add_argument("--compression-level", type=int, default=4, choices=range(1, 10), metavar="1..9")
    parser.add_argument("--verify", action="store_true",
                        help="Reparse sources and verify exact float64 equality after writing.")
    parser.add_argument("--apply", action="store_true", help="Write groups; otherwise print the plan only.")
    parser.add_argument("--overwrite-groups", action="store_true",
                        help="Replace existing generated native_processed groups in HDF5 files.")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def numeric_matrix(frame: pd.DataFrame, columns: list[str], source: Path) -> np.ndarray:
    try:
        return frame.loc[:, columns].to_numpy(dtype=np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Numeric conversion failed for {source}: {columns}") from exc


def schema(source: Path) -> tuple[list[str], list[str], list[str]]:
    sample = pd.read_csv(source, nrows=4_096)
    columns = sample.columns.tolist()
    numeric = [column for column in columns if pd.api.types.is_numeric_dtype(sample[column])]
    text = [column for column in columns if column not in numeric]
    return columns, numeric, text


def discover(root: Path, selected_takes: set[str]) -> list[Path]:
    sources: list[Path] = []
    for path in sorted(root.rglob("*.csv")):
        relative = path.relative_to(root)
        if len(relative.parts) < 4 or path.name == "synced_data.csv":
            continue
        take = relative.parent.as_posix()
        if selected_takes and take not in selected_takes:
            continue
        sources.append(path)
    return sources


def hdf5_path(source: Path, input_root: Path, hdf5_root: Path) -> Path:
    return hdf5_root / source.parent.relative_to(input_root).with_suffix(".h5")


def group_name(source: Path) -> str:
    return source.stem


def write_group(source: Path, destination: Path, input_root: Path, hdf5_root: Path,
                compression: int, overwrite: bool) -> dict[str, object]:
    columns, numeric_columns, text_columns = schema(source)
    if not columns:
        raise ValueError(f"No columns in {source}")
    source_digest = sha256(source)
    name = group_name(source)
    string_dtype = h5py.string_dtype("utf-8")
    rows = 0
    with h5py.File(destination, "a") as handle:
        root = handle.require_group("native_processed")
        if name in root:
            if not overwrite:
                raise FileExistsError(f"Existing group {root[name].name}; use --overwrite-groups")
            del root[name]
        group = root.create_group(name)
        dataset = group.create_dataset(
            "numeric_data", shape=(0, len(numeric_columns)), maxshape=(None, len(numeric_columns)), dtype=np.float64,
            chunks=(CHUNK_ROWS, max(1, len(numeric_columns))), compression="gzip", compression_opts=compression, shuffle=True,
        )
        group.create_dataset("columns", data=np.asarray(columns, dtype=object), dtype=string_dtype)
        group.create_dataset("numeric_columns", data=np.asarray(numeric_columns, dtype=object), dtype=string_dtype)
        group.create_dataset("text_columns", data=np.asarray(text_columns, dtype=object), dtype=string_dtype)
        text_group = group.create_group("text")
        missing_group = group.create_group("text_missing")
        text_datasets = {
            column: text_group.create_dataset(
                f"column_{columns.index(column)}", shape=(0,), maxshape=(None,), dtype=string_dtype
            ) for column in text_columns
        }
        missing_datasets = {
            column: missing_group.create_dataset(
                f"column_{columns.index(column)}", shape=(0,), maxshape=(None,), dtype=np.bool_,
                chunks=(CHUNK_ROWS,), compression="gzip", compression_opts=compression, shuffle=True,
            ) for column in text_columns
        }
        for chunk in pd.read_csv(source, chunksize=CHUNK_ROWS):
            if chunk.columns.tolist() != columns:
                raise ValueError(f"Column order changed while reading {source}")
            values = numeric_matrix(chunk, numeric_columns, source)
            next_rows = rows + len(values)
            dataset.resize((next_rows, len(numeric_columns)))
            dataset[rows:next_rows] = values
            for column in text_columns:
                values_text = chunk[column]
                missing = values_text.isna().to_numpy(dtype=bool)
                encoded = values_text.fillna("").astype(str).to_numpy(dtype=object)
                text_datasets[column].resize((next_rows,))
                text_datasets[column][rows:next_rows] = encoded
                missing_datasets[column].resize((next_rows,))
                missing_datasets[column][rows:next_rows] = missing
            rows = next_rows
        group.attrs["source_relative_path"] = source.relative_to(input_root).as_posix()
        group.attrs["source_sha256"] = source_digest
        group.attrs["numeric_storage_dtype"] = "float64"
        group.attrs["conversion_utc"] = datetime.now(timezone.utc).isoformat()
        group.attrs["lossless_compression"] = "gzip + shuffle; no value downcasting"
        storage_bytes = int(dataset.id.get_storage_size()) + sum(
            int(item.id.get_storage_size()) for item in missing_datasets.values()
        )
    return {
        "source_relative_path": source.relative_to(input_root).as_posix(),
        "hdf5_relative_path": destination.relative_to(hdf5_root).as_posix(),
        "group": f"/native_processed/{name}",
        "rows": rows,
        "columns": len(columns),
        "source_bytes": source.stat().st_size,
        "hdf5_data_storage_bytes": storage_bytes,
        "source_sha256": source_digest,
    }


def verify_group(source: Path, destination: Path) -> None:
    columns, numeric_columns, text_columns = schema(source)
    with h5py.File(destination, "r") as handle:
        group = handle["native_processed"][group_name(source)]
        stored_columns = [item.decode() if isinstance(item, bytes) else str(item) for item in group["columns"][:]]
        if stored_columns != columns:
            raise AssertionError(f"Column mismatch: {source}")
        stored_numeric = [item.decode() if isinstance(item, bytes) else str(item) for item in group["numeric_columns"][:]]
        stored_text = [item.decode() if isinstance(item, bytes) else str(item) for item in group["text_columns"][:]]
        if stored_numeric != numeric_columns or stored_text != text_columns:
            raise AssertionError(f"Column-type mismatch: {source}")
        row = 0
        for chunk in pd.read_csv(source, chunksize=CHUNK_ROWS):
            values = numeric_matrix(chunk, numeric_columns, source)
            stored = group["numeric_data"][row:row + len(values)]
            if not np.array_equal(values, stored, equal_nan=True):
                raise AssertionError(f"Value mismatch: {source} row {row}")
            for column in text_columns:
                key = f"column_{columns.index(column)}"
                stored_values = [item.decode() if isinstance(item, bytes) else str(item)
                                 for item in group["text"][key][row:row + len(values)]]
                stored_missing = group["text_missing"][key][row:row + len(values)]
                expected_missing = chunk[column].isna().to_numpy(dtype=bool)
                expected_values = chunk[column].fillna("").astype(str).tolist()
                if not np.array_equal(expected_missing, stored_missing) or expected_values != stored_values:
                    raise AssertionError(f"Text-value mismatch: {source} column {column} row {row}")
            row += len(values)
        if row != group["numeric_data"].shape[0] or group.attrs["source_sha256"] != sha256(source):
            raise AssertionError(f"Integrity mismatch: {source}")


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    hdf5_root = args.hdf5_root.resolve()
    if not input_root.is_dir() or not hdf5_root.is_dir():
        raise NotADirectoryError("Both --input-root and --hdf5-root must exist")
    sources = discover(input_root, set(args.take))
    if not sources:
        raise FileNotFoundError("No matching native processed CSV files")
    plan = []
    for source in sources:
        destination = hdf5_path(source, input_root, hdf5_root)
        if not destination.is_file():
            raise FileNotFoundError(f"Missing synchronized HDF5 file for {source}: {destination}")
        plan.append({"source": source.relative_to(input_root).as_posix(),
                     "hdf5": destination.relative_to(hdf5_root).as_posix(),
                     "group": f"/native_processed/{group_name(source)}"})
    if not args.apply:
        print(json.dumps({"planned_csvs": len(plan), "plan": plan}, indent=2))
        return
    records: list[dict[str, object]] = []
    for index, source in enumerate(sources, start=1):
        destination = hdf5_path(source, input_root, hdf5_root)
        record = write_group(source, destination, input_root, hdf5_root,
                             args.compression_level, args.overwrite_groups)
        if args.verify:
            verify_group(source, destination)
            record["verified"] = True
        records.append(record)
        print(f"Converted {index}/{len(sources)}: {record['source_relative_path']}", flush=True)
    manifest = hdf5_root / "native_processed_hdf5_manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    summary = {
        "input_root": str(input_root), "hdf5_root": str(hdf5_root),
        "native_csv_count": len(records), "source_bytes": sum(int(row["source_bytes"]) for row in records),
        "hdf5_data_storage_bytes": sum(int(row["hdf5_data_storage_bytes"]) for row in records),
        "compression_level": args.compression_level, "verified": args.verify,
    }
    (hdf5_root / "native_processed_hdf5_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
