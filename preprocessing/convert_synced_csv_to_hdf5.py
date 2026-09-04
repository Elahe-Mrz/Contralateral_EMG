#!/usr/bin/env python3
"""Convert publication ``synced_data.csv`` tables to lossless compressed HDF5.

The converter is deliberately non-destructive: it reads an existing processed
CSV tree and writes a parallel tree of one HDF5 file per take. Each HDF5 file
stores the parsed numeric table as a float64 matrix, its ordered column names,
the source relative path, source SHA-256, and conversion parameters. Gzip
compression and HDF5 shuffle are lossless; no values are downcast.

The result is a compact *alternative analysis representation*, not a rewrite of
the canonical CSV derivative. Use ``--verify`` for a parsed-value round-trip
check on each written file. Run a small ``--participant`` trial before a full
release conversion.
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
    parser.add_argument("--input-root", type=Path, required=True,
                        help="Root containing processed per-take synced_data.csv files.")
    parser.add_argument("--output-root", type=Path, required=True,
                        help="New parallel HDF5 root; never the input root.")
    parser.add_argument("--participant", action="append", default=[],
                        help="Participant ID to convert; repeatable. Defaults to all.")
    parser.add_argument("--take", action="append", default=[],
                        help="Relative take path, e.g. P001/Pinch/<take>; repeatable.")
    parser.add_argument("--compression-level", type=int, default=4, choices=range(1, 10),
                        metavar="1..9", help="Lossless gzip level (default: 4).")
    parser.add_argument("--verify", action="store_true",
                        help="Reparse each source CSV and verify exact float64 equality.")
    parser.add_argument("--apply", action="store_true",
                        help="Write HDF5 files. Without this flag, print the conversion plan only.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace an existing HDF5 file in the output root only.")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def discover(root: Path, participants: set[str], takes: set[str]) -> list[Path]:
    candidates = sorted(root.rglob("synced_data.csv"))
    selected: list[Path] = []
    for path in candidates:
        relative_take = path.parent.relative_to(root).as_posix()
        participant = relative_take.split("/", 1)[0]
        if participants and participant not in participants:
            continue
        if takes and relative_take not in takes:
            continue
        selected.append(path)
    return selected


def source_schema(source: Path) -> list[str]:
    columns = pd.read_csv(source, nrows=0).columns.tolist()
    if not columns:
        raise ValueError(f"No columns in {source}")
    return columns


def numeric_matrix(chunk: pd.DataFrame, source: Path) -> np.ndarray:
    try:
        return chunk.to_numpy(dtype=np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        non_numeric = [column for column in chunk.columns if not pd.api.types.is_numeric_dtype(chunk[column])]
        raise ValueError(
            f"{source} contains non-numeric columns incompatible with the lossless numeric HDF5 layout: {non_numeric}"
        ) from exc


def write_take(source: Path, output: Path, input_root: Path, output_root: Path, compression: int) -> dict:
    columns = source_schema(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    source_digest = sha256(source)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    row_count = 0
    with h5py.File(output, "w") as handle:
        dataset = handle.create_dataset(
            "data",
            shape=(0, len(columns)),
            maxshape=(None, len(columns)),
            dtype=np.float64,
            chunks=(CHUNK_ROWS, len(columns)),
            compression="gzip",
            compression_opts=compression,
            shuffle=True,
        )
        handle.create_dataset("columns", data=np.asarray(columns, dtype=object), dtype=string_dtype)
        for chunk in pd.read_csv(source, chunksize=CHUNK_ROWS):
            if chunk.columns.tolist() != columns:
                raise ValueError(f"Column order changed while reading {source}")
            values = numeric_matrix(chunk, source)
            next_row = row_count + len(values)
            dataset.resize((next_row, len(columns)))
            dataset[row_count:next_row] = values
            row_count = next_row
        dataset.attrs["source_relative_path"] = source.relative_to(input_root).as_posix()
        dataset.attrs["source_sha256"] = source_digest
        dataset.attrs["numeric_storage_dtype"] = "float64"
        dataset.attrs["conversion_utc"] = datetime.now(timezone.utc).isoformat()
        dataset.attrs["lossless_compression"] = "gzip + shuffle; no value downcasting"
    return {
        "source_relative_path": source.relative_to(input_root).as_posix(),
        "output_relative_path": output.relative_to(output_root).as_posix(),
        "rows": row_count,
        "columns": len(columns),
        "source_bytes": source.stat().st_size,
        "output_bytes": output.stat().st_size,
        "source_sha256": source_digest,
    }


def verify_take(source: Path, output: Path) -> None:
    columns = source_schema(source)
    with h5py.File(output, "r") as handle:
        stored_columns = [value.decode() if isinstance(value, bytes) else str(value) for value in handle["columns"][:]]
        if stored_columns != columns:
            raise AssertionError(f"Column mismatch for {source}")
        row_offset = 0
        for chunk in pd.read_csv(source, chunksize=CHUNK_ROWS):
            values = numeric_matrix(chunk, source)
            stored = handle["data"][row_offset:row_offset + len(values)]
            if not np.array_equal(values, stored, equal_nan=True):
                raise AssertionError(f"Float64 value mismatch for {source} at row {row_offset}")
            row_offset += len(values)
        if row_offset != handle["data"].shape[0]:
            raise AssertionError(f"Row-count mismatch for {source}")
        if handle["data"].attrs["source_sha256"] != sha256(source):
            raise AssertionError(f"Source checksum mismatch for {source}")


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    if input_root == output_root or input_root in output_root.parents:
        raise ValueError("Output root must be outside the input root")
    if not input_root.is_dir():
        raise NotADirectoryError(input_root)
    sources = discover(input_root, set(args.participant), set(args.take))
    if not sources:
        raise FileNotFoundError("No synced_data.csv files match the requested selection")
    plan = [
        {
            "source": source.relative_to(input_root).as_posix(),
            "output": source.parent.relative_to(input_root).with_suffix(".h5").as_posix(),
            "source_bytes": source.stat().st_size,
        }
        for source in sources
    ]
    if not args.apply:
        print(json.dumps({"planned_takes": len(plan), "plan": plan}, indent=2))
        return

    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for index, source in enumerate(sources, start=1):
        take_relative = source.parent.relative_to(input_root)
        output = output_root / take_relative.with_suffix(".h5")
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite {output}; use --overwrite")
        if output.exists():
            output.unlink()
        record = write_take(source, output, input_root, output_root, args.compression_level)
        if args.verify:
            verify_take(source, output)
            record["verified"] = True
        rows.append(record)
        print(f"Converted {index}/{len(sources)}: {record['source_relative_path']}", flush=True)
    manifest = output_root / "hdf5_conversion_manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "take_count": len(rows),
        "source_bytes": sum(row["source_bytes"] for row in rows),
        "output_bytes": sum(row["output_bytes"] for row in rows),
        "compression_level": args.compression_level,
        "verified": args.verify,
        "representation": "float64 numeric matrix with ordered column-name dataset",
    }
    (output_root / "hdf5_conversion_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
