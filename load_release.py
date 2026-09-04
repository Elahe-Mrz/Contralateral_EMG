#!/usr/bin/env python3
"""Load synchronized and native processed dataset tables from CSV or HDF5.

The HDF5 layout is created by the scripts in ``preprocessing/``.  The returned
DataFrames reproduce the parsed table values and their original column order.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def _strings(values: np.ndarray) -> list[str]:
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]


def load_synced_csv(path: str | Path) -> pd.DataFrame:
    """Load a release ``synced_data.csv`` table."""
    return pd.read_csv(path)


def load_synced_hdf5(path: str | Path) -> pd.DataFrame:
    """Load the root-level synchronized HDF5 table into a DataFrame."""
    with h5py.File(path, "r") as handle:
        return pd.DataFrame(handle["data"][:], columns=_strings(handle["columns"][:]))


def list_native_modalities(path: str | Path) -> list[str]:
    """Return available native processed group names in one HDF5 take."""
    with h5py.File(path, "r") as handle:
        return sorted(handle.get("native_processed", {}).keys())


def load_native_hdf5(path: str | Path, modality: str) -> pd.DataFrame:
    """Load one native processed HDF5 group into a DataFrame.

    ``modality`` is one value returned by :func:`list_native_modalities`.
    Text columns and their missing-value masks are restored exactly as represented
    by the HDF5 conversion.
    """
    with h5py.File(path, "r") as handle:
        group = handle["native_processed"][modality]
        columns = _strings(group["columns"][:])
        numeric_columns = _strings(group["numeric_columns"][:])
        text_columns = _strings(group["text_columns"][:])
        frame = pd.DataFrame(group["numeric_data"][:], columns=numeric_columns)
        for column in text_columns:
            index = columns.index(column)
            key = f"column_{index}"
            values = _strings(group["text"][key][:])
            missing = group["text_missing"][key][:]
            frame[column] = pd.Series(values, dtype="object").mask(missing)
        return frame.loc[:, columns]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="CSV or HDF5 take file")
    parser.add_argument("--native", help="Native HDF5 modality name; omit for synchronized data")
    args = parser.parse_args()
    if args.path.suffix.lower() == ".csv":
        if args.native:
            parser.error("--native is only valid for HDF5 input")
        frame = load_synced_csv(args.path)
    elif args.path.suffix.lower() in {".h5", ".hdf5"}:
        frame = load_native_hdf5(args.path, args.native) if args.native else load_synced_hdf5(args.path)
    else:
        parser.error("path must end in .csv, .h5, or .hdf5")
    print(f"rows={len(frame)} columns={len(frame.columns)}")
    print("\n".join(frame.columns))


if __name__ == "__main__":
    main()
