# Contralateral EMG data-release code

This repository contains preprocessing, loading, visualization, feature
extraction, and technical-validation code for the contralateral upper-limb
multimodal dataset.

It contains no participant data. The associated deposit provides the original
recordings as CSV files under `Raw/` and the analysis-ready data as one HDF5
file per take. Most users should work directly with the HDF5 files.

## Contents

* `preprocessing/preprocess_publication_data.py` — creates native-rate
  processed sEMG, IMU, force, and offline-retargeted-angle tables and a
  synchronized CSV table from `Raw/`. It never changes its input root.
* `preprocessing/trim_preprocessed_publication.py` — applies the documented
  shared-coverage window and writes the final trimmed CSV derivative to a
  separate output root.
* `load_release.py` — lists, loads, or exports synchronized and native
  processed HDF5 tables as pandas DataFrames or CSV files.
* `tutorials/load_and_visualize_hdf5.py` — loads one released HDF5 take and
  plots synchronized sEMG, joint-angle, and fingertip-force signals.
* `analysis/emg_ridge_session_baseline.py` — implements the reported causal
  500-ms sEMG features and participant-specific ridge-regression validation.
* `docs/hdf5_structure.md` and `docs/data_dictionary.csv` — describe the
  deposited HDF5 layout and variables.

## Installation

```bash
python -m pip install -r requirements.txt
```

The pinned environment was tested with Python 3.10.18.

## Quick start: released HDF5 data

Inspect one file and list its native-rate modality groups:

```bash
python load_release.py /path/to/take.h5 --list-native
```

## Load one synchronized take

Load the synchronized HDF5 table:

```python
from load_release import load_synced_hdf5

frame = load_synced_hdf5(
    "/path/to/hdf5/P001/Pinch/<take>.h5"
)
```

The preprocessing workflow described below can also recreate synchronized CSV
tables. Those tables can be loaded with:

```python
from load_release import load_synced_csv

frame = load_synced_csv("/path/to/recreated/synced_data.csv")
```

Load a native processed modality from HDF5:

```python
from load_release import list_native_modalities, load_native_hdf5

path = "/path/to/hdf5/P001/Pinch/<take>.h5"
print(list_native_modalities(path))
emg = load_native_hdf5(path, "emg_<take>_2000Hz_preprocessed")
```

Export the synchronized HDF5 table to CSV:

```bash
python load_release.py /path/to/take.h5 \
  --output-csv /path/to/exported_synced_data.csv
```

Export one native processed modality by using a group name returned by
`--list-native`:

```bash
python load_release.py /path/to/take.h5 \
  --native "emg_<take>_2000Hz_preprocessed" \
  --output-csv /path/to/exported_emg.csv
```

Existing CSV outputs are not replaced unless `--overwrite` is supplied.

The HDF5 representation stores parsed numeric values as `float64`, preserves
column order and missing values, and uses gzip plus shuffle compression without
downcasting. It is lossless with respect to the parsed CSV tables, rather than
byte-identical to their text serialization.

The final data-release curation retains the canonical offline-retargeted angle
table and excludes standard online-angle provenance derivatives.

Create a synchronized multimodal plot directly from one released HDF5 file:

```bash
python tutorials/load_and_visualize_hdf5.py \
  /path/to/hdf5/P001/Power/<take>.h5 \
  --start-s 0 --duration-s 15 --output example.png
```

## Preprocess the raw CSV release

This workflow is provided for users who want to reproduce processing from the
raw CSV release. It is not required when using the deposited HDF5 files. Both
stages write to explicitly separate output roots and never modify `Raw/`.

The publication profile applies:

* sEMG detrending, 20--450 Hz band-pass filtering, a 60 Hz notch, and
  conservative isolated-spike repair;
* force baseline correction, 100-ms median filtering, 5-Hz low-pass filtering,
  non-negative clipping, and a small adaptive zero deadband;
* 200-ms median and 5-Hz low-pass filtering of offline-retargeted angles;
* 20-Hz low-pass filtering of accelerometer/gyroscope channels and 10-Hz
  low-pass filtering of magnetometer channels without removing physical DC;
* timestamp synchronization to the nominal 2-kHz sEMG grid, with bounded
  interpolation and no extrapolation beyond observed coverage.

First create the full processed CSV tree:

```bash
python preprocessing/preprocess_publication_data.py \
  /path/to/Raw --output-root /path/to/preprocessed
```

Then select the documented shared-coverage windows. Online-angle timing is used
when determining shared coverage, but `--exclude-online-output` prevents that
non-canonical derivative from being copied into the final output:

```bash
python preprocessing/trim_preprocessed_publication.py \
  --input-root /path/to/preprocessed \
  --output-root /path/to/final_processed_csv \
  --exclude-online-output --apply
```

Each stage writes JSON/CSV provenance reports alongside its outputs. The
deposited HDF5 files are the official compact representation; HDF5 packaging is
an internal release-engineering operation rather than a scientific processing
step.

## Technical-validation baseline

The ridge script contains the exact feature definitions used in the paper:
RMS, mean absolute value, waveform length, and zero crossings from a causal
500-ms window. It fits participant-specific multi-output models and holds out
one complete take from each gesture in every fold. P007 EMG channel 7 is
excluded consistently by the documented default. The script reads either the
released HDF5 tree directly or synchronized CSV files recreated by the
preprocessing workflow.

Run it directly on the HDF5 deposit:

```bash
python analysis/emg_ridge_session_baseline.py \
  --data-root /path/to/hdf5 \
  --data-format hdf5 \
  --output /path/to/baseline_results
```

Use `--data-format csv` for a recreated synchronized CSV tree. With `auto`, the
script accepts a root containing exactly one of those representations.

The output directory contains:

* `fold_metrics.csv` and `summary_by_output.csv` for the primary concatenated
  three-take held-out evaluation;
* `held_out_take_metrics.csv` and
  `held_out_take_summary_by_output_equal_participant.csv` for separately
  evaluated held-out takes;
* participant-level summaries, take eligibility, window inventory, loading
  failures, and `run_config.json` with the complete analysis settings.

## Citation and data access

Please cite the accompanying data paper and the dataset DOI once available.
The code repository should be archived at the release commit through Zenodo to
obtain a versioned software DOI.
