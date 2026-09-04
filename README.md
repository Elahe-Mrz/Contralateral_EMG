# Contralateral EMG data-release code

This repository contains the reproducible preprocessing and data-loading code
for the contralateral upper-limb multimodal dataset.

It contains no participant data. Obtain the release separately, then point the
commands below at its `Raw/`, `Preprocessed/`, or HDF5 release root.

## Contents

* `preprocessing/preprocess_publication_data.py` — creates native-rate
  processed sEMG, IMU, force, and offline-retargeted-angle tables and a
  synchronized table from the raw release. It never changes its input root.
* `preprocessing/trim_preprocessed_publication.py` — applies the documented
  final shared-coverage trimming to a separate output root.
* `load_release.py` — loads synchronized or native processed data from CSV or
  HDF5 into pandas DataFrames.
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

Python 3.10 or later is recommended.

## Load one synchronized take

CSV:

```python
from load_release import load_synced_csv

frame = load_synced_csv(
    "/path/to/Release/Preprocessed/P001/Pinch/<take>/synced_data.csv"
)
```

HDF5:

```python
from load_release import load_synced_hdf5

frame = load_synced_hdf5(
    "/path/to/hdf5/P001/Pinch/<take>.h5"
)
```

Load a native processed modality from HDF5:

```python
from load_release import list_native_modalities, load_native_hdf5

path = "/path/to/hdf5/P001/Pinch/<take>.h5"
print(list_native_modalities(path))
emg = load_native_hdf5(path, "emg_<take>_2000Hz_preprocessed")
```

List the native modality names stored in a take:

```bash
python load_release.py /path/to/take.h5 --list-native
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

## Preprocess the raw CSV release

All commands write to an explicitly separate output directory. Run the main
preprocessing script first, then the trimming script. Exact input/output paths,
participants, and special documented corrections are recorded in the release
provenance files.

```bash
python preprocessing/preprocess_publication_data.py \
  /path/to/Raw --output-root /path/to/preprocessed

python preprocessing/trim_preprocessed_publication.py \
  --input-root /path/to/preprocessed \
  --output-root /path/to/Preprocessed --apply
```

The deposited HDF5 files are the official analysis-ready representation. The
CSV-to-HDF5 packaging utility is retained as internal release-engineering code;
it is not needed to load or analyze the deposited data.

```bash
python tutorials/load_and_visualize_hdf5.py \
  /path/to/hdf5/P001/Power/<take>.h5 \
  --start-s 0 --duration-s 15 --output example.png
```

## Technical-validation baseline

The ridge script contains the exact feature definitions used in the paper:
RMS, mean absolute value, waveform length, and zero crossings from a causal
500-ms window. It fits participant-specific multi-output models and holds out
one complete take from each gesture in every fold. P007 EMG channel 7 is
excluded consistently by the documented default. Run it on the synchronized
CSV tree recreated by the preprocessing workflow:

```bash
python analysis/emg_ridge_session_baseline.py \
  --data-root /path/to/Preprocessed \
  --output /path/to/baseline_results
```

## Citation and data access

Please cite the accompanying data paper and the dataset DOI once available.
The code repository should be archived at the release commit through Zenodo to
obtain a versioned software DOI.
