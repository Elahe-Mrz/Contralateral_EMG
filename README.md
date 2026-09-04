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
* `preprocessing/convert_synced_csv_to_hdf5.py` — losslessly converts each
  `synced_data.csv` numeric table to compressed HDF5.
* `preprocessing/append_native_processed_csvs_to_hdf5.py` — adds the native
  processed modality tables to the matching HDF5 take file.
* `load_release.py` — loads synchronized or native processed data from CSV or
  HDF5 into pandas DataFrames.

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

The HDF5 representation stores parsed numeric values as `float64`, preserves
column order and missing values, and uses gzip plus shuffle compression without
downcasting. It is lossless with respect to the parsed CSV tables, rather than
byte-identical to their text serialization.

The final data-release curation retains the canonical offline-retargeted angle
table and excludes standard online-angle provenance derivatives.

## Reproduce the processed release

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

To make a compact HDF5 representation of the final processed release:

```bash
python preprocessing/convert_synced_csv_to_hdf5.py \
  --input-root /path/to/Preprocessed --output-root /path/to/hdf5 \
  --verify --apply

python preprocessing/append_native_processed_csvs_to_hdf5.py \
  --input-root /path/to/Preprocessed --hdf5-root /path/to/hdf5 \
  --verify --apply
```

## Citation and data access

Please cite the accompanying data paper and the dataset DOI once available.
The code repository should be archived at the release commit through Zenodo to
obtain a versioned software DOI.
