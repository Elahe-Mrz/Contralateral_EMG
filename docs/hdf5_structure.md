# HDF5 structure

The processed release contains one HDF5 file per recording take. Files retain
the participant/gesture hierarchy in their paths:

```text
P001/Power/20260523_172903_power_grasp_unguided_take07.h5
```

Each file contains:

```text
/
├── columns                  ordered UTF-8 names for /data
├── data                     synchronized float64 table
└── native_processed/
    ├── emg_*_preprocessed/
    ├── imu_*_preprocessed/
    ├── FSR_*_preprocessed/
    └── offline_retargeted_angles_*_preprocessed/
```

`/data` is the synchronized analysis table and uses `/columns` for its ordered
column labels. It includes the common timestamp, processed sEMG, IMU, canonical
offline-retargeted joint angles, and calibrated fingertip forces.

Every group below `/native_processed` preserves one modality at its native
processed sampling rate. A native group contains:

* `columns`: original ordered column names;
* `numeric_columns`: names represented by `numeric_data`;
* `numeric_data`: a two-dimensional float64 array;
* `text_columns`: names of any preserved text/provenance fields;
* `text/`: string values for text fields;
* `text_missing/`: Boolean missing-value masks for text fields.

The numeric datasets use lossless gzip and shuffle compression without
downcasting. Missing numeric values remain IEEE NaN values. The HDF5 tables are
lossless with respect to parsed CSV values and column order, but are not
byte-identical reproductions of CSV text serialization.

Use `load_release.py` to reconstruct synchronized or native tables as pandas
DataFrames. Consult `data_dictionary.csv` for variable definitions and units.
