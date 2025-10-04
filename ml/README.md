# MODIS Cloud Seeding Likelihood Model

This folder contains an experimental PyTorch training pipeline that estimates the
likelihood that atmospheric conditions are favorable for cloud seeding using NASA
MODIS (Moderate Resolution Imaging Spectroradiometer) cloud property products.

## Overview

1. **Acquire raw data** from the MOD06_L2 / MYD06_L2 cloud property archive.
2. **Preprocess granules** into tabular features (optical thickness, effective radius, etc.).
3. **Train** a feed-forward PyTorch model that predicts a cloud seeding likelihood score.

> ⚠️ You will need NASA Earthdata credentials (https://urs.earthdata.nasa.gov) and to
> download the granules manually (or with your own tooling) before running the training script.

## Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-ml.txt
```

## Data Acquisition

Download the desired MODIS granules from NASA's Earthdata portal (LP DAAC) using the
standard browser workflow or another tool of your choice, then store them under
`data/raw/<collection>/`. Keep track of the collection, dates, and bounding box you used so
the preprocessing step can reference the same subset.

## Preprocessing

Convert the downloaded granules into a feature table:

```python
from pathlib import Path
from ml.modis_cloud_seeding import preprocess

raw_paths = Path("data/raw/mod06").glob("*.hdf")
preprocess.export_features(raw_paths, Path("data/processed/example_features.csv"))
```

If you customise the feature extraction, export the configuration for reproducibility:

```python
from ml.modis_cloud_seeding import preprocess
preprocess.save_feature_config(preprocess.DEFAULT_FEATURES, Path("ml_artifacts/features.json"))
```

## Training

Update `ml/modis_cloud_seeding/configs/default.yaml` to point to your feature CSV and
appropriate label column. Then run:

```bash
python -m ml.modis_cloud_seeding.train --config ml/modis_cloud_seeding/configs/default.yaml
```

Artifacts (best model checkpoint and metrics) are written to the directory specified in the config.

## Next Steps

- Replace the placeholder label column with your domain-specific cloud seeding observations.
- Experiment with additional MODIS variables (e.g., relative humidity, aerosol optical depth).
- Add spatial/temporal context by aggregating multiple consecutive granules or reanalysis data.
- Integrate evaluation notebooks to visualise performance and feature importance.
