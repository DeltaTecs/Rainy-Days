# MODIS Cloud Seeding Likelihood Model

This folder contains an experimental PyTorch training pipeline that estimates the
likelihood that atmospheric conditions are favorable for cloud seeding using NASA
MODIS (Moderate Resolution Imaging Spectroradiometer) cloud property products.

## Overview

1. **Acquire raw data** from the MOD06_L2 / MYD06_L2 cloud property archive (via the
   provided download helper or your own tooling).
2. **Preprocess granules** into tabular features (optical thickness, effective radius, etc.).
3. **Train** a feed-forward PyTorch model that predicts a cloud seeding likelihood score.

> ⚠️ You will need NASA Earthdata credentials (https://urs.earthdata.nasa.gov) and to
> download the granules manually (or with your own tooling) before running the training script.

## Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-ml.txt

# HDF4 support (MOD06 granules)

MOD06_L2 granules are distributed as HDF4. The preprocessing helpers automatically
fall back to ``pyhdf`` when xarray/netCDF cannot open a file. Install ``pyhdf`` with
conda-forge for the smoothest experience:

```bash
conda install -n modis-ml -c conda-forge pyhdf
```

If you prefer pip, make sure the HDF4 development libraries are available locally
before running ``pip install pyhdf``.

## Data Acquisition

1. Create a free Earthdata account at https://urs.earthdata.nasa.gov.
2. Export credentials for the downloader (`earthaccess` checks these automatically):

   ```bash
   export EARTHDATA_USERNAME="<your-username>"
   export EARTHDATA_PASSWORD="<your-password>"
   ```

   Alternatively, write a `~/.netrc` with your credentials and `chmod 600` the file.

3. Use the helper to fetch granules:

   ```python
   from pathlib import Path
   from datetime import datetime
   from ml.modis_cloud_seeding.download_modis import BoundingBox, discover_and_download

   downloads = discover_and_download(
       output_dir=Path("data/raw/mod06"),
       bbox=BoundingBox(min_lon=-116, min_lat=36, max_lon=-114, max_lat=38),
       start_date=datetime(2023, 6, 1),
       end_date=datetime(2023, 6, 30),
       collection_key="mod06",
       limit=10,
   )
   print(downloads)
   ```

If you prefer manual downloads, place the HDF granules under `data/raw/<collection>/` and
keep the date/bounding box metadata handy for downstream steps.

## Preprocessing

Convert the downloaded granules into a feature table, masking the statistics to your
province bounding box. The helpers now accept a `bounding_box` argument and append
platform/timestamp metadata derived from the filename:

```python
from pathlib import Path
from ml.modis_cloud_seeding import preprocess

raw_paths = Path("workspace/download/downloads").glob("*.hdf")
province_bbox = (-120.0, 49.5, -109.0, 59.5)
# 0.25° grid (~25 km) capturing sub-regions across the province.
grid_deg = (0.25, 0.25)
preprocess.export_features(
    raw_paths,
    Path("data/processed/granule_features.csv"),
    bounding_box=province_bbox,
    grid_size=grid_deg,
)
```

If you customise the feature extraction, export the configuration for reproducibility:

```python
from ml.modis_cloud_seeding import preprocess
preprocess.save_feature_config(preprocess.DEFAULT_FEATURES, Path("ml_artifacts/features.json"))
```

### Daily aggregation

For forecasting we operate on daily aggregates (optionally split by Terra/Aqua). Convert
the per-granule table into daily statistics with the CLI helper:

```bash
python -m ml.modis_cloud_seeding.daily \
  --input data/processed/granule_features.csv \
  --output data/processed/daily_features.csv

# Add --split-platforms to keep Terra and Aqua features in separate columns.
```

## Training

Update `ml/modis_cloud_seeding/configs/default.yaml` to point to your feature CSV and
appropriate label column. Then run:

```bash
python -m ml.modis_cloud_seeding.train --config ml/modis_cloud_seeding/configs/default.yaml
```

Artifacts (best model checkpoint and metrics) are written to the directory specified in the config.

## Forecast & seedability scoring

Train short-horizon property forecasters (1, 2, 3, and 7 days ahead by default) and
derive a seedability percentage for every grid cell with:

```bash
python -m ml.modis_cloud_seeding.forecast \
  --daily-features data/processed/daily_features.csv \
  --output data/processed/daily_predictions.csv \
  --artifacts-dir ml_artifacts/forecasts
```

Each horizon trains a multi-output LightGBM model shared across all grid cells. Artefacts
(models, metrics, backtests) are saved under `ml_artifacts/forecasts/`, and the output
CSV lists the grid cell metadata (`region_id`, centre lat/lon), forecast date, predicted
cloud properties, `seedability_score`, and `seedability_percent` for each cell.

## Next Steps

- Replace the placeholder label column with your domain-specific cloud seeding observations.
- Experiment with additional MODIS variables (e.g., relative humidity, aerosol optical depth).
- Add spatial/temporal context by aggregating multiple consecutive granules or reanalysis data.
- Integrate evaluation notebooks to visualise performance and feature importance.
