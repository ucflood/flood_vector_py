# Changelog

All notable changes to this repository are recorded here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) · versioning: [SemVer](https://semver.org/) (`0.x` = research code, interfaces may change).

**How to use it:** add every change under `## [Unreleased]` in the same commit that makes it, grouped as
`Added` / `Changed` / `Fixed` / `Removed`. When you tag a release, rename `Unreleased` to the version and date and open a new empty `Unreleased`.
Record *why* when it affects results (e.g. a changed threshold, a different data source), not only what.

## [Unreleased]

### Added
- `00a_gee_export_layers.py`: Sentinel-2 option for NDVI/NDWI (`OPTICAL_SENSOR = "sentinel2"`, now the default; `"landsat"` reproduces the paper). Uses `COPERNICUS/S2_SR_HARMONIZED` (B3/B4/B8/B11), Cloud Score+ masking (`S2_CLOUD_METHOD = "cloudscore"`, threshold `S2_CLOUDSCORE_MIN`) or the SCL band (`"scl"`). Flood windows are still excluded via `EXCLUDE_RANGES`; both sensors share one index code path.

### Changed
- NDVI/NDWI default source is now Sentinel-2 (10/20 m bands averaged to the 30 m grid) instead of Landsat 8/9. Re-run `00a` and `00c`; rasters from the two sensors are not interchangeable.

### Fixed
-

## [0.3.0] - 2026-10-02
### Added
- Repository packaging: `README.md`, `Makefile` (one target per stage), `requirements.txt`, `environment.yml`, MIT `LICENSE`, `.gitignore` (data and model artefacts excluded).
- `tests/` with synthetic-data generators and 6 pytest tests covering every stage except the GEE scripts; GitHub Actions workflow running them on push.
- `docs/methodology.md` (design decisions and limitations) and `data/README.md` (expected data layout).
- This `CHANGELOG.md`.

### Changed
- Scripts moved into `pipeline/`. `gee_export_s1.py` renamed to `02a_gee_export_s1.py`; it now takes a Cloud project (`ee.Initialize(project=...)`).
- Stage 2 (`02_ama_flood_inventory.py`) prints a message and returns instead of failing when no flood pixels remain after masking.

## [0.2.0] - 2026-10-02
### Added
- Stage 0 indicator builders so all 12 indicators of Trong et al. (2023) can be produced at 30 m for the AMA:
  - `00a_gee_export_layers.py`: DEM (ALOS AW3D30 or Copernicus GLO-30), slope, aspect, relief amplitude, Landsat 8/9 NDVI/NDWI (flood windows excluded via `EXCLUDE_RANGES`), CHIRPS max 15-day rainfall, WorldCover LULC, optional OpenLandMap soil texture.
  - `00b_terrain_hydrology.py`: TWI and stream density from the DEM (pysheds D8), or from a vector stream network (`--streams`).
  - `00c_build_stack.py`: aligns all layers to one grid, clips to the AMA, rasterises geology/soil vectors, flags flat aspect, prints a QA table with warnings for coarse or low-information layers.

### Fixed
- Stream density no longer yields tiny negative values from FFT round-off (clipped at 0).

## [0.1.0] - 2026-10-02
### Added
- TensorFlow/Keras port of the Sen1Floods11 PyTorch training notebook (`01_train_sen1floods11.py`, `s1_common.py`): same preprocessing, class-weighted loss ignoring label 255, flood IoU and accuracy metrics; U-Net with GroupNorm replaces FCN-ResNet50.
- `gee_export_s1.py` (later `02a_`): Sentinel-1 VV/VH and JRC permanent-water export for Asuncion.
- `02_ama_flood_inventory.py`: sliding-window inference on large S1 scenes, flood = water(post) AND NOT water(pre) AND NOT permanent water, small-blob removal, polygonisation to GeoPackage.
- `03_susceptibility_1dcnn.py`: 1D-CNN reproducing the paper's architecture (71,088 parameters, Adam + MSE), frequency-ratio encoding of categorical indicators fitted on training data only, spatial-block train/val/test split, distance-buffered non-flood sampling, LR and RF baselines, probability and 5-class maps.
- `data/events/ama_events_filtered.csv`: event list filtered to the AMA with SAR-suitability notes (April 2024 urban floods marked validation-only; July 2021 flagged for verification).

### Fixed (during development)
- `polygonize()` raised "all scalar values" when building the GeoDataFrame; now one `event` value per polygon.
- Stage 1 test summary printed an opaque `compile_metrics` entry; now `evaluate(..., return_dict=True)`.
