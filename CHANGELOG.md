# Changelog

All notable changes to this repository are recorded here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) · versioning: [SemVer](https://semver.org/) (`0.x` = research code, interfaces may change).

**How to use it:** add every change under `## [Unreleased]` in the same commit that makes it, grouped as
`Added` / `Changed` / `Fixed` / `Removed`. When you tag a release, rename `Unreleased` to the version and date and open a new empty `Unreleased`.
Record *why* when it affects results (e.g. a changed threshold, a different data source), not only what.

## [Unreleased]

### Added
- Google Colab support: `notebooks/AMA_flood_susceptibility_colab.ipynb` runs stages 0a-4 in order with Drive persistence (skips finished stages, copies GEE exports from Drive and merges tiled exports), plus `requirements-colab.txt`. `tests/test_notebook.py` checks that every cell compiles and every pipeline call uses valid arguments.
- `pipeline/gee_utils.py`: Earth Engine init with a clear authentication message, AOI from `AMA_AOI_ASSET`, and `wait_tasks()` to block until exports finish (`GEE_WAIT=1`).
- `01_train_sen1floods11.py --resume`: saves the last epoch's weights every epoch and continues after a disconnect; `history.csv` is appended and the best checkpoint is only replaced by a better one.

### Changed
- Colab notebook: removed the `REPO_ZIP` option; the repository now always comes from GitHub (`REPO_URL`). A non-git folder at the repo path is moved aside to `*_old_<timestamp>` and the setup cell checks that `requirements-colab.txt` is at the repo root.
- 00a and 02a read project, AOI, Drive folder and (02a) the scene list from environment variables instead of edited constants; 04 defaults `--project`/`--aoi-asset` to them. All CLI scripts expose `parse_args()`.

### Fixed
- `.gitignore`, `.github/workflows/ci.yml` and the `data/*/.gitkeep`, `results/.gitkeep` placeholders were missing from the zips delivered on 2026-10-12 (lost when the working copy was restored); recreated. CI now also installs `nbformat`.
- `pipeline/04_verify_flash_floods.py` (+ `tests/test_verify_flash.py`, `make verify-flash`): checks which severe storms of `ama_flood_inventory.csv` became flash floods. Per storm: time window from the row (optionally narrowed to the IMERG V07 rain peak over the AMA), first Sentinel-1 scene within `--max-lag-h` (72 h) compared with earlier scenes of the same relative orbit (VV <= -15 dB and drop <= -3 dB), optional Sentinel-2 MNDWI change on Cloud Score+ clear pixels; permanent water, slopes > 5 deg and patches < 10 px removed. Verdict Confirmada (>= 0.05 km2 new water) / No observada / Sin imagen útil goes to `ama_flood_inventory_verificado.csv`; rows verified by hand are never overwritten. `--export-masks` sends confirmed masks to Drive. Thresholds are starting values to calibrate; mock-tested only.
- `data/events/ama_flood_inventory.csv`: single flood inventory for the AMA. Joins the user's event list (18 rows: fluvial, plus the Ciudad del Este urban event) with the 170 severe-storm rows, flagged as probable flash floods because storms at the thesis threshold produce flash floods in the AMA (user's criterion, to verify with imagery). The five April-2024 rows of the user's list are merged into the matching storm days, so April 2024 appears once. Columns `verificacion_imagenes`, `imagen_usada`, `fecha_imagen`, `resultado` are for the imagery check.
- `rain_events_AMA_2018_2025.csv`: new column `inundacion_repentina_ama`. `rain_events_10min.py` sets it to "Probable" when a candidate reaches the threshold and to "Poco probable" when it does not.
- `data/events/rain_events_AMA_2018_2025.csv`: intense-rain event list for station 86218 (Aeropuerto Silvio Pettirossi). 30 events classified in Báez Ferreira (2020) (Mar 2018-Apr 2019) + 135 candidate days with >= 20 mm/day (Jan-Feb 2018, May 2019-Aug 2025, from tutiempo.net SYNOP daily totals) + 5 press-only events. Candidates are `Pendiente` until classified with DINAC 10-min data. The daily screen recovers 28 of the 30 thesis events; 55 % of screened days in the thesis period were events. Pluvial events: not for the Sentinel-1 inventory (see `ama_events_filtered.csv`), useful for validation and pluvial analysis.
- `pipeline/rain_events_10min.py` (+ `tests/test_rain_events.py`): applies the thesis Tabla 5 (3 / 5 / 10 mm in 10 min) to a 10-min CSV and can fill the `Pendiente` rows of the list. The event-separation rule (60 min dry gap) is an assumption; the thesis does not define one.
- `00a_gee_export_layers.py`: Sentinel-2 option for NDVI/NDWI (`OPTICAL_SENSOR = "sentinel2"`, now the default; `"landsat"` reproduces the paper). Uses `COPERNICUS/S2_SR_HARMONIZED` (B3/B4/B8/B11), Cloud Score+ masking (`S2_CLOUD_METHOD = "cloudscore"`, threshold `S2_CLOUDSCORE_MIN`) or the SCL band (`"scl"`). Flood windows are still excluded via `EXCLUDE_RANGES`; both sensors share one index code path.

- `tests/test_gee_scripts.py`: mock-executes `00a` (all sensor / cloud-mask / DEM / soil branches) and `02a`, asserts the export list and the GEE asset IDs. `docs/gee_assets.md` records which assets and bands were verified against the catalog.

### Changed
- Licensing split: code stays MIT (now naming Universidad Católica "Nuestra Señora de la Asunción" as co-holder); data products are CC BY-NC 4.0 (`LICENSE-DATA.md`). Added `docs/data_licenses.md` (upstream terms, pre-publication checklist) and `make license-text`.
- NDVI/NDWI default source is now Sentinel-2 (10/20 m bands averaged to the 30 m grid) instead of Landsat 8/9. Re-run `00a` and `00c`; rasters from the two sensors are not interchangeable.

### Fixed
- `00a`: `EXCLUDE_RANGES` was a 3-window placeholder; it now has one window per AMA event in the full event list (2019 cycle, Jul 2021, Jan-Jun 2023, Apr 2024, Jun 2025). Jul 2021 and Apr 2024 were missing, so their floods could have leaked into the NDVI/NDWI composite. Windows come from the new `exclude_from_composite`, `exclude_start`, `exclude_end` columns of `data/events/ama_events_filtered.csv`, and `tests/test_gee_scripts.py::test_exclude_ranges_match_events_table` fails if the script and the table disagree. The Jul 2021 and Jun 2025 windows are placeholders until the Asuncion gauge confirms the dates.
- `00a`: default DEM moved from `JAXA/ALOS/AW3D30/V3_2` (marked superseded in the GEE catalog) to `V4_1`; same `DSM` band. The V4_1 update concerns Antarctica according to the catalog citation, so Paraguay values should be essentially unchanged (not compared).

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
