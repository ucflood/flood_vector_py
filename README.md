# AMA flood susceptibility (TensorFlow)

Fluvial flood susceptibility map for the Asuncion Metropolitan Area (AMA), combining
(i) a Sentinel-1 flood-water segmentation network trained on **Sen1Floods11** and
(ii) the **1D-CNN + 12 flood indicators** model of Trong et al. (2023).

```
GEE (00a)  DEM, slope, aspect, relief, NDVI, NDWI, rainfall, LULC ──┐
DEM -> 00b TWI, stream density ──────────────────────────────────────┤
national geology / soil vectors ─────────────────────────────────────┴─> 00c  12-layer stack (30 m)
                                                                                         │
Sen1Floods11 ─> 01 train U-Net ─> 02 apply to AMA S1 scenes (02a export) ─> flood polygons ─┴─> 03 1D-CNN ─> susceptibility map
```

## Setup
```bash
git clone https://github.com/ucflood/flood_vector_py && cd ama-flood-susceptibility
conda env create -f environment.yml && conda activate ama-flood     # or: pip install -r requirements.txt
earthengine authenticate                                             # only for 00a / 02a
make test
```
Colab: `pip install -r requirements.txt` (use the GPU runtime for stage 1).

## Run order
| step | script | what it does |
|---|---|---|
| 0a | `pipeline/00a_gee_export_layers.py` | exports DEM derivatives, Sentinel-2 (or Landsat) NDVI/NDWI, CHIRPS rainfall, WorldCover LULC (edit `PROJECT`, `AOI`, `EXCLUDE_RANGES`, `OPTICAL_SENSOR`) |
| 0b | `make hydro` | TWI + stream density (pysheds D8) |
| 0c | `make stack ...` | align, clip to AMA, rasterise geology/soil, QA report |
| 1 | `make sen1floods11-data && make train` | U-Net on Sen1Floods11 hand-labelled chips |
| 2a | `pipeline/02a_gee_export_s1.py` | export S1 pre/post scenes + JRC permanent water for each event |
| 2 | `make inventory EVENT=... POST=... PRE=... PERM=...` | water(post) - water(pre) - permanent water -> polygons |
| 3 | `make susceptibility` | 1D-CNN (71,088 params), spatial-block validation, LR/RF baselines, maps |

Event selection and dates: `data/events/ama_events_filtered.csv`. Details and limitations: `docs/methodology.md`.

## Outputs (`results/`)
`checkpoints/unet_s1f11_best.keras`, `inventory/flood_<event>.gpkg`, `susceptibility/susceptibility.tif` (0-1),
`susceptibility_class.tif` (1-5), `cnn1d_flood.keras`, `preproc.json` (scalers + frequency-ratio tables).

## References
* Trong, N.G. et al. (2023). Spatial Prediction of Fluvial Flood in High-Frequency Tropical Cyclone Area Using TensorFlow 1D-Convolution Neural Networks and Geospatial Data. *Remote Sensing* 15(22), 5429.
* Bonafilia, D., Tellman, B., Anderson, T., Issenberg, E. (2020). Sen1Floods11: a georeferenced dataset to train and test deep learning flood algorithms for Sentinel-1. *CVPR Workshops*.

## Status
All stages except the two GEE scripts are covered by synthetic-data tests (`pytest`). Not yet validated on real Asuncion data.
Changes to the code are tracked in `CHANGELOG.md` (update it in the same commit as each change).
License: MIT (replace `<YOUR NAME>` in `LICENSE`).
