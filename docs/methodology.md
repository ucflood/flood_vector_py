# Methodology notes and known limitations

## Why three stages
Sen1Floods11 is *image segmentation* (S1 VV/VH chips -> water mask). The Trong et al. (2023) 1D-CNN is *tabular*
(12 indicators per pixel -> flood probability). The segmentation net produces the flood inventory the tabular model needs.

## Design decisions that differ from the paper
* **Spatial split.** Train/val/test are split by 2 km blocks, not randomly (random pixel splits leak through spatial autocorrelation and inflate AUC).
* **Negatives** are sampled at least `--buffer-px` pixels from any flood pixel, optionally restricted to a plausible zone (`--zone`).
* **Categorical layers** (aspect, geology, LULC, soil) use frequency ratio fitted on training samples only.
* **Baselines** (LR, RF) are reported next to the 1D-CNN. The convolution slides over 12 features in an arbitrary order, so the
  "local receptive field" argument does not really apply; if RF is as good, say so.
* **Probabilities** from 50/50 balanced samples are a relative index, not a calibrated flood probability. Class breaks are arbitrary.

## Data caveats for the AMA
* Sentinel-1 captures long-lasting fluvial floods (Rio Paraguay) but almost surely misses hours-long urban flash floods (Apr 2024) and is unreliable in dense urban areas.
* The Rio Paraguay is permanent water: stage 2 subtracts JRC GSW occurrence and a low-water reference scene (same orbit).
* Exclude flood windows from the optical (Sentinel-2/Landsat) composite (`EXCLUDE_RANGES` in `00a`) so NDVI/NDWI do not contain the floods being predicted.
* CHIRPS (~5 km) gives only a handful of rainfall values over the AMA; `00c` warns about low-information layers.
* D8 TWI is unreliable on very flat floodplains (Banados).
* NDVI/NDWI: Sentinel-2 (default) uses B8 (10 m) and B11 (20 m), Landsat uses B5/B6 (30 m); both are averaged to the 30 m grid. Do not mix sensors between runs.
* The paper's "NDWI" (Eq. 2) is (NIR-SWIR1)/(NIR+SWIR1), closer to NDMI. Choose via `NDWI_DEF` in `00a`.

## Testing status
`pytest` runs every stage except the two Google Earth Engine scripts (`00a`, `02a`) on synthetic data. The GEE scripts need your account and are untested.
