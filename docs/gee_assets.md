# Google Earth Engine assets used (`00a`, `02a`)

The scripts have **never been run live** (no GEE access where they were written). What was checked, and when:

| Asset | Used for | Bands / properties | Status |
|---|---|---|---|
| `JAXA/ALOS/AW3D30/V4_1` | DEM (default) | `DSM` | Checked in catalog 2026-10-02. V3_2 (used by the paper) is marked superseded by V4_1. |
| `COPERNICUS/DEM/GLO30` | DEM (alternative) | `DEM` | Not re-checked |
| `COPERNICUS/S2_SR_HARMONIZED` | NDVI/NDWI (default) | `B3 B4 B8 B11 SCL`, `CLOUDY_PIXEL_PERCENTAGE`, scale 0.0001 | Checked 2026-10-02 (plain `COPERNICUS/S2_SR` is the superseded one) |
| `GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED` | S2 cloud mask | `cs_cdf`, linked with `linkCollection` | Checked 2026-10-02 |
| `LANDSAT/LC08/C02/T1_L2`, `LANDSAT/LC09/C02/T1_L2` | NDVI/NDWI (paper option) | `SR_B3..B6`, `QA_PIXEL` | Not re-checked |
| `UCSB-CHG/CHIRPS/DAILY` | rainfall | `precipitation` | Not re-checked |
| `ESA/WorldCover/v200` | LULC | `Map` | Not re-checked |
| `OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02` | soil fallback | single `ee.Image`, band `b0` (0 cm) | Checked 2026-10-02 |
| `COPERNICUS/S1_GRD` | S1 scenes (02a, 04) | `VV VH` (dB), `instrumentMode`, `orbitProperties_pass`, `relativeOrbitNumber_start`, `transmitterReceiverPolarisation` | Checked 2026-10-12 |
| `JRC/GSW1_4/GlobalSurfaceWater` | permanent water (02a, 04) | `occurrence` | Checked 2026-10-12 |
| `NASA/GPM_L3/IMERG_V07` | storm timing (04) | `precipitation` (mm/h, 30 min) | Checked 2026-10-12; provisional after 2025-09-30 (provider moving to V08) |

Client side: `ee.batch.Export.image.toDrive` keyword arguments and `ee.Initialize(project=...)` were verified against the installed `earthengine-api` (1.7.46).
`tests/test_gee_scripts.py` mock-executes both scripts (all sensor / cloud-mask / DEM branches) and asserts the asset IDs above.

**First live run:** start with a tiny AOI, check the printed scene counts, open the exported GeoTIFFs and look at them before running `00b`/`00c`.
