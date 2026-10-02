"""Step 0a - export the GEE-derived flood indicators for the Asuncion Metropolitan Area (30 m, EPSG:32721).

Exports to Google Drive (folder AMA_layers):
  elevation, slope, aspect, relief_amplitude      <- DEM (ALOS AW3D30 like the paper, or Copernicus GLO-30)
  ndvi, ndwi                                      <- Sentinel-2 SR (default) or Landsat 8/9 SR median composite, see OPTICAL_SENSOR
  rainfall                                        <- maximum 15-day accumulated rainfall in the period (CHIRPS, ~5 km)
  lulc                                            <- ESA WorldCover 2021 (10 m) -> mode at 30 m  (replace with a national LULC if you have one)
  soil_openlandmap (optional, coarse 250 m)       <- only a fallback if you have no national soil map
TWI and stream density are NOT here: they need flow routing, done locally in 00b.
Geology has no global source: rasterize your national map in 00c.

NOT TESTED (needs your GEE account). Run `earthengine authenticate` once. Missing values are exported as -9999.
Layers are exported over AOI + BUFFER_M so flow accumulation in 00b is not truncated at the AOI edge; 00c clips to the AOI.
"""
import datetime as dt
import ee

PROJECT = "your-gee-cloud-project"          # <- EDIT
ee.Initialize(project=PROJECT)

CRS, SCALE, FOLDER = "EPSG:32721", 30, "AMA_layers"
AOI = ee.Geometry.Rectangle([-57.75, -25.55, -57.30, -25.05])    # <- EDIT: placeholder box; e.g. ee.FeatureCollection('projects/<p>/assets/AMA').geometry()
BUFFER_M = 10000
REGION = AOI.buffer(BUFFER_M).bounds()

DEM_SOURCE = "ALOS"                          # "ALOS" (paper) or "COPERNICUS" (usually better vertical accuracy)
RELIEF_RADIUS_M = 300                        # focal window for relief amplitude (paper does not state it)
OPTICAL_SENSOR = "sentinel2"                 # "sentinel2" (10/20 m bands, ~5-day revisit) or "landsat" (30 m, as in the paper)
S2_CLOUD_METHOD = "cloudscore"               # "cloudscore" (Cloud Score+, recommended) or "scl" (scene-classification band)
S2_CLOUDSCORE_MIN = 0.6                      # keep pixels with cs_cdf >= this (higher = stricter)
S2_MAX_SCENE_CLOUD_PCT = 70                  # drop scenes cloudier than this before compositing
OPTICAL_START, OPTICAL_END = "2019-01-01", "2025-12-31"   # Sentinel-2 L2A is global from ~Dec 2018
# Exclude flood windows from the optical composite, otherwise NDWI/NDVI "see" the flood you are trying to predict:
EXCLUDE_RANGES = [("2019-03-01", "2020-03-01"), ("2023-01-01", "2023-06-30"), ("2025-05-15", "2025-07-15")]
NDWI_DEF = "paper"                           # "paper" = (NIR-SWIR1)/(NIR+SWIR1) [Eq.2, really NDMI]; "mcfeeters" = (G-NIR); "xu" = (G-SWIR1)
RAIN_START, RAIN_END, RAIN_WINDOW_DAYS, RAIN_STEP_DAYS = "2019-01-01", "2025-12-31", 15, 5
SOIL_OPENLANDMAP = False


def export(img, name, bilinear=True):
    img = img.toFloat()
    if bilinear:
        img = img.resample("bilinear")
    ee.batch.Export.image.toDrive(image=img.unmask(-9999), description=name, folder=FOLDER, fileNamePrefix=name,
                                  region=REGION, crs=CRS, scale=SCALE, maxPixels=1e10, fileFormat="GeoTIFF").start()
    print("started:", name)


# ------------------------------------------------------------------ DEM ---
if DEM_SOURCE == "ALOS":
    col = ee.ImageCollection("JAXA/ALOS/AW3D30/V3_2").select("DSM")
else:
    col = ee.ImageCollection("COPERNICUS/DEM/GLO30").select("DEM")
dem = col.mosaic().setDefaultProjection(col.first().projection()).rename("elevation")

export(dem, "elevation")
export(ee.Terrain.slope(dem), "slope")
export(ee.Terrain.aspect(dem), "aspect", bilinear=False)           # angle: never interpolate
relief = (dem.focalMax(RELIEF_RADIUS_M, "square", "meters")
             .subtract(dem.focalMin(RELIEF_RADIUS_M, "square", "meters")))
export(relief, "relief_amplitude")

# ------------------------------------------------------- NDVI / NDWI ---
# Both sensors are renamed to the same band names so the indices below do not depend on the sensor.
COMMON = ["green", "red", "nir", "swir1"]


def prep_landsat(img):
    qa = img.select("QA_PIXEL")
    clear = (qa.bitwiseAnd(1 << 1).eq(0)        # dilated cloud
             .And(qa.bitwiseAnd(1 << 3).eq(0))  # cloud
             .And(qa.bitwiseAnd(1 << 4).eq(0))) # cloud shadow
    sr = img.select(["SR_B3", "SR_B4", "SR_B5", "SR_B6"]).multiply(0.0000275).add(-0.2)
    return sr.rename(COMMON).updateMask(clear)


def prep_s2_scl(img):
    scl = img.select("SCL")
    clear = scl.neq(1).And(scl.neq(3)).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11))
    return img.select(["B3", "B4", "B8", "B11"]).divide(10000).rename(COMMON).updateMask(clear)


def prep_s2_cs(img):
    clear = img.select("cs_cdf").gte(S2_CLOUDSCORE_MIN)
    return img.select(["B3", "B4", "B8", "B11"]).divide(10000).rename(COMMON).updateMask(clear)


if OPTICAL_SENSOR == "landsat":
    opt = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").merge(ee.ImageCollection("LANDSAT/LC09/C02/T1_L2"))
           .filterBounds(REGION).filterDate(OPTICAL_START, OPTICAL_END))
    prep = prep_landsat
elif OPTICAL_SENSOR == "sentinel2":
    opt = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
           .filterBounds(REGION).filterDate(OPTICAL_START, OPTICAL_END)
           .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", S2_MAX_SCENE_CLOUD_PCT)))
    if S2_CLOUD_METHOD == "cloudscore":
        cs_plus = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
        opt = opt.linkCollection(cs_plus, ["cs_cdf"])
        prep = prep_s2_cs
    else:
        prep = prep_s2_scl
else:
    raise ValueError("OPTICAL_SENSOR must be 'sentinel2' or 'landsat'")

for a, b in EXCLUDE_RANGES:
    opt = opt.filter(ee.Filter.date(a, b).Not())
comp = opt.map(prep).median()
print(f"{OPTICAL_SENSOR} scenes in composite:", opt.size().getInfo())

# Sentinel-2 bands are 10/20 m: EE averages them down to the 30 m export grid (no bilinear resample needed)
ndvi = comp.normalizedDifference(["nir", "red"])
nd_pair = {"paper": ["nir", "swir1"], "mcfeeters": ["green", "nir"], "xu": ["green", "swir1"]}[NDWI_DEF]
ndwi = comp.normalizedDifference(nd_pair)
export(ndvi, "ndvi", bilinear=(OPTICAL_SENSOR == "landsat"))
export(ndwi, "ndwi", bilinear=(OPTICAL_SENSOR == "landsat"))

# -------------------------------------------------------------- rainfall ---
# Interpretation of the paper's "5 years, maximum of 15 days": max 15-day accumulated rainfall in the period.
chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").select("precipitation")
d0, d_end = dt.date.fromisoformat(RAIN_START), dt.date.fromisoformat(RAIN_END)
wins, d = [], d0
while d + dt.timedelta(days=RAIN_WINDOW_DAYS) <= d_end:
    e = d + dt.timedelta(days=RAIN_WINDOW_DAYS)
    wins.append(chirps.filterDate(d.isoformat(), e.isoformat()).sum())
    d += dt.timedelta(days=RAIN_STEP_DAYS)
rain = ee.ImageCollection.fromImages(wins).max().rename("rainfall")
export(rain, "rainfall")                       # CHIRPS is ~5 km: expect only a handful of distinct values over the AMA

# ------------------------------------------------------------------ LULC ---
wc = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")
lulc = (wc.reduceResolution(reducer=ee.Reducer.mode(), maxPixels=1024)
          .reproject(crs=CRS, scale=SCALE).rename("lulc"))
export(lulc, "lulc", bilinear=False)

if SOIL_OPENLANDMAP:
    soil = ee.Image("OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02").select("b0").rename("soil")
    export(soil, "soil_openlandmap", bilinear=False)

print("Monitor tasks at https://code.earthengine.google.com/tasks ; then download the GeoTIFFs from Drive/AMA_layers into one folder.")
