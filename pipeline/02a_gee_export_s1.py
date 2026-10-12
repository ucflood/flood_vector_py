"""Export Sentinel-1 GRD (VV, VH in dB) for the Asunción Metropolitan Area from Google Earth Engine.
Sen1Floods11 was built from the same GEE collection, so using it keeps the input distribution consistent.
Mock-tested (tests/test_gee_scripts.py) and checked against the GEE catalog, but never run live. Run `earthengine authenticate` first.

Settings: GEE_PROJECT, AMA_AOI_ASSET, GEE_DRIVE_FOLDER (default AMA_S1), GEE_WAIT=1, and S1_SCENES as JSON
{"name": ["start", "end", "ASCENDING|DESCENDING"], ...}; in Colab use %run after ee.Authenticate().
Use the SAME orbit direction / relative orbit for the pre (low-water) and post (flood) scene.
"""
import json
import os
import sys
if "__file__" in globals():                 # %run in Colab does not add the script folder to sys.path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gee_utils import init_ee, aoi_geometry, wait_requested, wait_tasks
import ee

PROJECT = os.environ.get("GEE_PROJECT", "your-gee-cloud-project")      # <- or set GEE_PROJECT
init_ee(PROJECT)

AOI = aoi_geometry(ee, [-57.75, -25.55, -57.30, -25.10])   # AMA_AOI_ASSET, else this placeholder box
FOLDER = os.environ.get("GEE_DRIVE_FOLDER", "AMA_S1")
SCENES = {   # name: (start, end, orbit pass)  -- placeholders; pick dates from the river gauge + S1 availability
    "2019_flood": ("2019-05-20", "2019-05-31", "DESCENDING"),
    "2019_ref_lowwater": ("2019-01-01", "2019-01-31", "DESCENDING"),
}
if os.environ.get("S1_SCENES"):
    SCENES = {k: tuple(v) for k, v in json.loads(os.environ["S1_SCENES"]).items()}
TASKS = []

for name, (d0, d1, orbit) in SCENES.items():
    col = (ee.ImageCollection("COPERNICUS/S1_GRD")
           .filterBounds(AOI).filterDate(d0, d1)
           .filter(ee.Filter.eq("instrumentMode", "IW"))
           .filter(ee.Filter.eq("orbitProperties_pass", orbit))
           .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
           .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")))
    print(name, "scenes:", col.size().getInfo())
    img = col.select(["VV", "VH"]).mosaic().clip(AOI)             # already in dB in GEE
    task = ee.batch.Export.image.toDrive(image=img, description=f"S1_{name}", folder=FOLDER,
                                         region=AOI, scale=10, crs="EPSG:32721", maxPixels=1e10)
    task.start()
    TASKS.append(task)

# Permanent water (needed in stage 2): JRC Global Surface Water occurrence, 0-100 %
gsw = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").unmask(0).clip(AOI)
task = ee.batch.Export.image.toDrive(image=gsw, description="GSW_occurrence", folder=FOLDER,
                                     region=AOI, scale=30, crs="EPSG:32721", maxPixels=1e10)
task.start()
TASKS.append(task)
print(f"files land in Drive/{FOLDER}")
if wait_requested():
    wait_tasks(TASKS)
