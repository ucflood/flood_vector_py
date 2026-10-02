"""Export Sentinel-1 GRD (VV, VH in dB) for the Asunción Metropolitan Area from Google Earth Engine.
Sen1Floods11 was built from the same GEE collection, so using it keeps the input distribution consistent.
NOT TESTED here (needs your GEE account): run `earthengine authenticate` first.

Edit PROJECT, AOI and SCENES. Use the SAME orbit direction / relative orbit for the pre (low-water) and post (flood) scene.
"""
import ee
PROJECT = "your-gee-cloud-project"   # <- EDIT
ee.Initialize(project=PROJECT)

AOI = ee.Geometry.Rectangle([-57.75, -25.55, -57.30, -25.10])   # placeholder box: replace with your AMA polygon
SCENES = {   # name: (start, end, orbit pass)  -- dates are placeholders, pick them from the river gauge + S1 availability
    "2019_flood": ("2019-05-20", "2019-05-31", "DESCENDING"),
    "2019_ref_lowwater": ("2019-01-01", "2019-01-31", "DESCENDING"),
}

for name, (d0, d1, orbit) in SCENES.items():
    col = (ee.ImageCollection("COPERNICUS/S1_GRD")
           .filterBounds(AOI).filterDate(d0, d1)
           .filter(ee.Filter.eq("instrumentMode", "IW"))
           .filter(ee.Filter.eq("orbitProperties_pass", orbit))
           .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
           .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")))
    print(name, "scenes:", col.size().getInfo())
    img = col.select(["VV", "VH"]).mosaic().clip(AOI)             # already in dB in GEE
    ee.batch.Export.image.toDrive(image=img, description=f"S1_{name}", folder="AMA_S1",
                                  region=AOI, scale=10, crs="EPSG:32721", maxPixels=1e10).start()

# Permanent water (needed in stage 2): JRC Global Surface Water occurrence, 0-100 %
gsw = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").unmask(0).clip(AOI)
ee.batch.Export.image.toDrive(image=gsw, description="GSW_occurrence", folder="AMA_S1",
                              region=AOI, scale=30, crs="EPSG:32721", maxPixels=1e10).start()
