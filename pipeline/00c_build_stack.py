"""Step 0c - assemble the 12-layer stack that 03_susceptibility_1dcnn.py expects.

 - puts every layer on ONE grid (the grid of elevation.tif), clipped to the AMA polygon
 - rasterises geology / soil from your national vector maps (integer class codes + a code table CSV)
 - aspect -> -1 where the terrain is flat
 - prints a QA table (coverage, range, number of distinct values) and warns about layers that carry little spatial information

  python 00c_build_stack.py --layers-dir layers --aoi AMA.gpkg \
      --geology geologia.shp --geology-field FORMACION --soil suelos.shp --soil-field SUELO --out-dir stack
Files looked up in --layers-dir: elevation slope aspect relief_amplitude twi stream_density ndvi ndwi rainfall lulc [geology soil]
(GEE exports use -9999 for missing; that is handled.)  If you have no soil vector, rename soil_openlandmap.tif to soil.tif.
"""
import argparse, os
import numpy as np
import pandas as pd
import rasterio
from rasterio import features
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
import geopandas as gpd

ORDER = ["elevation", "slope", "aspect", "relief_amplitude", "twi", "stream_density",
         "geology", "lulc", "soil", "ndvi", "ndwi", "rainfall"]
CATEGORICAL = ["aspect", "geology", "lulc", "soil"]
NEAREST = ["aspect", "geology", "lulc", "soil"]


def read_on_grid(path, ref, resampling):
    with rasterio.open(path) as src:
        nd = src.nodata if src.nodata is not None else np.nan
        with WarpedVRT(src, crs=ref.crs, transform=ref.transform, width=ref.width, height=ref.height,
                       resampling=resampling, nodata=nd) as vrt:
            a = vrt.read(1, masked=True).astype(np.float32).filled(np.nan)
    a[a <= -9998] = np.nan
    return a


def rasterize_vector(path, field, ref, name, out_dir):
    g = gpd.read_file(path).to_crs(ref.crs)
    cats = sorted(g[field].dropna().unique(), key=str)
    code = {c: i + 1 for i, c in enumerate(cats)}
    pd.DataFrame({"code": list(code.values()), field: list(code.keys())}).to_csv(
        os.path.join(out_dir, f"{name}_codes.csv"), index=False)
    shapes = ((geom, code[v]) for geom, v in zip(g.geometry, g[field]) if v in code)
    arr = features.rasterize(shapes, out_shape=(ref.height, ref.width), transform=ref.transform, fill=0, dtype="int16")
    arr = arr.astype(np.float32); arr[arr == 0] = np.nan
    return arr


def main(a):
    os.makedirs(a.out_dir, exist_ok=True)
    ref_path = os.path.join(a.layers_dir, "elevation.tif")
    with rasterio.open(ref_path) as ref:
        prof, H, W = ref.profile, ref.height, ref.width
        aoi = gpd.read_file(a.aoi).to_crs(ref.crs)
        aoi_mask = features.rasterize(((g, 1) for g in aoi.geometry), out_shape=(H, W), transform=ref.transform,
                                      fill=0, dtype="uint8").astype(bool)
        layers = {}
        for n in ORDER:
            path = os.path.join(a.layers_dir, f"{n}.tif")
            vec = {"geology": (a.geology, a.geology_field), "soil": (a.soil, a.soil_field)}.get(n)
            if vec and vec[0]:
                layers[n] = rasterize_vector(vec[0], vec[1], ref, n, a.out_dir)
            elif os.path.exists(path):
                layers[n] = read_on_grid(path, ref, Resampling.nearest if n in NEAREST else Resampling.bilinear)
            else:
                print(f"!! missing layer: {n}")
    missing = [n for n in ORDER if n not in layers]
    if missing:
        raise SystemExit(f"Cannot build the stack, missing: {missing}")

    if "slope" in layers:                                    # flat -> -1 (class 0 in stage 3)
        layers["aspect"] = np.where(layers["slope"] < a.flat_slope_deg, -1.0, layers["aspect"])

    prof.update(count=1, dtype="float32", nodata=np.nan, compress="deflate")
    rows = []
    for n in ORDER:
        arr = np.where(aoi_mask, layers[n], np.nan).astype(np.float32)
        with rasterio.open(os.path.join(a.out_dir, f"{n}.tif"), "w", **prof) as d:
            d.write(arr, 1)
        v = arr[aoi_mask & np.isfinite(arr)]
        rows.append(dict(layer=n, coverage_pct=100 * len(v) / aoi_mask.sum(), min=v.min() if len(v) else np.nan,
                         max=v.max() if len(v) else np.nan, std=v.std() if len(v) else np.nan,
                         n_distinct=len(np.unique(v)) if len(v) else 0))
    qa = pd.DataFrame(rows).set_index("layer")
    print(qa.round(3).to_string())

    for n, r in qa.iterrows():
        if r.coverage_pct < 99:
            print(f"WARNING {n}: only {r.coverage_pct:.1f}% of the AMA has data (those pixels will be dropped in stage 3)")
        if n not in CATEGORICAL and r.n_distinct < 50:
            print(f"WARNING {n}: only {int(r.n_distinct)} distinct values (coarse source?). Min-max scaling will stretch it to 0-1 and "
                  "blocky cells can act as a location fingerprint. Consider dropping it or using a finer source / station interpolation.")
        if n in CATEGORICAL and n != "aspect" and r.n_distinct < 4:
            print(f"WARNING {n}: only {int(r.n_distinct)} classes in the AMA, little discriminating power.")
    print("stack written to", a.out_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--layers-dir", required=True)
    p.add_argument("--aoi", required=True, help="AMA polygon (any format geopandas reads)")
    p.add_argument("--geology"); p.add_argument("--geology-field")
    p.add_argument("--soil"); p.add_argument("--soil-field")
    p.add_argument("--flat-slope-deg", type=float, default=1.0)
    p.add_argument("--out-dir", default="stack")
    main(p.parse_args())
