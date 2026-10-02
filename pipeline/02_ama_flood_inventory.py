"""Stage 2 - apply the Sen1Floods11-trained network to Asunción Sentinel-1 scenes and build a flood inventory.

For each event:   flood = water(post) AND NOT water(pre / low-water reference) AND NOT permanent water
Output: probability GeoTIFF per scene + flood polygons (GeoPackage) = the inventory used in stage 3.

  python 02_ama_flood_inventory.py --model checkpoints/unet_s1f11_best.keras \
      --event 2019_05 --post S1_2019_flood.tif --pre S1_2019_ref_lowwater.tif \
      --perm-water GSW_occurrence.tif --out-dir inventory
"""
import argparse, os
import numpy as np
import rasterio
from rasterio import features
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from scipy import ndimage as ndi
import geopandas as gpd
from shapely.geometry import shape
from tensorflow import keras
from s1_common import preprocess_s1


def predict_water_prob(model, tif, out_tif, tile=512, stride=384, batch=4):
    """Sliding-window inference over a large scene; overlapping tiles are averaged."""
    with rasterio.open(tif) as src:
        x_db, prof = src.read(), src.profile
    nodata = np.all(np.isnan(x_db), axis=0) | np.all(x_db == 0, axis=0)
    x = preprocess_s1(x_db)                                   # (H, W, 2)
    H, W = x.shape[:2]
    Hp, Wp = max(H, tile), max(W, tile)
    xp = np.zeros((Hp, Wp, 2), np.float32); xp[:H, :W] = x
    prob, cnt = np.zeros((Hp, Wp), np.float32), np.zeros((Hp, Wp), np.float32)

    ys = sorted(set(list(range(0, Hp - tile + 1, stride)) + [Hp - tile]))
    xs = sorted(set(list(range(0, Wp - tile + 1, stride)) + [Wp - tile]))
    pos = [(i, j) for i in ys for j in xs]
    for k in range(0, len(pos), batch):
        chunk = pos[k:k + batch]
        out = model.predict(np.stack([xp[i:i + tile, j:j + tile] for i, j in chunk]), verbose=0)
        p = np.exp(out - out.max(-1, keepdims=True)); p = (p / p.sum(-1, keepdims=True))[..., 1]
        for (i, j), pi in zip(chunk, p):
            prob[i:i + tile, j:j + tile] += pi; cnt[i:i + tile, j:j + tile] += 1
    prob = (prob / np.maximum(cnt, 1))[:H, :W]
    prob[nodata] = np.nan

    prof.update(count=1, dtype="float32", nodata=np.nan, compress="deflate")
    with rasterio.open(out_tif, "w", **prof) as dst:
        dst.write(prob.astype(np.float32), 1)
    return prob


def read_on_grid(path, ref_path, resampling=Resampling.bilinear):
    """Read `path` resampled onto the grid of `ref_path`."""
    with rasterio.open(ref_path) as ref, rasterio.open(path) as src:
        with WarpedVRT(src, crs=ref.crs, transform=ref.transform, width=ref.width, height=ref.height,
                       resampling=resampling) as vrt:
            return vrt.read(1).astype(np.float32)


def remove_small(mask, min_px):
    lab, n = ndi.label(mask, structure=np.ones((3, 3)))
    if n == 0:
        return mask
    sizes = np.bincount(lab.ravel())
    keep = sizes >= min_px; keep[0] = False
    return keep[lab]


def polygonize(mask, ref_tif, event_id):
    with rasterio.open(ref_tif) as src:
        tr, crs = src.transform, src.crs
    geoms = [shape(g) for g, v in features.shapes(mask.astype(np.uint8), mask=mask, transform=tr) if v == 1]
    return gpd.GeoDataFrame({"event": [event_id] * len(geoms)}, geometry=geoms, crs=crs)


def main(a):
    os.makedirs(a.out_dir, exist_ok=True)
    model = keras.models.load_model(a.model, compile=False)
    post = predict_water_prob(model, a.post, os.path.join(a.out_dir, f"{a.event}_post_prob.tif"))
    water_post = np.nan_to_num(post) > a.thr

    flood = water_post.copy()
    if a.pre:
        pre = predict_water_prob(model, a.pre, os.path.join(a.out_dir, f"{a.event}_pre_prob.tif"))
        flood &= ~(np.nan_to_num(pre) > a.thr)
    if a.perm_water:
        occ = read_on_grid(a.perm_water, a.post)               # JRC occurrence 0-100
        flood &= ~(occ >= a.perm_thr)
    flood = remove_small(flood, a.min_px)

    gdf = polygonize(flood, a.post, a.event)
    if gdf.empty:
        print(f"{a.event}: no flood pixels left after masking - check dates, orbit, --thr and the pre/post scenes")
        return
    out = os.path.join(a.out_dir, f"flood_{a.event}.gpkg")
    gdf.to_file(out, driver="GPKG")
    print(f"{a.event}: {flood.sum()} flood px ({flood.sum() * 100 / 1e6:.2f} km2), {len(gdf)} polygons -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--event", required=True, help="event id, e.g. 2019_05")
    p.add_argument("--post", required=True, help="S1 GeoTIFF (VV,VH dB) during the flood")
    p.add_argument("--pre", help="S1 GeoTIFF at low water / before the event (same orbit as --post)")
    p.add_argument("--perm-water", help="JRC GSW occurrence raster (0-100)")
    p.add_argument("--perm-thr", type=float, default=50)
    p.add_argument("--thr", type=float, default=0.5)
    p.add_argument("--min-px", type=int, default=20, help="drop flood blobs smaller than this (10 m pixels)")
    p.add_argument("--out-dir", default="inventory")
    main(p.parse_args())
