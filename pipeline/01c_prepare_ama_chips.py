"""Stage 1c - cut 512x512 chips from Asuncion Sentinel-1 scenes and pre-label them with the stage-1 network,
so that you only have to CORRECT the water polygons in QGIS instead of drawing them from scratch.

  python 01c_prepare_ama_chips.py --model results/checkpoints_v2/unet_s1f11_best.keras --thr 0.7 \\
      --scene data/interim/S1_2019_flood.tif --event 2019_05 --out-dir data/labels/2019_05 \\
      --n-water 15 --n-other 8

Writes into --out-dir:
  S1/<event>_r<row>_c<col>_S1.tif   2-band chips (VV, VH in dB), same georeference as the scene
  labels.gpkg                       layers: `chips` (footprints) and `water` (the pre-label polygons: EDIT THIS ONE)
How the chips are chosen: `--n-water` chips that mix water and land (edges are what the network gets wrong) plus
`--n-other` chips with almost no predicted water (hard negatives: city, slopes, crops). Chips with > 5 % no-data are skipped.

Labelling in QGIS (about 5-10 min per chip):
  1. Open labels.gpkg (layers chips + water), a Sentinel-2 true-colour / MNDWI image of the SAME flood (or the closest
     clear date) and the S1 chips as reference. Style `chips` as outline only.
  2. Select layer `water` -> toggle editing. Delete polygons that are not water (radar shadow, wet crops, roofs),
     add the missing water with "Add polygon feature", reshape borders. Everything not inside a polygon becomes "no water".
  3. Doubtful areas (cannot tell): create a new GeoPackage layer `ignore.gpkg` (polygon, same CRS) in the same folder and draw
     over them; they are excluded from training and validation (label 255 = ignore, like in Sen1Floods11).
  4. Save edits. Then run 01d_finetune_ama.py.
Label permanent water too (the river is water): stage 2 removes it afterwards with the JRC layer.
Hold out one whole event (or one area) for validation: label it but pass it to 01d with --val-events.
"""
import argparse, importlib.util, os, pathlib
import numpy as np
import rasterio
import geopandas as gpd
from rasterio import features
from rasterio.windows import Window
from shapely.geometry import box, shape
from tensorflow import keras
from s1_common import preprocess_s1, predict_prob_array

HERE = pathlib.Path(__file__).resolve().parent
CHIP = 512


def _load_stage2():
    spec = importlib.util.spec_from_file_location("inv02", HERE / "02_ama_flood_inventory.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def choose_chips(prob, nodata, thr, n_water, n_other, min_water, max_water, rng, log=None):
    """Non-overlapping CHIP x CHIP grid -> (mixed, other) lists of (row, col, water_fraction)."""
    H, W = prob.shape
    mixed, other, fracs = [], [], []
    for r in range(0, H - CHIP + 1, CHIP):
        for c in range(0, W - CHIP + 1, CHIP):
            if nodata[r:r + CHIP, c:c + CHIP].mean() > 0.05:
                continue
            wf = float(np.nanmean(prob[r:r + CHIP, c:c + CHIP] > thr))
            fracs.append(wf)
            (mixed if min_water <= wf <= max_water else other if wf < min_water else []).append((r, c, wf))
    if log is not None:
        log.extend(fracs)
    mixed.sort(key=lambda t: t[2] * (1 - t[2]), reverse=True)            # most edge content first
    pool = mixed[: max(2 * n_water, n_water)]
    sel_w = [pool[i] for i in rng.choice(len(pool), min(n_water, len(pool)), replace=False)] if pool else []
    sel_o = [other[i] for i in rng.choice(len(other), min(n_other, len(other)), replace=False)] if other else []
    return sel_w, sel_o


def main(a):
    gpkg = os.path.join(a.out_dir, "labels.gpkg")
    if os.path.exists(gpkg) and not a.overwrite:
        raise SystemExit(f"{gpkg} already exists and may hold your hand edits: nothing done. "
                         "Use another --out-dir, or --overwrite to start this event again from the model's pre-label.")
    inv = _load_stage2()
    rng = np.random.default_rng(a.seed)
    model = keras.models.load_model(a.model, compile=False)
    os.makedirs(os.path.join(a.out_dir, "S1"), exist_ok=True)
    with rasterio.open(a.scene) as src:
        x_db, crs = src.read(), src.crs
        nodata = np.all(np.isnan(x_db), axis=0) | np.all(x_db == 0, axis=0)
        prob = predict_prob_array(model, preprocess_s1(x_db), tile=512, stride=384, tta=a.tta)
        prob[nodata] = np.nan
        fr = []
        sel_w, sel_o = choose_chips(prob, nodata, a.thr, a.n_water, a.n_other, a.min_water, a.max_water, rng, log=fr)
        if not (sel_w or sel_o):
            detail = (f"water fraction per chip: min {min(fr):.2f}, median {np.median(fr):.2f}, max {max(fr):.2f}; "
                      f"adjust --thr / --min-water / --max-water") if fr else "scene smaller than 512 px or > 5 % no-data in every chip"
            raise SystemExit(f"no usable chips ({detail})")
        chips, water = [], []
        for kind, sel in (("mixed", sel_w), ("other", sel_o)):
            for r, c, wf in sel:
                cid = f"{a.event}_r{r}_c{c}"
                win = Window(c, r, CHIP, CHIP)
                tr = src.window_transform(win)
                prof = dict(driver="GTiff", height=CHIP, width=CHIP, count=x_db.shape[0], dtype="float32",
                            crs=crs, transform=tr, compress="deflate")
                with rasterio.open(os.path.join(a.out_dir, "S1", f"{cid}_S1.tif"), "w", **prof) as dst:
                    dst.write(x_db[:, r:r + CHIP, c:c + CHIP].astype("float32"))
                chips.append(dict(chip_id=cid, kind=kind, water_frac=round(wf, 3)))
                mask = inv.remove_small(np.nan_to_num(prob[r:r + CHIP, c:c + CHIP]) > a.thr, a.min_px)
                for g, v in features.shapes(mask.astype("uint8"), mask=mask, transform=tr):
                    water.append(dict(chip_id=cid, geometry=shape(g).simplify(a.simplify_m)))
    # footprints (rebuilt cleanly from the written chips)
    rows = []
    for ch in chips:
        with rasterio.open(os.path.join(a.out_dir, "S1", f"{ch['chip_id']}_S1.tif")) as d:
            rows.append({**ch, "geometry": box(*d.bounds)})
    if os.path.exists(gpkg):
        os.remove(gpkg)
    gpd.GeoDataFrame(rows, geometry="geometry", crs=crs).to_file(gpkg, layer="chips", driver="GPKG")
    gw = gpd.GeoDataFrame(water, geometry="geometry", crs=crs) if water else \
        gpd.GeoDataFrame({"chip_id": []}, geometry=gpd.GeoSeries([], crs=crs), crs=crs)
    gw.to_file(gpkg, layer="water", driver="GPKG", mode="a")
    print(f"{a.event}: {len(sel_w)} mixed + {len(sel_o)} other chips, {len(gw)} pre-label polygons -> {a.out_dir}\n"
          f"edit layer `water` in {gpkg}, then run 01d_finetune_ama.py --chips-dir {a.out_dir}")


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--scene", required=True, help="S1 GeoTIFF (VV, VH in dB) of a flood date")
    p.add_argument("--event", required=True, help="id used in chip names, e.g. 2019_05")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--thr", type=float, default=0.7, help="threshold for the pre-label (use the one from 01b)")
    p.add_argument("--n-water", type=int, default=15)
    p.add_argument("--n-other", type=int, default=8)
    p.add_argument("--min-water", type=float, default=0.02, help="chips below this predicted water fraction count as 'other'")
    p.add_argument("--max-water", type=float, default=0.98)
    p.add_argument("--min-px", type=int, default=20, help="drop pre-label blobs smaller than this")
    p.add_argument("--simplify-m", type=float, default=5.0, help="polygon simplification (m) so they are easy to edit")
    p.add_argument("--tta", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--overwrite", action="store_true", help="replace an existing labels.gpkg (DISCARDS your QGIS edits)")
    return p.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
