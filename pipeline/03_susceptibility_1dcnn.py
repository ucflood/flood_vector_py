"""Stage 3 - fluvial flood susceptibility with the 1D-CNN of Trong et al. (2023), Remote Sens. 15, 5429,
using the 12 flood indicators and the flood inventory produced in stage 2.

Inputs
  --stack-dir   folder with 12 co-registered GeoTIFFs (same CRS / grid / extent, ~30 m, projected e.g. EPSG:32721):
                elevation slope aspect relief_amplitude twi stream_density geology lulc soil ndvi ndwi rainfall
                (aspect in degrees, -1 = flat;  geology / lulc / soil = integer class codes)
  --inventory   one or more GeoPackages from stage 2 (flood polygons)
  --zone        (optional) raster, 1 = area where a non-flood sample is plausible (e.g. low HAND / floodplain)

  python 03_susceptibility_1dcnn.py --stack-dir stack --inventory inventory/flood_*.gpkg --out-dir results
"""
import argparse, json, os
import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
from rasterio import features
from scipy import ndimage as ndi
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, cohen_kappa_score, confusion_matrix, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

ORDER = ["elevation", "slope", "aspect", "relief_amplitude", "twi", "stream_density",
         "geology", "lulc", "soil", "ndvi", "ndwi", "rainfall"]
CATEGORICAL = ["aspect", "geology", "lulc", "soil"]


# ------------------------------------------------------------------ data ---
def load_stack(d):
    arrs, prof = {}, None
    for n in ORDER:
        with rasterio.open(os.path.join(d, f"{n}.tif")) as src:
            arrs[n] = src.read(1, masked=True).astype(np.float32).filled(np.nan)
            if prof is None:
                prof = src.profile
            else:
                assert src.transform == prof["transform"] and src.shape == (prof["height"], prof["width"]), \
                    f"{n}.tif is not on the same grid as {ORDER[0]}.tif"
    return arrs, prof


def aspect_to_class(a):
    """degrees (-1 flat) -> 0 flat, 1..8 = N, NE, E, SE, S, SW, W, NW"""
    c = 1 + (((a + 22.5) % 360) // 45)
    c = np.where(a < 0, 0, c)
    return np.where(np.isnan(a), np.nan, c)


def rasterize_inventory(paths, prof):
    gdf = gpd.GeoDataFrame(pd.concat([gpd.read_file(p) for p in paths], ignore_index=True))
    gdf = gdf.to_crs(prof["crs"])
    return features.rasterize(((g, 1) for g in gdf.geometry), out_shape=(prof["height"], prof["width"]),
                              transform=prof["transform"], fill=0, dtype="uint8")


def sample_pixels(flood, valid, zone, n_max, buffer_px, rng):
    """Positives = flood pixels; negatives = same number of pixels farther than buffer_px from any flood
    (and inside `zone` if given), so negatives are not trivially 'mountains'."""
    pos = np.flatnonzero((flood == 1) & valid)
    dist = ndi.distance_transform_edt(flood == 0)
    ok = valid & (dist > buffer_px)
    if zone is not None:
        ok &= zone
    neg = np.flatnonzero(ok)
    n = min(len(pos), len(neg), n_max)
    pos, neg = rng.choice(pos, n, replace=False), rng.choice(neg, n, replace=False)
    idx = np.concatenate([pos, neg])
    y = np.r_[np.ones(n, int), np.zeros(n, int)]
    return idx, y


def fr_table(vals, y):
    """Frequency ratio per class, computed on TRAINING samples only: (flood share in class) / (sample share in class)."""
    P, N, tbl = (y == 1).sum(), len(y), {}
    for c in np.unique(vals):
        m = vals == c
        tbl[float(c)] = float(((y[m] == 1).sum() / P) / (m.sum() / N))
    return tbl


def apply_fr(arr, tbl, default=1.0):
    out = np.full(arr.shape, default, np.float32)
    out[np.isnan(arr)] = np.nan
    for c, v in tbl.items():
        out[arr == c] = v
    return out


# ----------------------------------------------------------------- model ---
def build_1dcnn(n_feat=12):
    """Architecture from Section 3.3 / Fig. 5 of the paper (valid padding -> 71,088 parameters)."""
    m = keras.Sequential([
        keras.Input((n_feat, 1)),
        layers.Conv1D(32, 1, activation="relu"),
        layers.Conv1D(64, 3, activation="relu"),
        layers.MaxPooling1D(2),
        layers.Conv1D(64, 1, activation="relu"),
        layers.Conv1D(128, 3, activation="relu"),
        layers.MaxPooling1D(2),
        layers.Flatten(),
        layers.Dense(200, activation="relu"),
        layers.Dense(50, activation="relu"),
        layers.Dense(2, activation="sigmoid"),      # [non-flood, flood]
    ], name="cnn1d_flood")
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse", metrics=["accuracy"])   # ADAM + MSE as in the paper
    return m


def metrics(y, p, thr=0.5):
    yh = (p > thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yh).ravel()
    return dict(TP=tp, TN=tn, FP=fp, FN=fn, PPV=tp / (tp + fp), NPV=tn / (tn + fn), Sens=tp / (tp + fn),
                Spec=tn / (tn + fp), Acc=(tp + tn) / len(y), F=f1_score(y, yh), Kappa=cohen_kappa_score(y, yh),
                AUC=roc_auc_score(y, p))


def show(name, m):
    print(f"{name:7s} " + "  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in m.items()))


# ------------------------------------------------------------------ main ---
def main(a):
    os.makedirs(a.out_dir, exist_ok=True)
    rng = np.random.default_rng(a.seed); tf.keras.utils.set_random_seed(a.seed)

    raw, prof = load_stack(a.stack_dir)
    raw["aspect"] = aspect_to_class(raw["aspect"])
    H, W = prof["height"], prof["width"]
    valid = np.all([np.isfinite(raw[n]) for n in ORDER], axis=0)
    flood = rasterize_inventory(a.inventory, prof)
    zone = None
    if a.zone:
        with rasterio.open(a.zone) as z:
            zone = z.read(1) == 1

    idx, y = sample_pixels(flood, valid, zone, a.max_pos, a.buffer_px, rng)
    r, c = np.unravel_index(idx, (H, W))
    print(f"samples: {len(y)} ({y.sum()} flood / {(1 - y).sum()} non-flood), valid px: {valid.sum()}")

    # spatial split in blocks (random pixel splits leak: neighbouring pixels are near-duplicates)
    block = (r // a.block_px).astype(np.int64) * 100003 + (c // a.block_px)
    tv, te = next(GroupShuffleSplit(1, test_size=0.15, random_state=a.seed).split(idx, y, block))
    tr, va = next(GroupShuffleSplit(1, test_size=0.15 / 0.85, random_state=a.seed).split(tv, y[tv], block[tv]))
    tr, va = tv[tr], tv[va]
    print(f"train/val/test = {len(tr)}/{len(va)}/{len(te)}")

    # categorical -> frequency ratio (fit on train only), then min-max to 0-1 (Eq. 5 of the paper)
    layers_ = {}
    tables = {}
    for n in ORDER:
        if n in CATEGORICAL:
            tables[n] = fr_table(raw[n][r[tr], c[tr]], y[tr])
            layers_[n] = apply_fr(raw[n], tables[n])
        else:
            layers_[n] = raw[n]
    scaler = {}
    for n in ORDER:
        lo, hi = float(np.nanmin(layers_[n][valid])), float(np.nanmax(layers_[n][valid]))
        scaler[n] = (lo, hi)
        layers_[n] = (layers_[n] - lo) / (hi - lo + 1e-12)
    json.dump({"order": ORDER, "scaler": scaler, "fr_tables": tables}, open(os.path.join(a.out_dir, "preproc.json"), "w"), indent=1)

    F = np.stack([layers_[n] for n in ORDER])                 # (12, H, W)
    X = F[:, r, c].T.astype(np.float32)                       # (n, 12)

    print("\nPearson correlation with flood occurrence (all samples):")
    for n, col in zip(ORDER, X.T):
        print(f"  {n:17s} {np.corrcoef(col, y)[0, 1]:+.3f}")

    model = build_1dcnn()
    print("\nparameters:", model.count_params(), "(paper: 71,088)")
    Y2 = keras.utils.to_categorical(y, 2)
    model.fit(X[tr, :, None], Y2[tr], validation_data=(X[va, :, None], Y2[va]), epochs=a.epochs, batch_size=a.batch,
              callbacks=[keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)], verbose=2)
    model.save(os.path.join(a.out_dir, "cnn1d_flood.keras"))

    print("\nHeld-out spatial blocks (test):")
    show("1D-CNN", metrics(y[te], model.predict(X[te, :, None], verbose=0)[:, 1]))
    for name, clf in [("LR", LogisticRegression(max_iter=1000)), ("RF", RandomForestClassifier(300, n_jobs=-1, random_state=a.seed))]:
        clf.fit(X[tr], y[tr]); show(name, metrics(y[te], clf.predict_proba(X[te])[:, 1]))

    # susceptibility map for every valid pixel
    Xall = F[:, valid].T.astype(np.float32)
    p = np.concatenate([model.predict(Xall[i:i + 200000, :, None], verbose=0)[:, 1] for i in range(0, len(Xall), 200000)])
    out = np.full((H, W), -9999, np.float32); out[valid] = p
    prof.update(count=1, dtype="float32", nodata=-9999, compress="deflate")
    with rasterio.open(os.path.join(a.out_dir, "susceptibility.tif"), "w", **prof) as d:
        d.write(out, 1)
    cls = np.zeros((H, W), np.uint8)                          # 1 very low ... 5 very high
    cls[valid] = 1 + np.digitize(p, a.breaks)
    prof.update(dtype="uint8", nodata=0)
    with rasterio.open(os.path.join(a.out_dir, "susceptibility_class.tif"), "w", **prof) as d:
        d.write(cls, 1)
    print("\nwrote susceptibility.tif and susceptibility_class.tif in", a.out_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stack-dir", required=True)
    p.add_argument("--inventory", nargs="+", required=True)
    p.add_argument("--zone")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--max-pos", type=int, default=5000)
    p.add_argument("--buffer-px", type=float, default=15, help="min distance of non-flood samples to flood (px)")
    p.add_argument("--block-px", type=int, default=67, help="spatial block size (67 px ~ 2 km at 30 m)")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--breaks", type=float, nargs=4, default=[0.2, 0.4, 0.6, 0.8])
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())
