"""Stage 1d - fine-tune the Sen1Floods11 network on your hand-corrected Asuncion chips (made with 01c).

  python 01d_finetune_ama.py --model results/checkpoints_v2/unet_s1f11_best.keras \\
      --chips-dir data/labels/2019_05 data/labels/2025_06 --val-events 2025_06 \\
      --s1f11-train-csv data/raw/sen1floods11/flood_train_data.csv \\
      --s1f11-valid-csv data/raw/sen1floods11/flood_valid_data.csv \\
      --s1-dir data/raw/sen1floods11/files/S1 --lbl-dir data/raw/sen1floods11/files/Labels \\
      --out-dir results/checkpoints_ama

What it does
  1. Rasterises your edited polygons (layer `water` of labels.gpkg; optional ignore.gpkg) to label chips:
     1 = inside a water polygon, 0 = elsewhere, -1 (ignored) = `ignore` polygons and no-data pixels.
  2. Fine-tunes ALL weights with a small learning rate (cosine decay), mixing in Sen1Floods11 batches (--replay-weight)
     so that the network does not forget what water looks like elsewhere.
  3. Validates on the held-out Asuncion chips (--val-events: whole events held out; with --val-frac the split is random
     and neighbouring chips of one scene leak into each other, so the number is optimistic).
  4. Saves unet_ama_best.keras ONLY if it beats the starting model on the Asuncion validation chips, and prints
     before/after IoU on Asuncion and on Sen1Floods11 validation (to see forgetting).
Then run 01b_calibrate_threshold.py on the Asuncion validation CSV (written to --out-dir) to get the new threshold:
  python 01b_calibrate_threshold.py --model <out-dir>/unet_ama_best.keras --valid-csv <out-dir>/ama_valid.csv \\
      --test-csv "" --s1-dir "" --lbl-dir "" --out <out-dir>/threshold_ama.json
"""
import argparse, glob, importlib.util, os, pathlib, random
import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
import tensorflow as tf
from rasterio import features
from tensorflow import keras
from s1_common import make_masked_ce_dice, FloodIoU, MaskedAccuracy

HERE = pathlib.Path(__file__).resolve().parent


def _load_stage1():
    spec = importlib.util.spec_from_file_location("train01", HERE / "01_train_sen1floods11.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def rasterize_chips(chips_dir):
    """Edited polygons -> label GeoTIFFs (int16: 1 water, 0 no water, -1 ignore). Returns [(s1_path, label_path, chip_id)]."""
    d = pathlib.Path(chips_dir)
    water = gpd.read_file(d / "labels.gpkg", layer="water")
    ignore = gpd.read_file(d / "ignore.gpkg") if (d / "ignore.gpkg").exists() else None
    (d / "Labels").mkdir(exist_ok=True)
    out = []
    for s1 in sorted(glob.glob(str(d / "S1" / "*_S1.tif"))):
        cid = pathlib.Path(s1).name[:-len("_S1.tif")]
        with rasterio.open(s1) as src:
            x, tr, shape, crs = src.read(), src.transform, (src.height, src.width), src.crs
        lab = np.zeros(shape, np.int16)
        w = water[water.chip_id == cid] if "chip_id" in water else water
        if len(w):
            lab[features.rasterize([(g, 1) for g in w.to_crs(crs).geometry if g is not None and not g.is_empty],
                                   out_shape=shape, transform=tr, fill=0, dtype="int16") == 1] = 1
        if ignore is not None and len(ignore):
            m = features.rasterize([(g, 1) for g in ignore.to_crs(crs).geometry if g is not None and not g.is_empty],
                                   out_shape=shape, transform=tr, fill=0, dtype="uint8")
            lab[m == 1] = -1
        lab[np.all(np.isnan(x), axis=0) | np.all(x == 0, axis=0)] = -1
        lp = d / "Labels" / f"{cid}_Label.tif"
        with rasterio.open(lp, "w", driver="GTiff", height=shape[0], width=shape[1], count=1, dtype="int16",
                           crs=crs, transform=tr, compress="deflate") as dst:
            dst.write(lab, 1)
        out.append((s1, str(lp), cid))
    return out


def split(items, val_events, val_frac, seed):
    if val_events:
        va = [t for t in items if any(t[2].startswith(e + "_") for e in val_events)]
        tr = [t for t in items if t not in va]
    else:
        print("WARNING: no --val-events: random split; chips from the same scene are neighbours, validation is optimistic")
        idx = list(range(len(items))); random.Random(seed).shuffle(idx)
        k = max(1, int(round(len(items) * val_frac)))
        va = [items[i] for i in idx[:k]]; tr = [items[i] for i in idx[k:]]
    if not tr or not va:
        raise SystemExit(f"need chips in both train ({len(tr)}) and validation ({len(va)})")
    return tr, va


def write_csv(path, items):
    pd.DataFrame([(s, l) for s, l, _ in items]).to_csv(path, header=False, index=False)    # absolute paths


def main(a):
    t1 = _load_stage1()
    os.makedirs(a.out_dir, exist_ok=True)
    items = [t for d in a.chips_dir for t in rasterize_chips(d)]
    print(f"{len(items)} chips from {len(a.chips_dir)} folder(s)")
    tr, va = split(items, a.val_events, a.val_frac, a.seed)
    ctr, cva = os.path.join(a.out_dir, "ama_train.csv"), os.path.join(a.out_dir, "ama_valid.csv")
    write_csv(ctr, tr); write_csv(cva, va)
    Xtr, Ytr = t1.load_split(ctr, "", ""); Xva, Yva = t1.load_split(cva, "", "")
    steps = a.steps
    ama_train = t1.make_ds(Xtr, Ytr, a.batch, a.crop, train=True)
    ama_valid = t1.make_ds(Xva, Yva, 4)
    train_ds, s1_valid = ama_train, None
    if a.s1f11_train_csv and os.path.exists(a.s1f11_train_csv):
        Xs, Ys = t1.load_split(a.s1f11_train_csv, a.s1_dir, a.lbl_dir)
        train_ds = tf.data.Dataset.sample_from_datasets(
            [ama_train, t1.make_ds(Xs, Ys, a.batch, a.crop, train=True)], [1 - a.replay_weight, a.replay_weight])
    if a.s1f11_valid_csv and os.path.exists(a.s1f11_valid_csv):
        Xsv, Ysv = t1.load_split(a.s1f11_valid_csv, a.s1_dir, a.lbl_dir)
        s1_valid = t1.make_ds(Xsv, Ysv, 4)

    model = keras.models.load_model(a.model, compile=False)
    lr = keras.optimizers.schedules.CosineDecay(a.lr, decay_steps=steps * a.epochs, alpha=0.05)
    model.compile(optimizer=keras.optimizers.AdamW(lr), loss=make_masked_ce_dice((1.0, a.water_weight), a.dice_weight),
                  metrics=[FloodIoU(), MaskedAccuracy()])
    before = model.evaluate(ama_valid, verbose=0, return_dict=True)
    before_s1 = model.evaluate(s1_valid, verbose=0, return_dict=True) if s1_valid is not None else None

    ckpt = os.path.join(a.out_dir, "unet_ama_best.keras")
    if os.path.exists(ckpt):
        os.remove(ckpt)
    cbs = [keras.callbacks.ModelCheckpoint(ckpt, monitor="val_flood_iou", mode="max", save_best_only=True, verbose=0,
                                           initial_value_threshold=before["flood_iou"]),
           keras.callbacks.CSVLogger(os.path.join(a.out_dir, "history_ama.csv"))]
    model.fit(train_ds, validation_data=ama_valid, epochs=a.epochs, steps_per_epoch=steps, callbacks=cbs, verbose=2)

    print(f"\nAsuncion validation chips ({len(Xva)}), flood IoU at 0.5: before {before['flood_iou']:.3f}", end="")
    if os.path.exists(ckpt):
        best = keras.models.load_model(ckpt, compile=False)
        best.compile(loss=make_masked_ce_dice((1.0, a.water_weight), a.dice_weight), metrics=[FloodIoU(), MaskedAccuracy()])
        after = best.evaluate(ama_valid, verbose=0, return_dict=True)
        print(f" -> after {after['flood_iou']:.3f}")
        if s1_valid is not None:
            a_s1 = best.evaluate(s1_valid, verbose=0, return_dict=True)
            print(f"Sen1Floods11 validation (forgetting check): before {before_s1['flood_iou']:.3f} -> after {a_s1['flood_iou']:.3f}")
        print(f"saved {ckpt}. Next: 01b_calibrate_threshold.py --model {ckpt} --valid-csv {cva}")
    else:
        print("\nno improvement over the starting model on the Asuncion validation chips: nothing saved. "
              "Label more chips, lower --lr, or raise --replay-weight.")
    return before


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="stage-1 network to start from (.keras)")
    p.add_argument("--chips-dir", nargs="+", required=True, help="folder(s) made by 01c_prepare_ama_chips.py and edited in QGIS")
    p.add_argument("--val-events", nargs="*", default=None, help="event ids held out for validation (e.g. 2025_06)")
    p.add_argument("--val-frac", type=float, default=0.25, help="only used without --val-events (random split, optimistic)")
    p.add_argument("--s1f11-train-csv", default="", help="enables replay of Sen1Floods11 batches")
    p.add_argument("--s1f11-valid-csv", default="", help="forgetting check")
    p.add_argument("--s1-dir", default="files/S1")
    p.add_argument("--lbl-dir", default="files/Labels")
    p.add_argument("--replay-weight", type=float, default=0.5, help="share of batches taken from Sen1Floods11")
    p.add_argument("--out-dir", default="results/checkpoints_ama")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--steps", type=int, default=20, help="steps per epoch")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--crop", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4, help="peak LR (the original training used 5e-4)")
    p.add_argument("--water-weight", type=float, default=8.0)
    p.add_argument("--dice-weight", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
