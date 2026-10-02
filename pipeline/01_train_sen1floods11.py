"""Stage 1 - train a flood-water segmentation network on Sen1Floods11 (TensorFlow port of the Colab notebook).

Download the data first (public bucket, same as the notebook):
  gsutil cp gs://sen1floods11/v1.1/splits/flood_handlabeled/flood_{train,valid,test}_data.csv .
  gsutil -m rsync -r gs://sen1floods11/v1.1/data/flood_events/HandLabeled/S1Hand    files/S1
  gsutil -m rsync -r gs://sen1floods11/v1.1/data/flood_events/HandLabeled/LabelHand files/Labels

Run:  python 01_train_sen1floods11.py --epochs 100
"""
import argparse, csv, os
import numpy as np
import rasterio
import tensorflow as tf
from tensorflow import keras
from s1_common import (preprocess_s1, prep_label, build_unet, make_masked_wce,
                       FloodIoU, MaskedAccuracy, IGNORE)

CHIP = 512


def fit_size(a, value):
    """Crop / pad (H,W[,C]) array to CHIP x CHIP (Sen1Floods11 chips are 512 but a few differ)."""
    a = a[:CHIP, :CHIP]
    ph, pw = CHIP - a.shape[0], CHIP - a.shape[1]
    if ph or pw:
        pad = [(0, ph), (0, pw)] + [(0, 0)] * (a.ndim - 2)
        a = np.pad(a, pad, constant_values=value)
    return a


def load_split(csv_path, s1_dir, lbl_dir):
    X, Y = [], []
    with open(csv_path) as f:
        for row in csv.reader(f):
            ps, pl = os.path.join(s1_dir, row[0]), os.path.join(lbl_dir, row[1])
            if not (os.path.exists(ps) and os.path.exists(pl)):
                continue
            with rasterio.open(ps) as s, rasterio.open(pl) as l:
                x, y = s.read(), l.read(1)
            X.append(fit_size(preprocess_s1(x), 0.0))
            Y.append(fit_size(prep_label(y), IGNORE))
    print(f"{csv_path}: {len(X)} chips")
    return np.stack(X), np.stack(Y)


def make_ds(X, Y, batch, crop=None, train=False):
    def gen():
        idx = np.random.permutation(len(X)) if train else range(len(X))
        for i in idx:
            yield X[i], Y[i]
    ds = tf.data.Dataset.from_generator(
        gen, output_signature=(tf.TensorSpec((CHIP, CHIP, 2), tf.float32),
                               tf.TensorSpec((CHIP, CHIP), tf.uint8)))
    if train:
        def aug(x, y):
            xy = tf.concat([x, tf.cast(y, tf.float32)[..., None]], -1)
            xy = tf.image.random_crop(xy, (crop, crop, 3))
            xy = tf.image.random_flip_left_right(xy)
            xy = tf.image.random_flip_up_down(xy)
            return xy[..., :2], tf.cast(xy[..., 2], tf.uint8)
        ds = ds.map(aug, num_parallel_calls=tf.data.AUTOTUNE).repeat()
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


def main(a):
    Xtr, Ytr = load_split(a.train_csv, a.s1_dir, a.lbl_dir)
    Xva, Yva = load_split(a.valid_csv, a.s1_dir, a.lbl_dir)
    steps = max(1, len(Xtr) // a.batch)

    train_ds = make_ds(Xtr, Ytr, a.batch, a.crop, train=True)
    valid_ds = make_ds(Xva, Yva, 4)

    model = build_unet(base=a.base)
    lr = keras.optimizers.schedules.CosineDecayRestarts(a.lr, first_decay_steps=steps * 10, t_mul=2.0, alpha=0.0)
    model.compile(optimizer=keras.optimizers.AdamW(lr), loss=make_masked_wce((1.0, a.water_weight)),
                  metrics=[FloodIoU(), MaskedAccuracy()])

    os.makedirs(a.out_dir, exist_ok=True)
    ckpt = os.path.join(a.out_dir, "unet_s1f11_best.keras")
    cbs = [keras.callbacks.ModelCheckpoint(ckpt, monitor="val_flood_iou", mode="max", save_best_only=True, verbose=1),
           keras.callbacks.CSVLogger(os.path.join(a.out_dir, "history.csv"))]
    model.fit(train_ds, validation_data=valid_ds, epochs=a.epochs, steps_per_epoch=steps, callbacks=cbs)

    if os.path.exists(a.test_csv):
        Xte, Yte = load_split(a.test_csv, a.s1_dir, a.lbl_dir)
        best = keras.models.load_model(ckpt, compile=False)
        best.compile(loss=make_masked_wce((1.0, a.water_weight)), metrics=[FloodIoU(), MaskedAccuracy()])
        print("TEST:", best.evaluate(make_ds(Xte, Yte, 4), verbose=0, return_dict=True))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train-csv", default="flood_train_data.csv")
    p.add_argument("--valid-csv", default="flood_valid_data.csv")
    p.add_argument("--test-csv", default="flood_test_data.csv")
    p.add_argument("--s1-dir", default="files/S1")
    p.add_argument("--lbl-dir", default="files/Labels")
    p.add_argument("--out-dir", default="checkpoints")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--crop", type=int, default=256)
    p.add_argument("--base", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--water-weight", type=float, default=8.0)
    main(p.parse_args())
