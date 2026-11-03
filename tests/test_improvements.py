"""Training / inference improvements: Dice loss, resume-aware LR schedule, EMA weights, TTA, slope filter, threshold sweep."""
import argparse, json
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from conftest import write_tif


def _ns(s1f11, **kw):
    d = dict(train_csv=str(s1f11 / "flood_train_data.csv"), valid_csv=str(s1f11 / "flood_valid_data.csv"),
             test_csv=str(s1f11 / "flood_test_data.csv"), s1_dir=str(s1f11 / "S1"), lbl_dir=str(s1f11 / "Labels"),
             out_dir=str(s1f11 / "ckpt"), epochs=2, batch=2, crop=128, base=8, lr=5e-4, water_weight=8.0,
             dice_weight=0.5, sched="cosine", ema=0.0, resume=False)
    d.update(kw)
    return argparse.Namespace(**d)


def test_dice_loss():
    import tensorflow as tf, s1_common as c
    rng = np.random.default_rng(0)
    logits = tf.constant(rng.normal(size=(2, 8, 8, 2)).astype("float32"))
    y = rng.integers(0, 2, (2, 8, 8)).astype(np.uint8); y[:, :2] = 255
    wce = float(c.make_masked_wce((1.0, 8.0))(y, logits))
    assert abs(float(c.make_masked_ce_dice((1.0, 8.0), 0.0)(y, logits)) - wce) < 1e-6      # dice off = old loss
    assert float(c.make_masked_ce_dice((1.0, 8.0), 0.5)(y, logits)) > wce                  # dice term is in (0, 1]
    # label 255 is ignored by the Dice term as well: changing logits on ignored pixels changes nothing
    l2 = logits.numpy().copy(); l2[:, :2] += 5.0
    f = c.make_masked_ce_dice((1.0, 8.0), 0.5)
    assert abs(float(f(y, logits)) - float(f(y, tf.constant(l2)))) < 1e-5
    # a perfect prediction has (almost) zero loss
    perfect = np.where((y == 1)[..., None], [-20.0, 20.0], [20.0, -20.0]).astype("float32")
    assert float(f(y, tf.constant(perfect))) < 1e-3


def test_offset_schedule_continues_after_resume():
    from tensorflow import keras
    import s1_common as c
    base = keras.optimizers.schedules.CosineDecay(1e-3, decay_steps=1000, alpha=0.01)
    off = c.OffsetSchedule(base, offset=400)
    assert abs(float(off(0)) - float(base(400))) < 1e-9 and abs(float(off(100)) - float(base(500))) < 1e-9
    cfg = off.get_config()
    assert abs(float(c.OffsetSchedule.from_config(cfg)(0)) - float(base(400))) < 1e-9


def test_stage1_ema_and_old_setup(load, s1f11):
    import pandas as pd
    m = load("01_train_sen1floods11")
    m.main(_ns(s1f11, ema=0.9, epochs=1))                                    # EMA + cosine + dice
    assert (s1f11 / "ckpt" / "unet_s1f11_best.keras").exists()
    m.main(_ns(s1f11, ema=0.9, epochs=2, resume=True))                       # resume keeps the EMA path working
    assert list(pd.read_csv(s1f11 / "ckpt" / "history.csv")["epoch"]) == [0, 1]
    m.main(_ns(s1f11, out_dir=str(s1f11 / "old"), sched="restarts", dice_weight=0.0, epochs=1))   # previous setup still runs


def test_ema_validates_with_average_and_restores_training_weights():
    import tensorflow as tf
    from tensorflow import keras
    import s1_common as c
    m = keras.Sequential([keras.Input((3,)), keras.layers.Dense(1, use_bias=False, kernel_initializer="ones")])
    cb = c.EMAWeights(0.5); cb.set_model(m)
    cb.on_train_batch_end(0)                                                 # shadow = 1
    m.set_weights([np.full((3, 1), 3.0, "float32")]); cb.on_train_batch_end(1)   # shadow = 0.5*1 + 0.5*3 = 2
    cb.on_test_begin(); assert np.allclose(m.get_weights()[0], 2.0)          # validation sees the average
    cb.on_epoch_end(0); assert np.allclose(m.get_weights()[0], 3.0)          # training weights are back


class _FlipEquivariantModel:
    """Fake network: logits depend only on the pixel value, so any flip of the input flips the output."""
    def predict(self, x, verbose=0):
        v = x[..., :1]
        return np.concatenate([-v, v], -1)


def test_predict_prob_array_blend_and_tta():
    import s1_common as c
    rng = np.random.default_rng(3)
    x = rng.normal(size=(700, 650, 2)).astype("float32")                     # not a multiple of the stride
    p0 = c.predict_prob_array(_FlipEquivariantModel(), x, tile=256, stride=192, tta=False)
    p1 = c.predict_prob_array(_FlipEquivariantModel(), x, tile=256, stride=192, tta=True)
    expect = 1 / (1 + np.exp(-2 * x[..., 0]))
    assert p0.shape == x.shape[:2] and np.allclose(p0, expect, atol=1e-4)     # blending identical tiles = identity
    assert np.allclose(p1, p0, atol=1e-4)                                    # equivariant model: TTA changes nothing
    small = c.predict_prob_array(_FlipEquivariantModel(), x[:100, :90], tile=256, stride=192)
    assert small.shape == (100, 90)


def test_stage2_slope_filter_and_tta(load, tmp_path):
    from tensorflow import keras
    import s1_common as c
    m = load("02_ama_flood_inventory")
    tr = from_origin(440000, 7180000, 10, 10)
    H, W = 520, 600
    rng = np.random.default_rng(2)
    yy, xx = np.mgrid[:H, :W]
    water = ((yy - 260) ** 2 + (xx - 300) ** 2) < 120 ** 2
    x = np.stack([np.where(water, -22, -9), np.where(water, -28, -16)]) + rng.normal(0, 1, (2, H, W))
    write_tif(tmp_path / "post.tif", x, tr, nodata=None)
    slope = np.zeros((H // 3, W // 3), "float32"); slope[:, : (300 // 3)] = 20          # coarse grid, steep west half
    write_tif(tmp_path / "slope.tif", slope, from_origin(440000, 7180000, 30, 30))
    # a stand-in "network": water where VV is low (so the test does not depend on training)
    class Net:
        def predict(self, xb, verbose=0):
            v = -xb[..., :1] * 4
            return np.concatenate([-v, v], -1)
    keras_model = c.build_unet(base=8); keras_model.save(tmp_path / "m.keras")
    m.keras = type("K", (), {"models": type("M", (), {"load_model": staticmethod(lambda p, compile=False: Net())})})
    base = dict(model="m", event="t", post=str(tmp_path / "post.tif"), pre=None, perm_water=None, perm_thr=50, thr=0.5,
                tta=True, slope=None, max_slope=5.0, min_px=20)
    m.main(argparse.Namespace(**base, out_dir=str(tmp_path / "a")))
    m.main(argparse.Namespace(**{**base, "slope": str(tmp_path / "slope.tif")}, out_dir=str(tmp_path / "b")))
    import geopandas as gpd
    ga, gb = gpd.read_file(tmp_path / "a" / "flood_t.gpkg"), gpd.read_file(tmp_path / "b" / "flood_t.gpkg")
    assert gb.area.sum() < 0.8 * ga.area.sum()                               # steep (west) part removed
    assert gb.total_bounds[0] >= 440000 + 295 * 10 - 40                      # nothing left on the steep side (x < ~300 px)


def test_calibrate_threshold(load, s1f11, tmp_path):
    import s1_common as c
    m1 = load("01_train_sen1floods11"); m1.main(_ns(s1f11, epochs=1))
    cal = load("01b_calibrate_threshold")
    a = argparse.Namespace(model=str(s1f11 / "ckpt" / "unet_s1f11_best.keras"), valid_csv=str(s1f11 / "flood_valid_data.csv"),
                           test_csv=str(s1f11 / "flood_test_data.csv"), s1_dir=str(s1f11 / "S1"), lbl_dir=str(s1f11 / "Labels"),
                           t_min=0.2, t_max=0.9, t_step=0.1, out=str(tmp_path / "thr.json"))
    res = cal.main(a)
    j = json.load(open(tmp_path / "thr.json"))
    assert 0.2 <= j["recommended"]["thr"] <= 0.9 and "test" in j
    assert all(0 <= j["valid_tta0"]["best"][k] <= 1 for k in ("iou", "precision", "recall"))
    # sweep() on a known case: thresholds trade precision for recall
    p = [np.array([[0.9, 0.6, 0.4, 0.1]])]; y = [np.array([[1, 1, 0, 255]], np.uint8)]
    r = cal.sweep(p, y, [0.5, 0.95])
    assert r[0]["iou"] == 1.0 and r[1]["recall"] == 0.0
