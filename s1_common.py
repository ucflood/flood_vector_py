"""Shared pieces for the Sen1Floods11 -> Asunción pipeline (TensorFlow/Keras).

Preprocessing is IDENTICAL to the original Sen1Floods11 PyTorch notebook:
  S1 (VV, VH in dB) -> NaN->0 -> clip [-50, 1] -> (x+50)/51 -> per-band mean/std.
Anything you feed the model later (Asunción scenes) MUST go through preprocess_s1().
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

IGNORE = 255
MEAN = np.array([0.6851, 0.5235], dtype=np.float32)   # from the original notebook
STD = np.array([0.0820, 0.1102], dtype=np.float32)


def preprocess_s1(x_db):
    """x_db: (2, H, W) float array in dB (VV, VH)  ->  (H, W, 2) float32, normalised."""
    x = np.nan_to_num(x_db.astype(np.float32))
    x = np.clip(x, -50, 1)
    x = (x + 50) / 51
    x = (x - MEAN[:, None, None]) / STD[:, None, None]
    return np.transpose(x, (1, 2, 0)).astype(np.float32)


def prep_label(y):
    """Sen1Floods11 labels: -1 = no data, 0 = no water, 1 = water -> uint8 with 255 = ignore."""
    y = y.astype(np.int16)
    y[y == -1] = IGNORE
    return y.astype(np.uint8)


# ----------------------------------------------------------------- model ---
def _block(x, f):
    for _ in range(2):
        x = layers.Conv2D(f, 3, padding="same", use_bias=False)(x)
        x = layers.GroupNormalization(groups=8)(x)   # GN instead of BN (small batches), as in the original
        x = layers.Activation("relu")(x)
    return x


def build_unet(base=32, depth=4, in_ch=2):
    """Fully-convolutional U-Net. Input spatial size must be divisible by 2**depth."""
    inp = keras.Input((None, None, in_ch))
    x, skips = inp, []
    for d in range(depth):
        x = _block(x, base * 2 ** d)
        skips.append(x)
        x = layers.MaxPool2D(2)(x)
    x = _block(x, base * 2 ** depth)
    for d in reversed(range(depth)):
        x = layers.UpSampling2D(2, interpolation="bilinear")(x)
        x = layers.Concatenate()([x, skips[d]])
        x = _block(x, base * 2 ** d)
    out = layers.Conv2D(2, 1)(x)                     # logits: [no-water, water]
    return keras.Model(inp, out, name="unet_s1")


# ------------------------------------------------- loss and metrics (mask 255) ---
def make_masked_wce(class_weights=(1.0, 8.0)):
    """Weighted cross-entropy that ignores label 255 (same as CrossEntropyLoss(weight=[1,8], ignore_index=255))."""
    cw = tf.constant(class_weights, tf.float32)

    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        valid = tf.cast(tf.not_equal(y_true, IGNORE), tf.float32)
        y_safe = tf.where(y_true == IGNORE, 0, y_true)
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_safe, logits=y_pred)
        w = tf.gather(cw, y_safe) * valid
        return tf.reduce_sum(ce * w) / (tf.reduce_sum(w) + 1e-7)
    return loss


def make_masked_ce_dice(class_weights=(1.0, 8.0), dice_weight=0.5, smooth=1.0):
    """make_masked_wce + dice_weight * (1 - soft Dice of the water class); label 255 is ignored in both terms.

    Dice looks at the overlap of the (minority) water class directly, like the flood IoU metric.
    dice_weight=0 gives exactly make_masked_wce."""
    wce = make_masked_wce(class_weights)

    def loss(y_true, y_pred):
        ce = wce(y_true, y_pred)
        if not dice_weight:
            return ce
        y = tf.cast(y_true, tf.int32)
        valid = tf.cast(tf.not_equal(y, IGNORE), tf.float32)
        p = tf.nn.softmax(tf.cast(y_pred, tf.float32), axis=-1)[..., 1] * valid
        t = tf.cast(tf.equal(y, 1), tf.float32)
        inter = tf.reduce_sum(p * t)
        dice = (2.0 * inter + smooth) / (tf.reduce_sum(p) + tf.reduce_sum(t) + smooth)
        return ce + dice_weight * (1.0 - dice)
    return loss


@keras.utils.register_keras_serializable(package="ama")
class OffsetSchedule(keras.optimizers.schedules.LearningRateSchedule):
    """Wraps a schedule so that step 0 of this run is step `offset` of the schedule.
    Used by --resume: a new optimizer starts at iteration 0, but the LR must continue where training stopped."""

    def __init__(self, schedule, offset=0):
        self.schedule, self.offset = schedule, int(offset)

    def __call__(self, step):
        return self.schedule(tf.cast(step, tf.float32) + float(self.offset))

    def get_config(self):
        return {"schedule": keras.optimizers.schedules.serialize(self.schedule), "offset": self.offset}

    @classmethod
    def from_config(cls, cfg):
        return cls(keras.optimizers.schedules.deserialize(cfg["schedule"]), cfg["offset"])


class EMAWeights(keras.callbacks.Callback):
    """Exponential moving average of the weights, used for VALIDATION and for the checkpoints.

    Keras' built-in `use_ema` overwrites the training weights every N steps; with only ~15 steps per epoch
    (252 Sen1Floods11 chips / batch 16) that keeps pulling the model back and slows learning. Here the
    training weights are never touched: during validation the averaged weights are swapped in, and swapped
    back at the end of the epoch. Put this callback LAST in the list, after the ModelCheckpoints, so that they
    save the averaged weights (callbacks run in list order)."""

    def __init__(self, decay=0.98):
        super().__init__()
        self.decay, self.shadow, self._backup = decay, None, None

    def on_train_batch_end(self, batch, logs=None):
        w = self.model.get_weights()
        if self.shadow is None:
            self.shadow = w
        else:
            d = self.decay
            self.shadow = [d * s + (1.0 - d) * x for s, x in zip(self.shadow, w)]

    def on_test_begin(self, logs=None):
        if self.shadow is not None and self._backup is None:
            self._backup = self.model.get_weights()
            self.model.set_weights(self.shadow)

    def on_epoch_end(self, epoch, logs=None):
        if self._backup is not None:
            self.model.set_weights(self._backup)
            self._backup = None


def predict_prob_array(model, x, tile=512, stride=384, batch=4, tta=False):
    """Sliding-window water probability for ONE preprocessed scene x (H, W, 2) -> (H, W) float32.

    Overlapping tiles are blended with a Hann window (no seams; tile centres count more than tile edges).
    tta=True averages 4 orientations (original, flip-H, flip-V, both): about 4x slower, smoother, usually a bit better.
    Scenes smaller than `tile` are zero-padded."""
    H, W = x.shape[:2]
    Hp, Wp = max(H, tile), max(W, tile)
    xp = np.zeros((Hp, Wp, x.shape[2]), np.float32)
    xp[:H, :W] = x
    prob, wsum = np.zeros((Hp, Wp), np.float32), np.zeros((Hp, Wp), np.float32)
    h = np.hanning(tile + 2)[1:-1].astype(np.float32)
    win = np.outer(h, h) + 1e-3
    ys = sorted(set(list(range(0, Hp - tile + 1, stride)) + [Hp - tile]))
    xs = sorted(set(list(range(0, Wp - tile + 1, stride)) + [Wp - tile]))
    pos = [(i, j) for i in ys for j in xs]
    flips = [(), (2,), (1,), (1, 2)] if tta else [()]          # axes of the (B, H, W, C) batch
    for k in range(0, len(pos), batch):
        chunk = pos[k:k + batch]
        tiles = np.stack([xp[i:i + tile, j:j + tile] for i, j in chunk])
        p = 0.0
        for ax in flips:
            t = np.ascontiguousarray(np.flip(tiles, ax)) if ax else tiles
            lo = np.asarray(model.predict(t, verbose=0), np.float32)
            e = np.exp(lo - lo.max(-1, keepdims=True))
            pr = (e / e.sum(-1, keepdims=True))[..., 1]
            p = p + (np.flip(pr, [a for a in ax]) if ax else pr)
        p = p / len(flips)
        for (i, j), pi in zip(chunk, p):
            prob[i:i + tile, j:j + tile] += pi * win
            wsum[i:i + tile, j:j + tile] += win
    return (prob / np.maximum(wsum, 1e-6))[:H, :W]


class _MaskedBase(keras.metrics.Metric):
    def _prep(self, y_true, y_pred):
        y_true = tf.reshape(tf.cast(y_true, tf.int32), [-1])
        pred = tf.reshape(tf.argmax(y_pred, axis=-1, output_type=tf.int32), [-1])
        valid = tf.not_equal(y_true, IGNORE)
        return tf.boolean_mask(y_true, valid), tf.boolean_mask(pred, valid)


class FloodIoU(_MaskedBase):
    def __init__(self, name="flood_iou", **kw):
        super().__init__(name=name, **kw)
        self.tp = self.add_weight(name="tp", initializer="zeros")
        self.fp = self.add_weight(name="fp", initializer="zeros")
        self.fn = self.add_weight(name="fn", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        t, p = self._prep(y_true, y_pred)
        t, p = tf.cast(t, tf.float32), tf.cast(p, tf.float32)
        self.tp.assign_add(tf.reduce_sum(t * p))
        self.fp.assign_add(tf.reduce_sum((1 - t) * p))
        self.fn.assign_add(tf.reduce_sum(t * (1 - p)))

    def result(self):
        return (self.tp + 1e-7) / (self.tp + self.fp + self.fn + 1e-7)

    def reset_state(self):
        for v in (self.tp, self.fp, self.fn):
            v.assign(0.0)


class MaskedAccuracy(_MaskedBase):
    def __init__(self, name="acc", **kw):
        super().__init__(name=name, **kw)
        self.ok = self.add_weight(name="ok", initializer="zeros")
        self.n = self.add_weight(name="n", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        t, p = self._prep(y_true, y_pred)
        self.ok.assign_add(tf.reduce_sum(tf.cast(tf.equal(t, p), tf.float32)))
        self.n.assign_add(tf.cast(tf.size(t), tf.float32))

    def result(self):
        return self.ok / (self.n + 1e-7)

    def reset_state(self):
        self.ok.assign(0.0)
        self.n.assign(0.0)
