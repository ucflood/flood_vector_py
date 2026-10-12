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
