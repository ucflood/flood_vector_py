"""Stage 1b - choose the water-probability threshold of the stage-1 network on the Sen1Floods11 validation split.

Sweeps thresholds (with and without 4-flip TTA), reports flood IoU / precision / recall, picks the threshold that
maximises IoU on VALIDATION and then reports that same threshold on TEST (never choose it on test).
Use the result in stage 2:  02_ama_flood_inventory.py --thr <value> [--tta]

  python 01b_calibrate_threshold.py --model results/checkpoints/unet_s1f11_best.keras \\
      --valid-csv data/raw/sen1floods11/flood_valid_data.csv --test-csv data/raw/sen1floods11/flood_test_data.csv \\
      --s1-dir data/raw/sen1floods11/files/S1 --lbl-dir data/raw/sen1floods11/files/Labels --out results/threshold.json

Caveat: the threshold is tuned on Sen1Floods11 (other biomes/seasons). On Asuncion scenes it is a starting point;
check a few scenes by eye (or against hand-drawn polygons) before trusting it.
"""
import argparse, importlib.util, json, os, pathlib
import numpy as np
import pandas as pd
from tensorflow import keras
from s1_common import predict_prob_array, IGNORE

HERE = pathlib.Path(__file__).resolve().parent


def _load_stage1():
    spec = importlib.util.spec_from_file_location("train01", HERE / "01_train_sen1floods11.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def sweep(P, Y, thresholds):
    """P: list of (H,W) probabilities, Y: list of (H,W) uint8 labels with 255 = ignore -> list of dicts."""
    out = []
    for t in thresholds:
        tp = fp = fn = 0
        for p, y in zip(P, Y):
            v = y != IGNORE
            pred, gt = (p >= t) & v, (y == 1) & v
            tp += int((pred & gt).sum()); fp += int((pred & ~gt).sum()); fn += int((~pred & gt).sum())
        out.append(dict(thr=round(float(t), 2), iou=tp / max(tp + fp + fn, 1),
                        precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1)))
    return out


def probs(model, X, tta):
    return [predict_prob_array(model, x, tile=x.shape[0], stride=x.shape[0], tta=tta) for x in X]


def main(a):
    t1 = _load_stage1()
    model = keras.models.load_model(a.model, compile=False)
    thr = np.arange(a.t_min, a.t_max + 1e-9, a.t_step)
    Xva, Yva = t1.load_split(a.valid_csv, a.s1_dir, a.lbl_dir)
    res = {}
    for tta in (False, True):
        r = sweep(probs(model, Xva, tta), list(Yva), thr)
        best = max(r, key=lambda d: d["iou"])
        at05 = min(r, key=lambda d: abs(d["thr"] - 0.5))
        print(f"\nVALID  TTA={tta}\n" + pd.DataFrame(r).round(3).to_string(index=False))
        print(f"-> best thr {best['thr']}  IoU {best['iou']:.3f}  (at 0.5: {at05['iou']:.3f})  "
              f"precision {best['precision']:.3f}  recall {best['recall']:.3f}")
        if best["thr"] in (round(float(thr[0]), 2), round(float(thr[-1]), 2)):
            print("   (best value is at the edge of the sweep: widen --t-min/--t-max)")
        res[f"valid_tta{int(tta)}"] = dict(best=best, at_0_5=at05)
    use_tta = res["valid_tta1"]["best"]["iou"] > res["valid_tta0"]["best"]["iou"]
    chosen = res[f"valid_tta{int(use_tta)}"]["best"]
    res["recommended"] = dict(thr=chosen["thr"], tta=use_tta)
    if a.test_csv and os.path.exists(a.test_csv):
        Xte, Yte = t1.load_split(a.test_csv, a.s1_dir, a.lbl_dir)
        t = sweep(probs(model, Xte, use_tta), list(Yte), [chosen["thr"]])[0]
        print(f"\nTEST at thr {chosen['thr']} (TTA={use_tta}): IoU {t['iou']:.3f}  precision {t['precision']:.3f}  recall {t['recall']:.3f}")
        res["test"] = t
    b = res["valid_tta0"]["best"]
    if b["recall"] < b["precision"] - 0.1:
        print("Diagnosis: recall << precision -> the model misses water; raise --water-weight or --dice-weight.")
    elif b["precision"] < b["recall"] - 0.1:
        print("Diagnosis: precision << recall -> the model over-detects; lower --water-weight, raise --thr, use --slope in stage 2.")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)
    print(f"saved {a.out}\nstage 2: --thr {chosen['thr']}" + (" --tta" if use_tta else ""))
    return res


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--valid-csv", default="flood_valid_data.csv")
    p.add_argument("--test-csv", default="flood_test_data.csv")
    p.add_argument("--s1-dir", default="files/S1")
    p.add_argument("--lbl-dir", default="files/Labels")
    p.add_argument("--t-min", type=float, default=0.2)
    p.add_argument("--t-max", type=float, default=0.95)
    p.add_argument("--t-step", type=float, default=0.05)
    p.add_argument("--out", default="results/threshold.json")
    return p.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
