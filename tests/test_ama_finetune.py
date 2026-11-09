"""01c (chips + pre-label) -> hand edit -> 01d (rasterise + fine-tune), on a synthetic scene."""
import argparse, types
import numpy as np
import rasterio
import geopandas as gpd
from rasterio.transform import from_origin
from shapely.geometry import box
from conftest import write_tif


class _Net:                                   # stand-in network: water where VV is low (no training needed)
    def predict(self, xb, verbose=0):
        v = -xb[..., :1] * 4
        return np.concatenate([-v, v], -1)


def _scene(path, seed):
    rng = np.random.default_rng(seed)
    H, W = 1300, 1500
    yy, xx = np.mgrid[:H, :W]
    water = (((yy - 300) ** 2 + (xx - 400) ** 2) < 200 ** 2) | (((yy - 900) ** 2 + (xx - 1100) ** 2) < 150 ** 2)
    x = np.stack([np.where(water, -22, -9), np.where(water, -28, -16)]) + rng.normal(0, 1, (2, H, W))
    write_tif(path, x, from_origin(440000, 7180000, 10, 10), nodata=None)


def _prepare(load, tmp_path, event, seed):
    m = load("01c_prepare_ama_chips")
    m.keras = types.SimpleNamespace(models=types.SimpleNamespace(load_model=lambda p, compile=False: _Net()))
    _scene(tmp_path / f"{event}.tif", seed)
    out = tmp_path / event
    m.main(argparse.Namespace(model="x", scene=str(tmp_path / f"{event}.tif"), event=event, out_dir=str(out), thr=0.5,
                              n_water=3, n_other=1, min_water=0.02, max_water=0.98, min_px=20, simplify_m=5.0,
                              tta=False, seed=0, overwrite=False))
    return out


def test_prepare_chips(load, tmp_path):
    out = _prepare(load, tmp_path, "A_2019", 1)
    chips = gpd.read_file(out / "labels.gpkg", layer="chips")
    water = gpd.read_file(out / "labels.gpkg", layer="water")
    assert 2 <= len(chips) <= 4 and set(chips.kind) <= {"mixed", "other"} and "mixed" in set(chips.kind)
    assert len(water) > 0 and set(water.chip_id) <= set(chips.chip_id)
    for cid in chips.chip_id:
        with rasterio.open(out / "S1" / f"{cid}_S1.tif") as s:
            assert s.shape == (512, 512) and s.count == 2


def test_rasterize_edited_polygons(load, tmp_path):
    out = _prepare(load, tmp_path, "A_2019", 1)
    ft = load("01d_finetune_ama")
    water = gpd.read_file(out / "labels.gpkg", layer="water")
    cid = water.chip_id.value_counts().index[0]
    # hand edit: delete every polygon of one chip (all "false positives"), and mark an ignore area on another chip
    water[water.chip_id != cid].to_file(out / "labels.gpkg", layer="water", driver="GPKG")
    other = [c for c in set(water.chip_id) if c != cid]
    ign_cid = other[0] if other else cid
    with rasterio.open(out / "S1" / f"{ign_cid}_S1.tif") as s:
        b = s.bounds
    gpd.GeoDataFrame(geometry=[box(b.left, b.top - 100, b.left + 100, b.top)], crs="EPSG:32721").to_file(out / "ignore.gpkg", driver="GPKG")
    items = ft.rasterize_chips(out)
    labs = {c: rasterio.open(l).read(1) for _, l, c in items}
    assert set(np.unique(labs[cid])) <= {0, -1} and (labs[cid] == 1).sum() == 0            # deleted -> no water
    assert (labs[ign_cid][:10, :10] == -1).all()                                             # ignore polygon -> -1
    if other:
        assert (labs[other[0]] == 1).sum() > 0                                               # untouched chips keep water


def test_finetune_runs_and_reports(load, tmp_path, s1f11):
    from tensorflow import keras
    import s1_common as c
    A = _prepare(load, tmp_path, "A_2019", 1)
    B = _prepare(load, tmp_path, "B_2025", 2)
    c.build_unet(base=8).save(tmp_path / "start.keras")
    ft = load("01d_finetune_ama")
    a = ft.parse_args(["--model", str(tmp_path / "start.keras"), "--chips-dir", str(A), str(B), "--val-events", "B_2025",
                       "--s1f11-train-csv", str(s1f11 / "flood_train_data.csv"), "--s1f11-valid-csv", str(s1f11 / "flood_valid_data.csv"),
                       "--s1-dir", str(s1f11 / "S1"), "--lbl-dir", str(s1f11 / "Labels"), "--out-dir", str(tmp_path / "ft"),
                       "--epochs", "1", "--steps", "2", "--batch", "2", "--crop", "128"])
    before = ft.main(a)
    assert 0 <= before["flood_iou"] <= 1
    out = tmp_path / "ft"
    import pandas as pd
    tr = pd.read_csv(out / "ama_train.csv", header=None); va = pd.read_csv(out / "ama_valid.csv", header=None)
    assert all("B_2025" in p for p in va[0]) and not any("B_2025" in p for p in tr[0])       # held-out event stays out of training
    assert (out / "history_ama.csv").exists()
    # 01b works on the Asuncion validation chips (absolute paths in the CSV, empty dirs)
    cal = load("01b_calibrate_threshold")
    r = cal.main(cal.parse_args(["--model", str(tmp_path / "start.keras"), "--valid-csv", str(out / "ama_valid.csv"),
                                 "--test-csv", "", "--s1-dir", "", "--lbl-dir", "", "--t-step", "0.2",
                                 "--out", str(out / "threshold_ama.json")]))
    assert "recommended" in r and (out / "threshold_ama.json").exists()


def test_prepare_does_not_overwrite_edits(load, tmp_path):
    import pytest
    out = _prepare(load, tmp_path, "A_2019", 1)
    before = (out / "labels.gpkg").stat().st_mtime_ns
    m = load("01c_prepare_ama_chips")
    m.keras = types.SimpleNamespace(models=types.SimpleNamespace(load_model=lambda p, compile=False: _Net()))
    a = argparse.Namespace(model="x", scene=str(tmp_path / "A_2019.tif"), event="A_2019", out_dir=str(out), thr=0.5, n_water=3,
                           n_other=1, min_water=0.02, max_water=0.98, min_px=20, simplify_m=5.0, tta=False, seed=0, overwrite=False)
    with pytest.raises(SystemExit):
        m.main(a)
    assert (out / "labels.gpkg").stat().st_mtime_ns == before                  # untouched
    a.overwrite = True; m.main(a)                                               # explicit opt-in works
