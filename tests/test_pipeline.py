import argparse
import numpy as np
import rasterio
import geopandas as gpd
from rasterio.transform import from_origin
from shapely.geometry import Point
from conftest import write_tif


# ----------------------------------------------------------- stage 1 ---
def test_preprocess_and_label():
    import s1_common as c
    x = np.full((2, 4, 4), -9.0, np.float32); x[:, 0, 0] = np.nan
    out = c.preprocess_s1(x)
    assert out.shape == (4, 4, 2) and np.isfinite(out).all()
    y = c.prep_label(np.array([[-1, 0, 1]]))
    assert y.tolist() == [[255, 0, 1]]


def test_masked_loss_ignores_255():
    import tensorflow as tf, s1_common as c
    loss = c.make_masked_wce((1.0, 8.0))
    logits = tf.constant(np.random.randn(1, 4, 4, 2).astype("float32"))
    y = np.zeros((1, 4, 4), np.uint8); y[0, :2] = 1
    y_masked = y.copy(); y_masked[0, 2:] = 255
    a = float(loss(y_masked, logits))
    b = float(loss(y[:, :2], logits[:, :2]))                 # only the valid rows
    assert abs(a - b) < 1e-5


def test_stage1_trains(load, s1f11):
    m = load("01_train_sen1floods11")
    a = argparse.Namespace(train_csv=str(s1f11 / "flood_train_data.csv"), valid_csv=str(s1f11 / "flood_valid_data.csv"),
                           test_csv=str(s1f11 / "flood_test_data.csv"), s1_dir=str(s1f11 / "S1"), lbl_dir=str(s1f11 / "Labels"),
                           out_dir=str(s1f11 / "ckpt"), epochs=2, batch=2, crop=128, base=8, lr=5e-4, water_weight=8.0)
    m.main(a)
    assert (s1f11 / "ckpt" / "unet_s1f11_best.keras").exists()


# ----------------------------------------------------------- stage 2 ---
def test_stage2_pieces(load, tmp_path):
    from tensorflow import keras
    import s1_common as c
    m = load("02_ama_flood_inventory")
    tr = from_origin(440000, 7180000, 10, 10)
    H, W = 600, 700
    rng = np.random.default_rng(1)
    write_tif(tmp_path / "s1.tif", rng.normal(-12, 2, (2, H, W)), tr, nodata=None)
    model = c.build_unet(base=8)
    prob = m.predict_water_prob(model, str(tmp_path / "s1.tif"), str(tmp_path / "p.tif"))
    assert prob.shape == (H, W) and np.nanmin(prob) >= 0 and np.nanmax(prob) <= 1
    # remove_small + polygonize on a known mask
    mask = np.zeros((H, W), bool); mask[100:140, 100:150] = True; mask[300, 300] = True
    mask = m.remove_small(mask, 20)
    assert mask.sum() == 40 * 50
    g = m.polygonize(mask, str(tmp_path / "s1.tif"), "t")
    assert abs(g.area.sum() - 40 * 50 * 100) < 1e-6          # 10 m pixels -> 100 m2 each
    # full main() must not crash (output depends on the untrained model)
    model.save(tmp_path / "m.keras")
    m.main(argparse.Namespace(model=str(tmp_path / "m.keras"), event="t", post=str(tmp_path / "s1.tif"), pre=None,
                              perm_water=None, perm_thr=50, thr=0.5, min_px=20, out_dir=str(tmp_path / "inv")))


# ------------------------------------------------- stages 0b, 0c, 3 ---
def test_full_chain_0b_0c_3(load, ama):
    L = ama / "layers"
    hydro, build, sus = load("00b_terrain_hydrology"), load("00c_build_stack"), load("03_susceptibility_1dcnn")

    hydro.main(argparse.Namespace(dem=str(L / "elevation.tif"), out_dir=str(L), streams=None, stream_km2=1.0,
                                  radius_m=1000, min_slope_deg=0.1))
    for n in ["twi", "stream_density", "flowacc_km2"]:
        with rasterio.open(L / f"{n}.tif") as s:
            a = s.read(1); assert np.isfinite(a).all() and a.min() >= 0 or n == "twi"

    build.main(argparse.Namespace(layers_dir=str(L), aoi=str(ama / "aoi.gpkg"), geology=str(ama / "geo.gpkg"),
                                  geology_field="FORM", soil=str(ama / "soil.gpkg"), soil_field="S",
                                  flat_slope_deg=1.0, out_dir=str(ama / "stack")))
    shapes = set()
    for n in sus.ORDER:
        with rasterio.open(ama / "stack" / f"{n}.tif") as s:
            shapes.add((s.shape, s.transform)); 
    assert len(shapes) == 1, "stack layers are not on one grid"

    # fake inventory: lowest-elevation pixels
    with rasterio.open(ama / "stack" / "elevation.tif") as s:
        e, tr = s.read(1), s.transform
    yy, xx = np.where(np.isfinite(e) & (e < np.nanpercentile(e, 15)))
    sel = np.random.default_rng(1).choice(len(yy), 300, replace=False)
    pts = [Point(*(tr * (xx[i] + .5, yy[i] + .5))).buffer(60) for i in sel]
    gpd.GeoDataFrame({"event": ["t"] * 300}, geometry=pts, crs="EPSG:32721").to_file(ama / "inv.gpkg", driver="GPKG")

    assert sus.build_1dcnn().count_params() == 71088          # matches Trong et al. (2023)
    sus.main(argparse.Namespace(stack_dir=str(ama / "stack"), inventory=[str(ama / "inv.gpkg")], zone=None,
                                out_dir=str(ama / "res"), max_pos=5000, buffer_px=5, block_px=67, epochs=10, batch=64,
                                breaks=[0.2, 0.4, 0.6, 0.8], seed=42))
    with rasterio.open(ama / "res" / "susceptibility.tif") as s:
        p = s.read(1); p = p[p != -9999]
    assert len(p) > 0 and p.min() >= 0 and p.max() <= 1


def test_helpers(load):
    sus = load("03_susceptibility_1dcnn")
    assert np.array_equal(sus.aspect_to_class(np.array([-1., 0., 90., 359.])), [0, 1, 3, 1])
    t = sus.fr_table(np.array([1, 1, 2, 2]), np.array([1, 1, 0, 0]))
    assert t[1.0] > 1 > t[2.0]
