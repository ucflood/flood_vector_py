"""Synthetic data generators so the whole pipeline can be tested without real imagery."""
import argparse, csv, importlib.util, pathlib, sys
import numpy as np
import pytest
import rasterio
import geopandas as gpd
from rasterio.transform import from_origin
from scipy.ndimage import gaussian_filter
from shapely.geometry import box

PIPE = pathlib.Path(__file__).resolve().parents[1] / "pipeline"
sys.path.insert(0, str(PIPE))


@pytest.fixture(scope="session")
def load():
    def _load(name):
        spec = importlib.util.spec_from_file_location(name, PIPE / f"{name}.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m
    return _load


def write_tif(path, arr, transform, crs="EPSG:32721", nodata=-9999, dtype="float32"):
    arr = np.asarray(arr, dtype)
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[-2], width=arr.shape[-1],
                       count=1 if arr.ndim == 2 else arr.shape[0], dtype=dtype, crs=crs,
                       transform=transform, nodata=nodata) as d:
        d.write(arr, 1) if arr.ndim == 2 else d.write(arr)


@pytest.fixture()
def s1f11(tmp_path):
    """Tiny Sen1Floods11-like dataset: 512x512 chips, 2 bands (VV, VH dB), labels in {-1,0,1}."""
    rng = np.random.default_rng(0)
    (tmp_path / "S1").mkdir(); (tmp_path / "Labels").mkdir()
    tr = from_origin(0, 0, 1e-4, 1e-4)
    for split, n in [("train", 4), ("valid", 2), ("test", 2)]:
        rows = []
        for i in range(n):
            yy, xx = np.mgrid[:512, :512]
            water = ((yy - 256) ** 2 + (xx - 100 - i * 30) ** 2 < 80 ** 2).astype("int16")
            x = np.stack([np.where(water, -22, -9) + rng.normal(0, 1, (512, 512)),
                          np.where(water, -28, -16) + rng.normal(0, 1, (512, 512))]).astype("float32")
            x[:, :5, :5] = np.nan
            y = water.copy(); y[:20, :] = -1
            a, b = f"{split}{i}_S1Hand.tif", f"{split}{i}_LabelHand.tif"
            write_tif(tmp_path / "S1" / a, x, tr, crs="EPSG:4326", nodata=None)
            write_tif(tmp_path / "Labels" / b, y, tr, crs="EPSG:4326", nodata=None, dtype="int16")
            rows.append([a, b])
        csv.writer(open(tmp_path / f"flood_{split}_data.csv", "w")).writerows(rows)
    return tmp_path


@pytest.fixture()
def ama(tmp_path):
    """Synthetic 'GEE exports' for the AMA: different grids / -9999 gaps on purpose, plus AOI and vector maps."""
    rng = np.random.default_rng(5)
    H, W = 360, 420
    tr = from_origin(440000, 7180000, 30, 30)
    L = tmp_path / "layers"; L.mkdir()
    z = gaussian_filter(rng.normal(size=(H, W)), 20)
    z = (z - z.min()) / (z.max() - z.min()) * 50 + 60 + np.linspace(0, 30, W)[None, :]
    write_tif(L / "elevation.tif", z, tr)
    dzdy, dzdx = np.gradient(z, 30)
    write_tif(L / "slope.tif", np.degrees(np.arctan(np.hypot(dzdx, dzdy))), tr)
    write_tif(L / "aspect.tif", np.degrees(np.arctan2(-dzdx, dzdy)) % 360, tr)
    write_tif(L / "relief_amplitude.tif", gaussian_filter(rng.normal(size=(H, W)), 5) * 3 + 10, tr)
    nd = gaussian_filter(rng.normal(size=(H, W)), 6); nd[:4, :4] = -9999
    write_tif(L / "ndvi.tif", nd, tr); write_tif(L / "ndwi.tif", -nd, tr)
    rain = np.array([[1000, 1010, 1020], [1005, 1015, 1025]], float).repeat(100, 0).repeat(100, 1)
    write_tif(L / "rainfall.tif", rain, from_origin(439000, 7181000, 50, 50))            # coarse, other grid
    write_tif(L / "lulc.tif", rng.choice([10, 30, 40, 50, 80], size=(H * 3, W * 3)), from_origin(440000, 7180000, 10, 10))
    x0, y0, x1, y1 = 440000, 7180000 - 30 * H, 440000 + 30 * W, 7180000
    gpd.GeoDataFrame(geometry=[box(x0 + 500, y0 + 500, x1 - 500, y1 - 500)], crs="EPSG:32721").to_file(tmp_path / "aoi.gpkg", driver="GPKG")
    xs = [x0, x0 + 30 * 140, x0 + 30 * 280, x1]
    gpd.GeoDataFrame({"FORM": list("ABC")}, geometry=[box(xs[i], y0, xs[i + 1], y1) for i in range(3)],
                     crs="EPSG:32721").to_file(tmp_path / "geo.gpkg", driver="GPKG")
    ym = y1 - 30 * 180
    gpd.GeoDataFrame({"S": ["x", "y"]}, geometry=[box(x0, y0, x1, ym), box(x0, ym, x1, y1)],
                     crs="EPSG:32721").to_file(tmp_path / "soil.gpkg", driver="GPKG")
    return tmp_path
