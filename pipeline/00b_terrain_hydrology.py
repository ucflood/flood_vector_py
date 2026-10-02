"""Step 0b - TWI and stream density from the DEM (flow routing with pysheds, D8).

  python 00b_terrain_hydrology.py --dem layers/elevation.tif --out-dir layers
  python 00b_terrain_hydrology.py --dem layers/elevation.tif --streams osm_waterways.gpkg --out-dir layers   # use a vector network instead

Outputs in --out-dir: twi.tif, stream_density.tif (km/km2), flowacc_km2.tif (handy for QA).
Run on the BUFFERED DEM from 00a (the catchments feeding the AMA lie partly outside it); 00c clips to the AMA.
The DEM must be in a projected CRS (metres), e.g. EPSG:32721.
"""
import argparse, os
import numpy as np
if not hasattr(np, "in1d"):          # pysheds 0.5 still calls np.in1d, removed in NumPy 2
    np.in1d = np.isin
import rasterio
from rasterio.transform import Affine
from rasterio import features
from scipy import ndimage as ndi
from scipy.signal import fftconvolve
import geopandas as gpd
from pysheds.grid import Grid

DIAG = [2, 8, 32, 128]               # pysheds D8 codes that move diagonally


def read_dem(path):
    with rasterio.open(path) as src:
        z, prof = src.read(1).astype(np.float32), src.profile
    z[(z <= -9998) | ~np.isfinite(z)] = np.nan
    if np.isnan(z).any():            # fill rare voids with the nearest value
        idx = ndi.distance_transform_edt(np.isnan(z), return_distances=False, return_indices=True)
        z = z[tuple(idx)]
    return z, prof


def write(path, arr, prof):
    p = prof.copy(); p.update(count=1, dtype="float32", nodata=np.nan, compress="deflate")
    with rasterio.open(path, "w", **p) as d:
        d.write(arr.astype(np.float32), 1)


def route(dem_path, tmp_clean):
    """Fill depressions, resolve flats, D8 flow direction and accumulation (cells, incl. the cell itself)."""
    g = Grid.from_raster(tmp_clean)
    dem = g.read_raster(tmp_clean)
    dem = g.resolve_flats(g.fill_depressions(g.fill_pits(dem)))
    fdir = g.flowdir(dem)
    acc = g.accumulation(fdir)
    return np.asarray(fdir), np.asarray(acc)


def circular_density(length_km, valid, cell_m, radius_m):
    r = int(round(radius_m / cell_m))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    k = (yy ** 2 + xx ** 2 <= r ** 2).astype(np.float32)
    num = fftconvolve(length_km, k, mode="same")
    den = fftconvolve(valid.astype(np.float32) * (cell_m / 1000) ** 2, k, mode="same")   # km2 of valid area in the window
    return np.where(den > 0, np.maximum(num / np.maximum(den, 1e-9), 0), np.nan)


def vector_length_km(streams_path, prof, factor=3):
    """Approximate stream length (km) per cell by rasterising on a 3x finer grid and summing."""
    H, W, tr = prof["height"], prof["width"], prof["transform"]
    g = gpd.read_file(streams_path).to_crs(prof["crs"])
    fine = features.rasterize(((geom, 1) for geom in g.geometry), out_shape=(H * factor, W * factor),
                              transform=tr * Affine.scale(1 / factor), fill=0, dtype="uint8", all_touched=True)
    cell = abs(tr.a) / factor
    return fine.reshape(H, factor, W, factor).sum(axis=(1, 3)).astype(np.float32) * cell / 1000.0


def main(a):
    os.makedirs(a.out_dir, exist_ok=True)
    z, prof = read_dem(a.dem)
    assert prof["crs"].is_projected, "DEM must be in a projected CRS (metres)"
    cell = abs(prof["transform"].a)

    clean = os.path.join(a.out_dir, "_dem_clean.tif")
    p = prof.copy(); p.update(dtype="float32", nodata=-9999)
    with rasterio.open(clean, "w", **p) as d:
        d.write(z, 1)
    fdir, acc = route(a.dem, clean)
    os.remove(clean)
    write(os.path.join(a.out_dir, "flowacc_km2.tif"), acc * cell ** 2 / 1e6, prof)

    # TWI = ln( specific catchment area / tan(slope) ),  a = (acc * cell) [m]
    dzdy, dzdx = np.gradient(z, cell)
    slope = np.arctan(np.hypot(dzdx, dzdy))
    tan_b = np.tan(np.maximum(slope, np.radians(a.min_slope_deg)))       # floor avoids infinite TWI on flats
    twi = np.log(acc * cell / tan_b)
    write(os.path.join(a.out_dir, "twi.tif"), twi, prof)
    print(f"TWI range {np.nanmin(twi):.1f} .. {np.nanmax(twi):.1f}")

    # stream density (km of channel per km2, circular window)
    if a.streams:
        length_km = vector_length_km(a.streams, prof)
        src = f"vector network {a.streams}"
    else:
        thr_cells = a.stream_km2 * 1e6 / cell ** 2
        is_stream = acc >= thr_cells
        step = np.where(np.isin(fdir, DIAG), cell * np.sqrt(2), cell) / 1000.0
        length_km = (is_stream * step).astype(np.float32)
        src = f"DEM-derived channels (>= {a.stream_km2} km2 upstream)"
    dens = circular_density(length_km, np.ones_like(z, bool), cell, a.radius_m)
    write(os.path.join(a.out_dir, "stream_density.tif"), dens, prof)
    print(f"stream density from {src}; range {np.nanmin(dens):.2f} .. {np.nanmax(dens):.2f} km/km2")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dem", required=True)
    p.add_argument("--out-dir", default="layers")
    p.add_argument("--streams", help="optional vector network (OSM waterways / national hydrography)")
    p.add_argument("--stream-km2", type=float, default=1.0, help="upstream area that defines a channel when deriving it from the DEM")
    p.add_argument("--radius-m", type=float, default=1000, help="search radius for the density window")
    p.add_argument("--min-slope-deg", type=float, default=0.1)
    main(p.parse_args())
