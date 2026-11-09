"""Sentinel-1 catalog and scene picker for Earth Engine (helper, not a numbered stage).

Used by notebooks/AMA_finetune_labels_colab.ipynb to download Sentinel-1 scenes of the AMA from 2014 on:
  catalog()      every IW VV+VH acquisition over the AOI since `start`, with the share of the AOI it covers
  pick_scene()   best single acquisition date around a target date (same pass / relative orbit, AOI fully covered)
  export_scene() Export.toDrive with the same parameters as 02a_gee_export_s1.py (10 m, EPSG:32721, VV+VH in dB)

The picking logic is plain Python (tests/test_gee_s1.py); the Earth Engine calls were mock-tested only and have
never run against live Earth Engine. Sentinel-1A data in the GEE collection starts on 2014-10-03: earlier
floods (e.g. June 2014) cannot be mapped with it.
"""
import datetime as dt

S1_FIRST_DATE = "2014-10-03"            # first IW GRD scene in COPERNICUS/S1_GRD
COLLECTION = "COPERNICUS/S1_GRD"


def _day(s):
    return dt.date.fromisoformat(str(s)[:10])


def _col(ee, aoi, d0, d1):
    return (ee.ImageCollection(COLLECTION).filterBounds(aoi).filterDate(d0, d1)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")))


def _chunks(start, end, days=365):
    a, e = _day(start), _day(end)
    while a < e:
        b = min(a + dt.timedelta(days=days), e)
        yield a.isoformat(), b.isoformat()
        a = b


def catalog(ee, aoi, start=S1_FIRST_DATE, end=None, log=print):
    """List of dicts {date, pass, orbit, platform, cover}: one per Sentinel-1 slice that touches the AOI.

    `cover` = share of the AOI inside the slice footprint (0-1). Asked one year at a time to stay below the
    getInfo element limit."""
    end = end or dt.date.today().isoformat()
    start = max(str(start)[:10], S1_FIRST_DATE)
    aoi_s = aoi.simplify(200)
    area = aoi_s.area(200)

    def feat(im):
        cover = im.geometry().intersection(aoi_s, 200).area(200).divide(area)
        return ee.Feature(None, {"t": im.get("system:time_start"), "pass": im.get("orbitProperties_pass"),
                                 "orbit": im.get("relativeOrbitNumber_start"), "platform": im.get("platform_number"),
                                 "cover": cover})
    rows = []
    for a, b in _chunks(start, end):
        feats = _col(ee, aoi, a, b).map(feat).getInfo()["features"]
        log(f"  {a} a {b}: {len(feats)} escenas")
        for f in feats:
            p = f["properties"]
            rows.append(dict(date=dt.datetime.fromtimestamp(p["t"] / 1000, dt.timezone.utc).date().isoformat(),
                             **{"pass": p["pass"]}, orbit=int(p["orbit"]), platform=p.get("platform"),
                             cover=float(p["cover"])))
    return rows


def _by_acquisition(rows):
    """Group slices of the same day / pass / relative orbit (consecutive frames of one pass)."""
    g = {}
    for r in rows:
        k = (r["date"], r["pass"], r["orbit"])
        g.setdefault(k, dict(date=r["date"], **{"pass": r["pass"]}, orbit=r["orbit"], platform=r.get("platform"),
                             n_slices=0, cover=0.0))
        g[k]["n_slices"] += 1
        g[k]["cover"] = min(1.0, g[k]["cover"] + r["cover"])         # adjacent frames barely overlap
    return list(g.values())


def pick_scene(rows, target, tol_days=12, min_cover=0.9, orbit_pass=None, rel_orbit=None):
    """Best acquisition within +-tol_days of `target`: AOI covered >= min_cover, then closest in time.

    Returns (best or None, candidates sorted best first). Each candidate has date, pass, orbit, platform,
    n_slices, cover, dias (signed days from target) and ok (cover >= min_cover)."""
    t = _day(target)
    cands = []
    for a in _by_acquisition(rows):
        if orbit_pass and a["pass"] != orbit_pass:
            continue
        if rel_orbit is not None and a["orbit"] != rel_orbit:
            continue
        d = (_day(a["date"]) - t).days
        if abs(d) <= tol_days:
            cands.append(dict(a, dias=d, ok=a["cover"] >= min_cover))
    cands.sort(key=lambda a: (not a["ok"], abs(a["dias"]), -a["cover"], a["date"]))
    best = cands[0] if cands and cands[0]["ok"] else None
    return best, cands


def best_orbit(rows, min_cover=0.9):
    """(pass, relative orbit, n_days) with the most full-coverage acquisition days: a fixed geometry for a long series."""
    n = {}
    for a in _by_acquisition(rows):
        if a["cover"] >= min_cover:
            n[(a["pass"], a["orbit"])] = n.get((a["pass"], a["orbit"]), 0) + 1
    if not n:
        return None
    (p, o), k = max(n.items(), key=lambda kv: kv[1])
    return p, o, k


def summary_table(rows, min_cover=0.9):
    """Acquisition days per year and pass/orbit (columns) with the AOI covered >= min_cover."""
    import pandas as pd
    acq = [a for a in _by_acquisition(rows) if a["cover"] >= min_cover]
    if not acq:
        return pd.DataFrame()
    df = pd.DataFrame(acq)
    df["anio"] = df["date"].str[:4]
    df["orbita"] = df["pass"].str[:4] + " " + df["orbit"].astype(str)
    return df.pivot_table(index="anio", columns="orbita", values="date", aggfunc="count", fill_value=0)


def monthly_targets(start, end, day=15):
    """Mid-month target dates from start to end (inclusive months)."""
    a, e = _day(start), _day(end)
    y, m = a.year, a.month
    out = []
    while dt.date(y, m, 1) <= e:
        out.append(dt.date(y, m, day).isoformat())
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def export_scene(ee, aoi, pick, name, folder):
    """Start an Export.toDrive of the picked acquisition (mosaic of its slices); file `S1_<name>.tif` in Drive/<folder>."""
    d0 = _day(pick["date"])
    d1 = (d0 + dt.timedelta(days=1)).isoformat()
    col = (_col(ee, aoi, d0.isoformat(), d1)
           .filter(ee.Filter.eq("orbitProperties_pass", pick["pass"]))
           .filter(ee.Filter.eq("relativeOrbitNumber_start", pick["orbit"])))
    img = col.select(["VV", "VH"]).mosaic().clip(aoi).toFloat()      # already in dB in GEE
    task = ee.batch.Export.image.toDrive(image=img, description=f"S1_{name}", folder=folder,
                                         fileNamePrefix=f"S1_{name}", region=aoi, scale=10, crs="EPSG:32721",
                                         maxPixels=1e10)
    task.start()
    return task
