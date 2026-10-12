"""Step 4 - verify with satellite imagery which severe storms became flash floods in the AMA.

For every 'Repentina' row of data/events/ama_flood_inventory.csv:
  1. Storm time window, taken from the row (thesis start/end time, or "D-1 o D" for daily candidates).
     With --imerg it is narrowed to +/- 3 h around the half-hour of peak rain over the AMA (GPM IMERG V07).
  2. Sentinel-1: first IW scene over the AMA from the start of the storm to --max-lag-h after its end.
     Reference = median of the earlier scenes of the SAME relative orbit (same geometry), 2 to 60 days before.
     New water = post VV <= --s1-water-db  AND  post - pre <= --s1-drop-db, outside permanent water
     (JRC GSW occurrence >= 50 %) and slopes > --max-slope-deg, in patches of >= --min-px pixels.
  3. Sentinel-2 (--s2): first scene in the same window with >= --s2-min-clear of the AMA clear (Cloud Score+).
     New water = MNDWI(B3, B11) > 0 after and < 0 in the 60-day median before, same masks.
  4. Verdict, written to the inventory columns:
       Confirmada       new water >= --min-area-km2 in S1 or S2
       No observada     an image exists within the lag but shows no new water
       Sin imagen útil  no usable image within the lag
     'No observada' is NOT proof that there was no flash flood: they drain in hours, and C-band SAR misses
     water in dense built-up areas (double bounce). Detection works in open low ground: parks, vacant lots,
     arroyo floodplains, bañados. Check confirmed events visually (--export-masks) before using them.

Thresholds are starting values; calibrate them on known events (e.g. 15-abr-2024, Ñu Guasu park flooded).
Revisit: Sentinel-1 every 12 days over the AMA for most of the period (6 days while S1B flew, until Dec 2021),
so most storms will end as 'Sin imagen útil'. NOT TESTED against the live service (needs your GEE account).

  python pipeline/04_verify_flash_floods.py --project my-gee-project \
      --aoi-asset projects/my-gee-project/assets/AMA --imerg --s2 --out results/flash_flood_verification.csv
  python pipeline/04_verify_flash_floods.py --dry-run          # only prints the time windows, no GEE
"""
import argparse
import datetime as dt
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # %run in Colab

import pandas as pd

UTC = dt.timezone.utc
LOCAL_OFFSET_H = -3          # Paraguay (UTC-3) for rows that only have a calendar date

S1 = "COPERNICUS/S1_GRD"
S2 = "COPERNICUS/S2_SR_HARMONIZED"
CSPLUS = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"
GSW = "JRC/GSW1_4/GlobalSurfaceWater"
DEM = "JAXA/ALOS/AW3D30/V4_1"
IMERG = "NASA/GPM_L3/IMERG_V07"


# ------------------------------------------------------------------ pure logic (no GEE) ---
def event_window(fecha, descripcion=""):
    """(start, end) in UTC for one inventory row."""
    fecha = str(fecha).strip()
    if " o " in fecha:                                   # daily candidate: 24-h total ending 12 UTC of D
        d0, d1 = [dt.date.fromisoformat(x.strip()) for x in fecha.split(" o ")]
        return (dt.datetime(d0.year, d0.month, d0.day, 12, tzinfo=UTC),
                dt.datetime(d1.year, d1.month, d1.day, 12, tzinfo=UTC))
    d = dt.date.fromisoformat(fecha[:10])
    m = re.search(r"entre (\d{2}):(\d{2}) y (\d{2}):(\d{2}) UTC", str(descripcion))
    if m:                                                # thesis event with start/end time
        h0, m0, h1, m1 = map(int, m.groups())
        start = dt.datetime(d.year, d.month, d.day, h0, m0, tzinfo=UTC)
        end = dt.datetime(d.year, d.month, d.day, h1, m1, tzinfo=UTC)
        if end <= start:
            end += dt.timedelta(days=1)
        return start, end
    start = dt.datetime(d.year, d.month, d.day, tzinfo=UTC) - dt.timedelta(hours=LOCAL_OFFSET_H)
    return start, start + dt.timedelta(days=1)            # whole local day


def refine_with_peak(start, end, series, pad_h=3):
    """series: [(datetime, mm/h)] of AMA-mean IMERG. Window becomes peak +/- pad_h (inside the original one)."""
    wet = [(t, p) for t, p in series if p is not None and p > 0]
    if not wet:
        return start, end, None
    t_peak, p_peak = max(wet, key=lambda x: x[1])
    return (max(start, t_peak - dt.timedelta(hours=pad_h)), min(end, t_peak + dt.timedelta(hours=pad_h)),
            (t_peak, p_peak))


def verdict(s1, s2, min_area_km2):
    """s1/s2: dict with 'area_km2' when a usable image exists, else None."""
    areas = [x["area_km2"] for x in (s1, s2) if x is not None and x.get("area_km2") is not None]
    if not areas:
        return "Sin imagen útil"
    return "Confirmada" if max(areas) >= min_area_km2 else "No observada"


def _fmt(x):
    return f"{x:.2f}".replace(".", ",")


def describe(s1, s2, end):
    parts, used, dates = [], [], []
    for name, r in (("S1", s1), ("S2", s2)):
        if r is None:
            continue
        lag = (r["time"] - end).total_seconds() / 3600
        when = f"{lag:+.0f} h desde el fin de la tormenta"
        if r.get("area_km2") is None:
            parts.append(f"{name}: {r.get('reason', 'sin imagen útil')}")
            continue
        parts.append(f"{name}: {_fmt(r['area_km2'])} km² de agua nueva ({when})")
        used.append(f"{name} {r['id']}" + (f" {r['pass']} órb. {r['orbit']}" if name == "S1" else ""))
        dates.append(r["time"].strftime("%Y-%m-%d %H:%M UTC"))
    return "; ".join(parts) or "Sin imágenes en la ventana", " | ".join(used), " | ".join(dates)


def update_inventory(inv, results):
    """Fill the verification columns of rows still 'Pendiente'; rows verified by hand are left as they are."""
    inv = inv.copy()
    by_id = {r["id"]: r for r in results}
    for i, row in inv.iterrows():
        r = by_id.get(row["id"])
        if r is None or str(row.get("verificacion_imagenes", "")).strip() not in ("", "Pendiente"):
            continue
        inv.at[i, "verificacion_imagenes"] = r["verdict"]
        inv.at[i, "imagen_usada"] = r["imagen_usada"]
        inv.at[i, "fecha_imagen"] = r["fecha_imagen"]
        inv.at[i, "resultado"] = r["resultado"]
    return inv


# ------------------------------------------------------------------ GEE ---
def _size(col):
    return int(col.size().getInfo())


def _info(obj):
    return obj.getInfo()


def _ms(t):
    return int(t.timestamp() * 1000)


def _from_ms(ms):
    return dt.datetime.fromtimestamp(ms / 1000, tz=UTC)


class Verifier:
    def __init__(self, ee, aoi, a):
        self.ee, self.aoi, self.a = ee, aoi, a
        perm = ee.Image(GSW).select("occurrence").unmask(0).lt(50)
        dem_col = ee.ImageCollection(DEM).select("DSM")
        dem = dem_col.mosaic().setDefaultProjection(dem_col.first().projection())
        self.mask = perm.And(ee.Terrain.slope(dem).lte(a.max_slope_deg))

    def _date(self, t):
        return self.ee.Date(_ms(t))

    def clean(self, water):
        """New-water mask without permanent water, steep slopes and small patches."""
        water = water.And(self.mask).selfMask()
        return water.updateMask(water.connectedPixelCount(self.a.min_px + 1, True).gte(self.a.min_px))

    def _area_km2(self, water):
        ee = self.ee
        s = water.multiply(ee.Image.pixelArea()).reduceRegion(
            reducer=ee.Reducer.sum(), geometry=self.aoi, scale=20, maxPixels=1e10, bestEffort=True, tileScale=4)
        vals = _info(s.values()) or [0]
        return (vals[0] or 0) / 1e6

    def imerg_series(self, start, end):
        ee = self.ee
        col = ee.ImageCollection(IMERG).filterDate(self._date(start), self._date(end)).select("precipitation")
        aoi = self.aoi
        feats = col.map(lambda img: ee.Feature(None, {
            "t": img.get("system:time_start"),
            "p": img.reduceRegion(ee.Reducer.mean(), aoi, 10000).get("precipitation")}))
        rows = _info(feats.reduceColumns(ee.Reducer.toList(2), ["t", "p"]).get("list")) or []
        return [(_from_ms(t), p) for t, p in rows]

    def s1(self, start, end):
        ee, a = self.ee, self.a
        col = (ee.ImageCollection(S1).filterBounds(self.aoi)
               .filter(ee.Filter.eq("instrumentMode", "IW"))
               .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV")))
        post_col = col.filterDate(self._date(start), self._date(end + dt.timedelta(hours=a.max_lag_h))).sort("system:time_start")
        if _size(post_col) == 0:
            return None
        first = ee.Image(post_col.first())
        p = _info(ee.Dictionary({"t": first.get("system:time_start"), "orbit": first.get("relativeOrbitNumber_start"),
                                 "pass": first.get("orbitProperties_pass"), "id": first.get("system:index")}))
        t = _from_ms(p["t"])
        same = col.filter(ee.Filter.eq("relativeOrbitNumber_start", p["orbit"]))
        post = same.filterDate(self._date(t - dt.timedelta(hours=1)), self._date(t + dt.timedelta(hours=1))).select("VV").mosaic()
        pre_col = same.filterDate(self._date(start - dt.timedelta(days=60)), self._date(start - dt.timedelta(days=2))).select("VV")
        base = {"time": t, "id": p["id"], "orbit": int(p["orbit"]), "pass": p["pass"]}
        if _size(pre_col) == 0:
            return dict(base, area_km2=None, reason="sin escena de referencia de la misma órbita")
        post_f = post.focalMedian(30, "circle", "meters")
        pre_f = pre_col.median().focalMedian(30, "circle", "meters")
        water = self.clean(post_f.lte(a.s1_water_db).And(post_f.subtract(pre_f).lte(a.s1_drop_db)))
        return dict(base, area_km2=self._area_km2(water), water=water)

    def s2(self, start, end):
        ee, a = self.ee, self.a
        cs = ee.ImageCollection(CSPLUS)
        col = ee.ImageCollection(S2).filterBounds(self.aoi).linkCollection(cs, ["cs_cdf"])

        def mndwi(img):
            clear = img.select("cs_cdf").gte(0.6)
            out = img.normalizedDifference(["B3", "B11"]).rename("mndwi").updateMask(clear)
            return ee.Image(out.copyProperties(img, ["system:time_start", "system:index"]))   # copyProperties returns an Element

        post_col = col.filterDate(self._date(start), self._date(end + dt.timedelta(hours=a.max_lag_h))) \
                      .sort("system:time_start").map(mndwi)
        n = _size(post_col)
        if n == 0:
            return None
        lst = post_col.toList(n)
        aoi = self.aoi
        for i in range(n):                                   # first scene clear enough over the AMA
            img = ee.Image(lst.get(i))
            clear = _info(img.mask().reduceRegion(ee.Reducer.mean(), aoi, 100, maxPixels=1e9, bestEffort=True).get("mndwi"))
            if clear is not None and clear >= a.s2_min_clear:
                p = _info(ee.Dictionary({"t": img.get("system:time_start"), "id": img.get("system:index")}))
                pre = col.filterDate(self._date(start - dt.timedelta(days=60)), self._date(start - dt.timedelta(days=2))) \
                         .map(mndwi).median()
                water = self.clean(img.gt(0).And(pre.lt(0)))
                return dict(time=_from_ms(p["t"]), id=p["id"], area_km2=self._area_km2(water), water=water)
        last = _info(ee.Image(lst.get(n - 1)).get("system:time_start"))
        return dict(time=_from_ms(last), id="", area_km2=None, reason=f"{n} escena(s), ninguna despejada sobre el AMA")

    def export(self, row_id, source, water):
        """Export a confirmed event's new-water mask to Drive (folder AMA_flash) for visual checking."""
        task = self.ee.batch.Export.image.toDrive(
            image=water.toByte(), description=f"flash_{row_id}_{source}", folder="AMA_flash",
            fileNamePrefix=f"flash_{row_id}_{source}", region=self.aoi, scale=20, crs="EPSG:32721", maxPixels=1e10)
        task.start()


def run(inv, a, ee=None):
    rows = inv[inv["tipo"].astype(str).str.startswith("Repentina")]
    if a.ids:
        rows = rows[rows["id"].isin(a.ids)]
    if a.limit:
        rows = rows.head(a.limit)
    if ee is not None:
        aoi = ee.FeatureCollection(a.aoi_asset).geometry() if a.aoi_asset else ee.Geometry.Rectangle(a.bbox)
        v = Verifier(ee, aoi, a)
    results = []
    for _, row in rows.iterrows():
        start, end = event_window(row["fecha"], row.get("descripcion", ""))
        peak = None
        if ee is None:
            print(f"{row['id']:14s} {start:%Y-%m-%d %H:%M} -> {end:%Y-%m-%d %H:%M} UTC  (+{a.max_lag_h} h para imágenes)")
            continue
        if a.imerg:
            start, end, peak = refine_with_peak(start, end, v.imerg_series(start, end))
        r1 = v.s1(start, end)
        r2 = v.s2(start, end) if a.s2 else None
        res, used, dates = describe(r1, r2, end)
        vd = verdict(r1, r2, a.min_area_km2)
        if vd == "Confirmada" and a.export_masks:
            for name, r in (("S1", r1), ("S2", r2)):
                if r and r.get("area_km2") is not None and r["area_km2"] >= a.min_area_km2:
                    v.export(row["id"], name, r["water"])
        results.append(dict(id=row["id"], start_utc=start.isoformat(), end_utc=end.isoformat(),
                            imerg_peak_utc=peak[0].isoformat() if peak else "", imerg_peak_mm_h=peak[1] if peak else "",
                            s1_area_km2=r1.get("area_km2") if r1 else "", s2_area_km2=r2.get("area_km2") if r2 else "",
                            verdict=vd, imagen_usada=used, fecha_imagen=dates, resultado=res))
        print(f"{row['id']:14s} {vd:16s} {res}")
    return results


def main(a):
    inv = pd.read_csv(a.inventory, dtype=str, keep_default_na=False)
    if a.dry_run:
        run(inv, a, ee=None)
        return
    from gee_utils import init_ee
    ee = init_ee(a.project)
    results = run(inv, a, ee=ee)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    pd.DataFrame(results).to_csv(a.out, index=False)
    out_inv = os.path.join(os.path.dirname(a.out) or ".", "ama_flood_inventory_verificado.csv")
    update_inventory(inv, results).to_csv(out_inv, index=False)
    n = pd.Series([r["verdict"] for r in results]).value_counts().to_dict()
    print(f"\n{len(results)} tormentas verificadas: {n}\n-> {a.out}\n-> {out_inv}")


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--inventory", default="data/events/ama_flood_inventory.csv")
    p.add_argument("--out", default="results/flash_flood_verification.csv")
    p.add_argument("--project", default=os.environ.get("GEE_PROJECT"), help="GEE Cloud project (default: $GEE_PROJECT)")
    p.add_argument("--aoi-asset", default=os.environ.get("AMA_AOI_ASSET") or None, help="AMA polygon asset (default: $AMA_AOI_ASSET)")
    p.add_argument("--bbox", type=float, nargs=4, default=[-57.75, -25.55, -57.30, -25.05], help="fallback AOI (lon/lat)")
    p.add_argument("--ids", nargs="+", help="only these inventory ids")
    p.add_argument("--limit", type=int)
    p.add_argument("--max-lag-h", type=float, default=72, help="hours after the storm in which an image still counts")
    p.add_argument("--imerg", action="store_true", help="narrow the window to the IMERG rain peak over the AMA")
    p.add_argument("--s2", action="store_true", help="also check Sentinel-2")
    p.add_argument("--s1-water-db", type=float, default=-15.0)
    p.add_argument("--s1-drop-db", type=float, default=-3.0)
    p.add_argument("--s2-min-clear", type=float, default=0.3, help="minimum clear fraction of the AMA in the S2 scene")
    p.add_argument("--max-slope-deg", type=float, default=5.0)
    p.add_argument("--min-px", type=int, default=10, help="minimum patch size (pixels at 20 m)")
    p.add_argument("--min-area-km2", type=float, default=0.05, help="new water needed to confirm (0.05 km2 = 5 ha)")
    p.add_argument("--export-masks", action="store_true", help="export new-water masks of confirmed events to Drive")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
