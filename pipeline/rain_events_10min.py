"""Classify intense-rain events with the criterion of Báez Ferreira (2020), Tabla 5, from 10-minute station data.

  No intenso   3 mm <= rain in 10 min < 5 mm
  Intenso      5 mm <= rain in 10 min < 10 mm
  Muy intenso  rain in 10 min >= 10 mm

The thesis used station 86218 (Aeropuerto Silvio Pettirossi, DINAC). Input: a CSV with one row per 10-minute
record (timestamp + rain in mm). Missing records stay missing (they are not assumed dry).

  python pipeline/rain_events_10min.py --input dinac_86218_10min.csv --out results/rain_events_86218.csv
  python pipeline/rain_events_10min.py --input dinac_86218_10min.csv --out results/rain_events_86218.csv \
      --candidates data/events/rain_events_AMA_2018_2025.csv   # also fills the 'Pendiente' rows of the list

Event definition (assumption, the thesis does not state one): consecutive 10-min records with rain > 0, merged
when separated by less than --dry-gap-min minutes without rain. An event is kept if its 10-min peak is >= 3 mm.
"""
import argparse
import os
import numpy as np
import pandas as pd

CLASSES = [(10.0, "Muy intenso"), (5.0, "Intenso"), (3.0, "No intenso")]


def classify(peak_mm):
    for thr, name in CLASSES:
        if peak_mm >= thr:
            return name
    return None


def load_series(path, time_col, rain_col, tz, dayfirst=True):
    """dayfirst=True reads 05/06/2025 as 5 June (Paraguayan convention); ISO dates are unaffected."""
    df = pd.read_csv(path)
    raw = df[time_col].astype(str).str.strip()
    if raw.str.match(r"^\d{4}-\d{2}-\d{2}").all():
        t = pd.to_datetime(raw, format="ISO8601")      # dayfirst would swap month and day in ISO strings
    else:
        t = pd.to_datetime(raw, dayfirst=dayfirst)
    if t.dt.tz is None:
        t = t.dt.tz_localize(tz)
    s = pd.Series(pd.to_numeric(df[rain_col], errors="coerce").values, index=t.dt.tz_convert("UTC")).sort_index()
    s = s[~s.index.duplicated(keep="first")]
    full = pd.date_range(s.index.min().floor("10min"), s.index.max().ceil("10min"), freq="10min", tz="UTC")
    return s.reindex(full)          # gaps -> NaN


def find_events(s, dry_gap_min=60):
    wet = s.fillna(0) > 0
    idx = np.flatnonzero(wet.values)
    if len(idx) == 0:
        return pd.DataFrame(columns=["inicio_utc", "fin_utc", "duracion_h", "acumulado_mm", "pico_10min_mm",
                                     "intensidad_max_mm_h", "clase", "registros_faltantes"])
    max_gap = dry_gap_min // 10
    groups, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev - 1 >= max_gap:
            groups.append((start, prev)); start = i
        prev = i
    groups.append((start, prev))
    rows = []
    for a, b in groups:
        seg = s.iloc[a:b + 1]
        peak = float(seg.max())
        cls = classify(peak)
        if cls is None:
            continue
        rows.append(dict(inicio_utc=seg.index[0] - pd.Timedelta(minutes=10),   # a record covers the 10 min before it
                         fin_utc=seg.index[-1],
                         duracion_h=round((b - a + 1) / 6, 2),
                         acumulado_mm=round(float(seg.sum()), 1),
                         pico_10min_mm=peak,
                         intensidad_max_mm_h=round(peak * 6, 1),
                         clase=cls,
                         registros_faltantes=int(seg.isna().sum())))
    return pd.DataFrame(rows)


def daily_candidates(s, threshold=20.0):
    """Days (UTC) above the thesis download criterion: daily rain > 20 mm."""
    d = s.resample("1D").sum(min_count=1)
    return d[d > threshold]


def fill_candidates(cand_path, events, out_path, series=None):
    """Fill 'Pendiente' rows: events starting from 12 UTC of the day before the daily value to 12 UTC of that day.
    Rows whose window has no 10-min data stay 'Pendiente' (missing data is not the same as no event)."""
    c = pd.read_csv(cand_path, dtype=str, keep_default_na=False)
    for i, row in c.iterrows():
        if row.get("clase_10min") != "Pendiente" or not row.get("fecha_dato_diario"):
            continue
        d = pd.Timestamp(row["fecha_dato_diario"], tz="UTC")
        lo, hi = d - pd.Timedelta(hours=12), d + pd.Timedelta(hours=12)
        if series is not None and series[(series.index > lo) & (series.index <= hi)].notna().sum() == 0:
            continue
        hit = events[(events["inicio_utc"] >= lo) & (events["inicio_utc"] < hi)] if len(events) else events
        has_flag = "inundacion_repentina_ama" in c.columns
        if len(hit) == 0:
            c.at[i, "clase_10min"] = "Sin evento (pico < 3 mm/10 min)"
            if has_flag:
                c.at[i, "inundacion_repentina_ama"] = "Poco probable: no alcanza el umbral"
            continue
        top = hit.loc[hit["pico_10min_mm"].idxmax()]
        c.at[i, "clase_10min"] = top["clase"]
        if has_flag:   # storms at the thesis threshold produce flash floods in the AMA (to verify with imagery)
            c.at[i, "inundacion_repentina_ama"] = "Probable (verificar con imágenes)"
        c.at[i, "inicio_utc"] = top["inicio_utc"].strftime("%Y-%m-%d %H:%M")
        c.at[i, "fin_utc"] = top["fin_utc"].strftime("%Y-%m-%d %H:%M")
        c.at[i, "acumulado_mm"] = f"{top['acumulado_mm']:.1f}"
        c.at[i, "pico_10min_mm"] = f"{top['pico_10min_mm']:.1f}"
    c.to_csv(out_path, index=False)
    return c


def main(a):
    s = load_series(a.input, a.time_col, a.rain_col, a.tz, dayfirst=not a.monthfirst)
    ev = find_events(s, a.dry_gap_min)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    ev.to_csv(a.out, index=False)
    days = daily_candidates(s)
    by_class = ", ".join(f"{k} {v}" for k, v in ev["clase"].value_counts().items()) if len(ev) else "ninguno"
    print(f"{len(s)} registros de 10 min ({int(s.isna().sum())} faltantes); {len(days)} días > 20 mm; "
          f"{len(ev)} eventos ({by_class})")
    if a.candidates:
        out = a.out.replace(".csv", "_listado.csv")
        fill_candidates(a.candidates, ev, out, series=s)
        print("listado completado ->", out)


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="CSV of 10-minute rain records")
    p.add_argument("--out", required=True)
    p.add_argument("--time-col", default="timestamp")
    p.add_argument("--rain-col", default="precip_mm")
    p.add_argument("--tz", default="UTC", help="time zone of naive timestamps, e.g. America/Asuncion")
    p.add_argument("--monthfirst", action="store_true", help="dates like 06/05/2025 are month/day (default: day/month)")
    p.add_argument("--dry-gap-min", type=int, default=60)
    p.add_argument("--candidates", help="data/events/rain_events_AMA_2018_2025.csv to fill its 'Pendiente' rows")
    return p.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
