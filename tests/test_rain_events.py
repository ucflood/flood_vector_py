import argparse
import numpy as np
import pandas as pd


def _series(tmp_path, rows):
    t = pd.date_range("2025-06-04 00:00", periods=24 * 6 * 2, freq="10min")
    rain = np.zeros(len(t))
    for ts, mm in rows:
        rain[t.get_loc(pd.Timestamp(ts))] = mm
    df = pd.DataFrame({"timestamp": t.strftime("%Y-%m-%d %H:%M"), "precip_mm": rain})
    p = tmp_path / "rain.csv"; df.to_csv(p, index=False)
    return p


def test_classes_match_table_5(load):
    m = load("rain_events_10min")
    assert m.classify(2.9) is None
    assert m.classify(3.0) == "No intenso" and m.classify(4.99) == "No intenso"
    assert m.classify(5.0) == "Intenso" and m.classify(9.99) == "Intenso"
    assert m.classify(10.0) == "Muy intenso"


def test_events_and_candidates(load, tmp_path):
    m = load("rain_events_10min")
    p = _series(tmp_path, [
        ("2025-06-05 04:10", 2.0), ("2025-06-05 04:20", 12.5), ("2025-06-05 04:30", 6.0),   # muy intenso
        ("2025-06-05 05:10", 1.0),                                                         # same event (gap 30 min)
        ("2025-06-04 15:00", 4.2),                                                         # no intenso, day before
        ("2025-06-05 20:00", 1.5),                                                         # below 3 mm -> dropped
    ])
    s = m.load_series(p, "timestamp", "precip_mm", "UTC")
    ev = m.find_events(s, 60)
    assert list(ev["clase"]) == ["No intenso", "Muy intenso"]
    big = ev.iloc[1]
    assert big["acumulado_mm"] == 21.5 and big["pico_10min_mm"] == 12.5 and big["intensidad_max_mm_h"] == 75.0
    assert str(big["inicio_utc"]) == "2025-06-05 04:00:00+00:00"

    cand = tmp_path / "cand.csv"
    pd.DataFrame({"id": ["C-20250605", "T-x"], "fecha_dato_diario": ["2025-06-05", "2018-03-03"],
                  "clase_10min": ["Pendiente", "Intenso"], "inicio_utc": ["", "16:50"], "fin_utc": ["", "20:50"],
                  "acumulado_mm": ["", "26.7"], "pico_10min_mm": ["", "9.2"]}).to_csv(cand, index=False)
    out = m.fill_candidates(cand, ev, tmp_path / "out.csv")
    assert out.loc[0, "clase_10min"] == "Muy intenso"          # strongest event in the 12-12 UTC window
    assert out.loc[1, "clase_10min"] == "Intenso"              # thesis rows untouched


def test_missing_records_are_not_dry(load, tmp_path):
    m = load("rain_events_10min")
    df = pd.DataFrame({"timestamp": ["2025-01-01 00:00", "2025-01-01 00:30"], "precip_mm": [0, 0]})
    p = tmp_path / "gap.csv"; df.to_csv(p, index=False)
    s = m.load_series(p, "timestamp", "precip_mm", "UTC")
    assert len(s) == 4 and int(s.isna().sum()) == 2


def test_cli(load, tmp_path):
    m = load("rain_events_10min")
    p = _series(tmp_path, [("2025-06-05 04:20", 11.0)])
    m.main(argparse.Namespace(input=str(p), out=str(tmp_path / "ev.csv"), time_col="timestamp", rain_col="precip_mm",
                              tz="UTC", monthfirst=False, dry_gap_min=60, candidates=None))
    assert pd.read_csv(tmp_path / "ev.csv")["clase"].tolist() == ["Muy intenso"]


def test_dayfirst_local_time_and_missing_window(load, tmp_path):
    """05/06/2025 01:20 local (UTC-3) = 5 June 04:20 UTC; candidates outside the 10-min record stay Pendiente."""
    m = load("rain_events_10min")
    t = pd.date_range("2025-06-04 00:00", "2025-06-06 00:00", freq="10min")
    r = np.zeros(len(t)); r[t.get_loc(pd.Timestamp("2025-06-05 01:20"))] = 12.0
    p = tmp_path / "local.csv"
    pd.DataFrame({"fecha": t.strftime("%d/%m/%Y %H:%M"), "pp": r}).to_csv(p, index=False)
    s = m.load_series(p, "fecha", "pp", "America/Asuncion")
    assert s.index.min() == pd.Timestamp("2025-06-04 03:00", tz="UTC") and int(s.isna().sum()) == 0
    ev = m.find_events(s, 60)
    assert str(ev.iloc[0]["inicio_utc"]) == "2025-06-05 04:10:00+00:00"
    cand = tmp_path / "cand.csv"
    pd.DataFrame({"id": ["a", "b"], "fecha_dato_diario": ["2025-06-05", "2019-03-08"], "clase_10min": ["Pendiente"] * 2,
                  "inicio_utc": ["", ""], "fin_utc": ["", ""], "acumulado_mm": ["", ""], "pico_10min_mm": ["", ""]}).to_csv(cand, index=False)
    out = m.fill_candidates(cand, ev, tmp_path / "o.csv", series=s)
    assert out.loc[0, "clase_10min"] == "Muy intenso"
    assert out.loc[1, "clase_10min"] == "Pendiente"


def test_flash_flood_flag_follows_classification(load, tmp_path):
    m = load("rain_events_10min")
    p = _series(tmp_path, [("2025-06-05 04:20", 12.5), ("2025-06-04 20:00", 1.0)])
    s = m.load_series(p, "timestamp", "precip_mm", "UTC")
    ev = m.find_events(s, 60)
    cand = tmp_path / "c.csv"
    flag = "Probable si se confirma el umbral (verificar con imágenes)"
    pd.DataFrame({"id": ["a", "b"], "fecha_dato_diario": ["2025-06-05", "2025-06-06"], "clase_10min": ["Pendiente"] * 2,
                  "inicio_utc": ["", ""], "fin_utc": ["", ""], "acumulado_mm": ["", ""], "pico_10min_mm": ["", ""],
                  "inundacion_repentina_ama": [flag, flag]}).to_csv(cand, index=False)
    out = m.fill_candidates(cand, ev, tmp_path / "o.csv", series=s)
    assert out.loc[0, "inundacion_repentina_ama"] == "Probable (verificar con imágenes)"
    assert out.loc[1, "clase_10min"].startswith("Sin evento")
    assert out.loc[1, "inundacion_repentina_ama"].startswith("Poco probable")


def test_own_criterion_10mm_in_30min(load, tmp_path):
    """--intense-mm 10 --window-min 30: counts 10 mm in 30 min, even when no 10-min record reaches 3 mm of Tabla 5."""
    m = load("rain_events_10min")
    p = _series(tmp_path, [
        ("2025-06-04 03:00", 3.4), ("2025-06-04 03:10", 3.4), ("2025-06-04 03:20", 3.4),   # 10.2 mm/30 min, peak 3.4
        ("2025-06-04 12:00", 9.0),                                                         # 9 mm in one record: < 10 in 30 min
        ("2025-06-05 04:20", 4.0), ("2025-06-05 04:30", 4.0), ("2025-06-05 04:40", 2.0),   # exactly 10.0 mm/30 min
        ("2025-06-05 20:00", 12.0),                                                        # >= 10 in one record
    ])
    s = m.load_series(p, "timestamp", "precip_mm", "UTC")
    thesis = m.find_events(s, 60)
    assert list(thesis["clase"]) == ["No intenso", "Intenso", "No intenso", "Muy intenso"]   # unchanged default
    own = m.find_events(s, 60, window_min=30, intense_mm=10)
    assert list(own["clase"]) == ["Intenso", "Intenso", "Intenso"]                        # 9.0 mm burst is dropped
    assert own["max_ventana_mm"].tolist() == [10.2, 10.0, 12.0]
    assert m.find_events(s, 60, window_min=10, intense_mm=10)["max_ventana_mm"].tolist() == [12.0]


def test_own_criterion_missing_records_are_a_lower_bound(load, tmp_path):
    m = load("rain_events_10min")
    t = pd.date_range("2025-06-05 00:00", periods=12, freq="10min")
    r = pd.Series(0.0, index=t.tz_localize("UTC")); r.iloc[4] = 6.0; r.iloc[5] = np.nan; r.iloc[6] = 5.0
    ev = m.find_events(r, 60, window_min=30, intense_mm=10)       # 6 + (missing) + 5 = at least 11 mm
    assert len(ev) == 1 and ev.iloc[0]["max_ventana_mm"] == 11.0 and ev.iloc[0]["registros_faltantes"] == 1
    r.iloc[6] = 3.0                                                 # 6 + missing + 3 = at least 9: not enough to be sure
    assert len(m.find_events(r, 60, window_min=30, intense_mm=10)) == 0


def test_own_criterion_candidates_and_cli(load, tmp_path):
    m = load("rain_events_10min")
    p = _series(tmp_path, [("2025-06-05 04:20", 4.0), ("2025-06-05 04:30", 4.0), ("2025-06-05 04:40", 2.0)])
    cand = tmp_path / "cand.csv"
    pd.DataFrame({"id": ["a", "b"], "fecha_dato_diario": ["2025-06-05", "2025-06-06"], "clase_10min": ["Pendiente"] * 2,
                  "inicio_utc": ["", ""], "fin_utc": ["", ""], "acumulado_mm": ["", ""], "pico_10min_mm": ["", ""]}).to_csv(cand, index=False)
    a = m.parse_args(["--input", str(p), "--out", str(tmp_path / "ev.csv"), "--window-min", "30", "--intense-mm", "10",
                      "--candidates", str(cand)])
    m.main(a)
    out = pd.read_csv(tmp_path / "ev_listado.csv", dtype=str, keep_default_na=False)
    assert out.loc[0, "clase_10min"] == "Intenso" and out.loc[0, "max_30min_mm"] == "10.0" and out.loc[0, "pico_10min_mm"] == "4.0"
    assert out.loc[1, "clase_10min"] == "Sin evento (< 10 mm en 30 min)"


def test_window_without_threshold_is_rejected(load):
    import pytest
    m = load("rain_events_10min")
    with pytest.raises(SystemExit):
        m.parse_args(["--input", "x.csv", "--out", "y.csv", "--window-min", "30"])
    with pytest.raises(ValueError):
        m.find_events(pd.Series(dtype=float), 60, window_min=25, intense_mm=10)


def test_daylight_saving_changes_do_not_crash(load, tmp_path):
    """America/Asuncion had DST until Oct 2024: 23 Mar 2014 23:00-23:59 happens twice, 5 Oct 2014 00:00-00:59 does not exist."""
    m = load("rain_events_10min")
    # logger that follows civil time: the repeated hour is in the file twice, the missing hour is absent
    t = list(pd.date_range("2014-03-22 22:00", "2014-03-22 23:50", freq="10min"))
    t += list(pd.date_range("2014-03-22 23:00", "2014-03-23 02:00", freq="10min"))
    r = [0.0] * len(t); r[t.index(pd.Timestamp("2014-03-22 23:30"))] = 11.0
    p = tmp_path / "civil.csv"
    pd.DataFrame({"fecha": [x.strftime("%Y-%m-%d %H:%M") for x in t], "pp": r}).to_csv(p, index=False)
    s = m.load_series(p, "fecha", "pp", "America/Asuncion")
    assert s.index.is_monotonic_increasing and s.index.is_unique
    assert s.loc["2014-03-23 02:30":"2014-03-23 02:30"].iloc[0] == 11.0     # first pass, UTC-3 (23:30 -> 02:30 UTC)

    # logger on a fixed clock: it has a record at 2014-10-05 00:30, which does not exist in civil time
    t2 = pd.date_range("2014-10-04 23:00", "2014-10-05 02:00", freq="10min")
    p2 = tmp_path / "fixed.csv"
    pd.DataFrame({"fecha": t2.strftime("%Y-%m-%d %H:%M"), "pp": 0.0}).to_csv(p2, index=False)
    assert len(m.load_series(p2, "fecha", "pp", "America/Asuncion")) > 0
    s3 = m.load_series(p2, "fecha", "pp", "Etc/GMT+3")                       # fixed UTC-3: nothing is lost
    assert s3.notna().sum() == len(t2)

    # repeated hour present only once in the file: cannot infer -> dropped with a warning, no crash
    t4 = pd.date_range("2014-03-22 22:00", "2014-03-23 01:00", freq="10min")
    p4 = tmp_path / "once.csv"
    pd.DataFrame({"fecha": t4.strftime("%Y-%m-%d %H:%M"), "pp": 0.0}).to_csv(p4, index=False)
    assert len(m.load_series(p4, "fecha", "pp", "America/Asuncion")) > 0
