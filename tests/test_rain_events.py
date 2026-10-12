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
