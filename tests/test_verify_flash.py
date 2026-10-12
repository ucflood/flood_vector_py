"""Tests for 04_verify_flash_floods.py: pure logic + the GEE code path with `ee` mocked (no live service)."""
import datetime as dt
import pathlib
from unittest.mock import MagicMock
import pandas as pd
import pytest

UTC = dt.timezone.utc
REPO = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture()
def vf(load):
    return load("04_verify_flash_floods")


def test_event_windows(vf):
    # daily candidate "D-1 o D": 12 UTC of D-1 to 12 UTC of D
    s, e = vf.event_window("2025-06-04 o 2025-06-05")
    assert (s, e) == (dt.datetime(2025, 6, 4, 12, tzinfo=UTC), dt.datetime(2025, 6, 5, 12, tzinfo=UTC))
    # thesis row with times in the description
    s, e = vf.event_window("2018-10-07", "Evento muy intenso (tesis): pico de 14,5 mm en 10 min; 37,0 mm entre 10:40 y 12:40 UTC.")
    assert (s.hour, s.minute, e.hour, e.minute) == (10, 40, 12, 40)
    # crossing midnight
    s, e = vf.event_window("2018-11-13", "entre 22:10 y 01:00 UTC")
    assert e.date() == dt.date(2018, 11, 14)
    # press-only: whole local day (UTC-3)
    s, e = vf.event_window("2021-01-31", "172,5 mm en Asunción")
    assert s == dt.datetime(2021, 1, 31, 3, tzinfo=UTC) and e - s == dt.timedelta(days=1)


def test_refine_with_peak(vf):
    s, e = dt.datetime(2025, 6, 4, 12, tzinfo=UTC), dt.datetime(2025, 6, 5, 12, tzinfo=UTC)
    peak_t = dt.datetime(2025, 6, 5, 7, tzinfo=UTC)
    s2, e2, peak = vf.refine_with_peak(s, e, [(peak_t - dt.timedelta(hours=1), 5.0), (peak_t, 40.0), (peak_t, None)])
    assert peak == (peak_t, 40.0) and s2 == peak_t - dt.timedelta(hours=3) and e2 == peak_t + dt.timedelta(hours=3)
    assert vf.refine_with_peak(s, e, [(peak_t, 0.0)]) == (s, e, None)       # no rain: window unchanged


def test_verdict(vf):
    assert vf.verdict(None, None, 0.05) == "Sin imagen útil"
    assert vf.verdict({"area_km2": None}, None, 0.05) == "Sin imagen útil"      # S1 without reference scene
    assert vf.verdict({"area_km2": 0.01}, None, 0.05) == "No observada"
    assert vf.verdict({"area_km2": 0.01}, {"area_km2": 0.2}, 0.05) == "Confirmada"


def test_update_keeps_manual_rows(vf):
    inv = pd.DataFrame({"id": ["a", "b"], "verificacion_imagenes": ["Pendiente", "Confirmada"],
                        "imagen_usada": ["", "Planet 2024-04-15"], "fecha_imagen": ["", "2024-04-15"], "resultado": ["", "a mano"]})
    res = [dict(id=i, verdict="No observada", imagen_usada="S1 x", fecha_imagen="2024", resultado="0 km²") for i in "ab"]
    out = vf.update_inventory(inv, res)
    assert out.loc[0, "verificacion_imagenes"] == "No observada"
    assert out.loc[1, "verificacion_imagenes"] == "Confirmada" and out.loc[1, "resultado"] == "a mano"


def test_dry_run_on_real_inventory(vf, capsys):
    a = vf.parse_args(["--inventory", str(REPO / "data/events/ama_flood_inventory.csv"), "--dry-run"])
    vf.main(a)
    lines = [l for l in capsys.readouterr().out.splitlines() if l.strip()]
    inv = pd.read_csv(REPO / "data/events/ama_flood_inventory.csv", dtype=str, keep_default_na=False)
    assert len(lines) == inv["tipo"].str.startswith("Repentina").sum() == 170


def _queue(monkeypatch, vf, sizes, infos):
    sizes, infos = list(sizes), list(infos)
    monkeypatch.setattr(vf, "_size", lambda col: sizes.pop(0))
    monkeypatch.setattr(vf, "_info", lambda obj: infos.pop(0))
    return sizes, infos


def _inv():
    return pd.DataFrame({"id": ["C-20250605"], "fecha": ["2025-06-04 o 2025-06-05"], "tipo": ["Repentina (pluvial): probable"],
                         "descripcion": ["100-120 mm"], "verificacion_imagenes": ["Pendiente"], "imagen_usada": [""],
                         "fecha_imagen": [""], "resultado": [""]})


def test_mocked_gee_confirmed_with_export(vf, monkeypatch):
    t_post = dt.datetime(2025, 6, 6, 9, 30, tzinfo=UTC)
    ms = lambda t: int(t.timestamp() * 1000)
    sizes, infos = _queue(monkeypatch, vf, sizes=[1, 3, 2], infos=[
        [[ms(dt.datetime(2025, 6, 5, 6, tzinfo=UTC)), 5.0], [ms(dt.datetime(2025, 6, 5, 7, tzinfo=UTC)), 38.0]],  # IMERG
        {"t": ms(t_post), "orbit": 68, "pass": "DESCENDING", "id": "S1A_x"},                                       # S1 props
        [250000.0],                                                                                               # S1 area m2
        0.1, 0.9,                                                                                                 # S2 clear fractions
        {"t": ms(t_post + dt.timedelta(hours=4)), "id": "S2B_y"},                                                 # S2 props
        [0.0],                                                                                                    # S2 area
    ])
    ee = MagicMock()
    a = vf.parse_args(["--imerg", "--s2", "--export-masks"])
    res = vf.run(_inv(), a, ee=ee)
    assert not sizes and not infos                                   # every GEE call consumed in the expected order
    r = res[0]
    assert r["verdict"] == "Confirmada" and abs(r["s1_area_km2"] - 0.25) < 1e-9 and r["s2_area_km2"] == 0.0
    assert r["imerg_peak_utc"].startswith("2025-06-05T07:00") and "S1 S1A_x DESCENDING órb. 68" in r["imagen_usada"]
    assert ee.batch.Export.image.toDrive.call_count == 1             # only S1 passed the area threshold
    assert "0,25 km²" in r["resultado"]


def test_mocked_gee_no_image(vf, monkeypatch):
    _queue(monkeypatch, vf, sizes=[0, 0], infos=[])
    res = vf.run(_inv(), vf.parse_args(["--s2"]), ee=MagicMock())
    assert res[0]["verdict"] == "Sin imagen útil" and res[0]["resultado"] == "Sin imágenes en la ventana"


def test_mocked_gee_no_water(vf, monkeypatch):
    ms = int(dt.datetime(2025, 6, 6, 9, 30, tzinfo=UTC).timestamp() * 1000)
    _queue(monkeypatch, vf, sizes=[1, 2], infos=[{"t": ms, "orbit": 68, "pass": "DESCENDING", "id": "S1A_x"}, [1000.0]])
    res = vf.run(_inv(), vf.parse_args([]), ee=MagicMock())
    assert res[0]["verdict"] == "No observada"
