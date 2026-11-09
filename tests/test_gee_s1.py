"""Sentinel-1 catalog / picker (pipeline/gee_s1.py). Earth Engine itself is mocked: nothing here talks to GEE."""
import datetime as dt
import importlib.util
import pathlib
from unittest import mock

import pytest

PIPE = pathlib.Path(__file__).resolve().parents[1] / "pipeline"
spec = importlib.util.spec_from_file_location("gee_s1", PIPE / "gee_s1.py")
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)


def row(date, orbit=68, cover=0.5, p="DESCENDING"):
    return {"date": date, "pass": p, "orbit": orbit, "platform": "A", "cover": cover}


def full(date, **kw):
    """Two adjacent frames that together cover the AOI."""
    return [row(date, cover=0.55, **kw), row(date, cover=0.5, **kw)]


def test_frames_of_one_pass_are_grouped_and_cover_is_capped():
    acq = g._by_acquisition(full("2019-05-20"))
    assert len(acq) == 1 and acq[0]["n_slices"] == 2 and acq[0]["cover"] == 1.0


def test_pick_prefers_full_cover_then_closest_date():
    rows = full("2019-05-14") + [row("2019-05-19", cover=0.4)] + full("2019-05-24") + full("2019-05-08")
    best, cands = g.pick_scene(rows, "2019-05-20", tol_days=12)
    assert best["date"] == "2019-05-24" and best["dias"] == 4 and best["ok"]       # 05-19 is closer but covers 40 %
    assert [c["ok"] for c in cands][:3] == [True, True, True] and not cands[-1]["ok"]


def test_pick_returns_none_when_nothing_covers_the_aoi():
    best, cands = g.pick_scene([row("2019-05-20", cover=0.4)], "2019-05-20")
    assert best is None and len(cands) == 1 and not cands[0]["ok"]
    best, _ = g.pick_scene([row("2019-05-20", cover=0.4)], "2019-05-20", min_cover=0.3)
    assert best["date"] == "2019-05-20"


def test_pick_respects_window_and_orbit_filters():
    rows = full("2019-05-20", orbit=68) + full("2019-05-21", orbit=170, p="ASCENDING") + full("2019-07-01", orbit=68)
    best, cands = g.pick_scene(rows, "2019-05-20", tol_days=3)
    assert len(cands) == 2 and best["orbit"] == 68                                   # 07-01 outside the window
    best, _ = g.pick_scene(rows, "2019-05-20", tol_days=3, orbit_pass="ASCENDING")
    assert best["orbit"] == 170
    best, _ = g.pick_scene(rows, "2019-05-20", tol_days=3, rel_orbit=999)
    assert best is None


def test_best_orbit_and_summary_table():
    rows = (full("2019-05-02", orbit=68) + full("2019-05-14", orbit=68) + full("2019-05-08", orbit=170, p="ASCENDING")
            + [row("2019-05-20", orbit=68, cover=0.2)] + full("2020-01-03", orbit=68))
    assert g.best_orbit(rows) == ("DESCENDING", 68, 3)
    assert g.best_orbit([row("2019-01-01", cover=0.1)]) is None
    t = g.summary_table(rows)
    assert t.loc["2019", "DESC 68"] == 2 and t.loc["2019", "ASCE 170"] == 1 and t.loc["2020", "DESC 68"] == 1


def test_monthly_targets_and_chunks():
    assert g.monthly_targets("2014-10-03", "2015-01-20") == ["2014-10-15", "2014-11-15", "2014-12-15", "2015-01-15"]
    ch = list(g._chunks("2014-10-03", "2016-01-01"))
    assert ch[0][0] == "2014-10-03" and ch[-1][1] == "2016-01-01" and all(a < b for a, b in ch)


def test_catalog_clips_start_to_first_scene_and_parses_features():
    t = int(dt.datetime(2019, 5, 20, 9, 30, tzinfo=dt.timezone.utc).timestamp() * 1000)
    feats = {"features": [{"properties": {"t": t, "pass": "DESCENDING", "orbit": 68, "platform": "A", "cover": 0.7}}]}
    col = mock.MagicMock()
    col.map.return_value.getInfo.return_value = feats
    with mock.patch.object(g, "_col", return_value=col) as m:
        rows = g.catalog(mock.MagicMock(), mock.MagicMock(), start="2010-01-01", end="2016-01-01", log=lambda *a: None)
    assert m.call_args_list[0].args[2] == g.S1_FIRST_DATE                            # nothing is asked before Oct 2014
    assert len(m.call_args_list) == 2                                                # 2014-10-03 -> 2016-01-01: two year chunks
    assert rows[0] == dict(date="2019-05-20", orbit=68, platform="A", cover=0.7, **{"pass": "DESCENDING"})


def test_export_scene_uses_the_same_parameters_as_02a():
    ee = mock.MagicMock()
    aoi = mock.MagicMock()
    pick = dict(date="2019-05-20", orbit=68, **{"pass": "DESCENDING"})
    with mock.patch.object(g, "_col", return_value=mock.MagicMock()) as m:
        task = g.export_scene(ee, aoi, pick, "2019_05", "ama_flood_S1")
    assert m.call_args.args[2:] == ("2019-05-20", "2019-05-21")                       # a single acquisition day
    kw = ee.batch.Export.image.toDrive.call_args.kwargs
    assert kw["description"] == "S1_2019_05" and kw["fileNamePrefix"] == "S1_2019_05" and kw["folder"] == "ama_flood_S1"
    assert kw["scale"] == 10 and kw["crs"] == "EPSG:32721"
    task.start.assert_not_called() if False else ee.batch.Export.image.toDrive.return_value.start.assert_called_once()


def test_first_date_constant_is_oct_2014():
    assert g.S1_FIRST_DATE == "2014-10-03"
