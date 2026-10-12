"""Smoke tests for the two Google Earth Engine scripts with `ee` mocked out.

They catch Python errors (undefined names, bad branches, wrong export count) and typos in dataset IDs.
They do NOT prove the scripts work against the real GEE servers - see docs/gee_assets.md for what was checked by hand.
"""
import pathlib, sys
from unittest.mock import MagicMock
import pytest

PIPE = pathlib.Path(__file__).resolve().parents[1] / "pipeline"


def run(monkeypatch, name, **replace):
    src = (PIPE / name).read_text()
    for old, new in replace.items():
        assert old in src, f"{old!r} not found in {name}"
        src = src.replace(old, new)
    ee = MagicMock()
    monkeypatch.setitem(sys.modules, "ee", ee)
    exec(compile(src, name, "exec"), {"__name__": "gee_test"})
    ids = {c.args[0] for m in (ee.ImageCollection, ee.Image) for c in m.call_args_list if c.args and isinstance(c.args[0], str)}
    exports = [c.kwargs["description"] for c in ee.batch.Export.image.toDrive.call_args_list]
    return ids, exports


@pytest.mark.parametrize("sensor,cloud,ndwi,dem,expect_ids", [
    ("sentinel2", "cloudscore", "paper", "ALOS", {"COPERNICUS/S2_SR_HARMONIZED", "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED", "JAXA/ALOS/AW3D30/V4_1"}),
    ("sentinel2", "scl", "xu", "COPERNICUS", {"COPERNICUS/S2_SR_HARMONIZED", "COPERNICUS/DEM/GLO30"}),
    ("landsat", "scl", "mcfeeters", "ALOS", {"LANDSAT/LC08/C02/T1_L2", "LANDSAT/LC09/C02/T1_L2"}),
])
def test_00a(monkeypatch, sensor, cloud, ndwi, dem, expect_ids):
    ids, exports = run(monkeypatch, "00a_gee_export_layers.py",
                       **{'OPTICAL_SENSOR = "sentinel2"': f'OPTICAL_SENSOR = "{sensor}"',
                          'S2_CLOUD_METHOD = "cloudscore"': f'S2_CLOUD_METHOD = "{cloud}"',
                          'NDWI_DEF = "paper"': f'NDWI_DEF = "{ndwi}"',
                          'DEM_SOURCE = "ALOS"': f'DEM_SOURCE = "{dem}"'})
    assert exports == ["elevation", "slope", "aspect", "relief_amplitude", "ndvi", "ndwi", "rainfall", "lulc"]
    assert expect_ids <= ids
    assert {"UCSB-CHG/CHIRPS/DAILY", "ESA/WorldCover/v200"} <= ids


def test_00a_soil_option(monkeypatch):
    ids, exports = run(monkeypatch, "00a_gee_export_layers.py", **{"SOIL_OPENLANDMAP = False": "SOIL_OPENLANDMAP = True"})
    assert exports[-1] == "soil_openlandmap"
    assert "OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02" in ids


def test_00a_rejects_unknown_sensor(monkeypatch):
    with pytest.raises(ValueError):
        run(monkeypatch, "00a_gee_export_layers.py", **{'OPTICAL_SENSOR = "sentinel2"': 'OPTICAL_SENSOR = "modis"'})


def test_02a(monkeypatch):
    ids, exports = run(monkeypatch, "02a_gee_export_s1.py")
    assert {"COPERNICUS/S1_GRD", "JRC/GSW1_4/GlobalSurfaceWater"} <= ids
    assert exports[-1] == "GSW_occurrence" and len(exports) == 3


def test_exclude_ranges_match_events_table(monkeypatch):
    """The optical-composite exclusion windows in 00a must be exactly the AMA events flagged in the events table."""
    import csv
    src = (PIPE / "00a_gee_export_layers.py").read_text()
    monkeypatch.setitem(sys.modules, "ee", MagicMock())
    g = {"__name__": "gee_test"}
    exec(compile(src, "00a", "exec"), g)
    table = pathlib.Path(__file__).resolve().parents[1] / "data" / "events" / "ama_events_filtered.csv"
    with open(table, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["exclude_from_composite"].strip().lower() == "yes"]
    expected = sorted((r["exclude_start"], r["exclude_end"]) for r in rows)
    assert expected, "no events flagged in the table"
    assert sorted(g["EXCLUDE_RANGES"]) == expected
    assert all(a < b for a, b in expected)


# ---------------------------------------------------------------- Colab / env settings ---
class _Task:
    def __init__(self, name, states):
        self.name, self.states = name, list(states)

    def status(self):
        return {"state": self.states.pop(0) if len(self.states) > 1 else self.states[0], "description": self.name}


def test_wait_tasks(load):
    gu = load("gee_utils")
    logs = []
    gu.wait_tasks([_Task("a", ["RUNNING", "COMPLETED"]), _Task("b", ["COMPLETED"])], poll_s=0, log=logs.append, sleep=lambda s: None)
    assert "done: a" in logs and "done: b" in logs
    with pytest.raises(RuntimeError, match="export failed: c"):
        gu.wait_tasks([_Task("c", ["FAILED"])], poll_s=0, log=logs.append, sleep=lambda s: None)


def test_00a_env_settings_and_wait(monkeypatch):
    monkeypatch.setenv("AMA_AOI_ASSET", "projects/p/assets/AMA")
    monkeypatch.setenv("GEE_DRIVE_FOLDER", "MyLayers")
    monkeypatch.setenv("GEE_WAIT", "1")
    src = (PIPE / "00a_gee_export_layers.py").read_text()
    ee = MagicMock()
    ee.batch.Export.image.toDrive.return_value.status.return_value = {"state": "COMPLETED", "description": "x"}
    monkeypatch.setitem(sys.modules, "ee", ee)
    exec(compile(src, "00a", "exec"), {"__name__": "gee_test"})
    ee.FeatureCollection.assert_any_call("projects/p/assets/AMA")
    assert {c.kwargs["folder"] for c in ee.batch.Export.image.toDrive.call_args_list} == {"MyLayers"}
    assert ee.batch.Export.image.toDrive.return_value.status.called          # GEE_WAIT polled the tasks


def test_02a_scenes_from_env(monkeypatch):
    monkeypatch.setenv("S1_SCENES", '{"2025_06": ["2025-06-05", "2025-06-08", "DESCENDING"], "2025_ref": ["2025-04-01", "2025-04-20", "DESCENDING"]}')
    ids, exports = run(monkeypatch, "02a_gee_export_s1.py")
    assert exports == ["S1_2025_06", "S1_2025_ref", "GSW_occurrence"]
