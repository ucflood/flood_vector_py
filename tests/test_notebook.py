"""The Colab notebook: valid format, every cell compiles, and every pipeline call uses arguments the script accepts."""
import json
import pathlib
import re
import shlex
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
NB = REPO / "notebooks" / "AMA_flood_susceptibility_colab.ipynb"
CALL = re.compile(r"^(?:!python|%run) pipeline/(?P<script>[\w]+)\.py(?P<args>.*)$")
# the 00c call receives its arguments through a variable built in the same cell
EXPAND = {"{args}": "--layers-dir X --aoi X --out-dir X --geology X --geology-field F --soil X --soil-field F",
          "{EPOCHS}": "100"}


def _cells():
    nb = json.loads(NB.read_text())
    return [("".join(c["source"]) if isinstance(c["source"], list) else c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def test_notebook_format():
    nbformat = pytest.importorskip("nbformat")
    nbformat.validate(nbformat.read(NB, as_version=4))


def test_cells_compile():
    for i, src in enumerate(_cells()):
        py = "\n".join(re.sub(r"^(\s*)[!%].*$", r"\1pass", line) for line in src.splitlines())
        compile(py, f"cell {i}", "exec")


def _calls():
    out = []
    for src in _cells():
        for line in src.splitlines():
            m = CALL.match(line.strip())
            if m:
                args = m.group("args")
                for k, v in EXPAND.items():
                    args = args.replace(k, v)
                args = re.sub(r'"?\{[^}]+\}"?', "X", args)       # notebook variables -> dummy value
                out.append((m.group("script"), shlex.split(args)))
    return out


def test_every_stage_is_called():
    called = {s for s, _ in _calls()}
    expected = {p.stem for p in (REPO / "pipeline").glob("*.py") if p.stem[:2].isdigit() or p.stem == "rain_events_10min"}
    assert expected <= called, f"not in the notebook: {expected - called}"


@pytest.mark.parametrize("script,argv", _calls())
def test_call_arguments_are_valid(load, script, argv):
    if script in ("00a_gee_export_layers", "02a_gee_export_s1"):     # configured through environment variables
        assert argv == []
        return
    mod = load(script)
    mod.parse_args(argv)                                             # argparse exits (fails the test) on a bad flag
