# data/

Nothing large is committed (see `.gitignore`). Expected layout:

| folder | content | produced by |
|---|---|---|
| `events/` | `ama_events_filtered.csv` - flood events usable for the Sentinel-1 inventory, with notes (tracked in git) | you |
| `events/` | `ama_flood_inventory.csv` - all AMA flood events: fluvial (user's list) + flash floods from severe storms, with imagery-verification columns | you (verification) |
| `events/` | `rain_events_AMA_2018_2025.csv` - intense-rain events at station 86218 (thesis criterion; candidates pending 10-min data) | `pipeline/rain_events_10min.py` fills the pending rows |
| `external/` | `AMA.gpkg` (study-area polygon), national geology / soil vector maps, optional waterways | you |
| `raw/sen1floods11/` | split CSVs + `files/S1`, `files/Labels` (Sen1Floods11 HandLabeled) | `make sen1floods11-data` |
| `interim/layers/` | GEE exports (elevation, slope, ... lulc), S1 scenes, GSW occurrence, TWI, stream density | `00a`, `02a`, `00b` |
| `processed/stack/` | the 12 aligned indicator rasters + geology/soil code tables | `00c` |

Outputs derived from these data are licensed CC BY-NC 4.0 (`../LICENSE-DATA.md`); check `../docs/data_licenses.md` before redistributing anything.
