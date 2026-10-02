# data/

Nothing large is committed (see `.gitignore`). Expected layout:

| folder | content | produced by |
|---|---|---|
| `events/` | `ama_events_filtered.csv` - events usable for the inventory, with notes (tracked in git) | you |
| `external/` | `AMA.gpkg` (study-area polygon), national geology / soil vector maps, optional waterways | you |
| `raw/sen1floods11/` | split CSVs + `files/S1`, `files/Labels` (Sen1Floods11 HandLabeled) | `make sen1floods11-data` |
| `interim/layers/` | GEE exports (elevation, slope, ... lulc), S1 scenes, GSW occurrence, TWI, stream density | `00a`, `02a`, `00b` |
| `processed/stack/` | the 12 aligned indicator rasters + geology/soil code tables | `00c` |
