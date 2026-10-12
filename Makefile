# Usage examples:  make test | make hydro | make stack GEOLOGY=data/external/geologia.gpkg GEOLOGY_FIELD=FORMACION ...
.RECIPEPREFIX := >
PY ?= python
AOI ?= data/external/AMA.gpkg
LAYERS ?= data/interim/layers
STACK ?= data/processed/stack
S1F11 ?= data/raw/sen1floods11
CKPT ?= results/checkpoints
INV ?= results/inventory
OUT ?= results/susceptibility

.PHONY: test hydro stack sen1floods11-data train inventory susceptibility license-text verify-flash

test:
> $(PY) -m pytest -q

# 00b: TWI + stream density from the (buffered) DEM exported by 00a. Optional: STREAMS=data/external/waterways.gpkg
hydro:
> $(PY) pipeline/00b_terrain_hydrology.py --dem $(LAYERS)/elevation.tif --out-dir $(LAYERS) $(if $(STREAMS),--streams $(STREAMS),)

# 00c: align + clip + rasterise vectors -> 12-layer stack. Optional: GEOLOGY/GEOLOGY_FIELD, SOIL/SOIL_FIELD
stack:
> $(PY) pipeline/00c_build_stack.py --layers-dir $(LAYERS) --aoi $(AOI) --out-dir $(STACK) \
>   $(if $(GEOLOGY),--geology $(GEOLOGY) --geology-field $(GEOLOGY_FIELD),) \
>   $(if $(SOIL),--soil $(SOIL) --soil-field $(SOIL_FIELD),)

# public bucket; needs gsutil (Google Cloud SDK)
sen1floods11-data:
> mkdir -p $(S1F11)/files/S1 $(S1F11)/files/Labels
> for s in train test valid; do gsutil cp gs://sen1floods11/v1.1/splits/flood_handlabeled/flood_$${s}_data.csv $(S1F11)/; done
> gsutil -m rsync -r gs://sen1floods11/v1.1/data/flood_events/HandLabeled/S1Hand $(S1F11)/files/S1
> gsutil -m rsync -r gs://sen1floods11/v1.1/data/flood_events/HandLabeled/LabelHand $(S1F11)/files/Labels

train:
> $(PY) pipeline/01_train_sen1floods11.py --train-csv $(S1F11)/flood_train_data.csv --valid-csv $(S1F11)/flood_valid_data.csv \
>   --test-csv $(S1F11)/flood_test_data.csv --s1-dir $(S1F11)/files/S1 --lbl-dir $(S1F11)/files/Labels --out-dir $(CKPT)

# make inventory EVENT=2019_05 POST=data/interim/S1_2019_flood.tif PRE=data/interim/S1_2019_ref_lowwater.tif PERM=data/interim/GSW_occurrence.tif
inventory:
> $(PY) pipeline/02_ama_flood_inventory.py --model $(CKPT)/unet_s1f11_best.keras --event $(EVENT) --post $(POST) \
>   $(if $(PRE),--pre $(PRE),) $(if $(PERM),--perm-water $(PERM),) --out-dir $(INV)

susceptibility:
> $(PY) pipeline/03_susceptibility_1dcnn.py --stack-dir $(STACK) --inventory $(INV)/flood_*.gpkg --out-dir $(OUT) \
>   $(if $(ZONE),--zone $(ZONE),)

# downloads the official CC BY-NC 4.0 legal text (needs internet)
license-text:
> curl -L -o LICENSE-DATA-CC-BY-NC-4.0.txt https://creativecommons.org/licenses/by-nc/4.0/legalcode.txt

# 04: verify which severe storms became flash floods (Sentinel-1/2 + IMERG). Needs GEE: PROJECT=... AOI_ASSET=...
verify-flash:
> $(PY) pipeline/04_verify_flash_floods.py --project $(PROJECT) $(if $(AOI_ASSET),--aoi-asset $(AOI_ASSET),) --imerg --s2 \
>   --out results/flash_flood_verification.csv $(if $(EXPORT),--export-masks,)
