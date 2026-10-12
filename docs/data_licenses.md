# Upstream data terms and pre-publication checklist

Nothing below was re-verified when this file was written: confirm each licence at the source before publishing.
"As far as I know" entries are from general knowledge, not checked against the provider today.

| Input | Terms (as far as I know) | What to do |
|---|---|---|
| Sen1Floods11 (training only) | CC BY 4.0 | Credit Bonafilia et al. (2020). Model weights are derived from it. |
| Sentinel-1 / Sentinel-2 | Copernicus: free, full and open | Credit: "Contains modified Copernicus Sentinel data [year]" |
| Cloud Score+ | Google dataset, check terms on its catalog page | Check |
| Landsat (paper option) | USGS, no use restrictions | Cite USGS/NASA |
| ALOS AW3D30 (DEM) | JAXA terms of use, credit required | **Read the terms** for redistribution of derived products |
| Copernicus GLO-30 (alternative DEM) | Copernicus licence | Check attribution wording |
| CHIRPS (rainfall) | Open | Check and cite Funk et al. |
| ESA WorldCover | CC BY 4.0 | Credit ESA / WorldCover |
| JRC Global Surface Water | Copernicus/JRC open data | Credit Pekel et al. (2016) |
| **OpenLandMap soil texture** (optional fallback) | **CC BY-SA 4.0**. ShareAlike is not compatible with NonCommercial | **Do not use it for a CC BY-NC release**, or release that layer separately under BY-SA |
| National geology / soil maps | Set by the issuing institution | **Get written permission** to redistribute derived products |
| Flood event list (`ama_events_filtered.csv`) | Factual summaries with their sources named | Keep sources; do not copy article text |

## Before you publish under CC BY-NC 4.0
1. **Co-ownership.** A written agreement with UCA (research office / legal) that both are licensors and that CC BY-NC 4.0 is the chosen licence.
   Without it, one co-owner may not be able to license alone. Rules depend on Paraguayan law and UCA's IP policy, which I have not seen.
2. **Other claims.** Check that no employer, project, grant or supervisor agreement gives a third party rights over the data.
3. **Upstream terms.** Resolve every row above, especially the SA layer and the national maps.
4. **Fill the placeholders** in `LICENSE`, `LICENSE-DATA.md` (name, contact e-mail, citation, DOI).
5. **Publish the data** (not in git; rasters are ignored by `.gitignore`) on a repository that mints a DOI, e.g. Zenodo, choosing "CC BY-NC 4.0".
6. Run `make license-text` and commit the official legal text.
