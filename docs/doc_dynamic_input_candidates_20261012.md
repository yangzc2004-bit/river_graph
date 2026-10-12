# Prospective inputs beyond the daily-flow experiment

This note identifies a possible next data package; it is not part of the running frozen experiment and makes no claim of DOC skill improvement. The environment MLP, GRU and directional graph Transformer would remain the backbone.

The current added descriptors summarize observed daily discharge on the existing monitoring footprint. They cannot resolve the mismatch between auxiliary-variable availability at measured DOC months and at naturally missing DOC months. They also do not represent dynamic terrestrial carbon mobilization independently of discharge.

A small globally available candidate package is monthly precipitation, two-metre air temperature and upper-layer soil moisture from ERA5-Land. The official monthly product provides global 0.1-degree fields from 1950 onward, including these variables. Monthly release delay is approximately two to three months, so the latest project calendar months may need to be marked unavailable. These are reanalysis estimates, not river concentration observations or measured river discharge. Air temperature must not be relabelled as water temperature.

Sources checked on 2026-10-12: [Copernicus product overview](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land-monthly-means?tab=overview), [available variables](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land-monthly-means?tab=download), and [ECMWF accumulation and monthly-mean conventions](https://confluence.ecmwf.int/pages/viewpage.action?pageId=355336451). The accumulation variables in monthly means of daily means have per-day units; conversion to a monthly total requires the actual month length. Soil moisture and air temperature follow their own units and do not use this accumulation conversion.

If tested, catchment-area summaries would have a clearer spatial interpretation than assigning the station grid cell to the entire upstream catchment. Grid-to-catchment mapping, spatial coverage, units, release dates and calendar conventions must be fixed before fitting. Predictor normalization would use training basins only. No DOC would enter these inputs. A paired availability-matched original-input control and identical tree input packages would distinguish data benefit from architecture benefit.

Daily sample-date alignment is a separate target-design question. The current labels usually have a single concentration sampling day, whereas full-month covariates describe all days. A sample-date study would require a new explicitly versioned target and evaluation protocol. It should not silently reinterpret or replace the existing monthly scores.

Data acquisition and these experiments have not been started. The present four-recipe study should finish first, with native/log MAE, centered temporal skill and change error interpreted jointly.
