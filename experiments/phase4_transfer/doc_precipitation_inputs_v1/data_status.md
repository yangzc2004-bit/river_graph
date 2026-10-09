# Monthly precipitation information: access status

The completed source-auxiliary and reference-trajectory studies did not improve
the retained DOC model. Monthly precipitation and antecedent accumulation are
a proposed additional environmental information source, not a result or a
confirmed missing cause of DOC errors.

No local precipitation/reanalysis cache was found under project data/cache
paths. Existing daily discharge already retains all120,541 cached grid months;
its frozen monthly mask excludes zero available daily-flow months. Rebuilding
that footprint therefore cannot expand the hydrological information.

Official NASA POWER documentation confirms a monthly point service, native
meteorological resolution and1981-onward historical availability. These sources
inform data selection only:

- https://power.larc.nasa.gov/docs/services/api/temporal/monthly/
- https://power.larc.nasa.gov/docs/tutorials/service-data-request/api/
- https://gis.earthdata.nasa.gov/portal/home/item.html?id=8f19b884dbfd4441beeb2ae9e26abae5

A public2001 precipitation unit probe at40N/95W failed DNS resolution through
the UV-managed runtime (`api_probe.json`). The same public API response was
not accessible through the web tool. Documentation access succeeded; actual
parameter metadata, unit conversion and dataset retrieval remain unverified.
No weather data or new weather-trained model exists in this directory. Do not
invent precipitation, replace missing values with climate normals silently,
or change old datasets. Future ingestion should check response units, missing
sentinels, actual calendar coverage and grid/site alignment before making a
separate versioned label-free product. Request each native grid cell once, with
limited concurrency, and fit all downstream scalers on source training data.

This access problem affects the proposed new weather input. Offline DOC-aligned
environmental-reference development can continue under its separate study plan
without further user approval or any change to the retained portable model.
