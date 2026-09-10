# Benchmark summary

MAE in mg/L (lower is better); E1 entries are mean±std over seeds 42/43/44.

| model                   | e1_r20    | e1_r40    | e1_r60    |   e2a_strict |   e2b_partial | e3_spatial   |
|:------------------------|:----------|:----------|:----------|-------------:|--------------:|:-------------|
| B0_station_mean         | 1.46±0.06 | 1.48±0.04 | 1.48±0.02 |         1.14 |          1.12 | 2.59±0.45    |
| B1_kriging              | 2.15±0.10 | 2.17±0.04 | 2.22±0.02 |       nan    |          1.51 | 2.05±0.29    |
| B2_random_forest        | 1.42±0.06 | 1.44±0.04 | 1.46±0.02 |         1.11 |          1.11 | 2.07±0.20    |
| B3_mlp                  | 1.87±0.09 | 1.86±0.05 | 1.88±0.01 |         1.31 |          1.29 | 2.02±0.24    |
| G0_gcn_none             | 1.81±0.09 | 1.88±0.04 | 1.87±0.06 |         1.4  |          1.38 | 1.68         |
| G0_gcn_random           | 2.10±0.10 | 2.15±0.06 | 2.14±0.02 |         1.27 |          1.24 | 2.39         |
| G0_gcn_river            | 1.74±0.01 | 1.77±0.01 | 1.87±0.09 |         1.45 |          1.38 | 1.97±0.31    |
| H15_directed_river      | 1.69±0.08 | nan       | nan       |       nan    |          1.15 | nan          |
| H1_directed_river       | 1.50±0.05 | 1.55±0.05 | 1.62±0.04 |         1.09 |          1.07 | 2.03±0.23    |
| H2E_transport_river     | 1.41±0.07 | 1.42±0.02 | 1.50±0.01 |         1.18 |          1.13 | 1.81±0.21    |
| H2X_climate_river       | nan       | nan       | nan       |       nan    |          1.06 | 2.05±0.33    |
| H2X_hydro_river         | nan       | nan       | nan       |       nan    |          1.01 | 1.98±0.30    |
| H2X_landcover_river     | nan       | nan       | nan       |       nan    |          1.05 | 2.09±0.21    |
| H2X_soil_topo_river     | nan       | nan       | nan       |       nan    |          0.99 | 1.97±0.24    |
| H2X_transport_enc_river | 1.40±0.09 | nan       | nan       |       nan    |          0.96 | 1.79±0.28    |
| H2_transport_river      | 1.41±0.08 | 1.43±0.04 | 1.52±0.03 |         1.1  |          1.05 | 2.01±0.26    |

RMSE (mg/L), same layout:

| model                   |   e1_r20 |   e1_r40 |   e1_r60 |   e2a_strict |   e2b_partial |   e3_spatial |
|:------------------------|---------:|---------:|---------:|-------------:|--------------:|-------------:|
| B0_station_mean         |     4.74 |     5.35 |     5.29 |         1.73 |          1.7  |         7.03 |
| B1_kriging              |     5.38 |     5.97 |     5.93 |       nan    |          2.05 |         6.13 |
| B2_random_forest        |     4.75 |     5.39 |     5.33 |         1.77 |          1.76 |         6.28 |
| B3_mlp                  |     5.2  |     5.82 |     5.73 |         1.89 |          1.84 |         6.19 |
| G0_gcn_none             |     5.18 |     5.87 |     5.74 |         1.9  |          1.86 |         7.11 |
| G0_gcn_random           |     5.64 |     6.25 |     6.12 |         1.85 |          1.79 |         7.46 |
| G0_gcn_river            |     4.99 |     5.71 |     5.73 |         1.94 |          1.86 |         6.21 |
| H15_directed_river      |     4.96 |   nan    |   nan    |       nan    |          1.71 |       nan    |
| H1_directed_river       |     4.73 |     5.44 |     5.58 |         1.64 |          1.58 |         6.22 |
| H2E_transport_river     |     4.76 |     5.29 |     5.42 |         1.64 |          1.56 |         6.03 |
| H2X_climate_river       |   nan    |   nan    |   nan    |       nan    |          1.52 |         6.24 |
| H2X_hydro_river         |   nan    |   nan    |   nan    |       nan    |          1.49 |         6.12 |
| H2X_landcover_river     |   nan    |   nan    |   nan    |       nan    |          1.53 |         6.23 |
| H2X_soil_topo_river     |   nan    |   nan    |   nan    |       nan    |          1.44 |         6.15 |
| H2X_transport_enc_river |     4.73 |   nan    |   nan    |       nan    |          1.41 |         6.03 |
| H2_transport_river      |     4.73 |     5.27 |     5.43 |         1.6  |          1.54 |         6.22 |

R2, same layout:

| model                   |   e1_r20 |   e1_r40 |   e1_r60 |   e2a_strict |   e2b_partial |   e3_spatial |
|:------------------------|---------:|---------:|---------:|-------------:|--------------:|-------------:|
| B0_station_mean         |     0.43 |     0.36 |     0.35 |         0.25 |          0.23 |        -0.04 |
| B1_kriging              |     0.24 |     0.19 |     0.17 |       nan    |         -0.11 |         0.27 |
| B2_random_forest        |     0.43 |     0.35 |     0.33 |         0.21 |          0.18 |         0.21 |
| B3_mlp                  |     0.31 |     0.24 |     0.23 |         0.1  |          0.1  |         0.26 |
| G0_gcn_none             |     0.31 |     0.22 |     0.23 |         0.09 |          0.08 |        -2.34 |
| G0_gcn_random           |     0.17 |     0.11 |     0.12 |         0.14 |          0.15 |        -2.68 |
| G0_gcn_river            |     0.36 |     0.27 |     0.23 |         0.06 |          0.08 |         0.25 |
| H15_directed_river      |     0.37 |   nan    |   nan    |       nan    |          0.22 |       nan    |
| H1_directed_river       |     0.43 |     0.33 |     0.27 |         0.32 |          0.34 |         0.25 |
| H2E_transport_river     |     0.42 |     0.37 |     0.31 |         0.32 |          0.35 |         0.32 |
| H2X_climate_river       |   nan    |   nan    |   nan    |       nan    |          0.39 |         0.25 |
| H2X_hydro_river         |   nan    |   nan    |   nan    |       nan    |          0.41 |         0.28 |
| H2X_landcover_river     |   nan    |   nan    |   nan    |       nan    |          0.38 |         0.24 |
| H2X_soil_topo_river     |   nan    |   nan    |   nan    |       nan    |          0.45 |         0.27 |
| H2X_transport_enc_river |     0.43 |   nan    |   nan    |       nan    |          0.47 |         0.31 |
| H2_transport_river      |     0.43 |     0.38 |     0.31 |         0.35 |          0.37 |         0.23 |