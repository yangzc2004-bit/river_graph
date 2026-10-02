# K-session mask-aware cross-analyte source selection

Only pH/EC observations at the K DOC support months are visible on
target stations. Missing profile dimensions are excluded pairwise from
source distance; they are never encoded as zero.

selected_profile_k=0
selected_source_k=160

 profile_k  source_k      mae
         0        40 1.830354
         0        80 1.834758
         0       160 1.808301
         1        40 1.912021
         1        80 1.836970
         1       160 1.837128
         3        40 1.897497
         3        80 1.818158
         3       160 1.825649
         5        40 1.894365
         5        80 1.815963
         5       160 1.829938

Selected outer rows:
 seed role  profile_k  source_k     rmse      mae       r2  log_rmse  log_mae   log_r2     pbias    n
   42 test          0       160 5.327034 2.469755 0.455582  0.373645 0.263471 0.613453  9.615469 2531
   43 test          0       160 5.379732 2.536702 0.444758  0.381198 0.273542 0.597667 10.258118 2531
   44 test          0       160 5.319022 2.500197 0.457219  0.374954 0.269457 0.610740  8.572780 2531
