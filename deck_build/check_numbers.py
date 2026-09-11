import glob, re, os

must = [
 "8.9","11,948","570","571","562","407","33,048","1972","2026",
 "1.46","1.48","1.14","1.12","2.59","2.15","2.17","2.22","1.51","2.05",
 "1.42","1.44","1.11","2.07","1.87","1.86","1.88","1.31","1.29","2.02",
 "1.74","1.77","1.87","1.45","1.38","1.97","1.50","1.55","1.62","1.09","1.07","2.03",
 "1.41","1.43","1.52","1.10","1.05","2.01","1.18","1.13","1.81","1.40","0.96","1.79",
 "0.47","0.33","0.08","0.53","0.31","0.28","2.79","2.21","3.24","2.52","2.30","2.04","0.27",
 "1.06","1.05","1.01","0.99","460","450","137.5","58.2","6.8","53","6.9","0.99","445",
 "99.5","06438000","06437000","212","160","1980","1978","28","81","45","29","32",
]
alltext = ""
for f in sorted(glob.glob(r"D:\river_graph\deck_build\pages_v3\p*.xml")):
    alltext += open(f, encoding="utf-8").read()
missing = [m for m in must if m not in alltext]
print("missing:", missing)
# forbidden path / file tokens
forbidden = ["experiments/","src/",".md",".py","D:\\","frozen_results","parquet","pcode","NHDPlus","NWIS","StreamCat","benchmark","ablation","regime","clamp","dropout","GRU","Strahler","HUC","R-GCN","Kriging","Random Forest"]
hits = [w for w in forbidden if w in alltext]
print("forbidden hits:", hits)
