import re
t = open(r"D:\river_graph\deck_build\readback_v3.xml", encoding="utf-8").read()

forbidden = ["experiments/","src/",".md",".py","D:\\","frozen_results","parquet","pcode","NHDPlus","NWIS","StreamCat","benchmark","ablation","regime","clamp","dropout","GRU","Strahler","HUC","R-GCN","Kriging","Random Forest","NLDI","WQP","mask","MVP","Spearman","Fig ","Figure","MECHANISM","OUTLINE","seed"]
hits = {w: t.count(w) for w in forbidden if w in t}
print("forbidden in readback:", hits)

imgs = re.findall(r'<img[^>]*src="([^"]+)"', t)
print("img count:", len(imgs))
for i in imgs: print(" ", i)

# slides count
print("slides:", len(re.findall(r"<slide ", t)))
# notes count
print("notes:", len(re.findall(r"<note ", t)))
