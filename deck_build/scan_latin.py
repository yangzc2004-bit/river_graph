import re, glob, os

allowed = {
    # anchors
    "B0","B1","B2","B3","G0","H1","H2","H2E","H2X","H3",
    "E1","E2a","E2b","E3","P0","P1","P2","P3",
    # metric symbols
    "MAE","RMSE","DOC","pH",
}
pat = re.compile(r"[A-Za-z]{2,}")
for f in sorted(glob.glob(r"D:\river_graph\deck_build\pages_v3\p*.xml")):
    txt = open(f, encoding="utf-8").read()
    # remove formulas
    txt2 = re.sub(r"<formula>.*?</formula>", "", txt, flags=re.S)
    # take only <p>...</p> and note text
    for m in re.finditer(r"<p[^>]*>(.*?)</p>", txt2, flags=re.S):
        seg = m.group(1)
        words = pat.findall(seg)
        bad = [w for w in words if w not in allowed]
        if bad:
            print(os.path.basename(f), bad, "|", re.sub(r"<[^>]+>","",seg)[:80])
