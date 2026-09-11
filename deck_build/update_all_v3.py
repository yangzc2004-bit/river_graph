import subprocess, glob, os, sys, time

PRES = "EiRmsud2NlHZW9dUhrJcNLYqnhh"
CWD = r"D:\river_graph\deck_build"
files = sorted(glob.glob(os.path.join(CWD, "pages_v3", "p*.xml")))
results = []
for f in files:
    rel = "pages_v3/" + os.path.basename(f)
    sid = os.path.basename(f).split("_")[1].split(".")[0]
    cmd = ["lark-cli", "slides", "+update-slide",
           "--presentation", PRES,
           "--slide-id", sid,
           "--content", "@" + rel.replace("/", os.sep)]
    r = subprocess.run(cmd, cwd=CWD, capture_output=True, text=True, encoding="utf-8")
    ok = '"ok": true' in r.stdout or '"ok":true' in r.stdout
    print(os.path.basename(f), "->", "OK" if ok else "FAIL")
    if not ok:
        print("STDOUT:", r.stdout[:800])
        print("STDERR:", r.stderr[:800])
        sys.exit(1)
    results.append(sid)
    time.sleep(0.6)
print("ALL UPDATED:", len(results))
