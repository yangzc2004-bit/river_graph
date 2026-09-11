import subprocess, glob, os, sys, json

LINT = r"C:\Users\17493\AppData\Local\Doubao\User Data\Default\.doubao\agent_mode\workspace\.skills\ppt\scripts\xml_lint.py"
files = sorted(glob.glob(r"D:\river_graph\deck_build\pages_v3\p*.xml"))
bad = 0
for f in files:
    r = subprocess.run([sys.executable, LINT, "--input", f], capture_output=True, text=True, encoding="utf-8")
    out = r.stdout
    try:
        j = json.loads(out)
        s = j.get("summary", {})
        ec = s.get("error_count", "?")
        wc = s.get("warning_count", "?")
        issues = j.get("issues", [])
        errs = [i for i in issues if i.get("severity") == "error"]
        print(os.path.basename(f), "errors=", ec, "warnings=", wc)
        if ec not in (0, "0"):
            bad += 1
            for i in errs:
                print("   ERR:", i.get("code"), i.get("message","")[:160])
    except Exception as e:
        bad += 1
        print(os.path.basename(f), "LINT PARSE FAIL", e, out[:300], r.stderr[:300])
print("TOTAL BAD PAGES:", bad)
