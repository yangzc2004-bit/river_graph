import shutil, struct, pathlib

src = pathlib.Path(r'D:\river_graph')
dst = src / 'deck_build' / 'assets'
dst.mkdir(parents=True, exist_ok=True)

files = [
    r'docs\figures\fig1_scientific_question.png',
    r'docs\figures\fig2_experimental_framework.png',
    r'docs\figures\fig4_model_iteration.png',
    r'docs\figures\fig1a_station_map.png',
    r'docs\figures\fig1b_station_graph.png',
    r'experiments\analysis\e3_seed42_map.png',
]

def png_size(p):
    with open(p, 'rb') as f:
        head = f.read(24)
    w, h = struct.unpack('>II', head[16:24])
    return w, h

for rel in files:
    s = src / rel
    t = dst / s.name
    shutil.copy2(s, t)
    w, h = png_size(t)
    print(f'{s.name}: {w}x{h}  ratio={w/h:.4f}')
