import csv, statistics as st

rows = list(csv.DictReader(open(r'D:\river_graph\experiments\analysis\headwater_recovery.csv', encoding='utf-8')))
def f(x):
    return float(x)
hw = [r for r in rows if int(r['upstream_degree']) == 0]
huc = [r for r in hw if r['huc2'] in ('10', '11')]
for name, grp in [('all held-out', rows), ('headwater all', hw), ('headwater HUC10/11', huc)]:
    h2 = st.mean(f(r['mae_h2']) for r in grp)
    h2e = st.mean(f(r['mae_h2e']) for r in grp)
    print(f'{name}: n={len(grp)}  H2 MAE={h2:.3f}  H2E MAE={h2e:.3f}  drop={h2-h2e:.3f}')

# improvement-weighted: count improved
imp = [r for r in hw if f(r['improvement']) > 0]
print('headwater improved stations:', len(imp), '/', len(hw))
