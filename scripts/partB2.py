"""Part B detail: verdict flips and trigger shift among true models that are metastable."""
import sys, json
sys.path.insert(0, sys.argv[1])
from model import analyze
grid = [(S, T, R, Q) for S in range(2, 6) for T in range(2, 7) for R in range(1, 5) for Q in range(6, 13) if Q > T]
c = {}
def g(p):
    if p not in c:
        a = analyze(*p); c[p] = (bool(a["strong"]), a["trigger"])
    return c[p]
names = ["S", "T", "Rmax", "Qmax"]
for nm_k in [None, 0, 1, 2, 3]:
    lost = gained = n = same = within1 = 0; shifts = []
    for p in grid:
        for k in range(4):
            if nm_k is not None and k != nm_k: continue
            for d in (-1, 1):
                q = list(p); q[k] += d; q = tuple(q)
                if q[0] < 1 or q[1] < 0 or q[2] < 1 or q[3] <= q[1]: continue
                bv, bt = g(p); nv, nt = g(q)
                if not bv: 
                    gained += nv; continue
                n += 1
                if not nv: lost += 1; continue
                shifts.append(abs((nt or 0) - (bt or 0)))
    import numpy as np
    s = np.array(shifts)
    print(f"{'all' if nm_k is None else 'off in '+names[nm_k]:10s}: metastable truths' neighbours={n}, says not metastable={lost} ({lost/n:.1%}); "
          f"of the rest, trigger same {np.mean(s==0):.1%}, within 1 token {np.mean(s<=1):.1%}, max shift {s.max()}; "
          f"false alarms from non-metastable truths: {gained}")
