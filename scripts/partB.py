"""Part B: if a model is off by one token in one parameter, does the Section 3.5 search
find the same metastable states? Grid of true parameters; compare each with its 8 one-off neighbours."""
import sys, time, itertools, json
sys.path.insert(0, sys.argv[1])
from model import analyze
grid = [(S, T, R, Q) for S in range(2, 6) for T in range(2, 7) for R in range(1, 5) for Q in range(6, 13) if Q > T]
cache = {}
def get(p):
    if p not in cache:
        a = analyze(*p); cache[p] = (bool(a["strong"]), a["trigger"], frozenset(a["strong"]), frozenset(a["weak"]))
    return cache[p]
def jac(x, y): return 1.0 if not x and not y else len(x & y) / len(x | y)
t0 = time.time(); rows = []
names = ["S", "T", "Rmax", "Qmax"]
for p in grid:
    base = get(p)
    for k in range(4):
        for dlt in (-1, 1):
            q = list(p); q[k] += dlt; q = tuple(q)
            if q[0] < 1 or q[1] < 0 or q[2] < 1 or q[3] <= q[1]: continue
            nb = get(q)
            rows.append(dict(param=names[k], d=dlt, verdict_same=base[0] == nb[0], trig_same=base[1] == nb[1],
                             strong_same=base[2] == nb[2], weak_same=base[3] == nb[3],
                             jac_strong=jac(base[2], nb[2]), jac_weak=jac(base[3], nb[3]), base_has=base[0]))
n_true = len(grid); n_meta = sum(get(p)[0] for p in grid)
print(f"true parameter sets={n_true} (S 2-5, T 2-6, Rmax 1-4, Qmax 6-12, Qmax>T), with a strongly metastable state: {n_meta}; neighbours compared={len(rows)}; {time.time()-t0:.0f}s")
def summ(rs, label):
    n = len(rs)
    f = lambda k: sum(r[k] for r in rs) / n
    m = lambda k: sum(r[k] for r in rs) / n
    print(f"  {label:22s} n={n:4d}  same verdict {f('verdict_same'):6.1%}  same trigger {f('trig_same'):6.1%}  "
          f"identical strong set {f('strong_same'):6.1%}  identical weak set {f('weak_same'):6.1%}  "
          f"mean Jaccard strong {m('jac_strong'):.2f} weak {m('jac_weak'):.2f}")
summ(rows, "all")
summ([r for r in rows if r["base_has"]], "true model metastable")
for nm in names:
    summ([r for r in rows if r["param"] == nm], f"off by 1 in {nm}")
json.dump(rows, open(sys.argv[2], "w"))
