"""Part C: fit the retry-storm model to a synthetic incident from noisy window averages.
Truth: a parameter set from the Part B grid. Incident: Poisson base load (mean = S/2) for 300
rounds, one spike of Qmax tokens at round 100. Request counts per round are known exactly
(access logs). Internal metrics (served, retries, and optionally the queue gauge) are seen only
as W-round averages with multiplicative Gaussian noise sigma. Fit = least squares over the whole
grid. Optional 'jitter': the true server handles S-1, S or S+1 tokens per round (0.2/0.6/0.2).
Usage: python3 -I partC.py <scripts dir> <trials> <seed> [W,noise,gauge01,jitter01 ; ...]"""
import sys, itertools
import numpy as np
sys.path.insert(0, sys.argv[1])
from model import analyze
trials, seed = int(sys.argv[2]), int(sys.argv[3])
grid = [(S, T, R, Q) for S in range(2, 6) for T in range(2, 7) for R in range(1, 5) for Q in range(6, 13) if Q > T]
P = np.array(grid)                                   # vectorised simulation over all candidates
info = {p: analyze(*p) for p in grid}
def sim_all(inp, Ps, srv=None):
    n = len(Ps); S, T, R, Q = Ps.T
    w = np.zeros(n, int); sv = np.zeros(n, int); r = np.zeros(n, int)
    rec = np.zeros((3, len(inp), n))
    for t, i in enumerate(inp):
        rec[0, t], rec[1, t], rec[2, t] = sv, r, w
        Sx = S if srv is None else np.maximum(S + srv[t], 1)
        w, sv, r = np.clip(i + np.maximum(w - Sx, 0) + r, 0, Q), np.clip(w, 0, Sx), np.clip(w - T, 0, R)
    return rec                                        # metric x round x candidate
def windows(x, W): return x[:, : (x.shape[1] // W) * W].reshape(x.shape[0], -1, W, *x.shape[2:]).mean(2)
def jac(a, b): return 1.0 if not a and not b else len(a & b) / len(a | b)
rng = np.random.default_rng(seed)
settings = [(W, sg, gauge, jit) for jit in (False, True) for gauge in (True, False) for W in (1, 10, 30) for sg in (0.0, 0.1)]
if len(sys.argv) > 4:
    settings = [(int(a), float(b), c == '1', d == '1') for a, b, c, d in (x.split(',') for x in sys.argv[4].split(';'))]
print(f"trials per setting={trials}, candidates={len(grid)}, truths drawn uniformly from the grid ({sum(bool(info[p]['strong']) for p in grid)} of {len(grid)} metastable)")
for W, sg, gauge, jit in settings:
    res = []
    for _ in range(trials):
        p = grid[rng.integers(len(grid))]
        inp = rng.poisson(p[0] / 2, 300); inp[100] = p[3]
        srv = rng.choice([-1, 0, 1], 300, p=[.2, .6, .2]) if jit else None
        truth = sim_all(inp, np.array([p]), srv)[:, :, 0]
        obs = windows(truth, W) * (1 + sg * rng.standard_normal(windows(truth, W).shape))
        cand = windows(sim_all(inp, P), W)            # metric x window x candidate
        use = [0, 1, 2] if gauge else [0, 1]
        sse = ((cand[use] - obs[use][:, :, None]) ** 2).sum((0, 1))
        best = np.flatnonzero(sse <= sse.min() + 1e-9)
        f = grid[rng.choice(best)]; a, b = info[p], info[f]
        stuck_truth = truth[2, -30:].mean() > p[1]     # truth still above the timeout threshold at the end
        res.append(dict(exact=f == p, ties=len(best), maxoff=max(abs(x - y) for x, y in zip(f, p)),
                        verdict=bool(a["strong"]) == bool(b["strong"]), trig=a["trigger"] == b["trigger"],
                        jac=jac(a["strong"], b["strong"]), meta=bool(a["strong"]), Sok=f[0] == p[0],
                        storm=stuck_truth,
                        hi=(a['trigger'] is not None) and (b['trigger'] is None or b['trigger'] > a['trigger'])))
    m = lambda k, rs=res: np.mean([r[k] for r in rs])
    mm = [r for r in res if r["meta"]]
    nm = [r for r in res if not r["meta"]]
    fp = sum(not r["verdict"] for r in nm); fn = sum(not r["verdict"] for r in mm); tr = sum(r["trig"] for r in mm)
    print(f"  COUNTS W={W} noise={sg:.0%} gauge={'yes' if gauge else 'no'} jitter={'yes' if jit else 'no'}: metastable truths {len(mm)}, flagged {len(mm)-fn} ({(len(mm)-fn)/len(mm):.1%}), trigger right {tr} ({tr/len(mm):.1%}) | non-metastable truths {len(nm)}, falsely flagged metastable {fp} ({fp/len(nm):.1%})")
    print(f"W={W:2d} noise={sg:.0%} gauge={'yes' if gauge else 'no '} jitter={'yes' if jit else 'no '}: exact fit {m('exact'):5.0%}, "
          f"S right {m('Sok'):5.0%}, all params within 1 {np.mean([r['maxoff']<=1 for r in res]):5.0%}, tied best fits {np.mean([r['ties']>1 for r in res]):4.0%} | "
          f"verdict right {m('verdict'):5.0%}, trigger right {m('trig'):5.0%}, Jaccard strong {m('jac'):.2f} | metastable truths n={len(mm)}: verdict {m('verdict', mm):5.0%} trigger {m('trig', mm):5.0%}, fitted trigger too high or missing {m('hi', mm):5.0%}")
