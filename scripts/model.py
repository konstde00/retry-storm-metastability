"""Retry-storm model from "Modeling Metastability" (HotNets'25), Figure 1 / Section 3.2:
  wait(t)    = (in(t-1) + (wait(t-1) - S)_+ + rexmits(t-1)) clipped to [0, Qmax]
  served(t)  = wait(t-1) clipped to [0, S]
  rexmits(t) = (wait(t-1) - T) clipped to [0, Rmax]
  out(t)     = served(t-1)
State = (wait, served, rexmits). Section 3.5 search: weakly metastable states are reachable
states S for which the product FST started at (I, S) reaches a cycle with differing outputs.
Strongly metastable: condition (2) also holds after any finite extension, i.e. for every
state reachable from S."""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def step(s, i, S, T, R, Q):
    w, sv, r = s
    return (min(max(i + max(w - S, 0) + r, 0), Q), min(max(w, 0), S), min(max(w - T, 0), R))


def simulate(inputs, S, T, R, Q, s=(0, 0, 0)):
    out, states = [], []
    for i in inputs:
        out.append(s[1]); states.append(s)
        s = step(s, i, S, T, R, Q)
    return np.array(out), states, s


def analyze(S, T, R, Q, imax=None):
    """Return dict: reachable states, weak set, strong set, min single-spike trigger."""
    imax = Q if imax is None else imax
    I = (0, 0, 0)
    idx = {I: 0}; states = [I]; frontier = [I]
    while frontier:                                  # reachable states from I
        nxt = []
        for s in frontier:
            for i in range(imax + 1):
                t = step(s, i, S, T, R, Q)
                if t not in idx:
                    idx[t] = len(states); states.append(t); nxt.append(t)
        frontier = nxt
    n = len(states); m = imax + 1
    delta = np.array([[idx[step(s, i, S, T, R, Q)] for i in range(m)] for s in states])
    out = np.array([s[1] for s in states])
    # product graph over ordered pairs (a, b), a != b; merge -> terminate (no edge)
    A = np.repeat(np.arange(n), n); B = np.tile(np.arange(n), n)
    keep = A != B
    pid = -np.ones(n * n, dtype=np.int64); pid[keep] = np.arange(keep.sum())
    A, B = A[keep], B[keep]; npairs = len(A)
    src, dst = [], []
    for i in range(m):
        a2, b2 = delta[A, i], delta[B, i]
        ok = a2 != b2
        src.append(np.nonzero(ok)[0]); dst.append(pid[a2[ok] * n + b2[ok]])
    src = np.concatenate(src); dst = np.concatenate(dst)
    G = csr_matrix((np.ones(len(src), dtype=np.int8), (src, dst)), shape=(npairs, npairs))
    ncomp, lab = connected_components(G, directed=True, connection="strong")
    size = np.bincount(lab, minlength=ncomp)
    selfloop = np.zeros(npairs, bool); selfloop[src[src == dst]] = True
    cyclic = (size[lab] > 1) | selfloop
    differ = out[A] != out[B]
    # an SCC is "marked" if it is cyclic and contains a differing-output pair
    marked_comp = np.zeros(ncomp, bool); marked_comp[lab[cyclic & differ]] = True
    good = marked_comp[lab].copy()
    # backward reachability to marked SCCs
    Gt = G.T.tocsr(); stack = list(np.nonzero(good)[0])
    while stack:
        v = stack.pop()
        for u in Gt.indices[Gt.indptr[v]:Gt.indptr[v + 1]]:
            if not good[u]:
                good[u] = True; stack.append(u)
    m2 = np.zeros(n, bool)
    for k in range(1, n):
        m2[k] = good[pid[0 * n + k]]                 # pair (I, S)
    weak = {states[k] for k in range(n) if m2[k]}
    # strong: every state reachable from S satisfies condition (2)
    adj = [set(delta[k]) for k in range(n)]
    strong = set()
    for k in range(1, n):
        if not m2[k]:
            continue
        seen = {k}; st = [k]; ok = True
        while st and ok:
            v = st.pop()
            for u in adj[v]:
                if u not in seen:
                    if not m2[u]:
                        ok = False; break
                    seen.add(u); st.append(u)
        if ok:
            strong.add(states[k])
    trig = None                                       # smallest spike k, then zeros forever, that never recovers
    for k in range(0, imax + 1):
        s = step(I, k, S, T, R, Q); seen = set()
        while s not in seen and s != I:
            seen.add(s); s = step(s, 0, S, T, R, Q)
        if s != I:
            trig = k; break
    return dict(states=states, weak=weak, strong=strong, trigger=trig, npairs=npairs)
