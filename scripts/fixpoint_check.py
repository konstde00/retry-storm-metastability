# Independent check of the one-line derivation: with zero input, a nonzero stuck queue level w
# (fixed point of the Section 3.2 equations as coded in model.step) exists iff Rmax >= S and T + S <= Qmax.
# Also checks: any zero-input orbit from any state that never returns to (0,0,0) ends in such a fixed point.
def step(s, i, S, T, R, Q):
    w, sv, r = s
    return (min(max(i + max(w - S, 0) + r, 0), Q), min(max(w, 0), S), min(max(w - T, 0), R))
mism = 0; n = 0; cyc = 0
for S in range(1, 9):
    for T in range(1, 13):
        for R in range(0, 11):
            for Q in range(2, 21):
                if Q <= T: continue
                n += 1
                fp = any(w == min(max(w - S, 0) + min(max(w - T, 0), R), Q) for w in range(1, Q + 1))
                rule = R >= S and T + S <= Q
                mism += fp != rule
                # zero-input orbits from all states: any nontrivial cycle that is not a fixed point?
                for w in range(Q + 1):
                    for r in range(R + 1):
                        s = (w, min(w, S), r); seen = []
                        while s not in seen:
                            seen.append(s); s = step(s, 0, S, T, R, Q)
                        loop = seen[seen.index(s):]
                        if len(loop) > 1 and any(x[0] for x in loop): cyc += 1
print(f"{n} parameter sets, fixed-point vs rule mismatches: {mism}, zero-input cycles longer than 1 with w>0: {cyc}")
