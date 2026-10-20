"""Part A: reproduce the paper's Section 3.1 claims for S=3, Qmax=10, T=4, Rmax=3."""
import sys, time
sys.path.insert(0, sys.argv[1])
from model import analyze, simulate
S, T, R, Q = 3, 4, 3, 10
t0 = time.time(); a = analyze(S, T, R, Q)
print(f"reachable states={len(a['states'])} pairs={a['npairs']} weak={len(a['weak'])} strong={len(a['strong'])} trigger={a['trigger']} ({time.time()-t0:.1f}s)")
print("(4,0,0) weakly metastable:", (4,0,0) in a['weak'], "| strongly:", (4,0,0) in a['strong'])
print("(10,0,0) weakly metastable:", (10,0,0) in a['weak'], "| strongly:", (10,0,0) in a['strong'])
o1, _, _ = simulate([4] + [4, 0] * 10, S, T, R, Q); o0, _, _ = simulate([4, 0] * 10, S, T, R, Q, s=(0,0,0))
print("R1=(4,0) from (4,0,0) out:", list(o1[1:13]), " from I out:", list(o0[:12]))
o2, st, _ = simulate([10] + [0] * 12, S, T, R, Q)
print("spike 10 then zeros, out:", list(o2), "states tail:", st[-2:])
o3, _, _ = simulate([0]*9 + [8] + [0] * 12, S, T, R, Q)
print("spike 8 then zeros, out:", list(o3))
print("three zeros from (4,0,0):", simulate([0,0,0,0], S, T, R, Q, s=(4,0,0))[2])
print("weak sample:", sorted(a['weak'])[:12]); print("strong sample:", sorted(a['strong'])[:12])
