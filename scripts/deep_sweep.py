"""haeberlen deep test: mitigations vs strongly metastable states on the Figure 1 model.

Own code. Uses only the reviewed reimplementation model.py (analyze, step; sha256 checked) plus
numpy/scipy through it, and the standard library (heapq, multiprocessing, json). No network, no
pickle, no files read except model.py. Writes one JSON and prints a summary.

Parts:
  A  gates: Section 3.1 case, and the round-1 sweep reproduced exactly; JSON oddity explained
  B  structural rule check: strong state exists  <=>  Rmax >= S and T + S <= Qmax
  C  mitigation sweep on the round-1 grid and on a large grid, shares side by side
  D  pattern triggers: minimum total extra load and minimum peak per-round load that reach a
     strong state from I (any finite input sequence), before and after each mitigation

usage: python3 -I deep_sweep.py --model-dir DIR --model-sha256 HEX --out OUT.json [--procs N]
"""
import argparse
import hashlib
import heapq
import json
import os
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool

MODEL = None  # set in init


def init(model_dir):
    global MODEL
    sys.path.insert(0, model_dir)
    import model  # noqa: E402  (own reviewed code)
    MODEL = model


def pattern_triggers(a, S, T, R, Q):
    """Min total load and min peak per-round load from I to any strong state (None if none)."""
    strong = a["strong"]
    if not strong:
        return None, None
    states = a["states"]
    I = (0, 0, 0)
    step = MODEL.step
    nxt = {s: [step(s, i, S, T, R, Q) for i in range(Q + 1)] for s in states}
    # Dijkstra, edge cost = input tokens in that round
    dist = {I: 0}
    pq = [(0, I)]
    total = None
    while pq:
        d, s = heapq.heappop(pq)
        if d > dist.get(s, 1 << 30):
            continue
        if s in strong:
            total = d
            break
        for i, t in enumerate(nxt[s]):
            nd = d + i
            if nd < dist.get(t, 1 << 30):
                dist[t] = nd
                heapq.heappush(pq, (nd, t))
    peak = None
    for k in range(Q + 1):                       # smallest per-round cap k that still reaches a strong state
        seen = {I}
        st = [I]
        hit = False
        while st and not hit:
            s = st.pop()
            for t in nxt[s][:k + 1]:
                if t in strong:
                    hit = True
                    break
                if t not in seen:
                    seen.add(t)
                    st.append(t)
        if hit:
            peak = k
            break
    return total, peak


def evaluate(p):
    S, T, R, Q = p
    a = MODEL.analyze(S, T, R, Q)
    total, peak = pattern_triggers(a, S, T, R, Q)
    return p, {"strong": len(a["strong"]), "weak": len(a["weak"]), "states": len(a["states"]),
               "trigger": a["trigger"], "min_total": total, "min_peak": peak}


MITIGATIONS = {
    "Rmax-1": lambda S, T, R, Q: (S, T, R - 1, Q),
    "Rmax-2": lambda S, T, R, Q: (S, T, R - 2, Q),
    "T+1": lambda S, T, R, Q: (S, T + 1, R, Q),
    "T+2": lambda S, T, R, Q: (S, T + 2, R, Q),
    "S+1": lambda S, T, R, Q: (S + 1, T, R, Q),
    "Qmax-1": lambda S, T, R, Q: (S, T, R, Q - 1),
}


def in_domain(q):
    S, T, R, Q = q
    return R >= 0 and S >= 1 and Q >= 1 and T >= 0


def rule(p):
    S, T, R, Q = p
    return R >= S and T + S <= Q


def cmp(before, after):
    if after is None:
        return "removed_or_none"
    if before is None:
        return "lowered"          # was unreachable by this trigger, now reachable
    return "raised" if after > before else ("same" if after == before else "lowered")


def sweep(grid, res):
    """Round-1 classification (single-spike trigger) plus pattern-trigger comparisons."""
    out = {}
    for name, f in MITIGATIONS.items():
        oc, created, skipped_safe, skipped_meta = Counter(), 0, 0, 0
        tot_c, peak_c = Counter(), Counter()
        by_rule = defaultdict(Counter)       # does the mitigation cross the rule boundary?
        examples = {}
        for p in grid:
            q = f(*p)
            b = res[p]
            if not in_domain(q):
                if b["strong"]:
                    skipped_meta += 1
                else:
                    skipped_safe += 1
                continue
            a = res[q]
            if not b["strong"]:
                created += a["strong"] > 0
                continue
            S, T, R, Q = q
            if T >= Q:
                key = "timeout_disabled_removed" if not a["strong"] else "timeout_disabled_still"
            elif not a["strong"]:
                key = "removed"
            elif a["trigger"] is None:
                key = "unreachable"
            elif b["trigger"] is None:
                key = "trigger_lowered"
            elif a["trigger"] > b["trigger"]:
                key = "trigger_raised"
            elif a["trigger"] == b["trigger"]:
                key = "trigger_same"
            else:
                key = "trigger_lowered"
            oc[key] += 1
            examples.setdefault(key, {"before": p, "after": q, "trigger_before": b["trigger"],
                                      "trigger_after": a["trigger"]})
            if a["strong"]:
                tot_c[cmp(b["min_total"], a["min_total"])] += 1
                peak_c[cmp(b["min_peak"], a["min_peak"])] += 1
            else:
                tot_c["removed"] += 1
                peak_c["removed"] += 1
            by_rule["crosses_rule" if not rule(q) else "stays_inside_rule"][key] += 1
        n = sum(oc.values())
        out[name] = {
            "metastable_models": n,
            "skipped_out_of_domain_total": skipped_safe + skipped_meta,
            "skipped_out_of_domain_safe": skipped_safe,
            "skipped_out_of_domain_metastable": skipped_meta,
            "outcomes": dict(oc),
            "share": {k: round(v / n, 3) for k, v in oc.items()} if n else {},
            "pattern_min_total_load": dict(tot_c),
            "pattern_min_peak_load": dict(peak_c),
            "by_rule": {k: dict(v) for k, v in by_rule.items()},
            "created_metastability_in_safe_models": created,
            "example_per_outcome": examples,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--model-sha256", required=True)
    ap.add_argument("--out", default="deep_sweep.json")
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    args = ap.parse_args()
    with open(os.path.join(args.model_dir, "model.py"), "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    if digest != args.model_sha256:
        raise SystemExit(f"model.py changed since review ({digest}), stopping")
    init(args.model_dir)

    small = [(S, T, R, Q) for S in range(2, 6) for T in range(2, 7) for R in range(1, 5)
             for Q in range(6, 13) if Q > T]
    large = [(S, T, R, Q) for S in range(1, 9) for T in range(1, 13) for R in range(0, 11)
             for Q in range(2, 21) if Q > T]
    need = set(small) | set(large) | {(3, 4, 3, 10)}
    for g in (small, large):
        for p in g:
            for f in MITIGATIONS.values():
                q = f(*p)
                if in_domain(q):
                    need.add(q)
    need = sorted(need, key=lambda p: -p[3])     # big Qmax first for load balance
    print(f"evaluating {len(need)} parameter sets with {args.procs} processes", flush=True)
    res = {}
    with Pool(args.procs, initializer=init, initargs=(args.model_dir,)) as pool:
        for k, (p, r) in enumerate(pool.imap_unordered(evaluate, need, chunksize=4), 1):
            res[p] = r
            if k % 1000 == 0:
                print(f"  {k}/{len(need)}", flush=True)

    # A. gates
    g31 = res[(3, 4, 3, 10)]
    gate31 = g31["trigger"] == 10 and g31["strong"] == 13
    print(f"gate Section 3.1 (S3 T4 R3 Q10): trigger {g31['trigger']} strong {g31['strong']} "
          f"min_total {g31['min_total']} min_peak {g31['min_peak']} -> {'OK' if gate31 else 'FAIL'}")
    sm = sweep(small, res)
    r1 = sm["Rmax-1"]["outcomes"], sm["T+1"]["outcomes"]
    gate_r1 = (sum(1 for p in small if res[p]["strong"]) == 179 and r1[0].get("removed") == 86
               and r1[1].get("removed") == 22 and r1[1].get("trigger_raised") == 83
               and r1[1].get("unreachable") == 74)
    print(f"gate round-1 reproduction: {'OK' if gate_r1 else 'FAIL'}")
    if not (gate31 and gate_r1):
        raise SystemExit("gate failed, not reporting")
    meta_by_R = Counter(p[2] for p in small if res[p]["strong"])
    print(f"oddity: small-grid metastable models per Rmax {dict(sorted(meta_by_R.items()))}; "
          f"Rmax-2 skipped {sm['Rmax-2']['skipped_out_of_domain_total']} = "
          f"{sm['Rmax-2']['skipped_out_of_domain_safe']} safe + {sm['Rmax-2']['skipped_out_of_domain_metastable']} metastable")

    # B. rule check
    rules = {}
    for gname, g in (("small", small), ("large", large)):
        mism = [p for p in g if (res[p]["strong"] > 0) != rule(p)]
        rules[gname] = {"models": len(g), "metastable": sum(1 for p in g if res[p]["strong"]),
                        "rule_mismatches": len(mism), "mismatch_examples": mism[:50]}
        print(f"rule check {gname}: {len(g)} models, {rules[gname]['metastable']} metastable, "
              f"{len(mism)} mismatches with (Rmax >= S and T + S <= Qmax)")
    allp = sorted(res)
    mism_all = [p for p in allp if (res[p]["strong"] > 0) != rule(p)]
    rules["all_evaluated"] = {"models": len(allp), "rule_mismatches": len(mism_all),
                              "mismatch_examples": mism_all[:50]}
    print(f"rule check all {len(allp)} evaluated parameter sets: {len(mism_all)} mismatches")
    # how the single-spike and pattern triggers relate to the parameters (descriptive)
    trig_rows = [(p, res[p]) for p in large if res[p]["strong"]]
    unreach = sum(1 for _, r in trig_rows if r["trigger"] is None)
    print(f"large grid: {len(trig_rows)} metastable, {unreach} with no single-spike trigger "
          f"(all have a finite pattern trigger: {all(r['min_total'] is not None for _, r in trig_rows)})")

    # C/D. sweeps
    lg = sweep(large, res)
    cols = ["removed", "unreachable", "trigger_raised", "trigger_same", "trigger_lowered",
            "timeout_disabled_removed", "timeout_disabled_still"]
    for gname, sw in (("round-1 grid", sm), ("large grid", lg)):
        print(f"\n{gname}: share of metastable models")
        print("mitigation  n     " + "  ".join(f"{c[:14]:>14s}" for c in cols) + "  created")
        for name, r in sw.items():
            cells = "  ".join(f"{r['share'].get(c, 0):14.1%}" for c in cols)
            print(f"{name:10s} {r['metastable_models']:5d}  {cells}  {r['created_metastability_in_safe_models']:7d}")
        print("pattern triggers (min total load | min peak load): removed/raised/same/lowered")
        for name, r in sw.items():
            t, k = r["pattern_min_total_load"], r["pattern_min_peak_load"]
            n = r["metastable_models"]
            fmt = lambda c: "/".join(f"{c.get(x, 0) / n:.0%}" for x in ("removed", "raised", "same", "lowered"))
            print(f"  {name:10s} total {fmt(t):>22s} | peak {fmt(k):>22s}")
        print("outcome split by whether the mitigated model leaves the rule region")
        for name, r in sw.items():
            print(f"  {name:10s} {r['by_rule']}")

    report = {"model_py_sha256": digest, "gates": {"section_3_1": g31, "round1_reproduced": gate_r1},
              "oddity": {"small_grid_metastable_per_Rmax": dict(meta_by_R),
                         "rmax2_skipped": {k: sm["Rmax-2"][k] for k in sm["Rmax-2"] if k.startswith("skipped")}},
              "rule": rules, "small_grid": sm, "large_grid": lg,
              "per_model": {",".join(map(str, p)): r for p, r in sorted(res.items())}}
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
