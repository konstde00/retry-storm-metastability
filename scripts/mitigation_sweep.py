"""Q2: do mitigations remove strongly metastable states, or only raise the trigger?

Reuses the reimplementation in model.py (Figure 1 retry-storm FST,
Section 3.5 search). Needs numpy and scipy (already installed: numpy 2.3.3, scipy 1.16.2, the versions
the earlier haeberlen runs used). Writes one JSON and prints a table. No network.

Grid: the same 544 models as Part B (S 2..5, T 2..6, Rmax 1..4, Qmax 6..12, Qmax > T).
Mitigations applied to each model: Rmax-1, Rmax-2 (lower retry cap), T+1, T+2 (later timeout),
S+1 (more capacity, for comparison), Qmax-1 (smaller queue). For every model with a strongly
metastable state, the outcome after the mitigation is one of:
  removed           no strongly metastable state left
  unreachable       strongly metastable states remain but no single spike from I reaches them
  trigger_raised    a larger single spike is now needed
  trigger_same      same smallest stuck spike
  trigger_lowered   a smaller spike now suffices (the mitigation made it worse)
Also counted: models with no strong state before that have one after (mitigation creates one).
Mitigated parameters are kept in the model's domain (Rmax >= 0, S >= 1). A timeout T >= Qmax can never
fire, so those cases are reported separately as "timeout_disabled".
"""
import argparse
import hashlib
import json
import os
import sys
from collections import Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="directory holding the reviewed model.py")
    ap.add_argument("--model-sha256", default=None, help="abort unless model.py has this sha256")
    ap.add_argument("--out", default="mitigation_sweep.json")
    args = ap.parse_args()
    path = os.path.join(args.model_dir, "model.py")
    with open(path, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    if args.model_sha256 and digest != args.model_sha256:
        raise SystemExit(f"model.py changed since review ({digest}), stopping")
    sys.path.insert(0, args.model_dir)
    from model import analyze  # noqa: E402  (own reviewed code)

    grid = [(S, T, R, Q) for S in range(2, 6) for T in range(2, 7) for R in range(1, 5)
            for Q in range(6, 13) if Q > T]
    mitigations = {
        "Rmax-1": lambda S, T, R, Q: (S, T, R - 1, Q),
        "Rmax-2": lambda S, T, R, Q: (S, T, R - 2, Q),
        "T+1": lambda S, T, R, Q: (S, T + 1, R, Q),
        "T+2": lambda S, T, R, Q: (S, T + 2, R, Q),
        "S+1": lambda S, T, R, Q: (S + 1, T, R, Q),
        "Qmax-1": lambda S, T, R, Q: (S, T, R, Q - 1),
    }
    cache = {}

    def get(p):
        if p not in cache:
            a = analyze(*p)
            cache[p] = (len(a["strong"]) > 0, a["trigger"], len(a["strong"]), len(a["weak"]))
        return cache[p]

    results = {}
    for name, f in mitigations.items():
        outcome, created, skipped = Counter(), 0, 0
        examples = {}
        for p in grid:
            q = f(*p)
            S, T, R, Q = q
            if R < 0 or S < 1 or Q < 1:
                skipped += 1
                continue
            before, after = get(p), get(q)
            if not before[0]:
                created += after[0]
                continue
            if T >= Q:
                key = "timeout_disabled_removed" if not after[0] else "timeout_disabled_still"
            elif not after[0]:
                key = "removed"
            elif after[1] is None:
                key = "unreachable"
            elif before[1] is None:
                key = "trigger_lowered"          # was unreachable by a spike, now reachable
            elif after[1] > before[1]:
                key = "trigger_raised"
            elif after[1] == before[1]:
                key = "trigger_same"
            else:
                key = "trigger_lowered"
            outcome[key] += 1
            examples.setdefault(key, (p, q, before[1], after[1]))
        n = sum(outcome.values())
        results[name] = {"metastable_models": n, "skipped_out_of_domain": skipped,
                         "outcomes": dict(outcome), "share": {k: round(v / n, 3) for k, v in outcome.items()} if n else {},
                         "created_metastability_in_safe_models": created,
                         "example_per_outcome": {k: {"before": v[0], "after": v[1], "trigger_before": v[2],
                                                     "trigger_after": v[3]} for k, v in examples.items()}}
    n_meta = sum(get(p)[0] for p in grid)
    report = {"grid_models": len(grid), "grid_metastable": n_meta, "model_py_sha256": digest, "mitigations": results}
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"grid {len(grid)} models, {n_meta} with a strongly metastable state (Part B found 179)")
    cols = ["removed", "unreachable", "trigger_raised", "trigger_same", "trigger_lowered",
            "timeout_disabled_removed", "timeout_disabled_still"]
    print("mitigation  n    " + "  ".join(f"{c[:16]:>16s}" for c in cols) + "  created")
    for name, r in results.items():
        cells = "  ".join(f"{r['share'].get(c, 0):16.1%}" for c in cols)
        print(f"{name:10s} {r['metastable_models']:4d}  {cells}  {r['created_metastability_in_safe_models']:7d}")


if __name__ == "__main__":
    main()
