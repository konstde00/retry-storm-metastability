# retry-storm-metastability

An independent reimplementation of the retry-storm model from "Modeling Metastability" (HotNets '25), Figure 1 and Sections 3.1 to 3.5, plus a few experiments on top of it. It's not affiliated with the paper's authors, and any mistakes are mine.

## Model

`scripts/model.py` follows the Section 3.2 equations:

```
wait(t)    = (in(t-1) + (wait(t-1) - S)+ + rexmits(t-1)) clipped to [0, Qmax]
served(t)  = wait(t-1) clipped to [0, S]
rexmits(t) = (wait(t-1) - T) clipped to [0, Rmax]
```

Rmax caps the number of retransmissions per round (not retries per request). `analyze()` runs the Section 3.5 search: it enumerates the reachable states, builds the product graph over state pairs, and returns the weakly and strongly metastable states plus the smallest single spike that never recovers.

## Results

1. Section 3.1 case (S 3, T 4, Rmax 3, Qmax 10): smallest stuck spike 10, 13 strongly metastable states, as in the paper (`results/partA.txt`).
2. Exact rule: a strongly metastable state exists if and only if Rmax >= S and T + S <= Qmax. With zero input a stuck level w needs w = (w - S)+ + min((w - T)+, Rmax), i.e. min(w - T, Rmax) >= S (equality below the Qmax cap), which gives exactly these two conditions. The full search agrees on all 19,470 parameter sets evaluated (`results/deep_sweep.txt`), and `scripts/fixpoint_check.py` checks the fixed-point form on 14,256 sets with 0 mismatches.
3. Consequence for mitigations: lowering Rmax removes the stuck states only once Rmax drops below S. A later timeout removes them only once T + S passes Qmax, and before that it usually raises the load needed to get stuck by 1 token (`results/mitigation_sweep.txt`, `results/deep_sweep.txt`).
4. Fitting the model to a synthetic incident from window-averaged, noisy metrics (`scripts/partC.py`, 1,000 incidents per setting, `results/partC_focus_fp.txt`). With 10-round windows, 10% noise and service jitter, seeing only served and retry counts, the fitted model flags 90.3% of the 331 metastable truths, falsely flags 2.1% of the 669 others, and gets the smallest stuck spike right for 53.5%. Adding the queue gauge: 91.8% flagged, 3.5% false flags, 76.0% spike right. The incidents come from the same model family that is fitted, so the model is well specified here. Real telemetry is the open part.

## Running

Needs Python 3 with numpy and scipy (tested with numpy 2.3.3 and scipy 1.16.2). No network access.

```
python3 -I scripts/partA.py scripts
python3 -I scripts/partB.py scripts partB_rows.json
python3 -I scripts/partB2.py scripts
python3 -I scripts/partC.py scripts 1000 11 "1,0,1,0;10,0.1,1,1;10,0.1,0,1"
python3 -I scripts/mitigation_sweep.py --model-dir scripts --out mitigation_sweep.json
python3 -I scripts/deep_sweep.py --model-dir scripts --out deep_sweep.json --procs 8
python3 -I scripts/fixpoint_check.py
```

The large JSON outputs (`partB_rows.json`, `deep_sweep.json`) aren't committed, the scripts regenerate them.

## Caveats

- The paper's Figure 1 topology only, deterministic, inputs capped at Qmax per round.
- The rule is checked numerically over a finite grid. The fixed-point argument explains it, but this isn't a formal proof.
- The "trigger" summaries (single spike, minimum total load, minimum peak load) are my operational choices, not definitions from the paper.

## License

MIT
