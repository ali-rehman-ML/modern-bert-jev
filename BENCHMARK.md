# modern-bert-jev vs. jev-1.13.0

Head-to-head on the nine choice-primitive sources of `Praveenrajus/jev-bench`.

Reference numbers are the benchmark's own
[`results/jev-1.13.0/test_report.md`](https://huggingface.co/datasets/Praveenrajus/jev-bench/blob/main/results/jev-1.13.0/test_report.md).
Ours are `runs/base-full/metrics.json`, produced by `experiment.py` on the held-out test
split with the temperature fitted on a separate calibration split that never touches test.

**The test rows are the same.** Every `n` below matches the reference report exactly —
1,000 per source and 1,599 for ChaosNLI, 9,599 in total. Neither side is evaluating on a
different sample.

**It is not a whole-benchmark comparison.** jev-1.13.0 covers 22 sources across `choice`,
`noul` and `score` primitives. This model only does `choice`, so the other 13 sources have no
entry here. Lower is better for ECE, Brier and NLL; higher is better for accuracy.

## Per source

| source | choices | acc (ours) | acc (jev) | Δ | ECE (ours) | ECE (jev) | NLL (ours) | NLL (jev) | Brier (ours) | Brier (jev) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `banking77` | 77 | **0.864** | 0.796 | **+0.068** | **0.036** | 0.095 | **0.47** | 2.10 | **0.195** | 0.317 |
| `massive` | 60 | **0.844** | 0.808 | **+0.036** | **0.014** | 0.090 | **0.54** | 1.92 | **0.221** | 0.295 |
| `clinc150` | 151 | 0.832 | **0.893** | −0.061 | 0.037 | **0.033** | **0.64** | 0.82 | 0.249 | **0.159** |
| `mnli` | 3 | 0.772 | **0.883** | −0.111 | **0.031** | 0.032 | 0.55 | **0.50** | 0.313 | **0.176** |
| `ledgar` | 100 | **0.761** | 0.751 | **+0.010** | **0.026** | 0.117 | **0.88** | 2.29 | **0.347** | 0.374 |
| `go_emotions` | 28 | **0.586** | 0.282 | **+0.304** | **0.079** | 0.384 | **1.97** | 9.89 | **0.235** | 1.040 |
| `chaosnli` | 3 | 0.523 | **0.615** | −0.092 | 0.254 | **0.222** | **1.32** | 1.71 | **0.277** | 0.583 |
| `arc_challenge` | 4 | 0.350 | **0.979** | −0.629 | 0.046 | **0.010** | 1.36 | **0.19** | 0.735 | **0.037** |
| `mmlu` | 4 | 0.277 | **0.923** | −0.646 | 0.089 | **0.027** | 1.41 | **0.46** | 0.761 | **0.124** |

Accuracy wins: 4 of 9. ECE wins: 5 of 9. NLL wins: **6 of 9**.

## Aggregates

All nine sources:

| | ours | jev |
|---|---:|---:|
| micro accuracy (n = 9,599) | 0.6379 | **0.7603** |
| macro accuracy | 0.6455 | **0.7700** |
| macro ECE | **0.0679** | 0.1122 |
| macro NLL | **1.0161** | 2.2089 |
| macro Brier | 0.3704 | **0.3450** |

The two knowledge tasks dominate that gap. Excluding ARC-Challenge and MMLU — seven sources,
n = 7,599:

| | ours | jev |
|---|---:|---:|
| micro accuracy | **0.7233** | 0.7101 |
| macro accuracy | **0.7404** | 0.7183 |
| macro ECE | **0.0681** | 0.1390 |
| macro NLL | **0.9104** | 2.7471 |
| macro Brier | **0.2625** | 0.4206 |

## What the numbers say

**Two tasks account for the entire deficit.** ARC-Challenge (−0.629) and MMLU (−0.646) are
school and university exam questions, and they need facts a 150M-parameter encoder does not
hold. jev answers 97.9% and 92.3% of them; this model gets 35.0% and 27.7%, the latter barely
above the 25% you get by guessing. Remove those two and the accuracy gap inverts: 0.723 to
0.710 in our favour.

**On the seven tasks that are actually classification, it is ahead on every calibration
measure at once** — half the ECE, a third of the NLL, and 62% of the Brier score. That is the
behaviour the benchmark is built to measure, and it is where a small model tuned with a soft
target and a fitted temperature beats a much stronger general system.

**GoEmotions is the clearest case.** jev scores 0.282 accuracy with an NLL of **9.89**. The
uniform baseline over 28 choices is `ln 28 = 3.33`, so a system that answered uniformly at
random would score three times better on NLL. That is severe overconfidence on a soft-label
task — high probability repeatedly assigned to a label the human annotators did not pick. This
model gets 0.586 accuracy at 1.97 NLL, below uniform as it should be.

**ChaosNLI is the one place calibration is genuinely poor on both sides.** Our ECE of 0.254
and jev's 0.222 are the worst entries in the table. ChaosNLI has no training split at all, so
this model reaches it purely by transfer from MNLI, whose hard labels teach exactly the
overconfidence that a human label-distribution task punishes.

## Caveats

- **Metric definitions are assumed to line up.** Ours come from `experiment.py:metrics` —
  10-bin ECE against target agreement, summed multi-class Brier, target NLL against the full
  label distribution. The reference report does not state its definitions. The comparison is
  as sound as that assumption.
- **Calibration procedure differs, or may.** Our temperature is one scalar fitted on a
  held-out split by grid search. How jev-1.13.0 calibrates is not documented in the report.
- **One seed.** These are single-run numbers with no variance estimate on either side, so
  small gaps — LEDGAR's +0.010, MNLI's −0.111 — should not be over-read.
- **This model was stopped early.** Validation NLL was still falling monotonically at the
  final step; see `EXPERIMENT_RESULTS.md`. The numbers above are not a converged ceiling.
- Reference report read at dataset revision `18f88da81c28c2bec55edc31f63f2afdfba109ea`;
  evaluation data pinned at `cbcb6703ef02b698bee32af39d3270fe1f356187`.

## Reproducing

```bash
python -m modal run --detach modal_train.py --config modal_base_full --run base-full --gpu h100
python sync_modal.py --run base-full --artifacts
```

`runs/base-full/test_predictions.json` holds the raw per-candidate scores for all 9,599 rows,
so every figure above is recomputable without a GPU. It is also published on the
[model repo](https://huggingface.co/ali-rehman-ML/modern-bert-jev).
