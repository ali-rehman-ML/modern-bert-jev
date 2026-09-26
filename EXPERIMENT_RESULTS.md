# Initial ModernBERT-large experiment

Completed in `yolo_env` on the local RTX 3050 6GB GPU. Reproduction instructions
are in README.md. The local TensorBoard dashboard is at http://127.0.0.1:6006.

## Pilot scope

- Backbone: ModernBERT-large; 396,158,977 parameters including adapters and head.
- Trainable: 1,377,281 parameters (rank-8 attention LoRA and scalar scoring head).
- Datasets: MNLI, ARC-Challenge, MMLU; choice primitives only.
- Selected examples: 768 train, 96 validation, 96 calibration, 192 test.
- Training: 100 optimizer steps, four examples per step (400 examples processed).
- Best checkpoint: step 100. Training plus intermediate validation took 802 seconds.
- Peak PyTorch GPU allocation: 1.65 GiB. This excludes driver/other-process memory.

## Held-out results

| Metric | Value |
|---|---:|
| Overall accuracy | 33.85% |
| Uniform random-choice expected accuracy | 27.78% |
| MNLI accuracy (64 examples) | 32.81% |
| ARC-Challenge accuracy (64 examples) | 32.81% |
| MMLU accuracy (64 examples) | 35.94% |
| Uncalibrated target NLL | 1.2880 |
| Calibrated target NLL | 1.2914 |
| Uncalibrated 10-bin ECE | 0.0615 |
| Calibrated 10-bin ECE | 0.0207 |
| Calibration temperature | 0.1 (lower bound of search) |
| Candidate inputs with truncated context | 4 |

This is a short learning/pipeline pilot, not a converged model or a Jev comparison.
Calibration reduced test ECE but slightly worsened test NLL. The temperature reached
the search boundary; calibration should be revisited after substantial training
with more calibration examples. These sampled figures do not establish broad
decision-making quality or statistical significance over chance.

The complete choice-only dataset configuration has been downloaded and audited:
49,364 training rows, 1,897 validation rows, 1,900 calibration rows, and 9,599 test
rows after exact-overlap filtering. No full-choice training run was launched.

## Artifacts

- `runs/pilot/adapter.pt`: best trained adapter/head.
- `runs/pilot/metrics.json`: final metrics and per-source results.
- `runs/pilot/test_predictions.json`: per-example logits and targets.
- `runs/pilot/history.jsonl`: optimizer-step history.
- `runs/pilot/data_audit.json` and `split_ids.json`: provenance and selected rows.
- `runs/choice-all-ready/data_audit.json`: full choice-only data preparation audit.
- `runs/tensorboard/`: dashboard event files for completed and future runs.

# Modal A10 run `modal-6-all` (train ≤6 choices, test unlimited)

Status `complete`, 11,753 steps / 10 epochs, peak 3.78 GiB, test pass 1,123 s.
It learned: MNLI 84.5%, ChaosNLI 61.0%, overall 39.3% against a 15.1% uniform
expectation. The 2.1457 plateau did not recur.

## Overfitting

Validation NLL bottoms at **step 2000 (epoch 1.7), 0.9990** and then degrades
monotonically to 3.7310 by epoch 10; training loss reaches 0.0000 by epoch 7.7.
Accuracy peaks later (0.5948 at step 4500) while ECE climbs from 0.042 to 0.354,
so the extra epochs bought a little ranking at the cost of all calibration.
Checkpoint selection saved the run — test used the step-2000 adapter. **Roughly
two epochs is the useful budget**; the other eight were wasted compute.

## Per-source test accuracy (uncalibrated; accuracy is temperature-invariant)

| Source | K | Accuracy | Uniform | NLL (T=1) | ln K | entropy / ln K |
|---|---:|---:|---:|---:|---:|---:|
| mnli | 3 | 0.845 | 0.333 | 0.434 | 1.099 | 0.32 |
| chaosnli | 3 | 0.610 | 0.333 | 1.322 | 1.099 | 0.45 |
| arc_challenge | 4 | 0.462 | 0.250 | 1.216 | 1.386 | 0.87 |
| clinc150 | 151 | 0.446 | 0.007 | 3.455 | 5.017 | 0.92 |
| mmlu | 4 | 0.329 | 0.250 | 1.402 | 1.386 | 0.91 |
| massive | 60 | 0.313 | 0.017 | 2.890 | 4.094 | 0.86 |
| banking77 | 77 | 0.312 | 0.013 | 3.406 | 4.344 | 0.91 |
| go_emotions | 28 | 0.099 | 0.036 | 3.332 | 3.332 | **1.00** |
| ledgar | 100 | 0.009 | 0.010 | 4.605 | 4.605 | **1.00** |

Zero-shot transfer to large choice sets works (clinc150 at 151 choices reaches
44.6%), so the six-choice training limit is not the ceiling people assumed.

## Root cause of the go_emotions / ledgar uniform output

Not a data-mix or training-loop problem. `jev-bench` publishes **null criteria
descriptions** for exactly two sources, in every split:

    ledgar/{train,validation,test}       8000 / 500 / 1000 rows, 100% null
    go_emotions/{train,validation,test}  8000 / 500 / 1000 rows, 100% null

`encode()` renders each candidate as `Question: …\nCandidate: {description}`, so
every choice became the literal string `Candidate: None`. All candidates shared
one input, so the encoder returned **identical logits by construction** —
1000/1000 go_emotions rows and 422/1000 ledgar rows are bit-identical across
candidates, giving exactly `ln K` NLL and exactly `1/K` confidence. No amount of
training could have moved those two sources.

This also re-explains the earlier failed `modal-28` run: go_emotions was 45.9% of
that training mix with a hard loss floor of `ln 28 = 3.332`, and in its
predictions go_emotions is 100% identical-logit while every other source collapsed
to near-zero spread (0.03–0.37). The uniform collapse was seeded by unlearnable
rows, not by soft-label entropy.

Fixed in `bench_data.py:normalize` (fall back to the criterion ID, which is
human-readable: `admiration`, `Anti-Corruption Laws`) with a loud guard in
`choice_model.py:encode` so an all-identical candidate set can never train
silently again. Regression tests in `test_experiment.NullChoiceDescriptionTests`.
Both sources need a re-run before their numbers mean anything.

# Modal H100 run `base-full` (ModernBERT-base, full dataset, no choice limit)

`configs/modal_base_full.json`. Status `complete`, 6,171 steps / 2 epochs in 114
minutes, **26.1 GiB peak of 80**, test pass 323 s. First run to train on all nine
sources with no choice-count limit anywhere. Zero identical-logit rows in 9,599
test predictions, so the null-description fix holds end to end.

| | `modal-6-all` (large, ≤6 train) | `base-full` (base, unlimited) |
|---|---:|---:|
| Test accuracy | 0.3930 | **0.6379** |
| Calibrated NLL | 2.4225 | **1.0351** |
| Calibrated ECE | 0.1268 | **0.0525** |
| Temperature | 1.380 | 1.202 |

## Per-source accuracy

| Source | K | `base-full` | `modal-6-all` | Δ | uniform |
|---|---:|---:|---:|---:|---:|
| banking77 | 77 | 0.864 | 0.312 | **+0.552** | 0.013 |
| massive | 60 | 0.844 | 0.313 | **+0.531** | 0.017 |
| clinc150 | 151 | 0.832 | 0.446 | **+0.386** | 0.007 |
| mnli | 3 | 0.772 | 0.845 | −0.073 | 0.333 |
| ledgar | 100 | 0.761 | 0.009 | **+0.752** | 0.010 |
| go_emotions | 28 | 0.619 | 0.099 | **+0.520** | 0.036 |
| chaosnli | 3 | 0.528 | 0.610 | −0.082 | 0.333 |
| arc_challenge | 4 | 0.350 | 0.462 | −0.112 | 0.250 |
| mmlu | 4 | 0.277 | 0.329 | −0.052 | 0.250 |

The five large-label sources were previously untrained or broken; they now carry
the benchmark. LEDGAR went from chance to 0.761 purely from the description fix.

## The four small-K sources regressed, and the cause is exposure, not the mix

MNLI, ARC-Challenge, MMLU and ChaosNLI are the same 9,402 rows in both runs, but
`modal-6-all` trained on them for 10 epochs — 94,020 presentations — while
`base-full` gives them 2 epochs, 18,804 presentations. **They got 5x less
training**, which is sufficient to explain every regression without appealing to
the backbone change. The smaller backbone is a confound for the knowledge tasks
(ARC, MMLU) specifically and cannot be separated without an ablation.

MMLU remains unlearned in both runs: 0.277 against 0.250 chance, with NLL 1.413
against `ln 4 = 1.386`, i.e. still worse than answering uniformly. 284 published
train rows is the binding constraint.

## Two epochs under-trained; there is no overfitting to protect against

Validation NLL fell monotonically to the final step and the best checkpoint **is**
the last one:

| step | epoch | val NLL | val acc | ECE |
|---:|---:|---:|---:|---:|
| 250 | 0.08 | 1.6803 | 0.4590 | 0.058 |
| 1750 | 0.57 | 1.0985 | 0.6200 | 0.042 |
| 3250 | 1.05 | 0.9688 | 0.6560 | 0.034 |
| 4750 | 1.54 | 0.9420 | 0.6660 | 0.057 |
| 6171 | 2.00 | **0.9229** | 0.6700 | 0.043 |

The 2-epoch budget was set from `modal-6-all`, which overfit at epoch 1.7 — but
that run re-saw 9,402 rows, and this one has 49,364. The correct read is that
epochs were the wrong unit: overfitting tracks repeat exposure per row, not
optimizer steps. The next run should raise epochs to 4 and let best-NLL checkpoint
selection find the real bottom; that also restores small-K exposure toward the
level `modal-6-all` had.

Temperature fell from 1.380 to 1.202 and calibrated ECE more than halved,
consistent with a model that stopped short of memorizing rather than past it.
