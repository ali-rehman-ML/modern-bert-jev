# modern-bert-jev

**Pick the best option from a list — with percentages you can trust.**

[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-modern--bert--jev-yellow)](https://huggingface.co/ali-rehman-ML/modern-bert-jev)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ali-rehman-ML/modern-bert-jev/blob/main/demo.ipynb)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

Give it some background, a question, and a list of options. It reads every option against
the background and tells you how likely each one is.

```
Background:  "How long will it take for my ID to verify?"
Question:    Which banking support intent does this express?
Options:     Card arrival / Lost or stolen card / Unable to verify identity / ... (77 total)

          →  Unable to verify identity        ████████████████████  81%
             Why verify identity              ███                   11%
             Verify my identity               ██                     5%
```

Two options or 151 — it is the same model either way. Nothing is wired to a fixed list, so
you can hand it categories it has never seen and it will still rank them.

The percentages are **calibrated**: when it says 70%, it is right about 70% of the time.
That is the unusual part. Most classifiers are wildly overconfident.

## Try it

**In your browser, free, no setup** — [open the Colab notebook](https://colab.research.google.com/github/ali-rehman-ML/modern-bert-jev/blob/main/demo.ipynb)
and press play. It prints a link to a working demo in about two minutes.

**On your own machine:**

```bash
git clone https://github.com/ali-rehman-ML/modern-bert-jev
cd modern-bert-jev
pip install torch transformers gradio
python space/app.py
```

**From Python:**

```python
from predict import Predictor

predictor = Predictor("runs/base-full")
print(predictor({
    "question": "Which emotion does the comment primarily express?",
    "context": "i stay as quiet as i can until im caught",
    "choices": {"anger": "anger", "annoyance": "annoyance", "fear": "fear", "joy": "joy"},
}))
```

## What it is good at

Sorting text into a known list of categories. Tested on 9,599 examples it had never seen:

| Task | Options to choose from | Gets it right | Random guessing |
|---|---:|---:|---:|
| Bank support messages | 77 | **86%** | 1% |
| Voice assistant commands | 60 | **84%** | 2% |
| Customer service intents | 151 | **83%** | 1% |
| Sentence logic | 3 | **77%** | 33% |
| Legal contract clauses | 100 | **76%** | 1% |
| Emotion in a comment | 28 | **62%** | 4% |
| Disputed sentence logic | 3 | 53% | 33% |
| School science questions | 4 | 35% | 25% |
| University exam questions | 4 | 28% | 25% |

Overall: **64% correct** where random guessing gets 15%.

## What it is bad at

**Anything needing world knowledge.** The bottom two rows above are the honest warning. On
university exam questions it scores 28% against 25% for guessing — that is not a useful
model, it is noise. It is a small model and it has not memorised facts. Use a large language
model for trivia and exams.

It is also **English only**, was trained on **one random seed** so there is no error bar, and
it only reads the first ~400 words of your background text.

## How it works, briefly

Instead of one big output layer with a slot per category, it scores **one option at a time**.
Each option gets glued to your background and question, read by the model, and turned into a
single number. Softmax over those numbers gives the percentages.

```
background + question + option 1  →  model  →  4.2  ┐
background + question + option 2  →  model  →  1.8  ├→  softmax  →  81% / 11% / 5%
background + question + option 3  →  model  →  0.3  ┘
```

The model never sees the competing options, which is exactly why the number of them does not
matter. The cost is that it runs once per option, so 151 options means 151 passes.

Underneath is [ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base), frozen.
Only **1.6 million** of its 150 million settings were trained — a thin adapter on the
attention layers plus one small output layer. The trained part is 6.5 MB.

Trained on all nine option-picking tasks at once for 2 hours on one H100.

## One bug worth knowing about

The benchmark ships two of its nine tasks with **empty option descriptions**. Every option
therefore rendered as the same text, the model saw identical input for all of them, and it
returned identical scores by construction — perfectly uniform, untrainable at any budget.

Legal clause classification went from **1% to 76%** when that was fixed. The repo now falls
back to the option's name and refuses outright if a question's options all read the same.

---

<details>
<summary><b>Full technical reference</b> — training recipe, memory strategy, data provenance, reproduction commands</summary>

## Current recipe: full dataset on one H100

`configs/modal_base_full.json` has no choice-count limit on any split. It fits by
bounding the *training* candidate count instead of dropping examples:
`train_max_candidates` keeps every candidate carrying target mass plus random
negatives up to the limit, renormalizes the target over what it kept, and leaves
validation, calibration and test scoring every published candidate. The scorer is a
per-candidate scalar, so a model trained to rank 32 candidates ranks 151 of them
too; the calibration split still sees full candidate sets, so the fitted
temperature corrects the confidence shift that sampled training introduces.

| | value | why |
| --- | --- | --- |
| backbone | ModernBERT-base | 2.5x less activation memory than large |
| `train_max_candidates` | 32 | 25.9 mean candidates/example instead of 68.0 |
| `epochs` | 2 | see EXPERIMENT_RESULTS.md: this under-trained, raise to 4 |
| `accumulation` | 16 | 512 sequences worst case, 26.1 GiB peak |
| `lora_rank` / `alpha` | 16 / 32 | 1.6M trainable, for nine tasks instead of four |
| `learning_rate` | 2e-4, 5% warmup, cosine | smaller backbone tolerates more |
| `validation_per_source` | 250 | 1,000 validation + 1,000 calibration rows |
| `eval_every` | 250 | 24 checkpoint decisions; best validation NLL wins |

49,364 train / 1,000 validation / 1,000 calibration / 9,599 test rows, 6,171 steps.
MMLU (284 train rows) and ARC-Challenge (1,118) stay data-starved by the published
splits and improve only by transfer; ChaosNLI has no train split at all and is pure
zero-shot from MNLI. Sampled training moved the per-epoch workload from 241M padded
token-rows to roughly 65M.

## Choice-count limits and what each run covers

`max_choices` drops entire examples whose choice list is longer than the limit. It
never truncates a choice list or renormalizes a modified label. `test_max_choices`
overrides the limit for the test split alone, and an explicit `null` means no limit
there, so a run can train on small choice sets while being evaluated on every
published test input. Both limits are recorded in `data_audit.json`, along with the
number of rows each one dropped per source and split (`0` when a split is
deliberately unfiltered).

The dataset's choice counts are only {3, 4, 5, 28, 60, 77, 100, 151}: nothing has
7 to 20 choices, so any limit between 6 and 27 selects exactly the same rows as 6.
All nine choice-primitive sources are already in use; the remaining thirteen
manifest sources are `score` and `noul` primitives that `prepare` rejects.

Because every candidate repeats the full example context, cost scales with choices
multiplied by context length, not with row count. Per-epoch training workload,
measured in padded token-rows:

| limit | train rows | token-rows/epoch | relative |
| --- | --- | --- | --- |
| 6 | 9,402 | 2,526,412 | 1.0x |
| 28 | 17,385 | 10,725,388 | 4.2x |
| none | 49,364 | 241,664,490 | 95.7x |

`runs/modal-28` trains on at most 28 choices (MNLI, GoEmotions, ARC-Challenge,
MMLU) and tests on all 9,599 published test rows, so Banking77, CLINC150, MASSIVE
and LEDGAR are scored without ever being trained on. Its temperature is fitted on a
calibration split that is also limited to 28 choices, so read `uncalibrated`
`per_source` numbers for the larger label sets.

`runs/modal-all` removed the limit entirely and hit CUDA OOM at step 633 of 18,512
on an 80GB H100. `train_records` pads every candidate in a batch to the batch's
longest sequence, so batches mixing CLINC150's 151 candidates with one long example
reach 1,004 rows of 508 tokens. Replaying the deterministic batch order gives a
worst case of 510,032 padded token-rows against 377,952 at the failure, which
projects to roughly 85GB allocated before fragmentation: that configuration needs
H200-class memory or a smaller `accumulation`.

## Earlier cloud run: maximum six choices

`configs/modal_six.json` excludes entire examples with more than six choices on
every split. It never truncates a choice list or renormalizes a modified label.
This leaves MNLI, ARC-Challenge, MMLU, and test-only ChaosNLI: 9,402 train,
649 validation, 650 calibration, and 4,599 test examples after duplicate filtering.
Banking77, CLINC150, MASSIVE, LEDGAR, and GoEmotions are excluded by this limit.

The Modal experiment starts fresh from the pinned ModernBERT-large weights with
fresh LoRA adapters; the old unfiltered laptop checkpoint is preserved separately
in `runs/choice-full/last.pt`. Three epochs correspond to 28,206 training example
presentations and 3,526 optimizer steps. Eight examples are batched into one
differentiable forward pass, with per-example softmax losses averaged equally.
The last partial batch uses its actual example count. Validation runs every
500 steps, restart checkpoints every 50 steps, and final calibration/test use
their complete filtered splits.

```bash
# macOS: the Modal CLI lives in the modal_env conda environment.
export PATH="$HOME/miniconda3/envs/modal_env/bin:$PATH"
# Full dataset, ModernBERT-base, no choice limit anywhere, on an H100:
python -m modal run --detach modal_train.py --config modal_base_full --run base-full --gpu h100
# Three-step smoke test of the same code path on an A10 (~2 minutes):
python -m modal run --detach modal_train.py --config smoke_base --run smoke-base --gpu a10
# Six choices on an A10 (the defaults):
python -m modal run --detach modal_train.py
# At most 28 choices for training, every source at test time, on an H100:
python -m modal run --detach modal_train.py --config modal_28 --run modal-28 --gpu h100
# Resume any run by repeating its --config/--run/--gpu and adding --resume:
python -m modal run --detach modal_train.py --config modal_28 --run modal-28 --gpu h100 --resume
# Mirror logs into the existing local TensorBoard, then download final artifacts:
python sync_modal.py --run modal-28 --watch
# One-off synchronization, optionally including checkpoints:
python sync_modal.py --run modal-28 --artifacts
```

Modal app: `modernbert-jev-choices`; persistent volume:
`modernbert-jev-experiments`. `--gpu a10` requests one A10, four CPU cores and 16GB
host memory with a six-hour timeout; `--gpu h100` requests one H100, sixteen CPU
cores and 64GB with a 24-hour timeout, because removing the choice limit makes
tokenization (which re-encodes the context once per candidate) the bottleneck.
The image sets `expandable_segments:True` under both the old and new PyTorch
allocator variable names: padded batch shapes vary widely between steps, which
fragmented the default allocator badly enough to waste 13.6GB in the `modal-all`
OOM. GPU allocation ends when the
function finishes or times out; volume storage persists. Model/data preparation
runs on CPU before GPU allocation. No local credentials or dataset copies are
uploaded: code/configuration are mounted, and pinned public model/data files are
downloaded in the cloud. Modal authenticates through the existing CLI profile.

Cloud artifacts live under `/runs/modal-six` in the volume. The local sync worker
updates `runs/modal-six` and the `modal-six` TensorBoard series every minute;
it does not keep the training job alive. The launcher spawns the GPU function and
returns its `function_call_id` in `runs/<run>-cloud.json` without awaiting it, so
the cloud run is independent of this machine: closing the terminal, losing DNS, or
Ctrl-C cannot cancel it. `--detach` is required and the launcher refuses to start
without it, because Modal otherwise stops the app when the launcher returns. Set
`PYTHONIOENCODING=utf-8` so the Modal CLI's status glyphs cannot abort the launch
on a cp1252 console. Local charts resume updating only when the sync worker is
running, and `--watch` retries transient network errors with backoff instead of
exiting. It exits after completion/failure. Re-run sync after reconnecting.

The GPU function commits the volume every two minutes, so an abrupt container loss
costs at most that much training on top of the 50-step checkpoint interval. If the
function is cancelled or times out, it signals the training child, waits for it,
and records `interrupted` in `run.json` rather than leaving an orphan on the GPU.
A status that `experiment.py` set itself is never overwritten, so a real traceback
is not replaced by the wrapper's generic exit-code message.

Use `python -m modal app list` to inspect jobs and `python -m modal app stop APP_ID`
to stop a specific cloud job. `runs/modal-launch.*.log` captures the local launcher;
the cloud run also persists `console.log`. The earlier full unfiltered laptop run
was stopped when switching to this filtered experiment.

## Environment and setup

Run commands from this directory in PowerShell:

```powershell
conda activate yolo_env
python download_model.py
python -m unittest test_experiment test_tracking -v
```

The existing environment has PyTorch `2.9.0+cu130`, Transformers `5.17.0`,
Hugging Face Hub, and safetensors. TensorBoard and its missing dependencies are
installed locally in `.deps`. On a fresh workspace, install them with:

```powershell
python -m pip install --target .deps --no-deps -r requirements-dashboard.txt
```

Dataset
JSONL files are downloaded directly; `datasets`, `peft`, and FastAPI are not needed.
An equivalent activation-free prefix is `conda run --no-capture-output -n yolo_env python`.

Model revision: `45bb4654a4d5aaff24dd11d4781fa46d39bf8c13`.
Dataset revision: `cbcb6703ef02b698bee32af39d3270fe1f356187`.
Weights are local in `models/ModernBERT-large`; data hashes and source licenses are
recorded in each run's `data_audit.json`. The dataset's source-specific licenses
are preserved in `data/manifest.json` and `data/README.md`.

## Run experiments

```powershell
# Two-step integration check; these results do not measure useful model quality.
python experiment.py --config configs/smoke.json --output runs/smoke-new

# Small initial experiment: MNLI, ARC-Challenge, MMLU.
python experiment.py --config configs/pilot.json --output runs/pilot

# Full choice-only training: three epochs, with restartable checkpoints.
python experiment.py --config configs/choice_all.json --output runs/choice-full

# Resume an interrupted full run, using its saved configuration.
python experiment.py --output runs/choice-full --resume
```

Each output directory is single-use once training starts, preventing accidental
checkpoint replacement. Use `--prepare-only` to download and audit data before
training. `last.pt` contains the latest periodic restart checkpoint (adapter,
optimizer, shuffled sample order, cursor, RNG state, and progress); `adapter.pt`
is the best validation checkpoint used for prediction. Restart checkpoints are
written atomically after step 1, every configured `save_every` steps, and at the
end. Interrupted work since the latest restart checkpoint is replayed. Resuming
preserves the old history in a backup and removes superseded TensorBoard steps.
`progress.json` records the active phase and training microbatch.

## TensorBoard dashboard

```powershell
python dashboard.py
```

Open **http://127.0.0.1:6006**. Existing JSON logs are imported automatically,
including the pilot and smoke runs. New experiments stream events to
`runs/tensorboard/<run-name>` and flush after every optimizer step; the dashboard
reloads every five seconds. Select runs in the sidebar to compare them.

- `train/`: loss, learning rate, gradient norm, step duration and GPU memory.
- `validation/`: step-zero baseline and checkpoint-selection metrics.
- `calibration/`: temperature and before/after calibration metrics.
- `test/`: final calibrated/uncalibrated metrics and results by source.
- Text cards: configuration, dataset audit, parameter counts and run metadata.

Earlier JSON logs contain loss, elapsed time, validation and final results, but
not per-step gradient norm or GPU memory; those appear for new runs. Final test
metrics are logged only after training and calibration, not after every step.
TensorBoard smoothing can hide fluctuations; set smoothing to zero for raw loss.
Use `python dashboard.py --sync-only` to import logs without starting a server,
or `--port 6007` to use a different port. Importing an unchanged run is idempotent.
The server binds only to localhost. It is a local experiment viewer, not a hosted
service or an MLflow registry.

The pilot samples at most 256 training, 64 validation, and 64 test rows per source.
The sampled validation rows are divided equally into checkpoint selection and
temperature calibration sets. It runs 100 optimizer steps with four examples per
step, so it is a pipeline/learning pilot rather than a converged benchmark.
The full configuration uses every available choice row for three epochs: 148,092
example presentations, or 18,512 optimizer steps at accumulation 8. It uses
eight-candidate chunks, 3% learning-rate warmup followed by cosine decay,
validation every 2,000 steps, and restart checkpoints every 10 steps. The final
partial accumulation is normalized correctly. The random-head baseline is skipped
for this long run; calibration and the entire test split run after training.
ChaosNLI is test-only. This is full-dataset **LoRA** training; the pretrained
backbone weights remain frozen to fit the GPU.

## Memory strategy for the 6GB RTX 3050

- Freeze the pretrained weights; train rank-8 LoRA on attention QKV/output
  projections plus a scalar head (approximately 1.38M trainable parameters).
- Keep parameters in FP32, use BF16 autocast on CUDA, and checkpoint activations.
- Score candidates in small chunks (two in pilot, eight in full training). A first pass computes the complete distribution;
  a second pass backpropagates its exact softmax derivative by candidate chunk.
- Disable dropout for consistency between passes. A unit test verifies the
  chunked gradient against full soft-target cross-entropy.
- This saves memory, not computation: a 151-choice example still requires scoring
  all 151 candidates twice for training. The full benchmark is costly on a laptop.

LoRA is implemented locally in `choice_model.py`; adapters are project-specific
PyTorch state dictionaries, not PEFT-format checkpoints. Both train and prediction
use the same module and loading code. No generative language-model head is used.
The pretrained MLM head being reported as unused at load time is expected.

## Data and evaluation

- Published train/validation/test boundaries are retained. Training never uses test labels.
- Exact input duplicates are excluded from train if present in published validation
  or test, and from validation if present in published test. Cross-source exact
  overlaps among selected rows are also excluded. This is not a semantic or
  pretraining-contamination audit.
- Choice IDs remain external; only candidate descriptions are scored. No labels,
  soft labels, source metadata, or IDs enter the model input.
- Hard labels use one-hot targets; available human distributions use soft targets.
- Only context is truncated; question/candidate text is protected. Oversized
  question/candidate pairs produce an error. Test truncation counts are reported.
- Checkpoints are selected by validation target NLL. A separate calibration set
  fits a scalar temperature on a bounded 0.1–10 grid. Test is evaluated afterward.
- Reports include hard-label accuracy, target NLL, summed multiclass Brier score,
  10-bin ECE against target agreement, per-source metrics, and GPU peak allocation.
  For soft targets, Brier/TVD describe distance to the human distribution; they
  should not be confused with hard-label Brier or empirical accuracy.
- `baseline_validation.json` uses a random scoring head and is a sanity baseline,
  not the pretrained encoder's zero-shot ability. Reported test accuracy is sampled
  and in-domain; it does not establish general instruction following.

## JSON prediction

After a run completes:

```powershell
python predict.py --run runs/pilot --input example_request.json
python predict.py --run runs/pilot --serve --port 8000
# Starts a temporary server, validates actual HTTP requests, and stops it.
python check_endpoint.py --run runs/pilot
```

The local, single-request-at-a-time development server accepts `POST /predict`:

```json
{
  "question": "Which team should investigate?",
  "context": "Checkout stopped working after the deployment.",
  "choices": {
    "engineering": "Software failures and service outages",
    "sales": "Product and purchasing inquiries"
  }
}
```

Response fields: `choice`, `probabilities` keyed by supplied IDs,
`max_probability`, `context_truncated`, and `temperature`. The maximum probability
is not Jev's separate confidence statistic. Calibration only applies insofar as
new requests resemble the calibration distribution. The API requires 2–255
choices with one intended outcome; add an explicit abstention option in training
if your task needs it. Score and Noul primitives are outside this experiment.

## Files

- `bench_data.py`: pinned downloads, normalization, sampling and split checks.
- `choice_model.py`: encoder, LoRA, pooling, tokenization and schema validation.
- `experiment.py`: training, checkpoint selection, calibration and evaluation.
- `predict.py`: shared inference path for JSON files and HTTP.
- `test_experiment.py`: gradient, distribution, LoRA and validation tests.
- `check_endpoint.py`: real HTTP integration check, including choice permutation.
- `runs/<name>/metrics.json`: final held-out evaluation; `history.jsonl`: training log.

The smoke test and pilot must not be presented as reproducing Jev's architecture
or matching its performance. This is an independently designed encoder experiment.

</details>
