"""Run a reproducible choice-only ModernBERT-large/jev-bench experiment."""
import argparse
import contextlib
import json
import math
import random
import time
import sys
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F

from bench_data import prepare
from choice_model import ChoiceScorer, DEFAULT_MODEL, MODELS, chunks, encode, tokenizer
from tracking import Tracker


def dump(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def save_restart(output, model, optimizer, step, order, cursor, best, samples_seen, elapsed):
    state = {"model": model.adapter_state(), "optimizer": optimizer.state_dict(),
             "step": step, "order": order, "cursor": cursor, "best": best,
             "samples_seen": samples_seen, "elapsed": elapsed,
             "python_rng": random.getstate(), "torch_rng": torch.get_rng_state(),
             "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    temporary = output / "last.pt.tmp"
    torch.save(state, temporary)
    temporary.replace(output / "last.pt")


def learning_rate(config, step, total_steps):
    if "warmup_fraction" not in config:
        return config["learning_rate"]
    warmup = max(1, int(total_steps * config["warmup_fraction"]))
    factor = step / warmup if step <= warmup else 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup)))
    return config["learning_rate"] * factor


def subsample(record, limit):
    """Train an example against its gold candidates plus random negatives.

    Evaluation and calibration still score every published candidate. Without
    this, one ledgar/clinc150 example costs 100-151 encoder passes and the
    large-choice sources swamp the step budget; the scorer is a per-candidate
    scalar, so a model trained to rank a sampled subset ranks the full set too.
    Draws from the module RNG, whose state save_restart already checkpoints.
    """
    if not limit or len(record["target"]) <= limit:
        return record
    ids = list(record["choices"])
    keep = {i for i, value in enumerate(record["target"]) if value > 0} | {record["label"]}
    pool = [i for i in range(len(ids)) if i not in keep]
    keep = sorted(keep | set(random.sample(pool, max(0, limit - len(keep)))))
    mass = sum(record["target"][i] for i in keep)
    return record | {"choices": {ids[i]: record["choices"][ids[i]] for i in keep},
                     "target": [record["target"][i] / mass for i in keep],
                     "label": keep.index(record["label"])}


def precision(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else contextlib.nullcontext()


def metrics(predictions, temperature=1.0):
    if not predictions:
        return {}
    correct, nll, brier, confidence, agreement, human_tvd = [], [], [], [], [], []
    for item in predictions:
        p = torch.softmax(torch.tensor(item["logits"], dtype=torch.float64) / temperature, 0)
        target = torch.tensor(item["target"], dtype=torch.float64)
        index = int(p.argmax())
        correct.append(float(index == item["label"]))
        agreement.append(float(target[index]))
        confidence.append(float(p.max()))
        nll.append(float(-(target * p.clamp_min(1e-12).log()).sum()))
        brier.append(float(((p - target) ** 2).sum()))
        human_tvd.append(float((p - target).abs().sum() / 2))
    ece = 0.0
    for b in range(10):
        ids = [i for i, c in enumerate(confidence) if min(int(c * 10), 9) == b]
        if ids:
            ece += abs(sum(confidence[i] - agreement[i] for i in ids)) / len(predictions)
    return {"n": len(predictions), "accuracy": sum(correct) / len(correct),
            "target_nll": sum(nll) / len(nll), "brier_sum": sum(brier) / len(brier),
            "ece_10_target_agreement": ece, "mean_target_tvd": sum(human_tvd) / len(human_tvd),
            "mean_max_probability": sum(confidence) / len(confidence)}


def evaluate(model, records, tok, config, device):
    model.eval()
    predictions = []
    started = time.perf_counter()
    with torch.inference_mode():
        for record in records:
            tokens, truncated = encode(record, tok, config["max_length"])
            with precision(device):
                scores = torch.cat([model(batch) for batch in chunks(tokens, config["choice_chunk"], device)])
            predictions.append({k: record[k] for k in ("id", "source", "target", "label")} |
                               {"logits": scores.float().cpu().tolist(), "truncated_candidates": truncated})
    return predictions, time.perf_counter() - started


def fit_temperature(predictions):
    if not predictions:
        raise ValueError("A separate calibration split is required")
    # Deterministic bounded search: never fit on test or model-selection rows.
    candidates = torch.logspace(math.log10(0.1), math.log10(10), 101).tolist()
    return min(candidates, key=lambda t: metrics(predictions, t)["target_nll"])


def train_record(model, record, tok, config, device):
    tokens, _ = encode(record, tok, config["max_length"])
    target = torch.tensor(record["target"], device=device, dtype=torch.float32)
    # Two-pass gradient caching keeps candidate count from multiplying activation
    # memory: obtain dL/dlogits, then backpropagate each candidate chunk separately.
    # Dropout is disabled consistently in both passes; gradients still train LoRA.
    model.eval()
    with torch.no_grad(), precision(device):
        scores = torch.cat([model(batch) for batch in chunks(tokens, config["choice_chunk"], device)]).float()
    derivative = (scores.softmax(0) - target) / config["accumulation"]
    loss = -(target * scores.log_softmax(0)).sum()
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.eval()
    offset = 0
    for batch in chunks(tokens, config["choice_chunk"], device):
        with precision(device):
            logits = model(batch)
        gradient = derivative[offset:offset + len(logits)]
        (logits.float() * gradient).sum().backward()
        offset += len(logits)
    return float(loss)


def train_records(model, records, tok, config, device):
    """One differentiable pass over a batch of examples with variable choices."""
    encoded = [encode(record, tok, config["max_length"])[0] for record in records]
    widths = [len(tokens["input_ids"]) for tokens in encoded]
    length = max(tokens["input_ids"].shape[1] for tokens in encoded)
    tokens = {key: torch.cat([F.pad(part[key], (0, length - part[key].shape[1]),
                     value=tok.pad_token_id if key == "input_ids" else 0) for part in encoded]).to(device)
              for key in ("input_ids", "attention_mask")}
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.eval()
    with precision(device):
        logits = model(tokens).float()
        losses, offset = [], 0
        for record, width in zip(records, widths):
            target = torch.tensor(record["target"], device=device, dtype=torch.float32)
            losses.append(-(target * logits[offset:offset + width].log_softmax(0)).sum())
            offset += width
        loss = torch.stack(losses).mean()
    loss.backward()
    return float(loss.detach())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot.json")
    parser.add_argument("--output", default="runs/pilot")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume last.pt in --output using its saved config")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    output = Path(args.output)
    if args.resume:
        config = json.loads((output / "config.json").read_text())
        if not (output / "last.pt").exists():
            raise SystemExit("No restart checkpoint at --output/last.pt")
        if (output / "run.json").exists() and json.loads((output / "run.json").read_text())["status"] == "complete":
            raise SystemExit("Run already completed")
    elif (output / "adapter.pt").exists() or (output / "run.json").exists():
        raise SystemExit("Output already has an experiment. Choose a new --output directory.")
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config["seed"])
    random.seed(config["seed"])
    groups, audit = prepare(config)
    dump(output / "data_audit.json", audit)
    dump(output / "split_ids.json", {s: [r["id"] for r in rows] for s, rows in groups.items()})
    dump(output / "config.json", config)
    print(json.dumps(audit["selected_counts"]), flush=True)
    if args.prepare_only:
        return
    if any(not groups[s] for s in groups):
        raise ValueError("Training, validation, calibration, and test must all be nonempty")
    total_samples = int(config["epochs"] * len(groups["train"])) if "epochs" in config else config["steps"] * config["accumulation"]
    total_steps = math.ceil(total_samples / config["accumulation"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This configuration needs BF16 support; use FP32 CPU or adapt precision settings")
    import transformers
    backbone = config.get("model", DEFAULT_MODEL)
    dump(output / "run.json", {"model": backbone, "model_revision": MODELS[backbone],
        "torch": torch.__version__, "transformers": transformers.__version__, "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "status": "running", "scope": config.get("scope", "sampled choice-only pilot; not a full Jev benchmark"),
        "total_steps": total_steps, "total_training_examples": total_samples})
    tok = tokenizer(backbone)
    model = ChoiceScorer(config["lora_rank"], config["lora_alpha"], device, backbone)
    counts = {"total": sum(p.numel() for p in model.parameters()),
              "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    dump(output / "parameters.json", counts)
    print(json.dumps(counts), flush=True)
    restart = torch.load(output / "last.pt", map_location="cpu", weights_only=True) if args.resume else None
    tracker = Tracker(output, purge_step=restart["step"] + 1 if restart else None)
    tracker.text("config", config)
    tracker.text("data_audit", audit)
    tracker.text("parameters", counts)
    tracker.text("run", json.loads((output / "run.json").read_text()))
    if not args.resume and not config.get("skip_baseline", False):
        baseline, seconds = evaluate(model, groups["validation"], tok, config, device)
        dump(output / "baseline_validation.json", metrics(baseline) | {"seconds": seconds, "note": "Random scoring head; not zero-shot model capability"})
        tracker.scalars("validation", metrics(baseline), 0)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config["learning_rate"], weight_decay=0.01)
    order = list(range(len(groups["train"])))
    random.shuffle(order)
    cursor, best = 0, float("inf")
    first_step, samples_seen, previous_elapsed = 1, 0, 0.0
    if restart:
        model.load_state_dict(restart["model"], strict=False)
        optimizer.load_state_dict(restart["optimizer"])
        order, cursor, best = restart["order"], restart["cursor"], restart["best"]
        samples_seen, previous_elapsed = restart["samples_seen"], restart["elapsed"]
        first_step = restart["step"] + 1
        random.setstate(restart["python_rng"])
        torch.set_rng_state(restart["torch_rng"])
        if device == "cuda":
            torch.cuda.set_rng_state_all(restart["cuda_rng"])
        history_path = output / "history.jsonl"
        if history_path.exists():
            original = history_path.read_text()
            (output / f"history.before-resume-{time.time_ns()}.jsonl").write_text(original)
            retained = []
            for line in original.splitlines():
                try:
                    if json.loads(line)["step"] <= restart["step"]:
                        retained.append(line)
                except json.JSONDecodeError:
                    break
            history_path.write_text("\n".join(retained) + "\n")
        del restart
    start = time.perf_counter()
    for step in range(first_step, total_steps + 1):
        step_started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        accumulation = min(config["accumulation"], total_samples - samples_seen)
        lr = learning_rate(config, step, total_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        batch_records = []
        for micro in range(accumulation):
            if cursor == len(order):
                random.shuffle(order)
                cursor = 0
            record = subsample(groups["train"][order[cursor]], config.get("train_max_candidates"))
            dump(output / "progress.json", {"phase": "training", "step": step, "total_steps": total_steps,
                "microbatch": micro + 1, "accumulation": accumulation, "source": record["source"],
                "choices": len(record["choices"]), "samples_seen": samples_seen, "updated_at": time.time()})
            if config.get("batch_examples", False):
                batch_records.append(record)
            else:
                total_loss += train_record(model, record, tok, config | {"accumulation": accumulation}, device)
            cursor += 1
            samples_seen += 1
        if batch_records:
            total_loss = train_records(model, batch_records, tok, config, device) * accumulation
        grad_norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0, error_if_nonfinite=True)
        optimizer.step()
        entry = {"step": step, "loss": total_loss / accumulation, "elapsed_seconds": previous_elapsed + time.perf_counter() - start,
                 "epoch": samples_seen / len(groups["train"]), "samples_seen": samples_seen,
                 "learning_rate": optimizer.param_groups[0]["lr"], "gradient_norm": float(grad_norm),
                 "step_seconds": time.perf_counter() - step_started}
        if device == "cuda":
            entry["gpu_allocated_gib"] = torch.cuda.memory_allocated() / 1024**3
            entry["gpu_peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 1024**3
        if step % config["eval_every"] == 0 or step == total_steps:
            dump(output / "progress.json", {"phase": "validation", "step": step, "updated_at": time.time()})
            validation, _ = evaluate(model, groups["validation"], tok, config, device)
            entry["validation"] = metrics(validation)
            if entry["validation"]["target_nll"] < best:
                best = entry["validation"]["target_nll"]
                model.save_adapter(output / "adapter.pt")
                dump(output / "best.json", entry)
        tracker.entry(entry)
        with (output / "history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry) + "\n")
        print(json.dumps(entry), flush=True)
        if step == 1 or step % config.get("save_every", 50) == 0 or step == total_steps:
            save_restart(output, model, optimizer, step, order, cursor, best, samples_seen,
                         previous_elapsed + time.perf_counter() - start)
    model.load_adapter(output / "adapter.pt")
    dump(output / "progress.json", {"phase": "calibration", "step": total_steps, "updated_at": time.time()})
    calibration, _ = evaluate(model, groups["calibration"], tok, config, device)
    temperature = fit_temperature(calibration)
    dump(output / "calibration.json", {"temperature": temperature, "n": len(calibration),
        "before": metrics(calibration), "after": metrics(calibration, temperature)})
    tracker.scalars("calibration", json.loads((output / "calibration.json").read_text()), total_steps)
    dump(output / "progress.json", {"phase": "test", "step": total_steps, "updated_at": time.time()})
    predictions, test_seconds = evaluate(model, groups["test"], tok, config, device)
    dump(output / "test_predictions.json", predictions)
    by_source = {source: metrics([p for p in predictions if p["source"] == source], temperature) for source in config["sources"]}
    report = {"uncalibrated": metrics(predictions), "calibrated": metrics(predictions, temperature),
        "per_source": by_source, "temperature": temperature, "test_seconds": test_seconds,
        "truncated_candidates": sum(p["truncated_candidates"] for p in predictions),
        "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3 if device == "cuda" else None,
        "uniform_accuracy_expectation": sum(1 / len(p["logits"]) for p in predictions) / len(predictions)}
    dump(output / "metrics.json", report)
    tracker.scalars("test", report, total_steps)
    run = json.loads((output / "run.json").read_text())
    run["status"] = "complete"
    dump(output / "run.json", run)
    tracker.text("completed_run", run)
    tracker.close()
    dump(output / "progress.json", {"phase": "complete", "step": total_steps, "updated_at": time.time()})
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except (Exception, KeyboardInterrupt) as error:
        if "--output" in sys.argv:
            output = Path(sys.argv[sys.argv.index("--output") + 1])
            if (output / "run.json").exists():
                run = json.loads((output / "run.json").read_text())
                run.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
                dump(output / "run.json", run)
        traceback.print_exc()
        sys.exit(1)
