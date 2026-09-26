"""Pinned jev-bench choice data, with published splits and explicit sampling."""
import hashlib
import json
import random
import urllib.request
from pathlib import Path

REVISION = "cbcb6703ef02b698bee32af39d3270fe1f356187"
BASE = f"https://huggingface.co/datasets/Praveenrajus/jev-bench/resolve/{REVISION}/"


def fetch(relative, destination):
    destination = Path(destination)
    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    with urllib.request.urlopen(BASE + relative, timeout=120) as response:
        with temporary.open("wb") as stream:
            while block := response.read(1024 * 1024):
                stream.write(block)
    temporary.replace(destination)
    return destination


def decode(value):
    return json.loads(value) if isinstance(value, str) else value


def normalize(row):
    question = decode(row["question"])
    if question["type"] != "choice":
        raise ValueError("This experiment supports choice tasks only")
    criteria = question["criteria"]
    ids = list(criteria)
    if len(ids) < 2 or row["label"] not in criteria:
        raise ValueError(f"Invalid choices/label: {row['id']}")
    soft = decode(row.get("soft_label"))
    target = [float(soft.get(k, 0)) for k in ids] if soft is not None else [float(k == row["label"]) for k in ids]
    if any(not 0 <= v <= 1 for v in target) or abs(sum(target) - 1) > 1e-4:
        raise ValueError(f"Invalid probability distribution: {row['id']}")
    # go_emotions and ledgar publish null descriptions. Without this fallback every
    # candidate renders as "Candidate: None", so all choices share one input and the
    # model returns identical logits (exactly uniform) no matter how well it trained.
    choices = {k: v if isinstance(v, str) and v.strip() else k for k, v in criteria.items()}
    return {"id": row["id"], "source": row["source"],
            "context": decode(row["state"]), "question": question["instructions"],
            "choices": choices, "target": target, "label": ids.index(row["label"])}


def fingerprint(record):
    # Match identical model inputs even if source IDs differ. Ignore choice order.
    fields = {k: record[k] for k in ("context", "question", "choices")}
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()


def prepare(config, root="data"):
    root = Path(root)
    manifest = json.loads(fetch("manifest.json", root / "manifest.json").read_text(encoding="utf-8"))
    groups = {k: [] for k in ("train", "validation", "calibration", "test")}
    audit = {"dataset_revision": REVISION, "sources": {}, "excluded_exact_overlaps": {},
             "max_choices": config.get("max_choices"),
             "test_max_choices": config.get("test_max_choices", config.get("max_choices")),
             "excluded_choice_limit": {}}
    for source in config["sources"]:
        info = manifest["sources"][source]
        if info["primitive"] != "choice":
            raise ValueError(f"{source} is not a choice task")
        audit["sources"][source] = {"license": info["license"], "published_counts": info["counts"], "files": {}}
        source_groups = {}
        for split in ("train", "validation", "test"):
            if split not in info["files"]:
                source_groups[split] = []
                continue
            relative = info["files"][split]
            path = fetch(relative, root / "raw" / relative)
            raw = path.read_bytes()
            rows = [normalize(json.loads(line)) for line in raw.decode("utf-8").splitlines() if line.strip()]
            audit["sources"][source]["files"][split] = {"sha256": hashlib.sha256(raw).hexdigest(), "count": len(rows)}
            limit = config.get("max_choices")
            if split == "test" and "test_max_choices" in config:
                # Evaluate on every published test input even when training is
                # restricted to a smaller choice count. Explicit null means no limit.
                limit = config["test_max_choices"]
            retained = rows if limit is None else [r for r in rows if len(r["choices"]) <= limit]
            # Always audited, so an unfiltered split is visibly a choice, not an omission.
            audit["excluded_choice_limit"][f"{source}/{split}"] = len(rows) - len(retained)
            rows = retained
            rng = random.Random(f"{config['seed']}:{source}:{split}")
            rng.shuffle(rows)
            source_groups[split] = rows
        # Protect ALL published test inputs, not just the sampled test subset.
        test_keys = {fingerprint(r) for r in source_groups["test"]}
        validation_keys = {fingerprint(r) for r in source_groups["validation"]}
        for split in ("train", "validation"):
            excluded = test_keys | (validation_keys if split == "train" else set())
            seen = set()
            clean = []
            for record in source_groups[split]:
                key = fingerprint(record)
                if key not in excluded and key not in seen:
                    clean.append(record)
                    seen.add(key)
            audit["excluded_exact_overlaps"][f"{source}/{split}"] = len(source_groups[split]) - len(clean)
            source_groups[split] = clean
        for split, rows in source_groups.items():
            limit = config.get(f"{split}_per_source", 0)
            rows = rows[:limit] if limit else rows
            if split == "validation":
                # Disjoint sets: checkpoint selection vs temperature fitting.
                middle = len(rows) // 2
                groups["validation"].extend(rows[:middle])
                groups["calibration"].extend(rows[middle:])
            else:
                groups[split].extend(rows)
    # Cross-source overlap safeguard; test always takes priority.
    protected = set()
    for split in ("test", "calibration", "validation", "train"):
        before = len(groups[split])
        groups[split] = [r for r in groups[split] if fingerprint(r) not in protected]
        audit["excluded_exact_overlaps"][f"cross_source/{split}"] = before - len(groups[split])
        protected.update(fingerprint(r) for r in groups[split])
    audit["selected_counts"] = {k: len(v) for k, v in groups.items()}
    return groups, audit
