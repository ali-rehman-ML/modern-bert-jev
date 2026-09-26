"""Mirror Modal metrics into the local TensorBoard and fetch final artifacts."""
import argparse
import json
import time
from pathlib import Path

import modal
from tracking import import_run

MIRROR_WARNED = []

VOLUME = "modernbert-jev-experiments"
FILES = ("config.json", "run.json", "progress.json", "parameters.json", "data_audit.json",
         "split_ids.json", "history.jsonl", "best.json", "calibration.json", "metrics.json")


def download(volume, remote, local):
    content = b"".join(volume.read_file(remote))
    if local.exists() and local.read_bytes() == content:
        return False
    temporary = local.with_suffix(local.suffix + ".sync")
    temporary.write_bytes(content)
    temporary.replace(local)
    return True


def sync(run="modal-six", artifacts=False):
    if not run or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in run):
        raise ValueError("Run must be a simple directory name")
    volume = modal.Volume.from_name(VOLUME)
    output = Path("runs") / run
    output.mkdir(parents=True, exist_ok=True)
    try:
        available = {Path(entry.path).name for entry in volume.iterdir(f"/runs/{run}", recursive=False)}
    except (FileNotFoundError, modal.exception.NotFoundError):
        return "waiting for cloud run"
    changed = False
    for name in FILES:
        if name in available:
            modified = download(volume, f"/runs/{run}/{name}", output / name)
            changed |= modified and name != "progress.json"
    if changed and (output / "history.jsonl").exists():
        try:
            import_run(output)
        except Exception as error:
            # Local TensorBoard mirroring is optional; never let it stop the
            # metric and artifact download that the cloud run depends on.
            if not MIRROR_WARNED:
                print(json.dumps({"status": "tensorboard-mirror-disabled", "error": str(error)}), flush=True)
                MIRROR_WARNED.append(True)
    status = json.loads((output / "run.json").read_text()).get("status") if (output / "run.json").exists() else "starting"
    if artifacts or status == "complete":
        for name in ("adapter.pt", "last.pt", "test_predictions.json", "console.log"):
            if name in available:
                download(volume, f"/runs/{run}/{name}", output / name)
    print(json.dumps({"status": status, "local_run": str(output), "changed": changed}), flush=True)
    return status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="modal-six")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--artifacts", action="store_true")
    parser.add_argument("--interval", type=int, default=60)
    args = parser.parse_args()
    failures = 0
    while True:
        try:
            status = sync(args.run, args.artifacts)
            failures = 0
        except Exception as error:
            # A transient local network drop must not end the watch; the cloud run
            # is detached and keeps training regardless of this machine. Bad input
            # (an invalid run name) is a real error and still propagates.
            if not args.watch or isinstance(error, ValueError):
                raise
            failures += 1
            print(json.dumps({"status": "sync-error", "attempt": failures, "error": str(error)}), flush=True)
            time.sleep(min(max(args.interval, 10) * failures, 300))
            continue
        if not args.watch or status in ("complete", "failed", "interrupted"):
            break
        time.sleep(max(args.interval, 10))


if __name__ == "__main__":
    main()
