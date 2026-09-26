"""Run filtered ModernBERT training on Modal with persistent artifacts.

Six choices on an A10:
    python -m modal run --detach modal_train.py
All choice counts on an H100:
    python -m modal run --detach modal_train.py --config modal_all --run modal-all --gpu h100
Resume either one by adding --resume with the same --config/--run/--gpu.

The GPU job is spawned rather than awaited, so no local failure can cancel it.
Both halves are required: --detach keeps the app alive once this entrypoint
returns, and spawn keeps the queued input from being tied to this client's
connection (an awaited .remote() call is cancelled when the client disconnects).
"""
import json
import sys
from pathlib import Path

import modal

COMMIT_SECONDS = 120


class ChildFailed(RuntimeError):
    """The training process exited nonzero and already recorded its own error."""

app = modal.App("modernbert-jev-choices")
volume = modal.Volume.from_name("modernbert-jev-experiments", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.9.0", "transformers==5.17.0", "tensorboard==2.21.0",
                      "numpy==2.3.5", "safetensors==0.8.0", "onnx==1.20.0",
                      "onnxruntime==1.30.0", "onnxscript==0.6.0")
         .env({"HF_HOME": "/experiment/.cache/huggingface", "HF_HUB_DISABLE_XET": "1",
               "HF_HUB_DISABLE_PROGRESS_BARS": "1", "PYTHONUNBUFFERED": "1",
               # Padded batch shapes vary widely between steps, which fragments the
               # default caching allocator badly (13.6 GiB wasted in the modal-all OOM).
               # torch 2.9 renamed this; set both so it works either side of the rename.
               "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
               "PYTORCH_ALLOC_CONF": "expandable_segments:True"}))
for filename in ("bench_data.py", "choice_model.py", "experiment.py", "tracking.py",
                 "download_model.py", "export_onnx.py"):
    image = image.add_local_file(Path(__file__).parent / filename, f"/opt/project/{filename}")
image = image.add_local_dir(Path(__file__).parent / "configs", "/opt/project/configs")


def mark_status(output, status, error):
    """Record a terminal status so `sync_modal.py --watch` stops waiting.

    Never overwrite a status the training process set itself: its error text is
    the real diagnosis, and this wrapper only knows the exit code.
    """
    path = output / "run.json"
    if not path.exists():
        return
    try:
        run = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return
    if run.get("status") in ("complete", "failed", "interrupted"):
        return
    run["status"] = status
    if error is not None:
        run["error"] = error
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(run, indent=2))
    temporary.replace(path)


def run_training(config: str, run: str, resume: bool):
    """Shared body for every GPU variant; only the hardware differs."""
    import os
    import signal
    import subprocess
    import sys
    import threading
    os.chdir("/experiment")
    output = Path("runs") / run
    output.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-u", "/opt/project/experiment.py",
               "--config", f"/opt/project/configs/{config}.json", "--output", str(output)]
    if resume:
        command.append("--resume")
    process = None
    stop = threading.Event()

    def commit_periodically():
        # Checkpoints are written every save_every steps; flush them to the volume
        # so an abrupt container loss costs at most COMMIT_SECONDS of training.
        while not stop.wait(COMMIT_SECONDS):
            try:
                volume.commit()
            except Exception as error:
                print(f"periodic volume commit failed: {error}", flush=True)

    committer = threading.Thread(target=commit_periodically, daemon=True)
    committer.start()
    try:
        with (output / "console.log").open("a", buffering=1) as log:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
            if process.wait() != 0:
                # experiment.py already recorded the real error in run.json; leave it
                # alone so the traceback is not replaced by this generic wrapper text.
                raise ChildFailed(f"Training failed with exit code {process.returncode}; see persisted console.log")
        return json.loads((output / "metrics.json").read_text())
    except BaseException as error:
        # Cancellation and timeouts arrive as exceptions here. Stop the child
        # explicitly; otherwise it is orphaned and keeps burning the GPU.
        if isinstance(error, ChildFailed):
            mark_status(output, "failed", None)
            raise
        if process is not None and process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        mark_status(output, "interrupted", f"{type(error).__name__}: {error}")
        raise
    finally:
        stop.set()
        committer.join(timeout=10)
        volume.commit()


@app.function(image=image, volumes={"/experiment": volume}, cpu=4, memory=8192, timeout=3600)
def prepare_cloud(config: str, run: str):
    import os
    import subprocess
    import sys
    os.chdir("/experiment")
    settings = json.loads(Path(f"/opt/project/configs/{config}.json").read_text())
    backbone = [settings["model"]] if "model" in settings else []
    subprocess.run([sys.executable, "/opt/project/download_model.py", *backbone], check=True)
    subprocess.run([sys.executable, "/opt/project/experiment.py",
                    "--config", f"/opt/project/configs/{config}.json",
                    "--output", f"runs/{run}-ready", "--prepare-only"], check=True)
    volume.commit()


@app.function(image=image, gpu="A10", volumes={"/experiment": volume}, cpu=4,
              memory=16384, timeout=21600, max_containers=1)
def train_a10(config: str, run: str, resume: bool = False):
    return run_training(config, run, resume)


# Every choice repeats the example context, so removing the choice-count limit
# raises the token workload ~96x. That needs both a faster GPU and enough CPU to
# keep tokenization off the critical path.
@app.function(image=image, gpu="H100", volumes={"/experiment": volume}, cpu=16,
              memory=65536, timeout=86400, max_containers=1)
def train_h100(config: str, run: str, resume: bool = False):
    return run_training(config, run, resume)


@app.function(image=image, volumes={"/experiment": volume}, cpu=8, memory=32768, timeout=5400)
def export_onnx(run: str):
    """Fold the adapter into the weights and emit a browser-ready ONNX scorer."""
    import os
    import subprocess
    import sys
    os.chdir("/experiment")
    settings = json.loads(Path(f"runs/{run}/config.json").read_text())
    subprocess.run([sys.executable, "/opt/project/download_model.py", settings["model"]], check=True)
    subprocess.run([sys.executable, "-u", "/opt/project/export_onnx.py",
                    "--run", f"runs/{run}", "--output", f"runs/{run}/onnx"], check=True)
    volume.commit()
    return json.loads(Path(f"runs/{run}/onnx/export_report.json").read_text())


TRAINERS = {"a10": train_a10, "h100": train_h100}


@app.local_entrypoint()
def main(config: str = "modal_six", run: str = "modal-six", gpu: str = "a10",
         resume: bool = False, skip_prepare: bool = False):
    if "--detach" not in sys.argv:
        # Without --detach Modal stops the app as soon as this entrypoint returns,
        # which kills the spawned training call.
        raise SystemExit("Launch with --detach, for example:\n"
                         "  python -m modal run --detach modal_train.py"
                         f" --config {config} --run {run} --gpu {gpu}"
                         + (" --resume" if resume else ""))
    if gpu not in TRAINERS:
        raise SystemExit(f"--gpu must be one of {sorted(TRAINERS)}")
    if not Path(f"configs/{config}.json").exists():
        raise SystemExit(f"No such config: configs/{config}.json")
    if not (resume or skip_prepare):
        prepare_cloud.remote(config, run)
    call = TRAINERS[gpu].spawn(config, run, resume=resume)
    record = {"app_id": app.app_id, "function_call_id": call.object_id,
              "url": f"https://modal.com/apps/{app.app_id}", "volume": volume.name,
              "run": run, "config": config, "gpu": gpu}
    Path("runs").mkdir(exist_ok=True)
    Path(f"runs/{run}-cloud.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))
    print(f"Training is detached. Follow it with: python sync_modal.py --run {run} --watch", flush=True)
