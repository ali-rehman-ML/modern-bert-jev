"""TensorBoard tracking for live runs and importing earlier JSON logs."""
import json
import numbers
import sys
from pathlib import Path


def enable_local_dependencies():
    local = Path(__file__).resolve().parent / ".deps"
    if local.exists() and str(local) not in sys.path:
        sys.path.append(str(local))


class Tracker:
    def __init__(self, output, native=True, purge_step=None):
        enable_local_dependencies()
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as error:
                # SummaryWriter lives in torch, so this fires when torch is missing too.
            raise RuntimeError(f"Needs torch and TensorBoard ({error}); TensorBoard alone installs with "
                               "python -m pip install --target .deps --no-deps -r requirements-dashboard.txt") from error
        self.output = Path(output)
        self.directory = self.output.parent / "tensorboard" / self.output.name
        self.writer = SummaryWriter(str(self.directory), flush_secs=5, purge_step=purge_step)
        if native:
            (self.output / "tracking.json").write_text(json.dumps({"native_tensorboard": True, "log_dir": str(self.directory)}))

    def scalars(self, prefix, values, step):
        for name, value in values.items():
            tag = f"{prefix}/{name}" if prefix else name
            if isinstance(value, dict):
                self.scalars(tag, value, step)
            elif isinstance(value, numbers.Real) and not isinstance(value, bool):
                self.writer.add_scalar(tag, value, step)

    def text(self, tag, value):
        self.writer.add_text(tag, "```json\n" + json.dumps(value, indent=2) + "\n```", 0)

    def entry(self, entry):
        step = entry["step"]
        self.scalars("train", {k: v for k, v in entry.items() if k not in ("step", "validation")}, step)
        if "validation" in entry:
            self.scalars("validation", entry["validation"], step)
        self.writer.flush()

    def close(self):
        self.writer.close()


def import_run(output):
    output = Path(output)
    if (output / "tracking.json").exists():
        return "native tracking"
    if not (output / "history.jsonl").exists():
        return "no training history"
    marker = output / "tensorboard_import.json"
    state = json.loads(marker.read_text()) if marker.exists() else {"step": 0, "files": {}}
    tracker = Tracker(output, native=False)
    try:
        for name in ("config", "run", "parameters", "data_audit"):
            path = output / f"{name}.json"
            if path.exists() and name not in state["files"]:
                tracker.text(name, json.loads(path.read_text()))
                state["files"][name] = path.stat().st_mtime_ns
        for line in (output / "history.jsonl").read_text().splitlines():
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                break  # A running process may still be writing the last line.
            if entry["step"] > state["step"]:
                tracker.entry(entry)
                state["step"] = entry["step"]
        for name, tag in (("baseline_validation", "validation"), ("calibration", "calibration"), ("metrics", "test")):
            path = output / f"{name}.json"
            if path.exists() and state["files"].get(name) != path.stat().st_mtime_ns:
                tracker.scalars(tag, json.loads(path.read_text()), 0 if name == "baseline_validation" else state["step"])
                state["files"][name] = path.stat().st_mtime_ns
        marker.write_text(json.dumps(state, indent=2))
    finally:
        tracker.close()
    return f"imported through step {state['step']}"
