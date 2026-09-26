import json
import tempfile
import unittest
from pathlib import Path

from tracking import Tracker, enable_local_dependencies, import_run

enable_local_dependencies()
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


class TrackingTests(unittest.TestCase):
    def test_native_scalar_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "native"
            output.mkdir()
            tracker = Tracker(output)
            tracker.entry({"step": 1, "loss": 0.75, "gradient_norm": 1.2,
                           "validation": {"accuracy": 0.5}})
            tracker.close()
            events = EventAccumulator(str(tracker.directory)).Reload()
            self.assertEqual(events.Scalars("train/loss")[0].step, 1)
            self.assertAlmostEqual(events.Scalars("validation/accuracy")[0].value, 0.5)
            self.assertEqual(import_run(output), "native tracking")

    def test_reimport_does_not_duplicate_points(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "old"
            output.mkdir()
            (output / "history.jsonl").write_text(json.dumps({"step": 1, "loss": 0.8}) + "\n")
            (output / "metrics.json").write_text(json.dumps({"calibrated": {"accuracy": 0.6}}))
            import_run(output)
            import_run(output)
            events = EventAccumulator(str(output.parent / "tensorboard" / "old")).Reload()
            self.assertEqual(len(events.Scalars("train/loss")), 1)
            self.assertEqual(len(events.Scalars("test/calibrated/accuracy")), 1)


if __name__ == "__main__":
    unittest.main()
