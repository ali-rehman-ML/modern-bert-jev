import copy
import json
import random
import tempfile
from types import SimpleNamespace
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from bench_data import normalize, fingerprint, prepare
from choice_model import LoRALinear, validate_request, tokenizer, encode
from experiment import train_record, train_records, metrics, fit_temperature, save_restart, learning_rate, subsample


class SplitChoiceLimitTests(unittest.TestCase):
    def test_test_split_ignores_training_choice_limit(self):
        rows = {"train": [2, 3, 30], "validation": [2, 40], "test": [2, 50]}

        def fake_normalize(row):
            return {"id": row["id"], "source": "s", "context": row["id"], "question": "q",
                    "choices": {f"c{i}": f"d{i}" for i in range(row["n"])},
                    "target": [1.0] + [0.0] * (row["n"] - 1), "label": 0}

        manifest = {"sources": {"s": {"primitive": "choice", "license": "x",
                                      "counts": {}, "files": {k: f"{k}.jsonl" for k in rows}}}}

        def fake_fetch(relative, destination):
            return Path(str(destination))

        def fake_read_text(self, **kwargs):
            return json.dumps(manifest)

        def fake_read_bytes(self):
            split = Path(str(self)).name.split(".")[0]
            return "\n".join(json.dumps({"id": f"{split}{i}", "n": n})
                              for i, n in enumerate(rows[split])).encode()

        config = {"sources": ["s"], "max_choices": 28, "test_max_choices": None, "seed": 1}
        with patch("bench_data.fetch", fake_fetch),              patch("bench_data.normalize", fake_normalize),              patch.object(Path, "read_text", fake_read_text),              patch.object(Path, "read_bytes", fake_read_bytes):
            groups, audit = prepare(config)
        # Training and validation drop the >28-choice rows; test keeps all of them.
        self.assertEqual(sorted(len(r["choices"]) for r in groups["train"]), [2, 3])
        self.assertEqual(sorted(len(r["choices"]) for r in groups["test"]), [2, 50])
        self.assertEqual(audit["max_choices"], 28)
        self.assertIsNone(audit["test_max_choices"])
        self.assertEqual(audit["excluded_choice_limit"]["s/test"], 0)
        self.assertEqual(audit["excluded_choice_limit"]["s/train"], 1)


class ExperimentTests(unittest.TestCase):
    def test_batched_variable_choices_match_individual_gradients(self):
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 1)
            def forward(self, tokens):
                return self.linear(tokens["input_ids"]).squeeze(-1)
        torch.manual_seed(5)
        reference = Toy()
        batched = copy.deepcopy(reference)
        encoded = [{"input_ids": torch.randn(n, 3), "attention_mask": torch.ones(n, 3)} for n in (2, 3)]
        records = [{"target": [1.0, 0.0]}, {"target": [0.2, 0.5, 0.3]}]
        losses = [-(torch.tensor(r["target"]) * reference(t).log_softmax(0)).sum() for r, t in zip(records, encoded)]
        torch.stack(losses).mean().backward()
        with patch("experiment.encode", side_effect=[(t, 0) for t in encoded]):
            train_records(batched, records, SimpleNamespace(pad_token_id=0), {"max_length": 10}, "cpu")
        for a, b in zip(reference.parameters(), batched.parameters()):
            torch.testing.assert_close(a.grad, b.grad)

    def test_restart_restores_optimizer_and_parameters(self):
        class Toy(torch.nn.Linear):
            def adapter_state(self):
                return {k: v.detach().cpu() for k, v in self.state_dict().items()}
        model = Toy(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        x = torch.tensor([[0.2, 0.8]])
        model(x).square().sum().backward()
        optimizer.step()
        with tempfile.TemporaryDirectory() as directory:
            save_restart(Path(directory), model, optimizer, 1, [2, 0, 1], 1, 0.5, 1, 2.0)
            state = torch.load(Path(directory) / "last.pt", weights_only=True)
            restored = Toy(2, 1)
            restored.load_state_dict(state["model"])
            restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.01)
            restored_optimizer.load_state_dict(state["optimizer"])
            for current, opt in ((model, optimizer), (restored, restored_optimizer)):
                opt.zero_grad()
                current(x).square().sum().backward()
                opt.step()
            for a, b in zip(model.parameters(), restored.parameters()):
                torch.testing.assert_close(a, b)
            self.assertEqual(state["order"], [2, 0, 1])

    def test_warmup_cosine_schedule(self):
        config = {"learning_rate": 0.001, "warmup_fraction": 0.1}
        self.assertAlmostEqual(learning_rate(config, 1, 100), 0.0001)
        self.assertAlmostEqual(learning_rate(config, 10, 100), 0.001)
        self.assertAlmostEqual(learning_rate(config, 100, 100), 0.0)

    def test_soft_label_order(self):
        row = {"id": "x", "source": "x", "state": '{"text":"hello"}',
               "question": '{"type":"choice","instructions":"Which?","criteria":{"b":"B","a":"A"}}',
               "label": "a", "soft_label": '{"a":0.8,"b":0.2}'}
        record = normalize(row)
        self.assertEqual(record["target"], [0.2, 0.8])
        self.assertEqual(record["label"], 1)
        reordered = copy.deepcopy(record)
        reordered["choices"] = {"a": "A", "b": "B"}
        self.assertEqual(fingerprint(record), fingerprint(reordered))

    def test_invalid_request(self):
        with self.assertRaises(ValueError):
            validate_request({"question": "Q", "context": "C", "choices": {"a": "A"}})

    def test_lora_initial_equivalence_and_gradient(self):
        base = torch.nn.Linear(3, 2)
        base.requires_grad_(False)
        layer = LoRALinear(base, 2, 4)
        x = torch.randn(4, 3)
        torch.testing.assert_close(layer(x), base(x))
        layer(x).sum().backward()
        self.assertIsNone(base.weight.grad)
        self.assertGreater(layer.B.grad.abs().sum(), 0)

    def test_chunked_gradients_equal_full_soft_cross_entropy(self):
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 1)
            def forward(self, tokens):
                return self.linear(tokens["input_ids"]).squeeze(-1)
        torch.manual_seed(9)
        full = Toy()
        chunked = copy.deepcopy(full)
        tokens = {"input_ids": torch.randn(5, 3)}
        target = [0.1, 0.2, 0.0, 0.4, 0.3]
        loss = -(torch.tensor(target) * full(tokens).log_softmax(0)).sum() / 2
        loss.backward()
        with patch("experiment.encode", return_value=(tokens, 0)):
            train_record(chunked, {"target": target}, None,
                         {"max_length": 10, "choice_chunk": 2, "accumulation": 2}, "cpu")
        for a, b in zip(full.parameters(), chunked.parameters()):
            torch.testing.assert_close(a.grad, b.grad)

    def test_metrics_and_calibration(self):
        predictions = [dict(logits=[0.0, 0.0], target=[1.0, 0.0], label=0)]
        result = metrics(predictions)
        self.assertAlmostEqual(result["target_nll"], 0.69314718)
        self.assertAlmostEqual(result["brier_sum"], 0.5)
        self.assertAlmostEqual(result["ece_10_target_agreement"], 0.5)
        predictions = [dict(logits=[5.0, -5.0], target=[0.6, 0.4], label=0)]
        temperature = fit_temperature(predictions)
        self.assertLess(metrics(predictions, temperature)["target_nll"], metrics(predictions)["target_nll"])

    def test_context_truncation_preserves_question_and_choices(self):
        tok = tokenizer()
        record = {"context": "A very long irrelevant passage. " * 100,
                  "question": "Is the transaction approved?",
                  "choices": {"yes": "Approval granted", "no": "Approval denied"}}
        tokens, truncated = encode(record, tok, 64)
        self.assertEqual(truncated, 2)
        for index, description in enumerate(record["choices"].values()):
            text = tok.decode(tokens["input_ids"][index], skip_special_tokens=True)
            self.assertIn(record["question"], text)
            self.assertIn(description, text)
        self.assertLessEqual(tokens["input_ids"].shape[1], 64)

    def test_published_test_overlap_removed_and_calibration_disjoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            info = {"primitive": "choice", "license": "test", "counts": {},
                    "files": {s: s + ".jsonl" for s in ("train", "validation", "test")}}
            (root / "manifest.json").write_text(json.dumps({"sources": {"toy": info}}))
            def row(identifier, text):
                return {"id": identifier, "source": "toy", "state": text,
                        "question": {"type": "choice", "instructions": "Pick", "criteria": {"a": "A", "b": "B"}},
                        "label": "a", "soft_label": None}
            values = {"train": [row("train/1", '"overlap"'), row("train/2", '"safe"')],
                      "validation": [row(f"val/{i}", json.dumps(f"validation {i}")) for i in range(4)],
                      "test": [row("test/1", '"overlap"')]}
            excessive = row("train/too_many", '"too many choices"')
            excessive["question"]["criteria"] = {"a": "A", **{str(i): f"Option {i}" for i in range(6)}}
            values["train"].append(excessive)
            boundary = row("train/exactly_six", '"six choices"')
            boundary["question"]["criteria"] = {"a": "A", **{str(i): f"Option {i}" for i in range(5)}}
            values["train"].append(boundary)
            for split, rows in values.items():
                (root / (split + ".jsonl")).write_text("\n".join(json.dumps(r) for r in rows))
            with patch("bench_data.fetch", side_effect=lambda relative, destination: root / relative):
                groups, audit = prepare({"sources": ["toy"], "seed": 42, "max_choices": 6}, root)
            self.assertEqual({r["id"] for r in groups["train"]}, {"train/2", "train/exactly_six"})
            self.assertEqual(audit["excluded_choice_limit"]["toy/train"], 1)
            self.assertEqual(max(len(r["choices"]) for r in groups["train"]), 6)
            self.assertEqual(audit["excluded_exact_overlaps"]["toy/train"], 1)
            ids = [{r["id"] for r in rows} for rows in groups.values()]
            for i in range(len(ids)):
                for j in range(i):
                    self.assertFalse(ids[i] & ids[j])


if __name__ == "__main__":
    unittest.main()


class NullChoiceDescriptionTests(unittest.TestCase):
    """go_emotions/ledgar publish null criteria values; every candidate then
    rendered as "Candidate: None" and the model was forced to uniform output."""

    def row(self, criteria):
        return {"id": "s/test/1", "source": "s", "state": '"text"',
                "question": {"type": "choice", "instructions": "q", "criteria": criteria},
                "label": "a"}

    def test_null_description_falls_back_to_the_choice_id(self):
        record = normalize(self.row({"a": None, "b": None}))
        self.assertEqual(record["choices"], {"a": "a", "b": "b"})

    def test_real_descriptions_are_kept(self):
        record = normalize(self.row({"a": "Alpha", "b": "  "}))
        self.assertEqual(record["choices"], {"a": "Alpha", "b": "b"})

    def test_encode_rejects_records_whose_candidates_are_all_identical(self):
        record = {"id": "s/test/1", "context": "text", "question": "q",
                  "choices": {"a": "same", "b": "same"}}
        with self.assertRaisesRegex(ValueError, "same text"):
            encode(record, tokenizer(), 512)


class SubsampleTests(unittest.TestCase):
    """Training samples candidates; the target distribution must survive intact."""

    def record(self, n, target, label):
        return {"id": "s/1", "source": "s", "context": "c", "question": "q",
                "choices": {f"c{i}": f"d{i}" for i in range(n)},
                "target": target, "label": label}

    def test_small_records_pass_through_untouched(self):
        record = self.record(4, [0.0, 1.0, 0.0, 0.0], 1)
        self.assertIs(subsample(record, 32), record)
        self.assertIs(subsample(record, None), record)

    def test_gold_is_kept_and_the_rest_are_negatives(self):
        target = [0.0] * 100
        target[73] = 1.0
        for seed in range(20):
            random.seed(seed)
            sampled = subsample(self.record(100, target, 73), 16)
            self.assertEqual(len(sampled["choices"]), 16)
            self.assertEqual(len(sampled["target"]), 16)
            self.assertIn("c73", sampled["choices"])
            self.assertEqual(sampled["target"][sampled["label"]], 1.0)
            self.assertEqual(list(sampled["choices"])[sampled["label"]], "c73")

    def test_soft_label_mass_is_kept_and_renormalized(self):
        target = [0.0] * 100
        target[10], target[20], target[30] = 0.5, 0.3, 0.2
        random.seed(0)
        sampled = subsample(self.record(100, target, 10), 16)
        self.assertEqual(sorted(sampled["target"], reverse=True)[:3], [0.5, 0.3, 0.2])
        self.assertAlmostEqual(sum(sampled["target"]), 1.0)
        for name in ("c10", "c20", "c30"):
            self.assertIn(name, sampled["choices"])

    def test_limit_never_drops_gold_mass_even_when_it_exceeds_the_limit(self):
        target = [0.05] * 20 + [0.0] * 80
        random.seed(0)
        sampled = subsample(self.record(100, target, 0), 8)
        self.assertEqual(len(sampled["choices"]), 20)
        self.assertAlmostEqual(sum(sampled["target"]), 1.0)

    def test_choices_and_target_stay_aligned(self):
        target = [0.0] * 50
        target[7] = 1.0
        random.seed(3)
        record = self.record(50, target, 7)
        sampled = subsample(record, 10)
        for name, value in zip(sampled["choices"], sampled["target"]):
            self.assertEqual(value, record["target"][int(name[1:])])
