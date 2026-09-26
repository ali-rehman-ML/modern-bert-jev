"""Fold the LoRA adapter into the backbone and export one ONNX scorer.

The browser demo cannot run PyTorch, so the adapter has to stop being an adapter:
`base(x) + (x @ A.T @ B.T) * scale` is just `Linear(W + scale * B @ A)`, which
collapses into the frozen weight and leaves a plain ModernBERT plus the scoring
head. Export, quantize, then check the result still reproduces the PyTorch scores
on real test rows rather than trusting that the graph is equivalent.
"""
import argparse
import json
from pathlib import Path

import torch
from torch import nn

from bench_data import prepare
from choice_model import ChoiceScorer, LoRALinear, chunks, encode, tokenizer


class Scorer(nn.Module):
    """ChoiceScorer.forward with the tokens split out, so ONNX sees plain inputs."""

    def __init__(self, source):
        super().__init__()
        self.encoder = source.encoder
        self.head = source.head

    def forward(self, input_ids, attention_mask):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        return self.head(pooled).squeeze(-1)


def merge(model):
    """Replace every LoRALinear with an ordinary Linear holding the summed weight."""
    merged = 0
    for name, module in list(model.encoder.named_modules()):
        if not isinstance(module, LoRALinear):
            continue
        base = module.base
        base.weight.data += module.scale * (module.B.data @ module.A.data)
        parent, attribute = name.rsplit(".", 1)
        setattr(model.encoder.get_submodule(parent), attribute, base)
        merged += 1
    if not merged:
        raise RuntimeError("No LoRA layers found to merge")
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="runs/base-full")
    parser.add_argument("--output", default="runs/base-full/onnx")
    parser.add_argument("--check-rows", type=int, default=200)
    args = parser.parse_args()

    run = Path(args.run)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((run / "config.json").read_text())
    backbone = config["model"]

    model = ChoiceScorer(config["lora_rank"], config["lora_alpha"], "cpu", backbone)
    model.load_adapter(run / "adapter.pt")
    model.eval()
    tok = tokenizer(backbone)

    # Keep a reference copy of the real scores before anything is rewritten.
    groups, _ = prepare(config | {"test_per_source": max(1, args.check_rows // 9)})
    rows = groups["test"][:args.check_rows]
    encoded = [encode(record, tok, config["max_length"]) for record in rows]
    with torch.inference_mode():
        reference = [torch.cat([model(batch) for batch in chunks(tokens, 32, "cpu")]).clone()
                     for tokens, _ in encoded]

    print(json.dumps({"merged_lora_layers": merge(model)}), flush=True)
    model.encoder.gradient_checkpointing_disable()
    exportable = Scorer(model).eval()

    # Folding W and scale*B@A into one matmul reorders float accumulation, so a
    # drift of ~1e-3 on scores spanning several units is arithmetic, not error.
    # The invariant that actually matters is that no ranking decision moved.
    worst = 0.0
    with torch.inference_mode():
        for (tokens, _), before in zip(encoded, reference):
            after = exportable(tokens["input_ids"], tokens["attention_mask"])
            worst = max(worst, float((after - before).abs().max()))
            if int(after.argmax()) != int(before.argmax()):
                raise RuntimeError("Merging LoRA changed which candidate wins")
            if worst > 0.05:
                raise RuntimeError(f"Merging LoRA moved the scores by {worst}")
    print(json.dumps({"merge_verified": True, "max_score_drift": round(worst, 6)}), flush=True)

    sample = encoded[0][0]
    path = output / "model.onnx"
    torch.onnx.export(
        exportable, (sample["input_ids"], sample["attention_mask"]), str(path),
        input_names=["input_ids", "attention_mask"], output_names=["score"],
        dynamic_axes={"input_ids": {0: "batch", 1: "sequence"},
                      "attention_mask": {0: "batch", 1: "sequence"},
                      "score": {0: "batch"}},
        opset_version=17, do_constant_folding=True, dynamo=False)
    print(json.dumps({"exported_mib": round(path.stat().st_size / 1024**2, 1)}), flush=True)

    from onnxruntime.quantization import QuantType, quantize_dynamic
    quantized = output / "model_int8.onnx"
    quantize_dynamic(str(path), str(quantized), weight_type=QuantType.QUInt8)
    print(json.dumps({"quantized_mib": round(quantized.stat().st_size / 1024**2, 1)}), flush=True)

    import onnxruntime
    report = {}
    for label, candidate in (("fp32", path), ("int8", quantized)):
        session = onnxruntime.InferenceSession(str(candidate), providers=["CPUExecutionProvider"])
        worst, correct = 0.0, 0
        for (tokens, _), before, record in zip(encoded, reference, rows):
            scores = session.run(["score"], {"input_ids": tokens["input_ids"].numpy(),
                                             "attention_mask": tokens["attention_mask"].numpy()})[0]
            worst = max(worst, float(abs(scores - before.numpy()).max()))
            chosen = int(scores.argmax())
            correct += int(record["target"][chosen] == max(record["target"]))
        report[label] = {"max_score_drift": round(worst, 5),
                         "accuracy": round(correct / len(rows), 4)}
        print(json.dumps({label: report[label]}), flush=True)

    reference_accuracy = sum(
        int(record["target"][int(scores.argmax())] == max(record["target"]))
        for record, scores in zip(rows, reference)) / len(rows)
    report["pytorch"] = {"accuracy": round(reference_accuracy, 4), "n": len(rows)}
    (output / "export_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
