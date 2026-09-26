"""Torch-free scorer: ONNX weights plus the raw tokenizer.

Serving does not need PyTorch, and saying so is worth real money — the whole
runtime is onnxruntime and the Rust tokenizer, so the API fits the smallest
container Modal sells and cold-starts in seconds instead of a minute.

The tokenization here must match choice_model.encode exactly, including
`only_first` truncation, or the scores drift from the evaluated model.
"""
import json
from pathlib import Path

import numpy as np
import onnxruntime
from tokenizers import Tokenizer

MAX_LENGTH = 512
MAX_CHOICES = 255
# A 151-candidate request is 151 sequences of up to 512 tokens; feed the graph in
# slices so peak memory follows the chunk, not the caller's choice count.
CHUNK = 32


class Scorer:
    def __init__(self, weights, tokenizer_file, temperature, max_length=MAX_LENGTH):
        self.session = onnxruntime.InferenceSession(
            str(weights), providers=["CPUExecutionProvider"])
        self.tokenizer = Tokenizer.from_file(str(tokenizer_file))
        self.temperature = float(temperature)
        self.max_length = max_length
        pad = self.tokenizer.token_to_id("[PAD]")
        if pad is None:
            raise RuntimeError("Tokenizer has no [PAD] token")
        # Only the context may be trimmed; the question and candidate are the
        # part that distinguishes one option from another.
        self.tokenizer.enable_truncation(max_length=max_length, strategy="only_first")
        self.tokenizer.enable_padding(pad_id=pad, pad_token="[PAD]")

    def __call__(self, context, question, choices, temperature=None):
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a nonempty string")
        if not isinstance(choices, dict) or not 2 <= len(choices) <= MAX_CHOICES:
            raise ValueError(f"choices must map 2 to {MAX_CHOICES} ids to descriptions")
        if any(not isinstance(k, str) or not k or not isinstance(v, str) or not v.strip()
               for k, v in choices.items()):
            raise ValueError("Choice ids and descriptions must be nonempty strings")
        if not isinstance(context, str):
            context = json.dumps(context, ensure_ascii=False, sort_keys=True)

        right = [f"Question: {question}\nCandidate: {text}" for text in choices.values()]
        if len(set(right)) == 1:
            raise ValueError("All candidates render to the same text")
        pairs = [(f"Context: {context}", text) for text in right]
        encoded = self.tokenizer.encode_batch(pairs)
        ids = np.array([item.ids for item in encoded], dtype=np.int64)
        mask = np.array([item.attention_mask for item in encoded], dtype=np.int64)

        pieces = []
        for start in range(0, len(ids), CHUNK):
            piece = self.session.run(["score"], {"input_ids": ids[start:start + CHUNK],
                                                 "attention_mask": mask[start:start + CHUNK]})[0]
            pieces.append(np.asarray(piece, dtype=np.float64).reshape(-1))
        scores = np.concatenate(pieces)
        scaled = scores / (self.temperature if temperature is None else float(temperature))
        weights = np.exp(scaled - scaled.max())
        probabilities = weights / weights.sum()

        names = list(choices)
        order = np.argsort(-probabilities)
        return {
            "choice": names[int(order[0])],
            "probabilities": {names[i]: float(probabilities[i]) for i in order},
            "scores": {names[i]: float(scores[i]) for i in order},
            "max_probability": float(probabilities.max()),
            "temperature": self.temperature if temperature is None else float(temperature),
            "tokens": int(ids.shape[1]),
        }


def from_directory(root, temperature=None):
    """Build a Scorer from a directory holding the ONNX, tokenizer and calibration.

    fp32 on purpose. Dynamic int8 quantization, per-tensor and per-channel alike,
    moved the scores by 5-8 and flipped MNLI's argmax; fp32 tracks PyTorch to 1e-4.
    """
    root = Path(root)
    if temperature is None:
        temperature = json.loads((root / "calibration.json").read_text())["temperature"]
    return Scorer(root / "model.onnx", root / "tokenizer.json", temperature)
