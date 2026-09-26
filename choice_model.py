"""ModernBERT candidate-conditioned scorer, with dependency-free LoRA."""
import json
import math
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

MODELS = {"answerdotai/ModernBERT-large": "45bb4654a4d5aaff24dd11d4781fa46d39bf8c13",
          "answerdotai/ModernBERT-base": "8949b909ec900327062f0ebf497f51aef5e6f0c8"}
DEFAULT_MODEL = "answerdotai/ModernBERT-large"


def model_dir(model=DEFAULT_MODEL):
    if model not in MODELS:
        raise ValueError(f"Unpinned model: {model}. Add its revision to MODELS first.")
    return Path("models") / model.rsplit("/", 1)[-1]


class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        self.A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=base.weight.dtype))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=base.weight.dtype))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale


class ChoiceScorer(nn.Module):
    def __init__(self, rank=8, alpha=16, device="cuda", model=DEFAULT_MODEL):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            model_dir(model), local_files_only=True, dtype=torch.float32,
            attn_implementation="sdpa",
        )
        self.encoder.requires_grad_(False)
        targets = [(name, module) for name, module in self.encoder.named_modules()
                   if isinstance(module, nn.Linear) and (name.endswith("attn.Wqkv") or name.endswith("attn.Wo"))]
        for name, module in targets:
            parent, attribute = name.rsplit(".", 1)
            setattr(self.encoder.get_submodule(parent), attribute, LoRALinear(module, rank, alpha))
        if not targets:
            raise RuntimeError("No ModernBERT attention projections found for LoRA")
        self.head = nn.Linear(self.encoder.config.hidden_size, 1)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)
        self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.to(device)

    def forward(self, tokens):
        hidden = self.encoder(**tokens).last_hidden_state
        mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        return self.head(pooled).squeeze(-1)

    def save_adapter(self, path):
        torch.save(self.adapter_state(), path)

    def adapter_state(self):
        keys = {name for name, p in self.named_parameters() if p.requires_grad}
        return {k: v.detach().cpu() for k, v in self.state_dict().items() if k in keys}

    def load_adapter(self, path):
        state = torch.load(path, map_location="cpu", weights_only=True)
        expected = {name for name, p in self.named_parameters() if p.requires_grad}
        if set(state) != expected:
            raise ValueError("Adapter keys do not match model configuration")
        self.load_state_dict(state, strict=False)


def tokenizer(model=DEFAULT_MODEL):
    return AutoTokenizer.from_pretrained(model_dir(model), local_files_only=True)


def encode(record, tok, max_length):
    context = record["context"]
    if not isinstance(context, str):
        context = json.dumps(context, ensure_ascii=False, sort_keys=True)
    # Put question and candidate in the protected second sequence. Only context
    # is truncated; reject oversized instructions rather than silently deleting them.
    right = [f"Question: {record['question']}\nCandidate: {description}" for description in record["choices"].values()]
    left = [f"Context: {context}"] * len(right)
    if len(set(right)) == 1:
        # Every candidate would get an identical input, so the scores are forced
        # uniform. Fail loudly instead of silently training against noise.
        raise ValueError(f"All candidates render to the same text: {record.get('id', '?')}")
    overhead = tok.num_special_tokens_to_add(pair=True)
    if any(len(tok(t, add_special_tokens=False)["input_ids"]) + overhead >= max_length for t in right):
        raise ValueError("Question/candidate exceeds max_length; increase the token budget")
    full_lengths = [len(tok(a, add_special_tokens=False)["input_ids"]) + len(tok(b, add_special_tokens=False)["input_ids"]) + overhead for a, b in zip(left, right)]
    tokens = tok(left, right, padding=True, truncation="only_first", max_length=max_length, return_tensors="pt")
    tokens.pop("token_type_ids", None)
    return tokens, sum(n > max_length for n in full_lengths)


def chunks(tokens, size, device):
    for start in range(0, len(tokens["input_ids"]), size):
        yield {k: v[start:start + size].to(device) for k, v in tokens.items()}


def validate_request(value):
    if not isinstance(value, dict) or not isinstance(value.get("question"), str) or not value["question"].strip():
        raise ValueError("question must be a nonempty string")
    if "context" not in value or not isinstance(value["context"], (str, dict, list)):
        raise ValueError("context must be text, an object, or an array")
    choices = value.get("choices")
    if not isinstance(choices, dict) or not 2 <= len(choices) <= 255:
        raise ValueError("choices must map 2 to 255 unique IDs to descriptions")
    if any(not isinstance(k, str) or not k or not isinstance(v, str) or not v.strip() for k, v in choices.items()):
        raise ValueError("Choice IDs and descriptions must be nonempty strings")
    return value
