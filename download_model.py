"""Download only the pinned tokenizer, config, and safe model weights."""
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(".cache/huggingface").resolve()))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import snapshot_download

from choice_model import DEFAULT_MODEL, MODELS, model_dir

def fetch_backbone(model=DEFAULT_MODEL):
    """Put the pinned weights where model_dir expects them. Safe to call twice."""
    return snapshot_download(
        model,
        revision=MODELS[model],
        local_dir=str(model_dir(model)),
        allow_patterns=["config.json", "model.safetensors", "tokenizer.json",
                        "tokenizer_config.json", "special_tokens_map.json"],
    )


if __name__ == "__main__":
    for name in sys.argv[1:] or [DEFAULT_MODEL]:
        print(fetch_backbone(name))
