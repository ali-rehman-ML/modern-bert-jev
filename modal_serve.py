"""Serverless scoring API on Modal, billed per second of actual work.

Deploy:
    python -m modal deploy modal_serve.py

Cost shape: `min_containers=0` means nothing runs, and nothing is billed, until a
request arrives. `scaledown_window=60` keeps the warm container for a minute so a
burst of requests shares one cold start, then it exits. There is no hourly rate and
no reserved capacity; an idle day costs nothing.

It is deliberately torch-free. The int8 ONNX plus the Rust tokenizer fit the
smallest container Modal sells, where a full PyTorch image would need several times
the memory and a far longer cold start, and be billed for every second of it.
"""
import modal

MODEL_REPO = "ali-rehman-ML/modern-bert-jev"
BACKBONE = "answerdotai/ModernBERT-base"
BACKBONE_REVISION = "8949b909ec900327062f0ebf497f51aef5e6f0c8"
ASSETS = "/model"

app = modal.App("modern-bert-jev-api")


def bake_assets():
    """Pull the weights at image-build time so a cold start is only a session load."""
    import shutil
    from pathlib import Path

    from huggingface_hub import hf_hub_download

    root = Path(ASSETS)
    root.mkdir(parents=True, exist_ok=True)
    shutil.copy(hf_hub_download(MODEL_REPO, "onnx/model.onnx"), root / "model.onnx")
    shutil.copy(hf_hub_download(MODEL_REPO, "calibration.json"), root / "calibration.json")
    shutil.copy(hf_hub_download(BACKBONE, "tokenizer.json", revision=BACKBONE_REVISION),
                root / "tokenizer.json")


image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("onnxruntime==1.23.2", "tokenizers==0.23.2", "numpy==2.3.5",
                 "fastapi[standard]==0.120.4", "huggingface_hub>=0.34,<2.0")
    .env({"HF_HUB_DISABLE_PROGRESS_BARS": "1", "HF_HUB_DISABLE_XET": "1",
          "OMP_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"})
    .run_function(bake_assets)
    .add_local_file("onnx_scorer.py", "/opt/project/onnx_scorer.py", copy=True)
)


@app.cls(image=image, cpu=2, memory=3072, scaledown_window=60,
         min_containers=0, max_containers=4, timeout=120)
@modal.concurrent(max_inputs=4)
class API:
    @modal.enter()
    def load(self):
        import sys

        sys.path.append("/opt/project")
        from onnx_scorer import from_directory

        self.scorer = from_directory(ASSETS)

    @modal.asgi_app()
    def web(self):
        from fastapi import FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware

        api = FastAPI(title="modern-bert-jev", docs_url="/docs")
        # The demo is a static page on a different origin, so it needs CORS. The
        # endpoint is public and read-only: it holds no data and no credentials.
        api.add_middleware(CORSMiddleware, allow_origins=["*"],
                           allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["*"])

        @api.get("/health")
        def health():
            return {"ok": True, "temperature": self.scorer.temperature}

        @api.post("/predict")
        def predict(payload: dict):
            try:
                return self.scorer(payload.get("context", ""), payload.get("question"),
                                   payload.get("choices"), payload.get("temperature"))
            except ValueError as error:
                raise HTTPException(status_code=400, detail=str(error))

        return api


@app.local_entrypoint()
def main():
    """Smoke the deployed shape locally: `python -m modal run modal_serve.py`."""
    print("Deploy with: python -m modal deploy modal_serve.py")
