"""JSON CLI or localhost HTTP endpoint for a trained experiment adapter."""
import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import torch

from choice_model import ChoiceScorer, DEFAULT_MODEL, chunks, encode, tokenizer, validate_request
from experiment import precision


class Predictor:
    def __init__(self, run):
        run = Path(run)
        self.config = json.loads((run / "config.json").read_text())
        self.temperature = json.loads((run / "calibration.json").read_text())["temperature"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        backbone = self.config.get("model", DEFAULT_MODEL)
        self.tok = tokenizer(backbone)
        self.model = ChoiceScorer(self.config["lora_rank"], self.config["lora_alpha"], self.device, backbone)
        self.model.load_adapter(run / "adapter.pt")
        self.model.eval()

    def __call__(self, request):
        request = validate_request(request)
        if self.config.get("max_choices") and len(request["choices"]) > self.config["max_choices"]:
            raise ValueError(f"This run accepts at most {self.config['max_choices']} choices")
        tokens, truncated = encode(request, self.tok, self.config["max_length"])
        with torch.inference_mode(), precision(self.device):
            logits = torch.cat([self.model(batch) for batch in chunks(tokens, self.config["choice_chunk"], self.device)])
        p = (logits.float() / self.temperature).softmax(0).cpu().tolist()
        ids = list(request["choices"])
        return {"choice": ids[max(range(len(p)), key=p.__getitem__)],
                "probabilities": dict(zip(ids, p)), "max_probability": max(p),
                "context_truncated": bool(truncated), "temperature": self.temperature}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="runs/pilot")
    parser.add_argument("--input", help="JSON request file")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    predictor = Predictor(args.run)
    if not args.serve:
        if not args.input:
            parser.error("Supply --input or --serve")
        print(json.dumps(predictor(json.loads(Path(args.input).read_text(encoding="utf-8"))), indent=2))
        return

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path != "/predict":
                self.send_error(404)
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 1_000_000:
                    raise ValueError("Request must be between 1 byte and 1 MB")
                body = predictor(json.loads(self.rfile.read(size)))
                status = 200
            except (ValueError, TypeError, KeyError) as error:
                body, status = {"error": str(error)}, 400
            payload = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    print(f"Listening on http://127.0.0.1:{args.port}/predict", flush=True)
    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
