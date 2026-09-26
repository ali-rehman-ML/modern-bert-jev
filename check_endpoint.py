"""Integration check: start the real local server, send JSON, then stop it."""
import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="runs/smoke")
    args = parser.parse_args()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    environment = os.environ | {"HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1"}
    log_path = Path(args.run) / "endpoint_test.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen([sys.executable, "-u", "predict.py", "--run", args.run,
                                    "--serve", "--port", str(port)], stdout=log, stderr=log,
                                   env=environment, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        try:
            deadline = time.monotonic() + 120
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"Server exited; inspect {log_path}")
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=1):
                        break
                except OSError:
                    if time.monotonic() > deadline:
                        raise TimeoutError("Server startup timed out")
                    time.sleep(0.5)
            def post(body):
                request = urllib.request.Request(f"http://127.0.0.1:{port}/predict",
                    data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(request, timeout=60) as response:
                    return json.load(response)
            body = json.loads(Path("example_request.json").read_text())
            response = post(body)
            assert set(response["probabilities"]) == set(body["choices"])
            assert abs(sum(response["probabilities"].values()) - 1) < 1e-5
            assert response["choice"] in body["choices"]
            reversed_body = body | {"choices": dict(reversed(list(body["choices"].items())))}
            reversed_response = post(reversed_body)
            assert all(abs(response["probabilities"][k] - reversed_response["probabilities"][k]) < 0.01 for k in body["choices"])
            try:
                post(body | {"choices": {"only": "One option"}})
            except urllib.error.HTTPError as error:
                assert error.code == 400
            else:
                raise AssertionError("Invalid request was accepted")
            result = {"passed": True, "checks": ["checkpoint reload", "valid JSON response", "probability normalization", "choice permutation", "invalid request returns 400"], "example_response": response}
            (Path(args.run) / "endpoint_test.json").write_text(json.dumps(result, indent=2))
            print(json.dumps(result, indent=2))
        finally:
            process.terminate()
            process.wait(timeout=15)


if __name__ == "__main__":
    main()
