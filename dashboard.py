"""Import previous run logs and serve TensorBoard on localhost."""
import argparse
from pathlib import Path

from tracking import enable_local_dependencies, import_run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--port", type=int, default=6006)
    parser.add_argument("--sync-only", action="store_true")
    args = parser.parse_args()
    enable_local_dependencies()
    for config in sorted(Path(args.runs).glob("*/config.json")):
        print(f"{config.parent.name}: {import_run(config.parent)}", flush=True)
    if args.sync_only:
        return
    from tensorboard import program
    tb = program.TensorBoard()
    tb.configure(argv=["tensorboard", "--logdir", str(Path(args.runs) / "tensorboard"),
                       "--host", "127.0.0.1", "--port", str(args.port), "--reload_interval", "5"])
    tb.main()


if __name__ == "__main__":
    main()
