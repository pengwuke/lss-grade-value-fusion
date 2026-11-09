import argparse
import importlib.metadata
import json
import platform
from pathlib import Path

from .runner import run


def main():
    parser = argparse.ArgumentParser(description="Unpublished matched-statistics candidate; explicit CSV inputs only")
    parser.add_argument("--analysis", choices=["single", "fixed", "selected", "r11_natural", "r11_robustness"], required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=None,
                        help="Defaults: 100,000 for r11_robustness; 20,000 for the four earlier plans; smaller values are smoke checks")
    args = parser.parse_args()
    if args.iterations is None:
        args.iterations = 100000 if args.analysis == "r11_robustness" else 20000
    args.output = args.output.expanduser()
    if args.output.exists():
        parser.error("output already exists; select a new report path")
    if args.iterations < 1:
        parser.error("iterations must be positive")
    report = run(args.inputs, args.analysis, iterations=args.iterations)
    report["execution_environment"] = {"python": platform.python_version(),
                                       **{x: importlib.metadata.version(x) for x in ["numpy", "scikit-learn"]}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({"report_written": True, "analysis": args.analysis,
                      "paper_resampling_configuration": report["paper_resampling_configuration"],
                      "iterations": args.iterations}))


if __name__ == "__main__":
    main()
