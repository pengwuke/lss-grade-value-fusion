import argparse
import json
from collections import Counter

from .artifacts import checkpoint_path, read_fusion_config, read_manifest


def main():
    parser = argparse.ArgumentParser(description="Inspect or validate unpublished Paper2 model artifacts")
    parser.add_argument("command", choices=["inspect", "verify", "load-check"])
    parser.add_argument("--models", required=True, help="Extracted model_candidate directory")
    args = parser.parse_args()
    entries = read_manifest(args.models)
    report = {"entries": len(entries), "groups": dict(Counter(e["group"] for e in entries)),
              "unique_tensor_states": len({e["tensor_value_sha256"] for e in entries}),
              "model_manifest_sha256_pass": True,
              "patient_inference_performed": False,
              "synthetic_forward_performed": False}
    if args.command in {"verify", "load-check"}:
        read_fusion_config(args.models)
        report["fusion_config_sha256_pass"] = True
        for entry in entries:
            checkpoint_path(args.models, entry, verify=True)
        report["checkpoint_sha256_pass"] = len(entries)
    if args.command == "load-check":
        from .models import load_model
        count = 0
        for entry in entries:
            model, _ = load_model(args.models, entry["group"], entry["family"], entry["fold"])
            count += 1
            del model
        report["strict_architecture_load_pass"] = count
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
