from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an XFace or baseline model.")
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    name = config.get("experiment", {}).get("name", Path(args.config).stem)
    print(f"Loaded training config: {name}")
    print("This script is a reproducible entry point; connect the local dataset before launching full training.")


if __name__ == "__main__":
    main()
