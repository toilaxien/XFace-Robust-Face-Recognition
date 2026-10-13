from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a face recognition checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    print(f"Evaluation config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")


if __name__ == "__main__":
    main()
