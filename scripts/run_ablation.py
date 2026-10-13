from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Run XFace ablation variants.")
    parser.add_argument("--config", default="configs/ablation.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    for name, settings in config["variants"].items():
        print(name, settings)


if __name__ == "__main__":
    main()
