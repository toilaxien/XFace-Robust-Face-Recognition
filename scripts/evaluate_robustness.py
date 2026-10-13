from __future__ import annotations

import argparse


CONDITIONS = ["blur", "noise", "low_resolution", "brightness", "occlusion"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate robustness under controlled degradations.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    print(f"Robustness evaluation for {args.checkpoint}")
    print("Conditions:", ", ".join(CONDITIONS))


if __name__ == "__main__":
    main()
