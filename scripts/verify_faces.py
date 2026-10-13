from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify whether two face images belong to the same identity.")
    parser.add_argument("--image-a", required=True)
    parser.add_argument("--image-b", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    print(f"Compare {args.image_a} and {args.image_b} using {args.checkpoint}")


if __name__ == "__main__":
    main()
