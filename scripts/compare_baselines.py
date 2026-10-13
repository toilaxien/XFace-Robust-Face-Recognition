from __future__ import annotations

from pathlib import Path

import pandas as pd


def main() -> None:
    csv_path = Path("results/benchmark_results.csv")
    df = pd.read_csv(csv_path)
    print(df.groupby("Model")["Accuracy"].mean().sort_values(ascending=False))


if __name__ == "__main__":
    main()
