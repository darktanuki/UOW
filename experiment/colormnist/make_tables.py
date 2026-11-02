#!/usr/bin/env python3
"""Build Colored MNIST k-NN and linear-probing LaTeX tables."""

import argparse
import csv
import math
from pathlib import Path

from config import METHODS, SEEDS


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def mean_std(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(variance)


def metric_cell(rows: list[dict[str, str]], metric: str) -> str:
    if not rows:
        return "--"
    mean, std = mean_std([float(row[metric]) for row in rows])
    return rf"{mean:.1f} $\pm$ {std:.1f}\%"


def write_table(rows: list[dict[str, str]], eval_type: str, caption: str, label: str, output: Path) -> None:
    lines = [
        r"\begin{table}[t]",
        r"    \centering",
        rf"    \caption{{{caption}}}",
        rf"    \label{{{label}}}",
        r"    \begin{tabular}{llcc}",
        r"        \toprule",
        r"        \textbf{Method} & \textbf{Target Distribution} & \textbf{Target Acc. ($\uparrow$)} & \textbf{Bias Acc. ($\downarrow$)} \\",
        r"        \midrule",
    ]
    valid_seeds = {str(seed) for seed in SEEDS}
    for index, method in enumerate(METHODS):
        if index == 2:
            lines.append(r"        \midrule")
        selected = [
            row
            for row in rows
            if row.get("method") == method.key
            and row.get("eval_type") == eval_type
            and row.get("seed") in valid_seeds
        ]
        lines.append(
            rf"        {method.title} & {method.target_distribution} & "
            rf"{metric_cell(selected, 'target_acc')} & {metric_cell(selected, 'bias_acc')} \\"
        )
    lines.extend([r"        \bottomrule", r"    \end{tabular}", r"\end{table}", ""])
    output.write_text("\n".join(lines))


def build_tables(results_csv: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(results_csv)
    write_table(
        rows,
        "k-NN",
        r"Quantitative evaluation on Colored MNIST using a frozen $k$-NN classifier. Results are mean $\pm$ standard deviation over seeds 0, 1, and 2.",
        "tab:cmnist_knn",
        output_dir / "table_knn.tex",
    )
    write_table(
        rows,
        "LP",
        r"Quantitative evaluation on Colored MNIST using frozen representations and linear probing. Results are mean $\pm$ standard deviation over seeds 0, 1, and 2.",
        "tab:cmnist_linear_probe",
        output_dir / "table_linear_probe.tex",
    )
    print(f"Tables written to {output_dir}")


def main() -> None:
    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=directory / "results.csv")
    parser.add_argument("--output-dir", type=Path, default=directory)
    args = parser.parse_args()
    build_tables(args.results, args.output_dir)


if __name__ == "__main__":
    main()
