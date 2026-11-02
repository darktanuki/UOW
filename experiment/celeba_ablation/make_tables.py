#!/usr/bin/env python3
"""Build EO and accuracy matrices for the CelebA lambda-alpha ablation."""

import argparse
import csv
import math
from pathlib import Path

from config import ALPHAS, LAMBDAS


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


def cell(rows: list[dict[str, str]], lambda_label: str, alpha_label: str, metric: str) -> str:
    selected = [
        row for row in rows
        if row.get("lambda_label") == lambda_label and row.get("alpha_label") == alpha_label
    ]
    if not selected:
        return "--"
    mean, std = mean_std([float(row[metric]) for row in selected])
    if len(selected) == 1:
        return f"{mean:.1f}"
    return rf"{mean:.1f} $\pm$ {std:.1f}"


def write_matrix(rows: list[dict[str, str]], metric: str, output: Path) -> None:
    if metric == "eo":
        metric_title = r"Equalized Odds (EO, $\downarrow$)"
        label = "tab:celeba_lambda_alpha_eo"
    else:
        metric_title = r"target accuracy (Acc., $\uparrow$)"
        label = "tab:celeba_lambda_alpha_accuracy"

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{CelebA ($T$: Attractive, $S$: Male) UOW ablation over $\lambda$ and $\alpha$. Each cell reports {metric_title} (\%) for seed 0.}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{rccccc}",
        r"\toprule",
        r"$\lambda\backslash\alpha$ & $0$ & $0.25$ & $0.5$ & $0.75$ & $1$ \\",
        r"\midrule",
    ]
    for lambda_label, _, lambda_latex in LAMBDAS:
        cells = [cell(rows, lambda_label, alpha_label, metric) for alpha_label, _ in ALPHAS]
        lines.append(rf"${lambda_latex}$ & {' & '.join(cells)} \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    output.write_text("\n".join(lines))


def build_tables(results_csv: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(results_csv)
    write_matrix(rows, "eo", output_dir / "table_eo.tex")
    write_matrix(rows, "acc", output_dir / "table_accuracy.tex")
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
