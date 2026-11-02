#!/usr/bin/env python3
"""Build the intersectional CelebA LaTeX table from results.csv."""

import argparse
import csv
import math
from pathlib import Path

from config import METHODS, SEEDS


METRICS = ("target_acc", "eo_male", "eo_young", "eo_joint")


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


def summaries(rows: list[dict[str, str]]) -> dict[str, dict[str, tuple[float, float]]]:
    output = {}
    valid_seeds = {str(seed) for seed in SEEDS}
    for method in METHODS:
        selected = [row for row in rows if row.get("method_key") == method.key and row.get("seed") in valid_seeds]
        if selected:
            output[method.key] = {
                metric: mean_std([float(row[metric]) for row in selected])
                for metric in METRICS
            }
    return output


def format_value(value: tuple[float, float] | None, bold: bool) -> str:
    if value is None:
        return "--"
    text = rf"{value[0]:.1f} $\pm$ {value[1]:.1f}"
    return rf"\textbf{{{text}}}" if bold else text


def build_table(results_csv: Path, output: Path) -> None:
    summary = summaries(read_rows(results_csv))
    best = {}
    for metric in METRICS:
        candidates = [(key, values[metric][0]) for key, values in summary.items()]
        if candidates:
            chooser = max if metric == "target_acc" else min
            best[metric] = chooser(candidates, key=lambda item: item[1])[0]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Intersectional CelebA results with \emph{Attractive} as the target and \emph{Male} and \emph{Young} as observed biases. Separate conditioning uses one UOWReg term per attribute, whereas joint conditioning treats $(\text{Male},\text{Young})$ as a four-valued bias variable. Results are means $\pm$ standard deviations over three seeds.}",
        r"\label{tab:celeba_intersectional}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Method & Bias conditioning & Acc. ($\uparrow$) & EO Male ($\downarrow$) & EO Young ($\downarrow$) & EO Joint ($\downarrow$) \\",
        r"\midrule",
    ]
    for index, method in enumerate(METHODS):
        values = summary.get(method.key, {})
        method_cell = method.method if index == 0 else (r"\multirow{3}{*}{UOW}" if index == 1 else "")
        cells = [
            format_value(values.get(metric), best.get(metric) == method.key)
            for metric in METRICS
        ]
        lines.append(rf"{method_cell} & {method.conditioning} & {' & '.join(cells)} \\")
        if index == 0:
            lines.append(r"\midrule")
        elif index in {1, 2}:
            lines.append("")
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}", ""])
    output.write_text("\n".join(lines))
    print(f"Table written to {output}")


def main() -> None:
    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=directory / "results.csv")
    parser.add_argument("--output", type=Path, default=directory / "table.tex")
    args = parser.parse_args()
    build_table(args.results, args.output)


if __name__ == "__main__":
    main()
