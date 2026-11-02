#!/usr/bin/env python3
"""Build the paper-style LaTeX table from results.csv."""

import argparse
import csv
import math
from pathlib import Path

from config import METHODS, SECTION_TITLES, TASKS


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


def format_cell(rows: list[dict[str, str]], metric: str) -> str:
    if not rows:
        return "--"
    mean, std = mean_std([float(row[metric]) for row in rows])
    return rf"{mean:.1f} $\pm$ {std:.1f}"


def build_table(results_csv: Path, output_tex: Path) -> None:
    rows = read_rows(results_csv)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Real-world classification results (mean $\pm$ standard deviation over seeds).}",
        r"\label{tab:realworld}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"\multirow{3}{*}{Method} & \multicolumn{4}{c}{CelebA} & \multicolumn{2}{c}{Waterbirds} \\",
        r"\cmidrule(lr){2-5}\cmidrule(lr){6-7}",
        r"& \multicolumn{2}{c}{$T$: Attractive / $S$: Male} & \multicolumn{2}{c}{$T$: Big Nose / $S$: Male} & \multicolumn{2}{c}{$T$: Bird type / $S$: Background} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r"& EO ($\downarrow$) & Acc. ($\uparrow$) & EO ($\downarrow$) & Acc. ($\uparrow$) & EO ($\downarrow$) & Acc. ($\uparrow$) \\",
    ]

    current_section = None
    for method in METHODS:
        if method.section != current_section:
            lines.extend(
                [
                    r"\midrule",
                    rf"\multicolumn{{7}}{{l}}{{\textit{{{SECTION_TITLES[method.section]}}}}} \\",
                    r"\midrule",
                ]
            )
            current_section = method.section

        cells = []
        for task in TASKS:
            selected = [
                row
                for row in rows
                if row.get("task") == task.key and row.get("method") == method.key
            ]
            cells.extend((format_cell(selected, "eo"), format_cell(selected, "acc")))
        lines.append(rf"{method.title} & {' & '.join(cells)} \\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    output_tex.parent.mkdir(parents=True, exist_ok=True)
    output_tex.write_text("\n".join(lines))
    print(f"Table written to {output_tex}")


def main() -> None:
    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=directory / "results.csv")
    parser.add_argument("--output", type=Path, default=directory / "table.tex")
    args = parser.parse_args()
    build_table(args.results, args.output)


if __name__ == "__main__":
    main()
