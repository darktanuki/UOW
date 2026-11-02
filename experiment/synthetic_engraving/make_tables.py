#!/usr/bin/env python3
"""Generate both Synthetic Engraving LaTeX tables from results.csv."""

import argparse
import csv
import math
from pathlib import Path

from config import ALPHAS, FIXED_ALPHA_LABEL, FIXED_LAMBDA_LABEL, LAMBDAS, M_VALUES, SEEDS


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def matching(rows: list[dict[str, str]], **keys: object) -> list[dict[str, str]]:
    return [row for row in rows if all(row.get(key) == str(value) for key, value in keys.items())]


def mean_std(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(variance)


def write_ablation_table(rows: list[dict[str, str]], output: Path) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Synthetic Engraving ablation over the regularization strength $\lambda$ and global-anchor weight $\alpha$ in the $m=2$ regime. Each cell reports the best validation $\mathrm{mAP}_{|c}$ (\%) obtained using seed 0.}",
        r"\label{tab:engraving_lambda_alpha_ablation}",
        r"\begin{tabular}{rccccc}",
        r"\toprule",
        r"$\lambda\backslash\alpha$ & $0$ & $0.25$ & $0.5$ & $0.75$ & $1$ \\",
        r"\midrule",
    ]
    for lambda_label, _ in LAMBDAS:
        cells = []
        for alpha_label, _ in ALPHAS:
            found = matching(
                rows,
                loss_name="uow",
                nb_signs=2,
                lambda_label=lambda_label,
                alpha_label=alpha_label,
                seed=0,
            )
            cells.append("--" if not found else f"{float(found[-1]['best_val_map']):.2f}")
        lines.append(rf"${lambda_label}$ & {' & '.join(cells)} \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    output.write_text("\n".join(lines))


def comparison_cell(rows: list[dict[str, str]], loss_name: str, nb_signs: int) -> str:
    selected = [
        row
        for row in rows
        if row.get("loss_name") == loss_name
        and row.get("nb_signs") == str(nb_signs)
        and row.get("lambda_label") == FIXED_LAMBDA_LABEL
        and row.get("alpha_label") == (FIXED_ALPHA_LABEL if loss_name == "uow" else "")
        and row.get("seed") in {str(seed) for seed in SEEDS}
    ]
    if not selected:
        return "--"
    mean, std = mean_std([float(row["best_val_map"]) for row in selected])
    return rf"{mean:.2f} $\pm$ {std:.2f}"


def write_comparison_table(rows: list[dict[str, str]], output: Path) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Retrieval $\mathrm{mAP}_{|c}$ on the Synthetic Engraving challenge averaged over seeds 0, 1, and 2.}",
        r"\label{tab:uow_ow_map_comparison}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Signatures per code ($m$) & OW ($\mathrm{mAP}_{|c}$ \%) & UOW ($\mathrm{mAP}_{|c}$ \%) \\",
        r"\midrule",
    ]
    for nb_signs in M_VALUES:
        lines.append(
            rf"{nb_signs} & {comparison_cell(rows, 'ow', nb_signs)} & "
            rf"{comparison_cell(rows, 'uow', nb_signs)} \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    output.write_text("\n".join(lines))


def build_tables(results_csv: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(results_csv)
    write_ablation_table(rows, output_dir / "table_ablation.tex")
    write_comparison_table(rows, output_dir / "table_comparison.tex")
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
