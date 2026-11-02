#!/usr/bin/env python3
"""Plot the CelebA accuracy--EO Pareto frontier for the lambda/alpha grid."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
REQUIRED_COLUMNS = {"lambda_reg", "alpha", "seed", "acc", "eo"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=HERE / "results.csv",
        help="Ablation results CSV (default: experiment/celeba_ablation/results.csv).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "celeba_legacy_alpha_pareto.pdf",
        help="Output PDF path. A PNG preview is written beside it.",
    )
    parser.add_argument(
        "--annotate-pareto",
        action="store_true",
        help="Annotate Pareto-optimal points with (lambda, alpha).",
    )
    return parser.parse_args()


def read_results(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing columns {sorted(missing)} in {path}")
        rows = [
            {
                "lambda_reg": float(row["lambda_reg"]),
                "alpha": float(row["alpha"]),
                "seed": float(row["seed"]),
                "acc": float(row["acc"]),
                "eo": float(row["eo"]),
            }
            for row in reader
        ]
    if not rows:
        raise ValueError(f"No results found in {path}")
    return rows


def aggregate(rows: list[dict[str, float]]) -> list[dict[str, float]]:
    """Average repeated seeds while retaining seed-0-only grids unchanged."""
    groups: dict[tuple[float, float], list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        groups[(row["lambda_reg"], row["alpha"])].append(row)

    points = []
    for (lambda_reg, alpha), group in sorted(groups.items()):
        accuracies = np.asarray([row["acc"] for row in group], dtype=float)
        equalized_odds = np.asarray([row["eo"] for row in group], dtype=float)
        points.append(
            {
                "lambda_reg": lambda_reg,
                "alpha": alpha,
                "acc": float(accuracies.mean()),
                "eo": float(equalized_odds.mean()),
                "acc_std": float(accuracies.std(ddof=1)) if len(group) > 1 else 0.0,
                "eo_std": float(equalized_odds.std(ddof=1)) if len(group) > 1 else 0.0,
                "n": float(len(group)),
            }
        )
    return points


def pareto_mask(acc: np.ndarray, eo: np.ndarray) -> np.ndarray:
    """Return non-dominated points when accuracy is maximized and EO minimized."""
    return np.asarray(
        [
            not np.any(
                (acc >= acc_i)
                & (eo <= eo_i)
                & ((acc > acc_i) | (eo < eo_i))
            )
            for acc_i, eo_i in zip(acc, eo)
        ]
    )


def compact(value: float) -> str:
    return f"{value:g}"


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if output_path.suffix.lower() != ".pdf":
        raise ValueError("--output must have a .pdf extension")

    points = aggregate(read_results(input_path))
    lambdas = np.asarray(sorted({point["lambda_reg"] for point in points}))
    alphas = np.asarray(sorted({point["alpha"] for point in points}))
    expected = len(lambdas) * len(alphas)
    if len(points) != expected:
        print(f"Warning: plotting {len(points)}/{expected} lambda-alpha configurations.")

    markers = ("o", "s", "^", "D", "P")
    if len(alphas) > len(markers):
        raise ValueError("Not enough marker styles for the alpha values in the input.")
    colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, len(lambdas)))
    color_by_lambda = dict(zip(lambdas, colors))
    marker_by_alpha = dict(zip(alphas, markers))

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axis = plt.subplots(figsize=(5.4, 3.5), constrained_layout=True)

    for point in points:
        kwargs = {
            "marker": marker_by_alpha[point["alpha"]],
            "ms": 6.5,
            "color": color_by_lambda[point["lambda_reg"]],
            "mec": "white",
            "mew": 0.8,
            "linestyle": "",
            "zorder": 3,
        }
        if point["n"] > 1:
            axis.errorbar(
                point["acc"],
                point["eo"],
                xerr=point["acc_std"],
                yerr=point["eo_std"],
                elinewidth=0.75,
                capsize=2,
                capthick=0.75,
                **kwargs,
            )
        else:
            axis.plot(point["acc"], point["eo"], **kwargs)

    accuracies = np.asarray([point["acc"] for point in points])
    equalized_odds = np.asarray([point["eo"] for point in points])
    frontier = [point for point, keep in zip(points, pareto_mask(accuracies, equalized_odds)) if keep]
    frontier.sort(key=lambda point: point["acc"])
    axis.plot(
        [point["acc"] for point in frontier],
        [point["eo"] for point in frontier],
        color="0.25",
        linewidth=0.9,
        zorder=1,
    )
    axis.scatter(
        [point["acc"] for point in frontier],
        [point["eo"] for point in frontier],
        s=62,
        facecolors="none",
        edgecolors="0.15",
        linewidths=0.8,
        zorder=4,
    )

    if args.annotate_pareto:
        offsets = ((5, 5), (5, -11), (-48, 5), (-48, -11))
        for index, point in enumerate(frontier):
            dx, dy = offsets[index % len(offsets)]
            axis.annotate(
                rf"$({compact(point['lambda_reg'])},\,{compact(point['alpha'])})$",
                (point["acc"], point["eo"]),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=7,
                color="0.15",
            )

    lambda_handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=6,
            markerfacecolor=color,
            markeredgecolor="white",
            label=compact(value),
        )
        for value, color in zip(lambdas, colors)
    ]
    alpha_handles = [
        plt.Line2D(
            [],
            [],
            marker=marker,
            linestyle="",
            markersize=6,
            markerfacecolor="0.55",
            markeredgecolor="white",
            label=rf"$\alpha={compact(value)}$",
        )
        for value, marker in zip(alphas, markers)
    ]
    lambda_legend = axis.legend(
        handles=lambda_handles,
        title=r"$\lambda$",
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0,
        handletextpad=0.35,
    )
    axis.add_artist(lambda_legend)
    axis.legend(
        handles=alpha_handles,
        title="mixing weight",
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.01, 0.48),
        borderaxespad=0,
        handletextpad=0.35,
    )

    axis.set_xlabel("Attractive accuracy (%)")
    axis.set_ylabel("Equalized Odds (%)")
    axis.grid(color="0.9", linewidth=0.6)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.margins(x=0.07, y=0.10)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    preview_path = output_path.with_suffix(".png")
    figure.savefig(output_path, bbox_inches="tight")
    figure.savefig(preview_path, dpi=240, bbox_inches="tight")
    plt.close(figure)
    print(f"Plotted {len(points)} configurations from {input_path}")
    print(f"Pareto frontier contains {len(frontier)} configurations")
    print(f"Saved {output_path}")
    print(f"Saved {preview_path}")


if __name__ == "__main__":
    main()
