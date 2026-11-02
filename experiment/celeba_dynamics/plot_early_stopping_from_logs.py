#!/usr/bin/env python3
"""Plot CelebA early-stopping dynamics from multi-seed Lightning logs."""

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
DEFAULT_STEM = "celeba_scratch_uow_alpha0_early_stopping_dynamics"
METRICS = ("val/Acc", "val/EO", "train/Lreg")
COLORS = {
    "val/Acc": "#fde725",
    "val/EO": "#35b779",
    "train/Lreg": "#440154",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-dir",
        type=Path,
        required=True,
        help="Directory containing seed_*/training_logs/version_*/metrics.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE,
        help=f"Output directory (default: {HERE}).",
    )
    parser.add_argument(
        "--output-stem",
        default=DEFAULT_STEM,
        help=f"Output filename without extension (default: {DEFAULT_STEM}).",
    )
    parser.add_argument(
        "--transition-epoch",
        type=int,
        default=75,
        help="Epoch highlighted as the early-stopping transition (default: 75).",
    )
    return parser.parse_args()


def completed_epochs(path: Path) -> int:
    """Return the number of distinct epochs present in a Lightning CSV."""
    with path.open(newline="", encoding="utf-8") as handle:
        return len({row["epoch"] for row in csv.DictReader(handle) if row.get("epoch")})


def discover_metrics_files(runs_dir: Path) -> list[Path]:
    """Select the most complete metrics file for every seed directory."""
    selected: list[Path] = []
    for seed_dir in sorted(runs_dir.glob("seed_*")):
        candidates = list((seed_dir / "training_logs").glob("version_*/metrics.csv"))
        if not candidates:
            continue
        selected.append(max(candidates, key=lambda path: (completed_epochs(path), path.stat().st_mtime)))
    if len(selected) < 2:
        raise RuntimeError(
            f"Expected at least two seed logs below {runs_dir}, found {len(selected)}."
        )
    return selected


def read_epoch_means(path: Path) -> dict[str, dict[int, float]]:
    """Aggregate batch-level Lightning metrics into one value per epoch."""
    raw: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = set(METRICS).difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing columns {sorted(missing)} in {path}")
        for row in reader:
            if not row.get("epoch"):
                continue
            epoch = int(row["epoch"])
            for metric in METRICS:
                if row.get(metric):
                    value = float(row[metric])
                    if metric in {"val/Acc", "val/EO"}:
                        value *= 100.0
                    raw[metric][epoch].append(value)

    epoch_means = {
        metric: {
            epoch: float(np.mean(values))
            for epoch, values in values_by_epoch.items()
        }
        for metric, values_by_epoch in raw.items()
    }
    missing = [metric for metric in METRICS if not epoch_means.get(metric)]
    if missing:
        raise ValueError(f"No values for {missing} in {path}")
    return epoch_means


def aggregate(
    runs: list[dict[str, dict[int, float]]], metric: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shared_epochs = sorted(set.intersection(*(set(run[metric]) for run in runs)))
    if not shared_epochs:
        raise ValueError(f"No shared epochs for {metric}")
    values = np.asarray(
        [[run[metric][epoch] for epoch in shared_epochs] for run in runs],
        dtype=float,
    )
    return (
        np.asarray(shared_epochs),
        values.mean(axis=0),
        values.std(axis=0, ddof=1),
    )


def draw(
    axis: plt.Axes,
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    label: str,
    color: str,
) -> plt.Line2D:
    line = axis.plot(x, mean, color=color, linewidth=1.9, label=label)[0]
    axis.fill_between(x, mean - std, mean + std, color=color, alpha=0.16, linewidth=0)
    return line


def main() -> None:
    args = parse_args()
    runs_dir = args.runs_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    metrics_files = discover_metrics_files(runs_dir)
    runs = [read_epoch_means(path) for path in metrics_files]

    aggregated = {metric: aggregate(runs, metric) for metric in METRICS}
    max_epoch = int(min(values[0][-1] for values in aggregated.values()))
    transition_epoch = args.transition_epoch
    if not 0 <= transition_epoch <= max_epoch:
        raise ValueError(
            f"--transition-epoch must be between 0 and {max_epoch}, got {transition_epoch}."
        )

    figure, left = plt.subplots(figsize=(8.6, 4.5))
    right = left.twinx()
    handles = []
    for metric, label in (
        ("val/Acc", "Validation accuracy"),
        ("val/EO", "Validation EO"),
    ):
        handles.append(draw(left, *aggregated[metric], label, COLORS[metric]))
    handles.append(
        draw(right, *aggregated["train/Lreg"], r"Train $L_{\mathrm{reg}}$", COLORS["train/Lreg"])
    )

    left.axvspan(0, transition_epoch, color="#31688e", alpha=0.06)
    left.axvspan(transition_epoch, max_epoch, color="#fde725", alpha=0.06)
    left.axvline(transition_epoch, color="0.35", linestyle="--", linewidth=1.1)

    left.set_xlabel("Epoch")
    left.set_ylabel("Validation metrics (%)")
    left.set_ylim(0, 100)
    left.set_xlim(0, max_epoch)
    right.set_ylabel(r"Train $L_{\mathrm{reg}}$")
    left.grid(axis="both", alpha=0.2, linewidth=0.7)
    left.legend(
        handles,
        [handle.get_label() for handle in handles],
        frameon=False,
        ncol=3,
        loc="upper center",
    )
    figure.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix in ("pdf", "png"):
        output = output_dir / f"{args.output_stem}.{suffix}"
        figure.savefig(output, dpi=240, bbox_inches="tight")
        outputs.append(output)
    plt.close(figure)

    print(f"Aggregated {len(metrics_files)} runs:")
    for path in metrics_files:
        print(f"  - {path}")
    print("Created:")
    for output in outputs:
        print(f"  - {output}")


if __name__ == "__main__":
    main()
