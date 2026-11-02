#%%
"""Plot mean +/- standard deviation pretraining dynamics across CelebA seeds."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent


def seed_files() -> list[Path]:
    """Return the latest Lightning metrics file for each completed seed."""
    selected = []
    for seed_dir in sorted((ROOT / "artifacts" / "runs").glob("seed_*")):
        candidates = sorted(
            (seed_dir / "training_logs").glob("version_*/metrics.csv"),
            key=lambda path: path.stat().st_mtime,
        )
        if candidates:
            selected.append(candidates[-1])
    return selected

# Viridis, matching the palette used in the preceding ablation figures.
COLORS = {
    "train/Linv": "#440154",
    "train/Lreg": "#440154",
    "val/Lreg": "#31688e",
    "val/EO": "#35b779",
    "val/Acc": "#fde725",
}


def epoch_means(path: Path) -> dict[str, dict[int, float]]:
    raw: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    metrics = ("train/Linv", "train/Lreg", "val/Lreg", "val/EO", "val/Acc")
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            epoch = int(row["epoch"])
            for metric in metrics:
                if row[metric]:
                    value = float(row[metric])
                    if metric in {"val/EO", "val/Acc"}:
                        value *= 100.0
                    raw[metric][epoch].append(value)
    return {
        metric: {epoch: float(np.mean(values)) for epoch, values in values_by_epoch.items()}
        for metric, values_by_epoch in raw.items()
    }


def aggregate(runs: list[dict[str, dict[int, float]]], metric: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shared_epochs = sorted(set.intersection(*(set(run[metric]) for run in runs)))
    values = np.asarray([[run[metric][epoch] for epoch in shared_epochs] for run in runs])
    return np.asarray(shared_epochs), values.mean(axis=0), values.std(axis=0, ddof=1)


def draw(axis, x, mean, std, label, color):
    line = axis.plot(x, mean, color=color, linewidth=1.9, label=label)[0]
    axis.fill_between(x, mean - std, mean + std, color=color, alpha=0.16, linewidth=0)
    return line


def main() -> None:
    metrics_files = seed_files()
    if len(metrics_files) < 2:
        raise RuntimeError("At least two seed metrics files are required.")
    runs = [epoch_means(path) for path in metrics_files]

    fig, left = plt.subplots(figsize=(8.6, 4.5))
    right = left.twinx()

    handles = []
    for metric, label in (("val/Acc", "Validation accuracy"), ("val/EO", "Validation EO")):
        x, mean, std = aggregate(runs, metric)
        handles.append(draw(left, x, mean, std, label, COLORS[metric]))
    for metric, label in (("train/Lreg", r"Train $L_{\mathrm{reg}}$"), ("val/Lreg", r"Validation $L_{\mathrm{reg}}$")):
        x, mean, std = aggregate(runs, metric)
        handles.append(draw(right, x, mean, std, label, COLORS[metric]))

    left.set_xlabel("Epoch")
    left.set_ylabel("Validation metrics (%)")
    left.set_ylim(0, 100)
    right.set_ylabel(r"$L_{\mathrm{reg}}$")
    right.set_yscale("log")
    left.set_xlim(0, max(aggregate(runs, "val/Acc")[0]))
    left.grid(axis="both", alpha=0.2, linewidth=0.7)
    left.set_title(r"CelebA scratch UOW ($\alpha=0.5$): pretraining dynamics", pad=10)
    left.legend(handles, [handle.get_label() for handle in handles], frameon=False, ncol=2, loc="upper center")
    fig.tight_layout()

    for suffix in ("png", "pdf"):
        fig.savefig(ROOT / f"celeba_scratch_uow_alpha0p5_pretraining_dynamics.{suffix}", dpi=240, bbox_inches="tight")

    fig, left = plt.subplots(figsize=(8.6, 4.5))
    right = left.twinx()
    handles = []
    for metric, label in (("val/Acc", "Validation accuracy"), ("val/EO", "Validation EO")):
        x, mean, std = aggregate(runs, metric)
        handles.append(draw(left, x, mean, std, label, COLORS[metric]))
    x, mean, std = aggregate(runs, "train/Linv")
    handles.append(draw(right, x, mean, std, r"Train $L_{\mathrm{inv}}$", COLORS["train/Linv"]))

    left.set_xlabel("Epoch")
    left.set_ylabel("Validation metrics (%)")
    left.set_ylim(0, 100)
    right.set_ylabel(r"$L_{\mathrm{inv}}$")
    left.set_xlim(0, max(aggregate(runs, "val/Acc")[0]))
    left.grid(axis="both", alpha=0.2, linewidth=0.7)
    left.set_title(r"CelebA scratch UOW ($\alpha=0.5$): invariance dynamics", pad=10)
    left.legend(handles, [handle.get_label() for handle in handles], frameon=False, ncol=3, loc="upper center")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(ROOT / f"celeba_scratch_uow_alpha0p5_invariance_dynamics.{suffix}", dpi=240, bbox_inches="tight")

    fig, left = plt.subplots(figsize=(8.6, 4.5))
    right = left.twinx()
    handles = []
    for metric, label in (("val/Acc", "Validation accuracy"), ("val/EO", "Validation EO")):
        x, mean, std = aggregate(runs, metric)
        handles.append(draw(left, x, mean, std, label, COLORS[metric]))
    x, mean, std = aggregate(runs, "train/Lreg")
    handles.append(draw(right, x, mean, std, r"Train $L_{\mathrm{reg}}$", COLORS["train/Lreg"]))

    left.set_xlabel("Epoch")
    left.set_ylabel("Validation metrics (%)")
    left.set_ylim(0, 100)
    right.set_ylabel(r"Train $L_{\mathrm{reg}}$")
    left.set_xlim(0, max(aggregate(runs, "val/Acc")[0]))
    left.grid(axis="both", alpha=0.2, linewidth=0.7)
    #left.set_title(r"CelebA scratch UOW ($\alpha=0.5$): accuracy, EO, and regularization", pad=10)
    left.legend(handles, [handle.get_label() for handle in handles], frameon=False, ncol=3, loc="upper center")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(ROOT / f"celeba_scratch_uow_alpha0p5_summary_dynamics.{suffix}", dpi=240, bbox_inches="tight")

    # Paper figure: emphasize the transition where encoder-level fairness
    # deteriorates despite continued optimization of the projected features.
    fig, left = plt.subplots(figsize=(8.6, 4.5))
    right = left.twinx()
    handles = []
    for metric, label in (("val/Acc", "Validation accuracy"), ("val/EO", "Validation EO")):
        x, mean, std = aggregate(runs, metric)
        handles.append(draw(left, x, mean, std, label, COLORS[metric]))
    x, mean, std = aggregate(runs, "train/Lreg")
    handles.append(draw(right, x, mean, std, r"Train $L_{\mathrm{reg}}$", COLORS["train/Lreg"]))

    transition_epoch = 75
    left.axvspan(0, transition_epoch, color="#31688e", alpha=0.06)
    left.axvspan(transition_epoch, max(x), color="#fde725", alpha=0.06)
    left.axvline(transition_epoch, color="0.35", linestyle="--", linewidth=1.1)

    left.set_xlabel("Epoch")
    left.set_ylabel("Validation metrics (%)")
    left.set_ylim(0, 100)
    right.set_ylabel(r"Train $L_{\mathrm{reg}}$")
    right.set_xlim(0, max(x))
    left.grid(axis="both", alpha=0.2, linewidth=0.7)
    left.legend(handles, [handle.get_label() for handle in handles], frameon=False,
                ncol=3, loc="upper center")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(ROOT / f"celeba_scratch_uow_alpha0p5_early_stopping_dynamics.{suffix}",
                    dpi=240, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

# %%
