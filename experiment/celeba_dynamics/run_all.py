#!/usr/bin/env python3
"""Train all seeds and reproduce the CelebA early-stopping dynamics figure."""

import argparse
import csv
import shlex
import subprocess
import sys
import time
from pathlib import Path

from config import (
    ALPHA, BATCH_SIZE, EPOCHS, LAMBDA_REG, LEARNING_RATE, NUM_WORKERS,
    PROJECTION_DIM, ROOT_DATA, SEEDS, TEMPERATURE, WEIGHT_DECAY,
)


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))
from experiment.checkpoint_utils import checkpoint_complete
ARTIFACTS = HERE / "artifacts"
RESULTS_CSV = HERE / "results.csv"
FIELDS = (
    "seed", "alpha", "lambda_reg", "final_val_acc", "final_val_eo", "metrics_csv",
    "validation_history_csv", "checkpoint", "elapsed_seconds",
)


def plot_dynamics() -> None:
    from plot import main

    main()


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def latest_metrics(run_dir: Path) -> Path:
    files = sorted(
        (run_dir / "training_logs").glob("version_*/metrics.csv"),
        key=lambda path: path.stat().st_mtime,
    )
    if not files:
        raise FileNotFoundError(f"No metrics.csv below {run_dir / 'training_logs'}")
    return files[-1]


def store_row(row: dict[str, object]) -> None:
    rows = [existing for existing in read_rows(RESULTS_CSV) if existing["seed"] != str(row["seed"])]
    rows.append(row)
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    temporary = RESULTS_CSV.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(RESULTS_CSV)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="0")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only-plot", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS)
    args = parser.parse_args()

    if args.only_plot:
        plot_dynamics()
        return

    planned = list(SEEDS if args.seeds is None else args.seeds)
    completed = {
        int(row["seed"])
        for row in read_rows(RESULTS_CSV)
        if row.get("alpha") == str(ALPHA)
        and row.get("lambda_reg") == str(LAMBDA_REG)
    }
    total = len(planned)
    finished = 0
    for position, seed in enumerate(planned, start=1):
        if args.skip_existing and seed in completed:
            finished += 1
            print(f"= [{position}/{total} | terminées: {finished} | restantes: {total-finished}] déjà faite: seed {seed}")
            continue

        run_dir = ARTIFACTS / "runs" / f"seed_{seed}"
        checkpoint_path = run_dir / "checkpoints" / "last.ckpt"
        history_path = run_dir / "validation_history.csv"
        history = read_rows(history_path)
        command = [
            sys.executable, str(HERE / "train.py"),
            "--root_data", ROOT_DATA, "--output_dir", str(run_dir),
            "--epochs", str(EPOCHS), "--batch_size", str(BATCH_SIZE),
            "--num_workers", str(NUM_WORKERS), "--lr", str(LEARNING_RATE),
            "--weight_decay", str(WEIGHT_DECAY), "--lambda_reg", str(LAMBDA_REG),
            "--alpha", str(ALPHA),
            "--temperature", str(TEMPERATURE), "--proj_out_dim", str(PROJECTION_DIM),
            "--seed", str(seed), "--devices", args.device,
        ]
        if args.skip_existing and history and (
            int(history[-1]["epoch"]) >= EPOCHS
            or (checkpoint_path.exists() and checkpoint_complete(checkpoint_path, EPOCHS))
        ):
            elapsed = 0.0
            print(f"= [{position}/{total}] entraînement déjà terminé: {run_dir}", flush=True)
        else:
            if args.skip_existing and checkpoint_path.exists():
                command.extend(("--resume_from_checkpoint", str(checkpoint_path)))
            print(
                f"+ [{position}/{total} | terminées: {finished} | restantes: {total-finished}] "
                f"TRAIN {shlex.join(command)}",
                flush=True,
            )
            if not args.dry_run:
                start = time.monotonic()
                subprocess.run(command, cwd=REPO, check=True)
                elapsed = time.monotonic() - start
        if args.dry_run:
            continue
        history = read_rows(history_path)
        if not history:
            raise RuntimeError(f"No validation history in {history_path}")
        final = history[-1]
        store_row({
            "seed": seed,
            "alpha": ALPHA,
            "lambda_reg": LAMBDA_REG,
            "final_val_acc": 100.0 * float(final["val_acc"]),
            "final_val_eo": 100.0 * float(final["val_eo"]),
            "metrics_csv": latest_metrics(run_dir),
            "validation_history_csv": history_path,
            "checkpoint": run_dir / "checkpoints" / "last.ckpt",
            "elapsed_seconds": f"{elapsed:.1f}",
        })
        finished += 1
        print(f"= [{position}/{total} | terminées: {finished} | restantes: {total-finished}] OK", flush=True)

    if not args.dry_run:
        plot_dynamics()


if __name__ == "__main__":
    main()
