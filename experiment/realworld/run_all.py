#!/usr/bin/env python3
"""Run every real-world benchmark, evaluate it, and build table.tex."""

import argparse
import csv
import shlex
import subprocess
import sys
import time
from pathlib import Path

from config import (
    BATCH_SIZE,
    EPOCHS,
    EVAL_NUM_WORKERS,
    LAMBDA_REG,
    LEARNING_RATE,
    METHODS,
    NUM_WORKERS,
    PROJECTION_DIM,
    SEEDS,
    TASKS,
    TEMPERATURE,
    WEIGHT_DECAY,
)
from make_table import build_table


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))
from experiment.checkpoint_utils import checkpoint_complete
RESULTS_CSV = HERE / "results.csv"
ARTIFACTS = HERE / "artifacts"

FIELDS = (
    "task",
    "dataset",
    "method",
    "method_title",
    "loss_name",
    "reg_name",
    "alpha",
    "seed",
    "acc",
    "eo",
    "wga",
    "bias_acc",
    "checkpoint",
    "evaluation_csv",
    "train_seconds",
    "eval_seconds",
)


def run(command: list[str], dry_run: bool, progress: str) -> float:
    print(f"+ {progress} {shlex.join(command)}", flush=True)
    if dry_run:
        return 0.0
    start = time.monotonic()
    subprocess.run(command, cwd=REPO, check=True)
    return time.monotonic() - start


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def store_row(row: dict[str, object]) -> None:
    """Atomically insert or replace one (task, method, seed) result."""
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    key = (str(row["task"]), str(row["method"]), str(row["seed"]))
    rows = [
        existing
        for existing in read_rows(RESULTS_CSV)
        if (existing["task"], existing["method"], existing["seed"]) != key
    ]
    rows.append(row)

    temporary = RESULTS_CSV.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(RESULTS_CSV)


def read_lp_metrics(path: Path) -> dict[str, float]:
    for row in read_rows(path):
        if row.get("eval_type") == "LP":
            return {
                "acc": 100.0 * float(row["target_acc"]),
                "eo": 100.0 * float(row["eo"]),
                "wga": 100.0 * float(row["wga"]),
                "bias_acc": 100.0 * float(row["bias_acc"]),
            }
    raise RuntimeError(f"No LP result found in {path}")


def selected(value: str, requested: list[str] | None) -> bool:
    return requested is None or value in requested


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="0", help="GPU index used for training and evaluation.")
    parser.add_argument("--skip-existing", action="store_true", help="Resume using completed rows in results.csv.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument("--only-table", action="store_true", help="Only rebuild table.tex from results.csv.")
    parser.add_argument("--tasks", nargs="+", choices=[task.key for task in TASKS])
    parser.add_argument("--methods", nargs="+", choices=[method.key for method in METHODS])
    parser.add_argument("--seeds", nargs="+", type=int, choices=list(SEEDS))
    args = parser.parse_args()

    if args.only_table:
        build_table(RESULTS_CSV, HERE / "table.tex")
        return

    completed = {
        (row["task"], row["method"], int(row["seed"]))
        for row in read_rows(RESULTS_CSV)
    }

    planned = [
        (task, method, seed)
        for task in TASKS
        for method in METHODS
        for seed in SEEDS
        if selected(task.key, args.tasks)
        and selected(method.key, args.methods)
        and (args.seeds is None or seed in args.seeds)
    ]
    total = len(planned)
    finished = 0

    for position, (task, method, seed) in enumerate(planned, start=1):
        key = (task.key, method.key, seed)
        if args.skip_existing and key in completed:
            finished += 1
            print(
                f"= [{position}/{total} | terminées: {finished} | restantes: {total - finished}] "
                f"déjà faite: {task.key} / {method.key} / seed {seed}",
                flush=True,
            )
            continue

        artifact_name = f"{task.key}/{method.key}/seed_{seed}"
        checkpoint = ARTIFACTS / "checkpoints" / artifact_name / "last.ckpt"
        evaluation_csv = ARTIFACTS / "evaluations" / task.key / method.key / f"seed_{seed}.csv"
        feature_dir = ARTIFACTS / "features" / task.key / method.key / f"seed_{seed}"
        evaluation_ready = args.skip_existing and any(
            row.get("eval_type") == "LP" for row in read_rows(evaluation_csv)
        )

        train_command = [
            sys.executable,
            str(REPO / "run_pretraining.py"),
            "--name_data", task.dataset,
            "--root_data", task.root_data,
            "--loss_name", method.loss_name,
            "--temperature", str(TEMPERATURE),
            "--epochs", str(EPOCHS),
            "--batch_size", str(BATCH_SIZE),
            "--lr", str(LEARNING_RATE),
            "--weight_decay", str(WEIGHT_DECAY),
            "--proj_out_dim", str(PROJECTION_DIM),
            "--num_workers", str(NUM_WORKERS),
            "--devices", args.device,
            "--seed", str(seed),
            "--artifacts_root", str(ARTIFACTS),
            "--artifact_name", artifact_name,
        ]
        if method.loss_name in {"ow", "uow", "lejepa", "endunsupervised"}:
            train_command.extend(("--lambda_reg", str(LAMBDA_REG)))
        if method.reg_name is not None:
            train_command.extend(("--reg_name", method.reg_name))
        if method.alpha is not None:
            train_command.extend(("--alpha", str(method.alpha)))
        if task.experiment is not None:
            train_command.extend(("--experiment", task.experiment))
        if method.loss_name not in {"crossentropy", "ce"}:
            train_command.append("--online_linear_probing")

        counter = (
            f"[{position}/{total} | terminées: {finished} | "
            f"restantes: {total - finished}] TRAIN"
        )
        if evaluation_ready:
            train_seconds = 0.0
            print(f"= {counter} métriques déjà présentes, entraînement ignoré", flush=True)
        elif args.skip_existing and checkpoint.exists():
            if checkpoint_complete(checkpoint, EPOCHS):
                train_seconds = 0.0
                print(f"= {counter} entraînement terminé, checkpoint réutilisé: {checkpoint}", flush=True)
            else:
                train_command.extend(("--resume_from_checkpoint", str(checkpoint)))
                train_seconds = run(train_command, args.dry_run, counter)
        else:
            train_seconds = run(train_command, args.dry_run, counter)

        eval_command = [
            sys.executable,
            str(REPO / "run_eval.py"),
            "--checkpoint_path", str(checkpoint),
            "--root_data", task.root_data,
            "--batch_size", str(BATCH_SIZE),
            "--num_workers", str(EVAL_NUM_WORKERS),
            "--devices", args.device,
            "--C", str(task.probe_c),
            "--metrics_output_csv", str(evaluation_csv),
            "--benchmark_csv", str(ARTIFACTS / "benchmark.csv"),
            "--tsne_dir", str(feature_dir),
            "--no-recompute-features",
        ]
        if evaluation_ready:
            eval_seconds = 0.0
            print(f"= [{position}/{total}] évaluation déjà présente: {evaluation_csv}", flush=True)
        else:
            eval_seconds = run(eval_command, args.dry_run, f"[{position}/{total}] EVAL ")

        if not args.dry_run:
            store_row(
                {
                    "task": task.key,
                    "dataset": task.dataset,
                    "method": method.key,
                    "method_title": method.title,
                    "loss_name": method.loss_name,
                    "reg_name": method.reg_name or "",
                    "alpha": "" if method.alpha is None else method.alpha,
                    "seed": seed,
                    **read_lp_metrics(evaluation_csv),
                    "checkpoint": checkpoint,
                    "evaluation_csv": evaluation_csv,
                    "train_seconds": f"{train_seconds:.1f}",
                    "eval_seconds": f"{eval_seconds:.1f}",
                }
            )
            finished += 1
            print(
                f"= [{position}/{total} | terminées: {finished} | restantes: {total - finished}] OK",
                flush=True,
            )

    if not args.dry_run:
        build_table(RESULTS_CSV, HERE / "table.tex")


if __name__ == "__main__":
    main()
