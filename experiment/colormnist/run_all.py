#!/usr/bin/env python3
"""Train and evaluate the Colored MNIST Gaussian/spherical comparison."""

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
    LINEAR_PROBE_C,
    METHODS,
    NUM_WORKERS,
    PROJECTION_DIM,
    ROOT_DATA,
    SEEDS,
    TEMPERATURE,
    WEIGHT_DECAY,
)
from make_tables import build_tables


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))
from experiment.checkpoint_utils import checkpoint_complete
ARTIFACTS = HERE / "artifacts"
RESULTS_CSV = HERE / "results.csv"

FIELDS = (
    "method", "method_title", "target_distribution", "loss_name", "reg_name",
    "alpha", "lambda_reg", "seed", "eval_type", "target_acc", "bias_acc",
    "wga", "eo", "checkpoint", "evaluation_csv", "train_seconds", "eval_seconds",
)


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def store_rows(new_rows: list[dict[str, object]], method: str, seed: int) -> None:
    existing = [
        row for row in read_rows(RESULTS_CSV)
        if not (row.get("method") == method and row.get("seed") == str(seed))
    ]
    temporary = RESULTS_CSV.with_suffix(".csv.tmp")
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(existing + new_rows)
    temporary.replace(RESULTS_CSV)


def completed_keys() -> set[tuple[str, int]]:
    types: dict[tuple[str, int], set[str]] = {}
    for row in read_rows(RESULTS_CSV):
        key = (row["method"], int(row["seed"]))
        types.setdefault(key, set()).add(row["eval_type"])
    return {key for key, found in types.items() if {"LP", "k-NN"}.issubset(found)}


def run(command: list[str], dry_run: bool, progress: str) -> float:
    print(f"+ {progress} {shlex.join(command)}", flush=True)
    if dry_run:
        return 0.0
    start = time.monotonic()
    subprocess.run(command, cwd=REPO, check=True)
    return time.monotonic() - start


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="0")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only-tables", action="store_true")
    parser.add_argument("--methods", nargs="+", choices=[method.key for method in METHODS])
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS)
    args = parser.parse_args()

    if args.only_tables:
        build_tables(RESULTS_CSV, HERE)
        return

    planned = [
        (method, seed)
        for method in METHODS
        for seed in SEEDS
        if (args.methods is None or method.key in args.methods)
        and (args.seeds is None or seed in args.seeds)
    ]
    completed = completed_keys()
    total = len(planned)
    finished = 0

    for position, (method, seed) in enumerate(planned, start=1):
        key = (method.key, seed)
        if args.skip_existing and key in completed:
            finished += 1
            print(
                f"= [{position}/{total} | terminées: {finished} | restantes: {total - finished}] "
                f"déjà faite: {method.key} / seed {seed}",
                flush=True,
            )
            continue

        artifact_name = f"{method.key}/seed_{seed}"
        checkpoint = ARTIFACTS / "checkpoints" / artifact_name / "last.ckpt"
        evaluation_csv = ARTIFACTS / "evaluations" / method.key / f"seed_{seed}.csv"
        features = ARTIFACTS / "features" / method.key / f"seed_{seed}"
        evaluation_ready = args.skip_existing and {
            row.get("eval_type") for row in read_rows(evaluation_csv)
        } == {"LP", "k-NN"}

        train_command = [
            sys.executable, str(HERE / "train.py"),
            "--name_data", "colorMNIST",
            "--root_data", ROOT_DATA,
            "--loss_name", method.loss_name,
            "--reg_name", method.reg_name,
            "--lambda_reg", str(LAMBDA_REG),
            "--temperature", str(TEMPERATURE),
            "--epochs", str(EPOCHS),
            "--batch_size", str(BATCH_SIZE),
            "--num_workers", str(NUM_WORKERS),
            "--lr", str(LEARNING_RATE),
            "--weight_decay", str(WEIGHT_DECAY),
            "--proj_out_dim", str(PROJECTION_DIM),
            "--devices", args.device,
            "--seed", str(seed),
            "--online_linear_probing",
            "--artifacts_root", str(ARTIFACTS),
            "--artifact_name", artifact_name,
        ]
        if method.alpha is not None:
            train_command.extend(("--alpha", str(method.alpha)))
        if evaluation_ready:
            train_seconds = 0.0
            print(f"= [{position}/{total}] métriques déjà présentes, entraînement ignoré", flush=True)
        elif args.skip_existing and checkpoint.exists() and checkpoint_complete(checkpoint, EPOCHS):
            train_seconds = 0.0
            print(f"= [{position}/{total}] entraînement terminé, checkpoint réutilisé: {checkpoint}", flush=True)
        else:
            if args.skip_existing and checkpoint.exists():
                train_command.extend(("--resume_from_checkpoint", str(checkpoint)))
            train_seconds = run(
                train_command,
                args.dry_run,
                f"[{position}/{total} | terminées: {finished} | restantes: {total - finished}] TRAIN",
            )

        eval_command = [
            sys.executable, str(HERE / "evaluate.py"),
            "--checkpoint_path", str(checkpoint),
            "--root_data", ROOT_DATA,
            "--batch_size", str(BATCH_SIZE),
            "--num_workers", str(EVAL_NUM_WORKERS),
            "--devices", args.device,
            "--C", str(LINEAR_PROBE_C),
            "--metrics_output_csv", str(evaluation_csv),
            "--benchmark_csv", str(ARTIFACTS / "benchmark.csv"),
            "--tsne_dir", str(features),
            "--no-recompute-features",
        ]
        if evaluation_ready:
            eval_seconds = 0.0
            print(f"= [{position}/{total}] évaluation déjà présente: {evaluation_csv}", flush=True)
        else:
            eval_seconds = run(eval_command, args.dry_run, f"[{position}/{total}] EVAL ")

        if not args.dry_run:
            output_rows = []
            for metrics in read_rows(evaluation_csv):
                output_rows.append(
                    {
                        "method": method.key,
                        "method_title": method.title,
                        "target_distribution": method.target_distribution,
                        "loss_name": method.loss_name,
                        "reg_name": method.reg_name,
                        "alpha": "" if method.alpha is None else method.alpha,
                        "lambda_reg": LAMBDA_REG,
                        "seed": seed,
                        "eval_type": metrics["eval_type"],
                        "target_acc": 100.0 * float(metrics["target_acc"]),
                        "bias_acc": 100.0 * float(metrics["bias_acc"]),
                        "wga": 100.0 * float(metrics["wga"]),
                        "eo": 100.0 * float(metrics["eo"]),
                        "checkpoint": checkpoint,
                        "evaluation_csv": evaluation_csv,
                        "train_seconds": f"{train_seconds:.1f}",
                        "eval_seconds": f"{eval_seconds:.1f}",
                    }
                )
            if {row["eval_type"] for row in output_rows} != {"LP", "k-NN"}:
                raise RuntimeError(f"Expected LP and k-NN rows in {evaluation_csv}")
            store_rows(output_rows, method.key, seed)
            finished += 1
            print(
                f"= [{position}/{total} | terminées: {finished} | restantes: {total - finished}] OK",
                flush=True,
            )

    if not args.dry_run:
        build_tables(RESULTS_CSV, HERE)


if __name__ == "__main__":
    main()
