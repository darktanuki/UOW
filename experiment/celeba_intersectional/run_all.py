#!/usr/bin/env python3
"""Run OW and intersectional UOW on CelebA over seeds 0, 1, and 2."""

import argparse
import csv
import shlex
import subprocess
import sys
import time
from pathlib import Path

from config import (
    BATCH_SIZE, EPOCHS, EVAL_NUM_WORKERS, LAMBDA_REG, LEARNING_RATE,
    LINEAR_PROBE_C, METHODS, MIN_GROUP_SIZE, NUM_WORKERS, PROJECTION_DIM,
    ROOT_DATA, SEEDS, TEMPERATURE, WEIGHT_DECAY,
)
from make_table import build_table


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))
from experiment.checkpoint_utils import checkpoint_complete
ARTIFACTS = HERE / "artifacts"
RESULTS_CSV = HERE / "results.csv"
METRICS = (
    "target_acc", "eo_male", "eo_young", "eo_joint", "wga_joint",
    "bias_bal_acc_male", "bias_bal_acc_young", "bias_bal_acc_joint",
)
FIELDS = (
    "method_key", "method", "conditioning", "loss_name", "grouping", "alpha",
    "lambda_reg", "seed", *METRICS, "checkpoint", "evaluation_csv",
    "train_seconds", "eval_seconds",
)


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def store_row(row: dict[str, object]) -> None:
    rows = [
        existing for existing in read_rows(RESULTS_CSV)
        if not (existing["method_key"] == row["method_key"] and existing["seed"] == str(row["seed"]))
    ]
    rows.append(row)
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    temporary = RESULTS_CSV.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(RESULTS_CSV)


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
    parser.add_argument("--only-table", action="store_true")
    parser.add_argument("--methods", nargs="+", choices=[method.key for method in METHODS])
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS)
    args = parser.parse_args()

    if args.only_table:
        build_table(RESULTS_CSV, HERE / "table.tex")
        return

    planned = [
        (method, seed) for method in METHODS for seed in SEEDS
        if (args.methods is None or method.key in args.methods)
        and (args.seeds is None or seed in args.seeds)
    ]
    completed = {(row["method_key"], int(row["seed"])) for row in read_rows(RESULTS_CSV)}
    total = len(planned)
    finished = 0

    for position, (method, seed) in enumerate(planned, start=1):
        key = (method.key, seed)
        if args.skip_existing and key in completed:
            finished += 1
            print(f"= [{position}/{total} | terminées: {finished} | restantes: {total-finished}] déjà faite: {method.key} / seed {seed}")
            continue

        artifact_name = f"{method.key}/seed_{seed}"
        checkpoint = ARTIFACTS / "checkpoints" / artifact_name / "last.ckpt"
        evaluation_csv = ARTIFACTS / "evaluations" / method.key / f"seed_{seed}.csv"
        feature_path = ARTIFACTS / "features" / method.key / f"seed_{seed}.pt"
        evaluation_ready = args.skip_existing and any(
            all(row.get(metric) not in {None, ""} for metric in METRICS)
            for row in read_rows(evaluation_csv)
        )
        train_command = [
            sys.executable, str(HERE / "train.py"),
            "--name_data", "celeba_intersectional", "--root_data", ROOT_DATA,
            "--loss_name", method.loss_name,
            "--intersectional_grouping", method.grouping,
            "--min_group_size", str(MIN_GROUP_SIZE),
            "--reg_name", "mmd", "--kernel_name", "heat",
            "--temperature", str(TEMPERATURE), "--lambda_reg", str(LAMBDA_REG),
            "--epochs", str(EPOCHS), "--batch_size", str(BATCH_SIZE),
            "--num_workers", str(NUM_WORKERS), "--lr", str(LEARNING_RATE),
            "--proj_out_dim", str(PROJECTION_DIM), "--weight_decay", str(WEIGHT_DECAY),
            "--devices", args.device, "--seed", str(seed), "--online_linear_probing",
            "--artifacts_root", str(ARTIFACTS), "--artifact_name", artifact_name,
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
            train_seconds = run(train_command, args.dry_run, f"[{position}/{total} | terminées: {finished} | restantes: {total-finished}] TRAIN")

        eval_command = [
            sys.executable, str(HERE / "evaluate.py"),
            "--checkpoint_path", str(checkpoint), "--root_data", ROOT_DATA,
            "--batch_size", str(BATCH_SIZE), "--num_workers", str(EVAL_NUM_WORKERS),
            "--devices", args.device, "--C", str(LINEAR_PROBE_C),
            "--metrics_output_csv", str(evaluation_csv), "--feature_path", str(feature_path),
            "--no-recompute-features",
        ]
        if evaluation_ready:
            eval_seconds = 0.0
            print(f"= [{position}/{total}] évaluation déjà présente: {evaluation_csv}", flush=True)
        else:
            eval_seconds = run(eval_command, args.dry_run, f"[{position}/{total}] EVAL ")

        if not args.dry_run:
            metrics = read_rows(evaluation_csv)[0]
            store_row({
                "method_key": method.key, "method": method.method,
                "conditioning": method.conditioning, "loss_name": method.loss_name,
                "grouping": method.grouping, "alpha": "" if method.alpha is None else method.alpha,
                "lambda_reg": LAMBDA_REG, "seed": seed,
                **{metric: 100.0 * float(metrics[metric]) for metric in METRICS},
                "checkpoint": checkpoint, "evaluation_csv": evaluation_csv,
                "train_seconds": f"{train_seconds:.1f}", "eval_seconds": f"{eval_seconds:.1f}",
            })
            finished += 1
            print(f"= [{position}/{total} | terminées: {finished} | restantes: {total-finished}] OK", flush=True)

    if not args.dry_run:
        build_table(RESULTS_CSV, HERE / "table.tex")


if __name__ == "__main__":
    main()
