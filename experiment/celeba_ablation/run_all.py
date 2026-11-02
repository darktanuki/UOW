#!/usr/bin/env python3
"""Run the CelebA Attractive/Male UOW lambda-alpha ablation."""

import argparse
import csv
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from config import (
    ALPHAS,
    BATCH_SIZE,
    EPOCHS,
    EVAL_NUM_WORKERS,
    KERNEL_NAME,
    LAMBDAS,
    LEARNING_RATE,
    LINEAR_PROBE_C,
    NUM_WORKERS,
    PROJECTION_DIM,
    REG_NAME,
    ROOT_DATA,
    SEED,
    TARGET_EXPERIMENT,
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


@dataclass(frozen=True)
class RunSpec:
    lambda_label: str
    lambda_reg: float
    alpha_label: str
    alpha: float
    seed: int

    @property
    def key(self) -> str:
        return f"lambda={self.lambda_label}|alpha={self.alpha_label}|seed={self.seed}"

    @property
    def artifact_name(self) -> str:
        lambd = self.lambda_label.replace(".", "p")
        alpha = self.alpha_label.replace(".", "p")
        return f"lambda_{lambd}/alpha_{alpha}/seed_{self.seed}"


FIELDS = (
    "run_key", "lambda_label", "lambda_reg", "alpha_label", "alpha", "seed",
    "eo", "acc", "wga", "bias_acc", "checkpoint", "evaluation_csv",
    "train_seconds", "eval_seconds",
)


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def store_row(row: dict[str, object]) -> None:
    rows = [existing for existing in read_rows(RESULTS_CSV) if existing["run_key"] != row["run_key"]]
    rows.append(row)
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    temporary = RESULTS_CSV.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(RESULTS_CSV)


def lp_metrics(path: Path) -> dict[str, float]:
    for row in read_rows(path):
        if row.get("eval_type") == "LP":
            return {
                "eo": 100.0 * float(row["eo"]),
                "acc": 100.0 * float(row["target_acc"]),
                "wga": 100.0 * float(row["wga"]),
                "bias_acc": 100.0 * float(row["bias_acc"]),
            }
    raise RuntimeError(f"No LP row found in {path}")


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
    parser.add_argument("--lambdas", nargs="+", choices=[item[0] for item in LAMBDAS])
    parser.add_argument("--alphas", nargs="+", choices=[item[0] for item in ALPHAS])
    args = parser.parse_args()

    if args.only_tables:
        build_tables(RESULTS_CSV, HERE)
        return

    planned = [
        RunSpec(lambda_label, lambda_reg, alpha_label, alpha, seed)
        for lambda_label, lambda_reg, _ in LAMBDAS
        for alpha_label, alpha in ALPHAS
        for seed in (SEED,)
        if (args.lambdas is None or lambda_label in args.lambdas)
        and (args.alphas is None or alpha_label in args.alphas)
    ]
    completed = {row["run_key"] for row in read_rows(RESULTS_CSV)}
    total = len(planned)
    finished = 0

    for position, spec in enumerate(planned, start=1):
        if args.skip_existing and spec.key in completed:
            finished += 1
            print(
                f"= [{position}/{total} | terminées: {finished} | restantes: {total - finished}] "
                f"déjà faite: {spec.key}",
                flush=True,
            )
            continue

        checkpoint = ARTIFACTS / "checkpoints" / spec.artifact_name / "last.ckpt"
        evaluation_csv = ARTIFACTS / "evaluations" / spec.artifact_name / "metrics.csv"
        features = ARTIFACTS / "features" / spec.artifact_name
        evaluation_ready = args.skip_existing and any(
            row.get("eval_type") == "LP" for row in read_rows(evaluation_csv)
        )
        train_command = [
            sys.executable, str(HERE / "train.py"),
            "--name_data", "celeba",
            "--root_data", ROOT_DATA,
            "--experiment", TARGET_EXPERIMENT,
            "--loss_name", "uow",
            "--reg_name", REG_NAME,
            "--kernel_name", KERNEL_NAME,
            "--lambda_reg", str(spec.lambda_reg),
            "--alpha", str(spec.alpha),
            "--temperature", str(TEMPERATURE),
            "--epochs", str(EPOCHS),
            "--batch_size", str(BATCH_SIZE),
            "--num_workers", str(NUM_WORKERS),
            "--lr", str(LEARNING_RATE),
            "--weight_decay", str(WEIGHT_DECAY),
            "--proj_out_dim", str(PROJECTION_DIM),
            "--devices", args.device,
            "--seed", str(spec.seed),
            "--online_linear_probing",
            "--artifacts_root", str(ARTIFACTS),
            "--artifact_name", spec.artifact_name,
        ]
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
            store_row(
                {
                    "run_key": spec.key,
                    "lambda_label": spec.lambda_label,
                    "lambda_reg": spec.lambda_reg,
                    "alpha_label": spec.alpha_label,
                    "alpha": spec.alpha,
                    "seed": spec.seed,
                    **lp_metrics(evaluation_csv),
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
        build_tables(RESULTS_CSV, HERE)


if __name__ == "__main__":
    main()
