#!/usr/bin/env python3
"""Run the Synthetic Engraving ablation and OW/UOW comparison."""

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
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    EVAL_NUM_WORKERS,
    FIXED_ALPHA,
    FIXED_ALPHA_LABEL,
    FIXED_LAMBDA,
    FIXED_LAMBDA_LABEL,
    KERNEL_NAME,
    LAMBDAS,
    LEARNING_RATE,
    M_VALUES,
    NUM_WORKERS,
    PROJECTION_DIM,
    REG_NAME,
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
SWEEP_ROOT = HERE / "temp_vit_s8_lr_sweep"


@dataclass(frozen=True)
class RunSpec:
    loss_name: str
    nb_signs: int
    lambda_label: str
    lambda_reg: float
    alpha_label: str
    alpha: float | None
    seed: int

    @property
    def key(self) -> str:
        alpha = self.alpha_label or "none"
        return f"{self.loss_name}|m={self.nb_signs}|lambda={self.lambda_label}|alpha={alpha}|seed={self.seed}"

    @property
    def artifact_name(self) -> str:
        alpha = self.alpha_label.replace(".", "p") if self.alpha_label else "none"
        lambd = self.lambda_label.replace("/", "over")
        return f"{self.loss_name}/m_{self.nb_signs}/lambda_{lambd}/alpha_{alpha}/seed_{self.seed}"


FIELDS = (
    "run_key", "loss_name", "nb_signs", "lambda_label", "lambda_reg",
    "alpha_label", "alpha", "seed", "best_val_map", "best_epoch",
    "test_map_emb", "test_map_z", "metrics_csv", "best_checkpoint",
    "last_checkpoint", "elapsed_seconds",
)


def ablation_runs() -> list[RunSpec]:
    return [
        RunSpec("uow", 2, lambda_label, lambda_reg, alpha_label, alpha, 0)
        for lambda_label, lambda_reg in LAMBDAS
        for alpha_label, alpha in ALPHAS
    ]


def comparison_runs() -> list[RunSpec]:
    runs = []
    for nb_signs in M_VALUES:
        for seed in SEEDS:
            runs.append(RunSpec("ow", nb_signs, FIXED_LAMBDA_LABEL, FIXED_LAMBDA, "", None, seed))
            runs.append(
                RunSpec(
                    "uow", nb_signs, FIXED_LAMBDA_LABEL, FIXED_LAMBDA,
                    FIXED_ALPHA_LABEL, FIXED_ALPHA, seed,
                )
            )
    return runs


def planned_runs(phase: str) -> list[RunSpec]:
    candidates = []
    if phase in {"ablation", "all"}:
        candidates.extend(ablation_runs())
    if phase in {"comparison", "all"}:
        candidates.extend(comparison_runs())
    # The fixed UOW m=2 seed-0 point occurs in both phases; execute it once.
    return list(dict.fromkeys(candidates))


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def store_row(row: dict[str, object]) -> None:
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows = [existing for existing in read_rows(RESULTS_CSV) if existing["run_key"] != row["run_key"]]
    rows.append(row)
    temporary = RESULTS_CSV.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(RESULTS_CSV)


def metric_summary(metrics_csv: Path, test_metrics_csv: Path) -> dict[str, float | int]:
    best_value = None
    best_epoch = None
    test_map_emb = None
    test_map_z = None
    training_csvs = sorted(metrics_csv.parent.parent.glob("version_*/metrics.csv"))
    for row in [row for path in training_csvs for row in read_rows(path)] + read_rows(test_metrics_csv):
        raw_val = row.get("val/mAP/emb")
        if raw_val not in {None, ""} and (best_value is None or float(raw_val) > best_value):
            best_value = float(raw_val)
            best_epoch = int(float(row["epoch"]))
        raw_test_emb = row.get("test/mAP/emb")
        raw_test_z = row.get("test/mAP/z")
        if raw_test_emb not in {None, ""}:
            test_map_emb = float(raw_test_emb)
        if raw_test_z not in {None, ""}:
            test_map_z = float(raw_test_z)
    if best_value is None or best_epoch is None:
        raise RuntimeError(f"No val/mAP/emb found in {metrics_csv}")
    if test_map_emb is None or test_map_z is None:
        raise RuntimeError(f"No test mAP metrics found in {metrics_csv}")
    return {
        "best_val_map": 100.0 * best_value,
        "best_epoch": best_epoch,
        "test_map_emb": 100.0 * test_map_emb,
        "test_map_z": 100.0 * test_map_z,
    }


def latest_metrics(spec: RunSpec) -> Path:
    files = sorted(
        (ARTIFACTS / "logs" / spec.artifact_name).glob("version_*/metrics.csv"),
        key=lambda path: path.stat().st_mtime,
    )
    if not files:
        raise FileNotFoundError(f"No metrics.csv under {ARTIFACTS / 'logs' / spec.artifact_name}")
    return files[-1]


def has_test_metrics(path: Path) -> bool:
    return any(
        row.get("test/mAP/emb") not in {None, ""}
        and row.get("test/mAP/z") not in {None, ""}
        for row in read_rows(path)
    )


def has_logged_test_metrics(spec: RunSpec) -> bool:
    return any(
        has_test_metrics(path)
        for path in (ARTIFACTS / "logs" / spec.artifact_name).glob("version_*/metrics.csv")
    )


def reuse_vit_sweep_run(spec: RunSpec, backbone: str, lr: float) -> bool:
    """Import the already evaluated fixed m=2, seed-0 ViT run from the LR sweep."""
    if not (
        backbone == "vit_s8"
        and lr == 1e-4
        and spec == RunSpec("uow", 2, FIXED_LAMBDA_LABEL, FIXED_LAMBDA,
                            FIXED_ALPHA_LABEL, FIXED_ALPHA, 0)
    ):
        return False

    sweep_rows = [row for row in read_rows(SWEEP_ROOT / "results.csv") if float(row["lr"]) == lr]
    if not sweep_rows:
        return False
    artifact_name = "vit_s8/m_2/seed_0/lr_0.0001"
    checkpoint_dir = SWEEP_ROOT / "checkpoints" / artifact_name
    best_checkpoint = checkpoint_dir / "best.ckpt"
    test_metrics_csv = checkpoint_dir / "test_metrics.csv"
    training_csvs = list((SWEEP_ROOT / "logs" / artifact_name).glob("version_*/metrics.csv"))
    if not best_checkpoint.exists() or not has_test_metrics(test_metrics_csv) or not training_csvs:
        return False

    metrics_csv = max(training_csvs, key=lambda path: path.stat().st_mtime)
    summary = metric_summary(metrics_csv, test_metrics_csv)
    if abs(summary["best_val_map"] - float(sweep_rows[-1]["best_val_map"])) > 0.01:
        raise ValueError("The ViT sweep CSV and validation logs disagree for lr=1e-4")
    store_row({
        "run_key": spec.key,
        "loss_name": spec.loss_name,
        "nb_signs": spec.nb_signs,
        "lambda_label": spec.lambda_label,
        "lambda_reg": spec.lambda_reg,
        "alpha_label": spec.alpha_label,
        "alpha": spec.alpha,
        "seed": spec.seed,
        **summary,
        "metrics_csv": metrics_csv,
        "best_checkpoint": best_checkpoint,
        "last_checkpoint": checkpoint_dir / "last.ckpt",
        "elapsed_seconds": "0.0",
    })
    print(f"= reused ViT sweep run: {spec.key} ({best_checkpoint})", flush=True)
    return True


def run_one(spec: RunSpec, device: str, accelerator: str, backbone: str, lr: float, dry_run: bool, progress: str, skip_existing: bool) -> None:
    command = [
        sys.executable, str(HERE / "train.py"),
        "--backbone", backbone,
        "--loss_name", spec.loss_name,
        "--reg_name", REG_NAME,
        "--kernel_name", KERNEL_NAME,
        "--proj_out_dim", str(PROJECTION_DIM),
        "--temperature", str(TEMPERATURE),
        "--lambda_reg", str(spec.lambda_reg),
        "--nb_signs", str(spec.nb_signs),
        "--seed", str(spec.seed),
        "--epochs", str(EPOCHS),
        "--batch_size", str(BATCH_SIZE),
        "--num_workers", str(NUM_WORKERS),
        "--eval_num_workers", str(EVAL_NUM_WORKERS),
        "--lr", str(lr),
        "--weight_decay", str(WEIGHT_DECAY),
        "--early_stopping_patience", str(EARLY_STOPPING_PATIENCE),
        "--devices", device,
        "--accelerator", accelerator,
        "--artifacts_root", str(ARTIFACTS),
        "--artifact_name", spec.artifact_name,
    ]
    if spec.alpha is not None:
        command.extend(("--alpha", str(spec.alpha)))

    checkpoint_dir = ARTIFACTS / "checkpoints" / spec.artifact_name
    best_checkpoint = checkpoint_dir / "best.ckpt"
    last_checkpoint = checkpoint_dir / "last.ckpt"
    test_metrics_csv = checkpoint_dir / "test_metrics.csv"
    if skip_existing and best_checkpoint.exists():
        has_metrics = has_test_metrics(test_metrics_csv) or has_logged_test_metrics(spec)
        complete = (checkpoint_dir / ".training_complete").exists() or (
            last_checkpoint.exists() and checkpoint_complete(last_checkpoint, EPOCHS)
        )
        if has_metrics:
            command = []
        elif complete or not last_checkpoint.exists():
            command.append("--test-only")
        else:
            command.extend(("--resume_from_checkpoint", str(last_checkpoint)))
        if not command or "--test-only" in command:
            print(f"= {progress} entraînement réutilisé: {best_checkpoint}", flush=True)
    elif skip_existing and last_checkpoint.exists():
        command.extend(("--resume_from_checkpoint", str(last_checkpoint)))
    if command:
        print(f"+ {progress} {shlex.join(command)}", flush=True)
    if dry_run:
        return
    start = time.monotonic()
    if command:
        subprocess.run(command, cwd=REPO, check=True)
    elapsed = time.monotonic() - start

    metrics_csv = latest_metrics(spec)
    store_row(
        {
            "run_key": spec.key,
            "loss_name": spec.loss_name,
            "nb_signs": spec.nb_signs,
            "lambda_label": spec.lambda_label,
            "lambda_reg": spec.lambda_reg,
            "alpha_label": spec.alpha_label,
            "alpha": "" if spec.alpha is None else spec.alpha,
            "seed": spec.seed,
            **metric_summary(metrics_csv, test_metrics_csv),
            "metrics_csv": metrics_csv,
            "best_checkpoint": checkpoint_dir / "best.ckpt",
            "last_checkpoint": checkpoint_dir / "last.ckpt",
            "elapsed_seconds": f"{elapsed:.1f}",
        }
    )


def main() -> None:
    global ARTIFACTS, RESULTS_CSV
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("ablation", "comparison", "all"), default="all")
    parser.add_argument("--backbone", choices=("dinov3_convnext", "resnet50", "vit_s8"), default="dinov3_convnext")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--device", default="0")
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only-tables", action="store_true")
    parser.add_argument("--losses", nargs="+", choices=("ow", "uow"))
    parser.add_argument("--m-values", nargs="+", type=int, choices=M_VALUES)
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS)
    args = parser.parse_args()

    if args.lr <= 0:
        parser.error("--lr must be positive")
    output_dir = HERE if args.backbone == "dinov3_convnext" and args.lr == LEARNING_RATE else HERE / f"{args.backbone}_lr_{args.lr:g}"
    ARTIFACTS = output_dir / "artifacts"
    RESULTS_CSV = output_dir / "results.csv"

    if args.only_tables:
        build_tables(RESULTS_CSV, output_dir)
        return

    planned = [
        spec for spec in planned_runs(args.phase)
        if (args.losses is None or spec.loss_name in args.losses)
        and (args.m_values is None or spec.nb_signs in args.m_values)
        and (args.seeds is None or spec.seed in args.seeds)
    ]
    completed = {row["run_key"] for row in read_rows(RESULTS_CSV)}
    total = len(planned)
    finished = 0

    for position, spec in enumerate(planned, start=1):
        if args.skip_existing and not args.dry_run and reuse_vit_sweep_run(spec, args.backbone, args.lr):
            finished += 1
            continue
        if args.skip_existing and spec.key in completed:
            finished += 1
            print(
                f"= [{position}/{total} | terminées: {finished} | restantes: {total - finished}] "
                f"déjà faite: {spec.key}",
                flush=True,
            )
            continue
        progress = f"[{position}/{total} | terminées: {finished} | restantes: {total - finished}] TRAIN+EVAL"
        run_one(spec, args.device, args.accelerator, args.backbone, args.lr, args.dry_run, progress, args.skip_existing)
        if not args.dry_run:
            finished += 1
            print(
                f"= [{position}/{total} | terminées: {finished} | restantes: {total - finished}] OK",
                flush=True,
            )

    if not args.dry_run:
        build_tables(RESULTS_CSV, output_dir)


if __name__ == "__main__":
    main()
