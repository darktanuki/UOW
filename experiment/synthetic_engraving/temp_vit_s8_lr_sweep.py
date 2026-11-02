#!/usr/bin/env python3
"""Temporary pretrained ViT-S/8 learning-rate sweep (m=2, seed=0)."""

import argparse
import csv
import math
import subprocess
import sys
from pathlib import Path

from config import (
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    EVAL_NUM_WORKERS,
    FIXED_ALPHA,
    FIXED_LAMBDA,
    KERNEL_NAME,
    NUM_WORKERS,
    PROJECTION_DIM,
    REG_NAME,
    TEMPERATURE,
    WEIGHT_DECAY,
)


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
ROOT = HERE / "temp_vit_s8_lr_sweep"
RESULTS = ROOT / "results.csv"
LEARNING_RATES = (1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2)
FIELDS = ("lr", "best_val_map", "test_map_emb", "best_checkpoint")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def save_result(row: dict[str, str]) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    rows = [old for old in read_csv(RESULTS) if old["lr"] != row["lr"]]
    rows.append(row)
    temporary = RESULTS.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(RESULTS)


def remove_result(lr: float) -> None:
    if not RESULTS.exists():
        return
    rows = [row for row in read_csv(RESULTS) if row["lr"] != str(lr)]
    with RESULTS.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def best_validation_map(artifact_name: str) -> float:
    logs = ROOT / "logs" / artifact_name
    files = list(logs.glob("version_*/metrics.csv"))
    if not files:
        raise RuntimeError(f"No metrics.csv found under {logs}")
    latest = max(files, key=lambda path: int(path.parent.name.removeprefix("version_")))
    values = [
        float(row["val/mAP/emb"])
        for row in read_csv(latest)
        if row.get("val/mAP/emb") not in (None, "")
    ]
    if not values:
        raise RuntimeError(f"No val/mAP/emb found under {logs}")
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"Non-finite validation mAP in {latest}")
    return 100.0 * max(values)


def print_table() -> None:
    rows = {row["lr"]: row for row in read_csv(RESULTS)}
    print("\nViT-S/8 — synthetic engraving, m=2, seed=0")
    print("Meilleur mAP|c de validation sur embeddings (%)")
    print("+---------+------------+")
    print("| LR      | mAP|c (%)  |")
    print("+---------+------------+")
    for lr in LEARNING_RATES:
        row = rows.get(str(lr))
        value = f"{float(row['best_val_map']):.2f}" if row else "--"
        print(f"| {lr:<7g} | {value:>10} |")
    print("+---------+------------+")
    print(f"Résultats : {RESULTS}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only-table", action="store_true")
    args = parser.parse_args()

    if args.only_table:
        print_table()
        return

    failures = []
    existing = {row["lr"]: row for row in read_csv(RESULTS)}
    for position, lr in enumerate(LEARNING_RATES, start=1):
        artifact_name = f"vit_s8/m_2/seed_0/lr_{lr:g}"
        checkpoint = ROOT / "checkpoints" / artifact_name / "best.ckpt"
        if args.skip_existing and str(lr) in existing and checkpoint.exists():
            print(f"[{position}/{len(LEARNING_RATES)}] {lr:g} : déjà terminé", flush=True)
            continue

        command = [
            sys.executable, str(HERE / "train.py"),
            "--backbone", "vit_s8",
            "--loss_name", "uow",
            "--reg_name", REG_NAME,
            "--kernel_name", KERNEL_NAME,
            "--lambda_reg", str(FIXED_LAMBDA),
            "--alpha", str(FIXED_ALPHA),
            "--temperature", str(TEMPERATURE),
            "--proj_out_dim", str(PROJECTION_DIM),
            "--nb_signs", "2",
            "--seed", "0",
            "--lr", str(lr),
            "--weight_decay", str(WEIGHT_DECAY),
            "--batch_size", str(BATCH_SIZE),
            "--epochs", str(args.epochs),
            "--num_workers", str(args.num_workers),
            "--eval_num_workers", str(EVAL_NUM_WORKERS),
            "--early_stopping_patience", str(EARLY_STOPPING_PATIENCE),
            "--accelerator", args.accelerator,
            "--devices", args.device,
            "--artifacts_root", str(ROOT),
            "--artifact_name", artifact_name,
        ]
        print(f"[{position}/{len(LEARNING_RATES)}] LR={lr:g}: {' '.join(command)}", flush=True)
        if args.dry_run:
            continue
        remove_result(lr)
        if subprocess.run(command, cwd=REPO, check=False).returncode != 0:
            failures.append(lr)
            print(f"Échec pour LR={lr:g}; passage au suivant.", file=sys.stderr, flush=True)
            continue

        try:
            test_rows = read_csv(ROOT / "checkpoints" / artifact_name / "test_metrics.csv")
            if not test_rows or test_rows[-1].get("test/mAP/emb") in (None, ""):
                raise RuntimeError("Métrique de test absente")
            val_map = best_validation_map(artifact_name)
            test_map = 100.0 * float(test_rows[-1]["test/mAP/emb"])
            if not math.isfinite(test_map):
                raise ValueError("Non-finite test mAP")
        except (OSError, RuntimeError, ValueError) as error:
            failures.append(lr)
            print(f"Métrique absente pour LR={lr:g}: {error}", file=sys.stderr, flush=True)
            continue
        save_result({
            "lr": str(lr),
            "best_val_map": str(val_map),
            "test_map_emb": str(test_map),
            "best_checkpoint": str(checkpoint),
        })

    if not args.dry_run:
        print_table()
    if failures:
        raise SystemExit(f"Échecs pour les LR : {', '.join(f'{lr:g}' for lr in failures)}")


if __name__ == "__main__":
    main()
