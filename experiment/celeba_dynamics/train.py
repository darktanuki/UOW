#!/usr/bin/env python3
"""Train CelebA UOW (spherical) from scratch and plot validation metrics."""

import argparse
import csv
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from dataloaders.celeba import get_celeba_loaders
from lightning_module import PretrainedWithLinearProbing, compute_equalized_odds_torch
from parameters import SSLParams


class CelebAScratchUOW(PretrainedWithLinearProbing):
    """CelebA model that records validation accuracy and EO after every epoch."""

    def __init__(self, params, output_dir: Path):
        super().__init__(params)
        self.output_dir = Path(output_dir)
        self.validation_history: list[dict[str, float | int]] = []

    def on_validation_epoch_end(self, mode="val"):
        should_record = (
            not self.trainer.sanity_checking
            and bool(self.val_preds_labels)
            and (self.linear_prob_labels is not None or self.use_cross_entropy)
        )

        if should_record:
            predictions = torch.cat(self.val_preds_labels)
            targets = torch.cat(self.val_y)
            biases = torch.cat(self.val_bias)
            accuracy = (predictions == targets).float().mean()
            equalized_odds = compute_equalized_odds_torch(targets, predictions, biases)
            self.validation_history.append(
                {
                    "epoch": self.current_epoch + 1,
                    "val_acc": float(accuracy.cpu()),
                    "val_eo": float(equalized_odds.cpu()),
                }
            )

        super().on_validation_epoch_end(mode=mode)

        if should_record and self.trainer.is_global_zero:
            self.save_validation_history()

    def on_fit_end(self):
        if self.trainer.is_global_zero and self.validation_history:
            self.save_validation_history()

    def save_validation_history(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = self.output_dir / "validation_history.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["epoch", "val_acc", "val_eo"])
            writer.writeheader()
            writer.writerows(self.validation_history)

        epochs = [int(row["epoch"]) for row in self.validation_history]
        accuracies = [100.0 * float(row["val_acc"]) for row in self.validation_history]
        equalized_odds = [100.0 * float(row["val_eo"]) for row in self.validation_history]

        figure, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
        axes[0].plot(epochs, accuracies, color="tab:blue", linewidth=2, marker="o", markersize=3)
        axes[0].set_ylabel("Validation accuracy (%)")
        axes[0].set_title("CelebA — UOW spherical, T: Attractive / S: Male")
        axes[0].grid(alpha=0.3)

        axes[1].plot(epochs, equalized_odds, color="tab:orange", linewidth=2, marker="o", markersize=3)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Validation EO (%)")
        axes[1].grid(alpha=0.3)

        figure.tight_layout()
        figure.savefig(self.output_dir / "validation_accuracy_eo.png", dpi=200)
        figure.savefig(self.output_dir / "validation_accuracy_eo.pdf")
        plt.close(figure)


def build_argparser():
    parser = argparse.ArgumentParser(
        "CelebA training from scratch: UOW spherical, T=Attractive, S=Male"
    )
    parser.add_argument("--root_data", default="./data")
    parser.add_argument("--output_dir", default="results/celeba_scratch_uow_alpha0p5")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-2)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--lambda_reg", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=4 / 64)
    parser.add_argument("--proj_out_dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument(
        "--fast_dev_run",
        action="store_true",
        help="Run one training and validation batch to verify the pipeline.",
    )
    return parser


def build_params(args) -> SSLParams:
    return SSLParams(
        root_data=args.root_data,
        name_data="celeba",
        experiment="T=a/S=m",
        pretrained_backbone=False,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        accelerator=args.accelerator,
        devices=args.devices,
        proj_out_dim=args.proj_out_dim,
        loss_name="uow",
        reg_name="mmd",
        kernel_name="heat",
        temperature=args.temperature,
        lambda_reg=args.lambda_reg,
        alpha=args.alpha,
        seed=args.seed,
        online_linear_probing=True,
    )


def main():
    args = build_argparser().parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    params = build_params(args)
    if params.devices != "auto":
        params.devices = [int(params.devices)]

    pl.seed_everything(params.seed, workers=True)
    train_loader, val_loader, _ = get_celeba_loaders(
        root=params.root_data,
        batch_size=params.batch_size,
        num_workers=params.num_workers,
        experiment=params.experiment,
    )

    model = CelebAScratchUOW(params, output_dir)
    history_path = output_dir / "validation_history.csv"
    if args.resume_from_checkpoint and history_path.exists():
        with history_path.open(newline="", encoding="utf-8") as handle:
            model.validation_history = [
                {"epoch": int(row["epoch"]), "val_acc": float(row["val_acc"]), "val_eo": float(row["val_eo"])}
                for row in csv.DictReader(handle)
            ]
    checkpoint = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="last",
        save_top_k=0,
        save_last=True,
        every_n_epochs=1,
    )
    logger = CSVLogger(save_dir=output_dir, name="training_logs")
    trainer = pl.Trainer(
        accelerator=params.accelerator,
        devices=params.devices,
        max_epochs=params.epochs,
        callbacks=[checkpoint],
        logger=logger,
        log_every_n_steps=50,
        fast_dev_run=args.fast_dev_run,
    )

    print(
        "Configuration: CelebA, T=Attractive, S=Male, UOW spherical, "
        f"alpha={params.alpha}, random ResNet-18 initialization, epochs={params.epochs}."
    )
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=args.resume_from_checkpoint,
    )

    if model.validation_history:
        final = model.validation_history[-1]
        print(
            f"Final validation: Acc={100 * float(final['val_acc']):.2f}% | "
            f"EO={100 * float(final['val_eo']):.2f}%"
        )
        print(f"Metrics and plots saved in: {output_dir}")


if __name__ == "__main__":
    main()
