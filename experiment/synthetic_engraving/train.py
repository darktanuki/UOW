#!/usr/bin/env python3
import argparse
import csv
import os
import sys
from pathlib import Path
from torch.utils.data import DataLoader


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from lightning_module import PretrainedWithRetrieval
from parameters import SSLParams
from dataloaders.synthetic_engraving import get_synthetic_engraving_loaders


def build_argparser():
    p = argparse.ArgumentParser("Synthetic engraving pretraining launcher")

    p.add_argument("--backbone", choices=["dinov3_convnext", "resnet18", "resnet50", "vit_s8"], default="dinov3_convnext")
    p.add_argument("--loss_name", type=str, default="uow", choices=["uow", "ow", "infonce", "endunsupervised"])
    p.add_argument("--reg_name", type=str, default="kl", choices=["kl", "ksd", "mmd"])
    p.add_argument("--kernel_name", type=str, default="heat")
    p.add_argument("--temperature", type=float, default=2 / 256)
    p.add_argument("--lambda_reg", type=float, default=0.5)
    p.add_argument("--lambda_end", type=float, default=0.1, help="EnDUnsupervised regularization weight.")
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--lambda_uow_reg", type=float, default=None, help="Deprecated alias for --alpha.")
    p.add_argument("--proj_out_dim", type=int, default=256)
    p.add_argument("--nb_signs", type=int, default=2)

    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--eval_num_workers", type=int, default=0)
    p.add_argument("--test-only", action="store_true", help="Evaluate the saved best checkpoint without training.")
    p.add_argument("--resume_from_checkpoint", default=None)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=5e-4)

    p.add_argument("--accelerator", type=str, default="auto")
    p.add_argument("--devices", type=str, default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run_tag", default="", help="Optional suffix used to distinguish otherwise identical runs.")
    p.add_argument(
        "--artifacts_root",
        default=".",
        help="Root directory containing logs/ and checkpoints/ (default: repository root).",
    )
    p.add_argument(
        "--artifact_name",
        default=None,
        help="Optional explicit logs/checkpoints name for structured experiment suites.",
    )
    p.add_argument("--early_stopping_patience", type=int, default=8)
    p.add_argument("--log_every_n_steps", type=int, default=50)
    return p


def build_params(args):
    params = SSLParams()
    for k, v in vars(args).items():
        if hasattr(params, k):
            setattr(params, k, v)

    params.name_data = "synthetic_engraving"
    params.online_linear_probing = False

    if params.devices != "auto":
        params.devices = [int(params.devices)]  # type: ignore

    return params


def build_experiment_name(params):
    alpha = params.alpha if params.alpha is not None else params.lambda_uow_reg
    alpha = 0.5 if alpha is None else alpha
    tag = f"{params.run_tag}-" if params.run_tag else ""
    return (
        f"synthetic_engraving-"
        f"backbone={params.backbone}-"
        f"{params.loss_name}-"
        f"nb_signs={params.nb_signs}-"
        f"reg={params.reg_name}-"
        f"temp={params.temperature}-"
        f"alpha={alpha}-"
        f"{tag}"
        f"epochs={params.epochs}-"
        f"batch={params.batch_size}-"
        f"dim={params.proj_out_dim}-"
        f"lr={params.lr}-"
        f"wd={params.weight_decay}"
    )


def main():
    args = build_argparser().parse_args()
    params = build_params(args)

    pl.seed_everything(args.seed, workers=True)

    train_loader, val_loader, original_test_loader = get_synthetic_engraving_loaders(
        batch_size=params.batch_size,
        nb_signatures_per_code=params.nb_signs,
        num_workers=params.num_workers,
    )
    test_loader = DataLoader(
        original_test_loader.dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.eval_num_workers,
        collate_fn=original_test_loader.collate_fn,
    )

    model = PretrainedWithRetrieval(params)
    experiment_name = args.artifact_name or build_experiment_name(params)
    artifacts_root = os.path.abspath(args.artifacts_root)
    logger = CSVLogger(os.path.join(artifacts_root, "logs"), name=experiment_name)

    ckpt_dir = os.path.join(artifacts_root, "checkpoints", experiment_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    if not args.test_only and not args.resume_from_checkpoint:
        for fname in os.listdir(ckpt_dir):
            if fname.startswith("last") or fname.startswith("best") or fname in {"test_metrics.csv", ".training_complete"}:
                os.remove(os.path.join(ckpt_dir, fname))

    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="best",
            monitor="val/mAP/emb",
            mode="max",
            save_top_k=1,
            save_last=True,
            enable_version_counter=False,
        ),
        EarlyStopping(
            monitor="val/mAP/emb",
            mode="max",
            patience=args.early_stopping_patience,
            verbose=True,
        ),
    ]

    trainer = pl.Trainer(
        accelerator=params.accelerator,
        devices=params.devices,
        max_epochs=params.epochs,
        log_every_n_steps=args.log_every_n_steps,
        callbacks=callbacks,
        logger=False if args.test_only else logger,
    )

    if not args.test_only:
        trainer.fit(
            model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=args.resume_from_checkpoint,
        )
        Path(ckpt_dir, ".training_complete").touch()
    best_checkpoint = os.path.join(ckpt_dir, "best.ckpt")
    if not os.path.exists(best_checkpoint):
        raise FileNotFoundError(best_checkpoint)
    test_results = trainer.test(model, dataloaders=test_loader, ckpt_path=best_checkpoint)
    if test_results:
        output_path = os.path.join(ckpt_dir, "test_metrics.csv")
        with open(output_path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(test_results[0]))
            writer.writeheader()
            writer.writerow(test_results[0])


if __name__ == "__main__":
    main()
