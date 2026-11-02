#!/usr/bin/env python3
import argparse
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from parameters import SSLParams
from lightning_module import (
    PretrainedWithLinearProbing,
    PretrainedWithRetrieval,
)

from dataloaders.colored_mnist import get_colored_mnist_loaders, get_mnist_loaders
from dataloaders.celeba import get_celeba_loaders
from dataloaders.celeba_intersectional import get_celeba_intersectional_loaders
from dataloaders.synthetic_engraving import get_synthetic_engraving_loaders
from other.birds import get_waterbirds_loaders
from other.fairface import get_fairface_loaders


# =========================================================
# ARGPARSE
# =========================================================
def build_argparser():
    p = argparse.ArgumentParser("Pretraining launcher")

    # -----------------------------
    # Dataset
    # -----------------------------
    p.add_argument(
        "--name_data",
        type=str,
        default="colorMNIST",
        choices=["MNIST", "colorMNIST", "birds", "fairface", "celeba", "celeba_intersectional", "synthetic_engraving"],
    )
    p.add_argument("--root_data", type=str, default="./data")
    p.add_argument("--experiment", type=str, default=None)
    p.add_argument(
        "--run_tag",
        type=str,
        default="",
        help="Optional artifact-only tag to avoid sharing logs/checkpoints between runs.",
    )
    p.add_argument(
        "--artifacts_root",
        type=str,
        default=".",
        help="Root directory containing logs/ and checkpoints/ (default: repository root).",
    )
    p.add_argument(
        "--artifact_name",
        type=str,
        default=None,
        help="Optional explicit logs/checkpoints name. Useful for structured experiment suites.",
    )
    p.add_argument("--resume_from_checkpoint", default=None)

    # -----------------------------
    # Training
    # -----------------------------
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)

    # -----------------------------
    # SSL / Loss
    # -----------------------------
    p.add_argument("--loss_name", type=str, default="uow", choices=["uow", "uow_intersectional", "ow", "infonce", "lejepa", "endunsupervised", "fscl+", "fscl*", "crossentropy", "ce"])
    p.add_argument("--reg_name", type=str, default="mmd")
    p.add_argument("--kernel_name", type=str, default="heat")
    p.add_argument("--temperature", type=float, default=4 / 64) #use 3/32 for colorMNIST
    p.add_argument("--ell", type=int, default=2)
    p.add_argument("--lambda_reg", type=float, default=0.5, help="Regularization weight in [0, 1].")
    p.add_argument(
        "--lambda_lejepa",
        type=float,
        default=None,
        help="Deprecated explicit LeJEPA override; by default LeJEPA uses --lambda_reg.",
    )
    p.add_argument(
        "--lambda_end",
        type=float,
        default=None,
        help="Deprecated explicit EnD override; by default EnD uses --lambda_reg.",
    )
    p.add_argument("--alpha", type=float, default=None, help="Fraction of UOW regularization assigned to OWReg, in [0, 1].")
    p.add_argument("--lambda_uow_reg", type=float, default=None, help="Deprecated alias for --alpha.")
    p.add_argument(
        "--intersectional_grouping",
        choices=["male", "young", "marginal", "joint", "all"],
        default="all",
        help="Grouping used by the dedicated CelebA intersectional UOW loss.",
    )
    p.add_argument("--min_group_size", type=int, default=2)
    p.add_argument("--online_linear_probing", action="store_true", help="Enable online linear probing")    
    p.add_argument("--nb_signs", type=int, default=256)
    
    # -----------------------------
    # Lightning
    # -----------------------------
    p.add_argument("--accelerator", type=str, default="auto")
    p.add_argument("--devices", type=str, default="auto")
    p.add_argument("--proj_out_dim", type=int, default=64)
    return p

# =========================================================
# LOADERS
# =========================================================
def get_loaders(params):
    if params.name_data == "colorMNIST":
        return get_colored_mnist_loaders(
            root=params.root_data,
            batch_size=params.batch_size,
            num_workers=params.num_workers,
        )

    elif params.name_data == "MNIST":
        return get_mnist_loaders(
            root=params.root_data,
            batch_size=params.batch_size,
            num_workers=params.num_workers,
        )

    elif params.name_data == "celeba":
        return get_celeba_loaders(
            root=params.root_data,
            batch_size=params.batch_size,
            num_workers=params.num_workers,
            experiment=params.experiment,
        )

    elif params.name_data == "celeba_intersectional":
        return get_celeba_intersectional_loaders(
            root=params.root_data,
            batch_size=params.batch_size,
            num_workers=params.num_workers,
        )

    elif params.name_data == "birds":
        return get_waterbirds_loaders(
            root=params.root_data,
            batch_size=params.batch_size,
            num_workers=params.num_workers,
        )

    elif params.name_data == "fairface":
        return get_fairface_loaders(
            root=params.root_data,
            batch_size=params.batch_size,
            num_workers=params.num_workers,
        )

    elif params.name_data == "synthetic_engraving":
        return get_synthetic_engraving_loaders(
            batch_size=params.batch_size,
            num_workers=params.num_workers,
            nb_signatures_per_code=params.nb_signs,
        )

    else:
        raise ValueError(f"Unknown dataset {params.name_data}")


# =========================================================
# MODEL
# =========================================================
def get_model(params):
    if params.name_data == "synthetic_engraving":
        return PretrainedWithRetrieval(params)
    else:
        return PretrainedWithLinearProbing(params)


def build_experiment_name(params):
    tag = f"{params.run_tag}-" if params.run_tag else ""
    return (
        f"{params.name_data}-"
        f"{params.experiment or 'default'}-"
        f"{params.loss_name}-"
        f"{params.epochs}-"
        f"{params.proj_out_dim}-"
        f"{params.lr}-"
        f"{params.weight_decay}-"
        f"{params.lambda_reg}-"
        f"{tag}"
        f"seed={params.seed}"
    )

# =========================================================
# MAIN
# =========================================================
def main():
    args = build_argparser().parse_args()

    # Init params
    params = SSLParams()

    # Override params
    for k, v in vars(args).items():
        if hasattr(params, k):
            setattr(params, k, v)

    # Devices parsing
    if params.devices != "auto":
        params.devices = [int(params.devices)] #type: ignore

    pl.seed_everything(params.seed, workers=True)

    # Loaders
    train_loader, val_loader, _ = get_loaders(params) # type: ignore
    # train_loader, _, val_loader = get_loaders(params) # type: ignore

    # Model
    model = get_model(params)

    # Experiment name
    experiment_name = args.artifact_name or build_experiment_name(params)
    artifacts_root = os.path.abspath(args.artifacts_root)

    # Logger
    logger = CSVLogger(os.path.join(artifacts_root, "logs"), name=experiment_name)

    # Checkpoint: keep only the last checkpoint for this experiment.
    ckpt_dir = os.path.join(artifacts_root, "checkpoints", experiment_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Remove old last/best checkpoints from previous runs.
    if not args.resume_from_checkpoint:
        for fname in os.listdir(ckpt_dir):
            if fname.startswith("last") or fname.startswith("best"):
                os.remove(os.path.join(ckpt_dir, fname))

    ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="last",
        save_top_k=0,
        save_last=True,
        every_n_epochs=1,
    )
    callbacks = ckpt_cb
  
    # Trainer
    trainer = pl.Trainer(
        accelerator=params.accelerator,
        devices=params.devices,
        max_epochs=params.epochs,
        log_every_n_steps=50,
        callbacks=callbacks, # 
        logger=logger,
    )

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=args.resume_from_checkpoint,
    )

if __name__ == "__main__":
    main()
