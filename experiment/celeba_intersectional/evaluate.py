#!/usr/bin/env python3
"""Frozen linear-probe evaluation for CelebA Male x Young intersections."""

import argparse
import csv
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dataloaders.celeba_intersectional import decode_intersectional_bias, get_celeba_intersectional_loaders
from lightning_module import PretrainedWithLinearProbing
from parameters import SSLParams
from run_eval import compute_equalized_odds, compute_target_acc, compute_wga, extract_features


METRIC_FIELDS = [
    "target_acc",
    "eo_male",
    "eo_young",
    "eo_joint",
    "wga_joint",
    "bias_bal_acc_male",
    "bias_bal_acc_young",
    "bias_bal_acc_joint",
]


def _linear_probe(C):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, solver="lbfgs", C=C),
    )


def _group_rows(y_true, y_pred, joint_bias):
    rows = []
    for group in sorted(np.unique(joint_bias)):
        mask = joint_bias == group
        group_y = y_true[mask]
        group_pred = y_pred[mask]
        positive = group_y == 1
        negative = group_y == 0
        rows.append({
            "joint_group": int(group),
            "male": int(group) // 2,
            "young": int(group) % 2,
            "count": int(mask.sum()),
            "accuracy": compute_target_acc(group_y, group_pred),
            "tpr": float((group_pred[positive] == 1).mean()) if positive.any() else float("nan"),
            "fpr": float((group_pred[negative] == 1).mean()) if negative.any() else float("nan"),
        })
    return rows


def evaluate(args):
    device = torch.device(f"cuda:{args.devices}" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    params = SSLParams()
    for key, value in checkpoint["hyper_parameters"].items():
        if hasattr(params, key):
            setattr(params, key, value)
    params.name_data = "celeba_intersectional"
    params.root_data = args.root_data
    params.batch_size = args.batch_size
    params.num_workers = args.num_workers

    model = PretrainedWithLinearProbing(params)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    train_loader, _, test_loader = get_celeba_intersectional_loaders(
        root=args.root_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        eval_mode=True,
    )

    feature_path = Path(args.feature_path)
    if args.no_recompute_features and feature_path.exists():
        features = torch.load(feature_path, map_location="cpu")
        emb_train, y_train, b_train = (features["train"][key] for key in ("emb", "y", "b"))
        emb_test, y_test, b_test = (features["test"][key] for key in ("emb", "y", "b"))
    else:
        emb_train, y_train, b_train = extract_features(model, train_loader, device)
        emb_test, y_test, b_test = extract_features(model, test_loader, device)
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "train": {"emb": emb_train, "y": y_train, "b": b_train},
                "test": {"emb": emb_test, "y": y_test, "b": b_test},
            },
            feature_path,
        )

    X_train = emb_train.numpy()
    X_test = emb_test.numpy()
    y_train_np = y_train.numpy()
    y_test_np = y_test.numpy()
    joint_train = b_train.numpy()
    joint_test = b_test.numpy()
    components_train = decode_intersectional_bias(b_train).numpy()
    components_test = decode_intersectional_bias(b_test).numpy()

    target_probe = _linear_probe(args.C)
    target_probe.fit(X_train, y_train_np)
    target_pred = target_probe.predict(X_test)

    bias_predictions = {}
    bias_targets = {
        "male": (components_train[:, 0], components_test[:, 0]),
        "young": (components_train[:, 1], components_test[:, 1]),
        "joint": (joint_train, joint_test),
    }
    for name, (train_labels, test_labels) in bias_targets.items():
        probe = _linear_probe(args.C)
        probe.fit(X_train, train_labels)
        bias_predictions[name] = probe.predict(X_test)

    metrics = {
        "target_acc": compute_target_acc(y_test_np, target_pred),
        "eo_male": compute_equalized_odds(y_test_np, target_pred, components_test[:, 0]),
        "eo_young": compute_equalized_odds(y_test_np, target_pred, components_test[:, 1]),
        "eo_joint": compute_equalized_odds(y_test_np, target_pred, joint_test),
        "wga_joint": compute_wga(y_test_np, target_pred, joint_test),
        "bias_bal_acc_male": balanced_accuracy_score(components_test[:, 0], bias_predictions["male"]),
        "bias_bal_acc_young": balanced_accuracy_score(components_test[:, 1], bias_predictions["young"]),
        "bias_bal_acc_joint": balanced_accuracy_score(joint_test, bias_predictions["joint"]),
    }

    output_path = Path(args.metrics_output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        writer.writerow(metrics)

    group_path = output_path.with_name(f"{output_path.stem}_groups.csv")
    group_rows = _group_rows(y_test_np, target_pred, joint_test)
    with group_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(group_rows[0].keys()))
        writer.writeheader()
        writer.writerows(group_rows)

    print(f"Saved features to {feature_path}")
    print(f"Saved metrics to {output_path}")
    print(f"Saved intersectional group rates to {group_path}")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    return metrics


def main():
    parser = argparse.ArgumentParser("Evaluate the CelebA Male x Young experiment")
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--root_data", default="./data")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--devices", type=int, default=0)
    parser.add_argument("--C", type=float, default=0.01)
    parser.add_argument("--metrics_output_csv", required=True)
    parser.add_argument("--feature_path", required=True)
    parser.add_argument("--no-recompute-features", action="store_true")
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
