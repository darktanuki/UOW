#!/usr/bin/env python3
import argparse
import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch
# import torch.multiprocessing as mp
# mp.set_sharing_strategy('file_system')
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import torch.nn.functional as F

from dataloaders.celeba import get_celeba_loaders
from dataloaders.celeba_intersectional import get_celeba_intersectional_loaders
from dataloaders.colored_mnist import get_colored_mnist_loaders, get_mnist_loaders
from other.birds import get_waterbirds_loaders
from other.fairface import get_fairface_loaders
from lightning_module import PretrainedWithLinearProbing, PretrainedWithRetrieval
from parameters import SSLParams


def get_loaders(params):
    if params.name_data == "colorMNIST":
        return get_colored_mnist_loaders(root=params.root_data, batch_size=params.batch_size, eval_mode=True)
    if params.name_data == "MNIST":
        return get_mnist_loaders(root=params.root_data, batch_size=params.batch_size, eval_mode=True)
    if params.name_data == "celeba":
        return get_celeba_loaders(
            root=params.root_data,
            batch_size=params.batch_size,
            num_workers=params.num_workers,
            eval_mode=True,
            experiment=params.experiment,
        )
    if params.name_data == "celeba_intersectional":
        return get_celeba_intersectional_loaders(
            root=params.root_data,
            batch_size=params.batch_size,
            num_workers=params.num_workers,
            eval_mode=True,
        )
    if params.name_data == "birds":
        return get_waterbirds_loaders(
            root=params.root_data,
            batch_size=params.batch_size,
            num_workers=params.num_workers,
            eval_mode=True,
        )
    if params.name_data == "fairface":
        return get_fairface_loaders(
            root=params.root_data,
            batch_size=params.batch_size,
            num_workers=params.num_workers,
            eval_mode=True,
        )
    raise ValueError(f"Unknown dataset {params.name_data}")


def get_model(params):
    if params.name_data == "synthetic_engraving":
        return PretrainedWithRetrieval(params)
    return PretrainedWithLinearProbing(params)


@torch.no_grad()
def extract_features(model, loader, device):
    embs, labels, bias = [], [], []
    model = model.to(device).eval()

    for x, b, y in tqdm(loader, desc="Extracting features", leave=False):
        if isinstance(x, (list, tuple)):
            x = x[0]
        x = x.to(device)
        emb = model.f(x)
        # emb = F.normalize(emb, dim=-1)
        embs.append(emb.cpu())
        labels.append(y.cpu())
        bias.append(b.cpu())

    return torch.cat(embs), torch.cat(labels), torch.cat(bias)


_PAPER_COLORS = [
    "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00",
    "#56B4E9", "#F0E442", "#000000", "#882255", "#44AA99",
    "#AA4499", "#117733", "#999933", "#332288", "#DDCC77",
    "#88CCEE", "#CC6677", "#6699CC", "#661100", "#999999",
]


def _as_numpy_1d(values):
    return np.asarray(values).reshape(-1)


def _class_name(value):
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    return str(value)


def _fit_tsne_embedding(X, max_points=4000, seed=0, perplexity=30):
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"Expected a 2D feature matrix for t-SNE, got shape {X.shape}.")

    rng = np.random.default_rng(seed)
    if max_points is not None and max_points > 0 and X.shape[0] > max_points:
        idx = np.sort(rng.choice(X.shape[0], size=max_points, replace=False))
        X = X[idx]
    else:
        idx = np.arange(X.shape[0])

    X = StandardScaler().fit_transform(X)
    if X.shape[1] > 50 and X.shape[0] > 2:
        n_components = min(50, X.shape[1], X.shape[0] - 1)
        X = PCA(n_components=n_components, random_state=seed).fit_transform(X)

    if X.shape[0] < 3:
        raise ValueError("t-SNE requires at least 3 samples.")
    effective_perplexity = min(float(perplexity), max(1.0, (X.shape[0] - 1) / 3))
    tsne = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        init="pca",
        learning_rate="auto",
        max_iter=1000,
        metric="euclidean",
        random_state=seed,
    )
    X_2d = tsne.fit_transform(X)
    X_2d = (X_2d - X_2d.mean(axis=0)) / (X_2d.std(axis=0) + 1e-12)
    return X_2d, idx


def plot_tsne(
    X,
    y,
    filename,
    *,
    max_points=4000,
    seed=0,
    perplexity=30,
    label_names=None,
    label_colors=None,
):
    y = _as_numpy_1d(y)
    X_2d, idx = _fit_tsne_embedding(
        X,
        max_points=max_points,
        seed=seed,
        perplexity=perplexity,
    )
    y_plot = y[idx]
    labels = np.unique(y_plot)
    color_lookup = {
        label: (label_colors[label] if label_colors and label in label_colors else _PAPER_COLORS[i % len(_PAPER_COLORS)])
        for i, label in enumerate(labels)
    }
    colors = [color_lookup[label] for label in y_plot]

    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "legend.fontsize": 8,
            "figure.dpi": 160,
            "savefig.dpi": 400,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    ):
        fig, ax = plt.subplots(figsize=(4.35, 3.45), constrained_layout=True)
        ax.scatter(
            X_2d[:, 0],
            X_2d[:, 1],
            c=colors,
            s=10,
            alpha=0.76,
            linewidths=0.18,
            edgecolors="white",
            rasterized=True,
        )
        ax.set_axis_off()
        ax.set_aspect("equal", adjustable="datalim")

        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                label=label_names[label] if label_names and label in label_names else _class_name(label),
                markerfacecolor=color_lookup[label],
                markeredgecolor="white",
                markeredgewidth=0.4,
                markersize=5.5,
            )
            for label in labels
        ]
        ax.legend(
            handles=handles,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            ncol=1,
            frameon=False,
            handletextpad=0.35,
            borderaxespad=0.0,
        )

        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", pad_inches=0.03)
        fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
        plt.close(fig)


def _safe_rate(num, den):
    return float(num) / float(den) if den > 0 else 0.0


def compute_target_acc(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float((y_true == y_pred).mean())


def compute_wga(y_true, y_pred, sensitive):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    sensitive = np.asarray(sensitive)

    worst_group_acc = 1.0
    for label in np.unique(y_true):
        for bias in np.unique(sensitive):
            mask = (y_true == label) & (sensitive == bias)
            if mask.any():
                group_acc = float((y_pred[mask] == y_true[mask]).mean())
                worst_group_acc = min(worst_group_acc, group_acc)
    return worst_group_acc


def compute_equalized_odds(y_true, y_pred, sensitive):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    sensitive = np.asarray(sensitive)

    groups = np.sort(np.unique(sensitive))
    classes = np.sort(np.unique(np.concatenate([y_true, y_pred])))
    if groups.size < 2:
        return 0.0

    gaps = []
    for true_class in classes:
        true_mask = y_true == true_class
        for pred_class in classes:
            rates = []
            for group in groups:
                group_true_mask = true_mask & (sensitive == group)
                den = np.sum(group_true_mask)
                if den == 0:
                    continue
                rates.append(_safe_rate(np.sum((y_pred == pred_class) & group_true_mask), den))

            for i in range(len(rates)):
                for j in range(i + 1, len(rates)):
                    gaps.append(abs(rates[i] - rates[j]))

    return float(np.mean(gaps)) if gaps else 0.0


def compute_metrics_dict(y_true, y_pred, b_true, b_pred):
    return {
        "target_acc": compute_target_acc(y_true, y_pred),
        "wga": compute_wga(y_true, y_pred, b_true),
        "bias_acc": compute_target_acc(b_true, b_pred),
        "eo": compute_equalized_odds(y_true, y_pred, b_true),
    }


def build_linear_probe():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            # multi_class="auto",
        ),
    )


def build_knn(n_neighbors=5):
    return make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=n_neighbors),
    )


def evaluate_frozen_representations(emb_train, y_train, b_train, emb_val, y_val, b_val, C=1.0):
    X_train = emb_train.numpy()
    X_val = emb_val.numpy()
    y_train_np = y_train.numpy()
    y_val_np = y_val.numpy()
    b_train_np = b_train.numpy()
    b_val_np = b_val.numpy()
    def build_linear_probe_with_C(C_val):
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, solver="lbfgs", C=C_val),
        )

    lp_target = build_linear_probe_with_C(C)
    lp_target.fit(X_train, y_train_np)
    lp_y_pred = lp_target.predict(X_val)
    lp_bias = build_linear_probe_with_C(C)
    lp_bias.fit(X_train, b_train_np)
    lp_b_pred = lp_bias.predict(X_val)

    knn_target = build_knn()
    knn_target.fit(X_train, y_train_np)
    knn_y_pred = knn_target.predict(X_val)

    knn_bias = build_knn()
    knn_bias.fit(X_train, b_train_np)
    knn_b_pred = knn_bias.predict(X_val)

    return {
        "LP": compute_metrics_dict(y_val_np, lp_y_pred, b_val_np, lp_b_pred),
        "k-NN": compute_metrics_dict(y_val_np, knn_y_pred, b_val_np, knn_b_pred),
    }


def append_benchmark_rows(results, model_name, dataset_name, benchmark_csv):
    benchmark_path = Path(benchmark_csv)
    benchmark_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["model", "dataset", "eval_type", "target_acc", "wga", "bias_acc", "eo"]
    file_exists = benchmark_path.exists()

    with benchmark_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        for eval_type, metrics in results.items():
            writer.writerow(
                {
                    "model": model_name,
                    "dataset": dataset_name,
                    "eval_type": eval_type,
                    "target_acc": metrics["target_acc"],
                    "wga": metrics["wga"],
                    "bias_acc": metrics["bias_acc"],
                    "eo": metrics["eo"],
                }
            )


def write_metrics_output(results, output_csv):
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["eval_type", "target_acc", "wga", "bias_acc", "eo"]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for eval_type, metrics in results.items():
            writer.writerow(
                {
                    "eval_type": eval_type,
                    "target_acc": metrics["target_acc"],
                    "wga": metrics["wga"],
                    "bias_acc": metrics["bias_acc"],
                    "eo": metrics["eo"],
                }
            )


def evaluate_checkpoint(args):
    device = torch.device(f"cuda:{args.devices}" if torch.cuda.is_available() else "cpu") 
    ckpt = torch.load(args.checkpoint_path, map_location="cpu")
    params = SSLParams()
    for k, v in ckpt["hyper_parameters"].items():
        if hasattr(params, k):
            setattr(params, k, v)
    params.root_data = args.root_data
    params.batch_size = args.batch_size
    params.num_workers = args.num_workers

    # Legacy checkpoints used an unbounded regularization scale (for example
    # lambda_reg=19.0).  The current losses use coefficients constrained to
    # [0, 1], but the coefficient is irrelevant when evaluating a frozen
    # encoder.  Use a valid construction value without changing checkpoint
    # weights or the extracted representations.
    checkpoint_lambda_reg = getattr(params, "lambda_reg", 0.5)
    if not 0.0 <= checkpoint_lambda_reg <= 1.0:
        print(
            "Legacy checkpoint: replacing lambda_reg="
            f"{checkpoint_lambda_reg} with 0.5 for frozen evaluation only."
        )
        params.lambda_reg = 0.5

    model = get_model(params)
    model.load_state_dict(ckpt["state_dict"], strict=True)

    train_loader, _, test_loader = get_loaders(params)  # type: ignore

    model_name = Path(args.checkpoint_path).parent.name
    os.makedirs(args.plot_dir, exist_ok=True)
    features_path = os.path.join(args.plot_dir, f"{model_name}_features.pt")

    if args.no_recompute_features and os.path.exists(features_path):
        print(f"Loading saved features from {features_path}")
        data = torch.load(features_path, map_location="cpu")
        emb_train = data["train"]["emb"]
        y_train = data["train"]["y"]
        b_train = data["train"]["b"]
        emb_test = data["test"]["emb"]
        y_test = data["test"]["y"]
        b_test = data["test"]["b"]
    else:
        print("Extracting frozen features...")
        emb_train, y_train, b_train = extract_features(model, train_loader, device)
        emb_test, y_test, b_test = extract_features(model, test_loader, device)

        torch.save(
            {
                "train": {"emb": emb_train, "y": y_train, "b": b_train},
                "test": {"emb": emb_test, "y": y_test, "b": b_test},
            },
            features_path,
        )

    results = evaluate_frozen_representations(emb_train, y_train, b_train, emb_test, y_test, b_test, C=args.C)

    classes_path = os.path.join(args.plot_dir, f"{model_name}_tsne_labels.png")
    bias_path = os.path.join(args.plot_dir, f"{model_name}_tsne_bias.png")
    saved_tsne = False
    if params.name_data.lower() == "colormnist":
        plot_tsne(
            emb_test.numpy(),
            y_test.numpy(),
            classes_path,
            max_points=args.tsne_max_points,
            seed=args.tsne_seed,
            perplexity=args.tsne_perplexity,
            label_names={i: str(i) for i in range(10)},
        )
        plot_tsne(
            emb_test.numpy(),
            b_test.numpy(),
            bias_path,
            max_points=args.tsne_max_points,
            seed=args.tsne_seed,
            perplexity=args.tsne_perplexity,
            label_names={0: "red", 1: "green"},
            label_colors={0: "red", 1: "green"},
        )
        saved_tsne = True

    append_benchmark_rows(results, model_name=model_name, dataset_name=params.name_data, benchmark_csv=args.benchmark_csv)

    if args.metrics_output_csv is not None:
        write_metrics_output(results, args.metrics_output_csv)

    print(f"Saved features: {features_path}")
    if saved_tsne:
        print(f"Saved t-SNE plots: {classes_path} / {bias_path}")
        print(f"Saved t-SNE PDFs: {Path(classes_path).with_suffix('.pdf')} / {Path(bias_path).with_suffix('.pdf')}")
    else:
        print(f"Skipped t-SNE plots: only generated for colorMNIST, got {params.name_data}")
    print("Frozen evaluation results:")
    for eval_type, metrics in results.items():
        print(
            f"{eval_type}"
            f" → Target Acc={metrics['target_acc']:.4f}"
            f" WGA={metrics['wga']:.4f}"
            f" Bias Acc={metrics['bias_acc']:.4f}"
            f" EO={metrics['eo']:.4f}"
        )

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--root_data", type=str, default="./data")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--devices", type=int, default=0)
    parser.add_argument("--tsne_dir", "--umap_dir", dest="plot_dir", type=str, default="tsne_plots", help="Directory for saved features and t-SNE plots.")
    parser.add_argument("--metrics_output_csv", type=str, default=None)
    parser.add_argument("--benchmark_csv", type=str, default="results/results_benchmark.csv")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--no-recompute-features", dest="no_recompute_features", action="store_true", help="If set, load saved features from the plot directory and skip recomputing them if present.")
    parser.add_argument("--C", type=float, default=1.0, help="Regularization strength for linear probe (inverse of regularization).")
    parser.add_argument("--tsne_max_points", "--umap_max_points", dest="tsne_max_points", type=int, default=4000, help="Maximum number of test samples shown in each t-SNE plot; set <=0 to use all samples.")
    parser.add_argument("--tsne_seed", "--umap_seed", dest="tsne_seed", type=int, default=0, help="Random seed for t-SNE subsampling and initialization.")
    parser.add_argument("--tsne_perplexity", type=float, default=30.0, help="Requested t-SNE perplexity; automatically reduced for small datasets.")

    evaluate_checkpoint(parser.parse_args())


if __name__ == "__main__":
    main()
