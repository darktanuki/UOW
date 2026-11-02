#%%

#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import math
import inspect

def generic_compare(models, B, d, seed, device, title, yscale=True):
    """
    Generic comparison and plotting utility shared by sliced, non-sliced and RF methods.
    """
    import matplotlib.pyplot as plt
    torch.manual_seed(seed)

    def eval_models(z):
        z = z.to(device)
        out = {}
        for name, model in models.items():
            with torch.no_grad():
                try:
                    # case 1: regular nn.Module or callable taking z
                    val = model(z)
                except TypeError:
                    # case 2: builder expecting (L, seed)
                    try:
                        built = model(256, seed)
                    except TypeError:
                        built = model()
                    val = built(z)
            if isinstance(val, torch.Tensor):
                val = val.detach()
            out[name] = float(val)
        return out

    results = {}

    # Uniform on sphere
    z = torch.randn(B, d)
    z = z / z.norm(dim=-1, keepdim=True)
    results["uniform"] = eval_models(z)

    # Dirac
    z = torch.zeros(B, d)
    z[:, 0] = 1.0
    z = z / z.norm(dim=-1, keepdim=True)
    results["dirac"] = eval_models(z)

    # Correlated
    z = torch.randn(B, d)
    z[:, 0] *= 10
    z[:, 1] *= 5
    z = z / z.norm(dim=-1, keepdim=True)
    results["correlated"] = eval_models(z)

    # Low rank
    z = torch.randn(B, 4) @ torch.randn(4, d)
    z = z / z.norm(dim=-1, keepdim=True)
    results["low_rank"] = eval_models(z)

    # Two clusters
    c1 = torch.randn(1, d)
    c2 = torch.randn(1, d)
    c1 = c1 / c1.norm(dim=-1, keepdim=True)
    c2 = c2 / c2.norm(dim=-1, keepdim=True)

    z = torch.cat([
        c1 + 0.05 * torch.randn(B // 2, d),
        c2 + 0.05 * torch.randn(B - B // 2, d)
    ], dim=0)
    z = z / z.norm(dim=-1, keepdim=True)

    results["two_clusters"] = eval_models(z)

    print(f"\n=== {title} ===\n")
    for name, res in results.items():
        print(f"{name:12s} | " +
              " | ".join([f"{k}={v:.4e}" for k, v in res.items()]))

    labels = list(results.keys())
    method_names = list(models.keys())

    plt.figure(figsize=(10, 5))
    x = np.arange(len(labels))
    width = 0.8 / len(method_names)

    for i, m in enumerate(method_names):
        vals = [results[k][m] for k in labels]
        plt.bar(x + i * width, vals, width, label=m)

    plt.xticks(x + width * (len(method_names) / 2), labels, rotation=20)
    plt.ylabel("statistic value")
    plt.title(title)
    plt.grid(alpha=0.3)

    if yscale:
        plt.yscale("log")
        plt.ylim(1e-8, 2e0)

    plt.legend()
    plt.tight_layout()
    plt.show()

    return results

def generic_degeneracy_core(models_builders, B_values, num_trials, device, title):
    """
    Core generic implementation for degeneracy scaling.
    This function contains all the shared logic and is reused by
    RF, sliced and non-sliced variants.
    """

    import matplotlib.pyplot as plt

    if B_values is None:
        B_values = [32, 64, 128, 256, 512]

    d = 256

    all_variances = {name: [] for name in models_builders.keys()}
    all_means = {name: [] for name in models_builders.keys()}

    for B in B_values:
        for name, ctor in models_builders.items():
            stats = []

            for s in range(num_trials):
                torch.manual_seed(s)

                z = torch.randn(B, d)
                z = z / z.norm(dim=-1, keepdim=True)
                z = z.to(device)

                # Accept either a builder (callable) or a ready nn.Module
                if isinstance(ctor, nn.Module):
                    model = ctor  # use the provided model instance
                else:
                    try:
                        params = inspect.signature(ctor).parameters
                        if 'seed' in params:
                            model = ctor(seed=s)
                        else:
                            model = ctor()
                    except (TypeError, ValueError):
                        # Fallback: try without seed
                        model = ctor()

                with torch.no_grad():
                    val = model(z).item()

                stats.append(val)

            all_variances[name].append(np.var(stats))
            all_means[name].append(np.mean(np.abs(stats)))

    print(f"\n{title}")
    print("B values:", B_values)
    for name, vars in all_variances.items():
        print(f"{name:20s} :", vars)

    plt.figure(figsize=(7, 5))

    for name, vars in all_variances.items():
        plt.loglog(B_values, vars, marker="o", label=name)

    first = list(models_builders.keys())[0]

    ref1 = [all_variances[first][0] * (B_values[0] / B) for B in B_values]
    ref2 = [all_variances[first][0] * (B_values[0] / B) ** 2 for B in B_values]

    plt.loglog(B_values, ref1, "--", color="gray", label="1/B reference")
    plt.loglog(B_values, ref2, ":", color="black", label="1/B^2 reference")

    plt.xlabel("batch size B")
    plt.ylabel("empirical variance")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ---- Additional plot: mean and variance ----
    plt.figure(figsize=(7, 5))

    for name in models_builders.keys():
        plt.loglog(B_values, all_means[name], marker="s", linestyle="--", label=f"{name} mean")
        plt.loglog(B_values, all_variances[name], marker="o", linestyle="-", label=f"{name} variance")

    plt.xlabel("batch size B")
    plt.ylabel("statistic value")
    plt.title(title + " (mean & variance)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(all_means)
    return all_variances

def generic_variance_vs_L(models_builders, B, d, L_values, num_trials, device, title, x_label):
    """
    Core generic implementation shared by RF and sliced methods.
    Computes empirical variance as a function of L for a fixed batch size B.
    """

    import matplotlib.pyplot as plt

    all_variances = {name: [] for name in models_builders.keys()}

    print(f"\n{title}\n")

    for L in L_values:
        print(f"Computing for L = {L} ...")

        for name, ctor in models_builders.items():
            stats = []
            ctor.num_projections = L
            for s in range(num_trials):
                torch.manual_seed(s)

                # Sample uniform on sphere
                z = torch.randn(B, d)
                z = z / z.norm(dim=-1, keepdim=True)
                z = z.to(device)

                with torch.no_grad():
                    val = ctor(z).item()

                stats.append(val)

            all_variances[name].append(np.var(stats))

    # ---- Print results ----
    print("\nEmpirical variances:")
    print("L values:", L_values)
    for name, vars in all_variances.items():
        print(f"{name:20s} :", vars)

    # ---- Plot ----
    plt.figure(figsize=(7, 5))

    for name, vars in all_variances.items():
        plt.loglog(L_values, vars, marker="o", label=name)

    # Reference slopes
    first = list(models_builders.keys())[0]

    ref_sqrt = [all_variances[first][0] * (L_values[0] / L) ** 0.5 for L in L_values]
    ref_linear = [all_variances[first][0] * (L_values[0] / L) for L in L_values]

    plt.loglog(L_values, ref_sqrt, "--", color="gray", label="1/sqrt(L) reference")
    plt.loglog(L_values, ref_linear, ":", color="black", label="1/L reference")

    plt.xlabel(x_label)
    plt.ylabel("empirical variance")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return all_variances

# %%
