#%%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

import numpy as np
import torch
import torch.nn as nn


SIGREG_GAMMA = 0.5


def _rho_d_normalization_constant(d: int) -> float:
    from scipy.special import gammaln

    log_c = gammaln(d / 2.0) - 0.5 * math.log(math.pi) - gammaln((d - 1) / 2.0)
    return math.exp(log_c)


def _gauss_jacobi_quadrature_for_rho_d(d: int, n_quad: int):
    from scipy.special import roots_jacobi

    alpha = (d - 3.0) / 2.0
    nodes, jacobi_weights = roots_jacobi(n_quad, alpha, alpha)
    weights = _rho_d_normalization_constant(d) * jacobi_weights
    return nodes.astype(np.float64), weights.astype(np.float64)


class SIGRegInducedKernel(nn.Module):
    """
    Radial KerJEPA kernel induced by integrating the SIGReg slice direction.

    Starting from the sliced Gaussian kernel

        k_A(x, y) = exp(-0.5 * (A^T (x - y))^2),
        A ~ Unif(S^{d-1}),

    the direction-integrated kernel is

        kappa_d(x, y) = E_A[k_A(x, y)]
                      = integral exp(-0.5 * d * (||x-y||^2 / d) * u^2) rho_d(u) du,

    where rho_d is the first-coordinate density of a uniform sphere point.

    The forward API takes r2_over_d = ||x-y||^2 / d directly. It does not take
    raw Euclidean distances and it does not square its input.
    """

    def __init__(self, d: int, n_quad: int = 64):
        super().__init__()
        self.d = int(d)
        self.n_quad = int(n_quad)

        nodes, weights = _gauss_jacobi_quadrature_for_rho_d(self.d, self.n_quad)
        self.register_buffer("nodes", torch.from_numpy(nodes.copy()))
        self.register_buffer("weights", torch.from_numpy(weights.copy()))

    def forward(self, r2_over_d: torch.Tensor) -> torch.Tensor:
        """
        r2_over_d : (...,) tensor containing ||x-y||^2 / d.
        returns   : (...,) tensor containing kappa_d(x, y).
        """
        nodes = self.nodes.to(device=r2_over_d.device, dtype=r2_over_d.dtype)
        weights = self.weights.to(device=r2_over_d.device, dtype=r2_over_d.dtype)
        vals = torch.exp(-SIGREG_GAMMA * self.d * r2_over_d.unsqueeze(-1) * nodes.square()) # type: ignore
        return (vals * weights).sum(dim=-1) # type: ignore


class SIGRegInduced(nn.Module):
    """
    KerJEPA / direction-integrated MMD to N(0, I).

    The kernel is evaluated through SIGRegInducedKernel.forward only. Its input
    convention is always squared Euclidean distance divided by d:

        r2_over_d = ||x-y||^2 / d.

    The P-Q term uses the same forward API after analytically integrating the
    Gaussian target Y ~ N(0, sigma2 I).
    """

    def __init__(
        self,
        d: int,
        sigma2: float = 1.0,
        n_quad: int = 64,
    ):
        super().__init__()
        self.kernel = SIGRegInducedKernel(d=d, n_quad=n_quad)
        self.d = int(d)
        sigma2 = float(sigma2)
        denom = 1.0 + 2.0 * SIGREG_GAMMA * sigma2
        self.pq_prefactor = denom ** -0.5
        self.pq_kernel_scale = 1.0 / denom
        self.qq = (1.0 + 4.0 * SIGREG_GAMMA * sigma2) ** -0.5
        self.sigma2 = sigma2

    @staticmethod
    def pairwise_r2(X: torch.Tensor, Y: torch.Tensor | None = None) -> torch.Tensor:
        if Y is None:
            Y = X
        x2 = X.square().sum(dim=-1, keepdim=True)
        y2 = Y.square().sum(dim=-1, keepdim=True).T
        return (x2 + y2 - 2.0 * (X @ Y.T)).clamp_min(0.0)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        B = X.size(0)
        gram = self.kernel(self.pairwise_r2(X) / self.d)

        pp = gram.mean()

        # E_Y k(X,Y) = prefactor * k_effective(||X||^2 / d), Y ~ N(0, sigma2 I).
        pq = self.pq_prefactor * self.kernel(self.pq_kernel_scale * X.square().sum(dim=-1) / self.d).mean()
        qq = torch.as_tensor(self.qq, device=X.device, dtype=X.dtype)

        mmd2 = pp - 2.0 * pq + qq
        return mmd2


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    torch.manual_seed(0)
    D = 64
    sigma2 = 1.0

    kerjepa = SIGRegInduced(d=D, sigma2=sigma2, n_quad=96).to(dtype=torch.float64)

    print("KerJEPA SIGReg-induced radial kernel plot")
    print(f"D={D}, gamma={SIGREG_GAMMA}, sigma2={sigma2}")

    r2_over_d_grid = torch.linspace(0.0, 10.0, 800, dtype=torch.float64)
    kernel_values = kerjepa.kernel(r2_over_d_grid).detach().cpu()

    n_pairs = 100
    pair_seed = 321
    pair_generator = torch.Generator().manual_seed(pair_seed)
    r2_over_d = 10.0 * torch.rand(n_pairs, dtype=torch.float64, generator=pair_generator)
    directions = torch.randn(n_pairs, D, dtype=torch.float64, generator=pair_generator)
    directions = directions / directions.norm(dim=-1, keepdim=True)
    X_pairs = torch.randn(n_pairs, D, dtype=torch.float64, generator=pair_generator)
    Y_pairs = X_pairs + (D * r2_over_d).sqrt().unsqueeze(-1) * directions

    measured_r2_over_d = kerjepa.pairwise_r2(X_pairs, Y_pairs).diag() / D
    distance_grid_err = (measured_r2_over_d - r2_over_d).abs().max().item()
    exact_pair_kernel = kerjepa.kernel(measured_r2_over_d)

    plt.figure(figsize=(7, 4))
    plt.plot(r2_over_d_grid.cpu(), kernel_values, linewidth=2.0, label="kernel")
    plt.scatter(
        measured_r2_over_d.cpu(),
        exact_pair_kernel.detach().cpu(),
        s=14,
        alpha=0.6,
        color="red",
        marker="x",
        linewidths=0.7,
        label="10k random pairs",
    )
    plt.xlabel(r"$\|x-y\|_2^2 / d$")
    plt.ylabel(r"$\kappa_d(x,y)$")
    plt.title(f"SIGReg-induced KerJEPA kernel, d={D}, gamma={SIGREG_GAMMA}")
    plt.xlim(0.0, 10.0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plot_path = "sigreginduced_kernel.png"
    plt.savefig(plot_path, dpi=180)
    plt.close()

    print(
        "10k random-pair scatter "
        f"r2/d in [{measured_r2_over_d.min().item():.3f}, {measured_r2_over_d.max().item():.3f}] | "
        f"distance_err={distance_grid_err:.3e} | saved={plot_path}"
    )

# %%
