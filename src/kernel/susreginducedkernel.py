#%%
import math
import torch
import torch.nn as nn
import numpy as np


def _rho_d_normalization_constant(d: int) -> float:
    """Return C_d such that rho_d(t) = C_d (1 - t^2)^((d-3)/2) on [-1, 1]."""
    from scipy.special import gammaln

    log_c = gammaln(d / 2.0) - 0.5 * math.log(math.pi) - gammaln((d - 1) / 2.0)
    return math.exp(log_c)


def _gauss_jacobi_quadrature_for_rho_d(d: int, n_quad: int):
    """
    Build Gauss-Jacobi nodes/weights for the spherical projection density rho_d.

    Since
        rho_d(t) = C_d (1 - t^2)^alpha,   alpha = (d-3)/2,
    Gauss-Jacobi with parameters (alpha, alpha) absorbs the whole weight exactly.
    """
    from scipy.special import roots_jacobi

    alpha = (d - 3.0) / 2.0
    t_np, w_jacobi = roots_jacobi(n_quad, alpha, alpha)
    w_np = _rho_d_normalization_constant(d) * w_jacobi
    return t_np.astype(np.float64), w_np.astype(np.float64)


def exact_kernel_from_projections_torch(
    cos_theta: torch.Tensor,
    d: int,
    sigma2: float,
    n_quad: int = 64,
) -> torch.Tensor:
    """
    Differentiable torch version of exact_kernel_from_projections.

    It computes
        K(c) = ∫_{-1}^1 exp(-(1-c)t^2 / sigma2) rho_d(t) dt
    using Gauss-Jacobi quadrature, whose weight matches rho_d(t) up to the
    normalization constant. Autograd flows through the exponential naturally.

    Parameters
    ----------
    cos_theta : torch.Tensor
        Tensor of cosines <x, y> in [-1, 1].
    d : int
        Ambient dimension.
    sigma2 : float
        Gaussian bandwidth.
    n_quad : int
        Number of Gauss-Jacobi nodes.
    """
    if sigma2 <= 0:
        raise ValueError(f"sigma2 must be positive, got {sigma2}")

    t_np, w_np = _gauss_jacobi_quadrature_for_rho_d(int(d), int(n_quad))

    t_quad = torch.as_tensor(t_np, device=cos_theta.device, dtype=cos_theta.dtype)
    w_quad = torch.as_tensor(w_np, device=cos_theta.device, dtype=cos_theta.dtype)

    exponent = -(1.0 - cos_theta).unsqueeze(-1) * t_quad.square() / float(sigma2)
    return (torch.exp(exponent) * w_quad).sum(dim=-1)


class SUSRegInducedKernel(nn.Module):
    """
    Differentiable PyTorch implementation of the projection-averaged Gaussian kernel:

        K(x, y) = E_{a ~ Unif(S^{d-1})}[ exp(-(a^T(x-y))^2 / (2 sigma2)) ]

    Expressed as a 1-D integral via the marginal density rho_d of a^Tu:

        K(c) = integral_{-1}^{1} exp(-(1-c) * t^2 / sigma2) * rho_d(t) dt

    where c = <x, y> in [-1, 1] and rho_d(t) = C_d * (1 - t^2)^((d-3)/2).

    Implementation
    --------------
    Uses Gauss-Jacobi quadrature with weight (1-t^2)^alpha, alpha = (d-3)/2.
    This absorbs rho_d analytically, leaving only the smooth exponential to
    evaluate at the (fixed) quadrature nodes.  The whole computation is a
    dot-product of exp(...) values against precomputed buffers, so gradients
    w.r.t. c (and therefore x, y) flow through autograd without any special
    treatment.

    Analytical derivatives
    ----------------------
    Differentiating under the integral sign:

        K'(c)  = (1/sigma2)   * integral t^2 * exp(...) * rho_d(t) dt
        K''(c) = (1/sigma2)^2 * integral t^4 * exp(...) * rho_d(t) dt

    These use the same exp values with t^2- and t^4-weighted buffers.

    Parameters
    ----------
    d       : int    — ambient dimension (sphere S^{d-1})
    sigma2  : float  — Gaussian bandwidth; defaults to 1/d
    n_quad  : int    — number of Gauss-Jacobi nodes (default 64 gives ~machine
                       precision for smooth integrands)

    Note on sigma2 vs SUSReg
    ------------------------
    SUSReg uses a spectral window exp(-sigma2_s * t^2 / 2).  The induced
    spatial kernel corresponds to sigma2_gauss = sigma2_s / d.
    With the default sigma2_s=1 in SUSReg and d=256: sigma2_gauss = 1/256 = 1/d.
    """

    def __init__(self, d: int, sigma2: float = None, n_quad: int = 64):  # type: ignore
        super().__init__()
        self.d      = int(d)
        self.sigma2 = float(sigma2) if sigma2 is not None else 1.0 / d
        self.n_quad = int(n_quad)

        t_np, w_np = _gauss_jacobi_quadrature_for_rho_d(self.d, self.n_quad)

        # Keep the reference quadrature in float64 and cast on the fly to match
        # the incoming tensor dtype/device during evaluation.
        self.register_buffer("t_quad", torch.from_numpy(t_np.copy()))
        self.register_buffer("w0", torch.from_numpy(w_np.copy()))
        self.register_buffer("w2", torch.from_numpy((t_np ** 2 * w_np).copy()))
        self.register_buffer("w4", torch.from_numpy((t_np ** 4 * w_np).copy()))

    def _exp_vals(self, c: torch.Tensor) -> torch.Tensor:
        """
        exp(-(1-c) * t_k^2 / sigma2)   for each quadrature node k.

        c   : (...,)
        out : (..., n_quad)
        """
        # dist2 = ||x-y||^2 = 2*(1-c);  exponent = -dist2*t^2/(2*sigma2) = -(1-c)*t^2/sigma2
        t_quad = self.t_quad.to(device=c.device, dtype=c.dtype)
        return torch.exp(
            -(1.0 - c).unsqueeze(-1) * t_quad.square() / self.sigma2 # type: ignore
        )

    def phi(self, c: torch.Tensor) -> torch.Tensor:
        """K(c)   — shape (...)"""
        w0 = self.w0.to(device=c.device, dtype=c.dtype)
        return (self._exp_vals(c) * w0).sum(-1)  # type: ignore

    def phi_prime(self, c: torch.Tensor) -> torch.Tensor:
        """dK/dc  — shape (...)"""
        w2 = self.w2.to(device=c.device, dtype=c.dtype)
        return (self._exp_vals(c) * w2).sum(-1) / self.sigma2  # type: ignore

    def phi_second(self, c: torch.Tensor) -> torch.Tensor:
        """d²K/dc² — shape (...)"""
        w4 = self.w4.to(device=c.device, dtype=c.dtype)
        return (self._exp_vals(c) * w4).sum(-1) / (self.sigma2 ** 2)  # type: ignore

    def log_kernel(self, c: torch.Tensor) -> torch.Tensor:
        return torch.log(self.phi(c))

    def log_kernel_prime(self, c: torch.Tensor) -> torch.Tensor:
        """d/dc log K(c) = K'(c) / K(c)"""
        return self.phi_prime(c) / self.phi(c)

    def log_kernel_second(self, c: torch.Tensor) -> torch.Tensor:
        """d²/dc² log K(c) = K''(c)/K(c) − (K'(c)/K(c))²"""
        phi  = self.phi(c)
        phi1 = self.phi_prime(c)
        phi2 = self.phi_second(c)
        return phi2 / phi - (phi1 / phi) ** 2

    def _mmd_constant(self) -> float:
        """
        Compute E[k(x,y)] for x,y iid uniform on S^{d-1}.

        Since k(x,y) = phi(<x,y>) and <x,y> follows rho_d on [-1, 1],

            E[k(x,y)] = ∫_{-1}^1 phi(c) rho_d(c) dc.

        We evaluate this with the same Gauss-Jacobi rule used to define phi.

        Note:
            E[k(x,x)] = phi(1) = 1
        for this normalized kernel, but this is not the MMD uniform constant.
        """
        c_quad = self.t_quad
        w0 = self.w0
        return float((self.phi(c_quad) * w0).sum().item())  # type: ignore

    def pairwise(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        X, Y : (B, d) unit vectors on S^{d-1}
        Returns K(x_i, y_i) — shape (B,)
        """
        c = (X * Y).sum(-1).clamp(-1.0, 1.0)
        return self.phi(c)

    def gram(self, X: torch.Tensor, Y: torch.Tensor = None) -> torch.Tensor:  # type: ignore
        """
        X : (B, d),  Y : (N, d)  [defaults to X]
        Returns K(x_i, y_j) — shape (B, N)
        """
        if Y is None:
            Y = X
        c = (X @ Y.T).clamp(-1.0, 1.0)   # (B, N)
        return self.phi(c)

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        """c : (...,) cosines <x,y>  →  K(c) : (...,)"""
        return self.phi(c)


# %%
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

    import matplotlib.pyplot as plt

    # ----------------------------------------------------------------
    # Test 1: kernel shape & agreement with exact reference
    # ----------------------------------------------------------------
    d      = 256
    sigma2 = 1.0 / d
    kernel = SUSRegInducedKernel(d=d, sigma2=sigma2, n_quad=64)

    c_vals    = torch.linspace(-1.0, 1.0, 100)
    k_torch   = kernel(c_vals).detach().numpy()
    mmd_const = kernel._mmd_constant()

    plt.figure(figsize=(8, 4))
    plt.plot(c_vals.numpy(), k_torch, "--", label="SUSRegInducedKernel (torch)", linewidth=1.5)
    plt.xlabel(r"$c = \langle x, y \rangle$")
    plt.ylabel(r"$K(c)$")
    plt.title(f"SUSRegInducedKernel  —  d={d},  σ²=1/d")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ----------------------------------------------------------------
    # Test 2: gradient check via autograd
    # ----------------------------------------------------------------
    c_test = torch.linspace(-0.9, 0.9, 20, requires_grad=False)
    c_test = c_test.clone().requires_grad_(True)
    k      = kernel(c_test).sum()
    k.backward()
    grad_auto = c_test.grad.detach().clone()  # type: ignore

    grad_anal = kernel.phi_prime(c_test.detach())
    grad_err  = (grad_auto - grad_anal).abs().max().item()
    print(f"d={d}, sigma2={sigma2:.4f}")
    print(f"E[k(x,y)] = {mmd_const:.8f}")
    print(f"\nGradient check (autograd vs phi_prime): max err = {grad_err:.2e}")

    c_one = torch.tensor([1.0], dtype=torch.float64)
    print(f"K(1) = {kernel(c_one).item():.8f}")
    print(f"E[k(x,x)] = {kernel(c_one).item():.8f}   (exactly 1 for the normalized kernel)")

    torch.manual_seed(0)
    N = 20_000
    X_mc = torch.randn(N, d, dtype=torch.float64)
    X_mc = X_mc / X_mc.norm(dim=-1, keepdim=True)
    Y_mc = torch.randn(N, d, dtype=torch.float64)
    Y_mc = Y_mc / Y_mc.norm(dim=-1, keepdim=True)
    c_mc = (X_mc * Y_mc).sum(dim=-1).clamp(-1.0, 1.0)
    mc_est = kernel.phi(c_mc).mean().item()

    print(f"\nMonte Carlo check for E[k(x,y)] with N={N}")
    print(f"MC estimate : {mc_est:.8f}")
    print(f"Quadrature  : {mmd_const:.8f}")
    print(f"Abs error   : {abs(mc_est - mmd_const):.3e}")

# %%
