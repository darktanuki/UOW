#%%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn


class SobolevKernel(nn.Module):
    """
    Sobolev kernel on S^{d-1}.

    k(x,y) = phi(c) where c = x^T y

    phi(c) = sum_{k=1}^ell (1 + 2k/(d-2)) C_k^{(alpha)}(c)

    NOTE: This is NOT a probability kernel, so log_density is defined up to a constant.
    """

    def __init__(self, d: int, ell: int, eps: float = 1e-15):
        super().__init__()
        self.d = int(d)
        self.ell = int(ell)
        self.alpha = (d - 2) / 2
        self.eps = eps

        self._build_polynomial_coefficients()

        # Precompute Dirac normalization constant
        dirac_const = sum(self.eigenspace_dim(k) for k in range(1, self.ell + 1))
        self.register_buffer("dirac_const", torch.tensor(float(dirac_const)))

        # Coefficients are normalized in _build_polynomial_coefficients so that phi(1)=1
        self.register_buffer("norm_const", torch.tensor(1.0))

    def forward(self, c: torch.Tensor):
        return self.phi(c)

    def eigenspace_dim(self, k: int) -> int:
        p = self.d
        return math.comb(p + k - 3, p - 2) + math.comb(p + k - 2, p - 2)

    def _poly_add(self, p, q):
        n = max(len(p), len(q))
        out = [0.0] * n
        for i in range(len(p)):
            out[i] += p[i]
        for i in range(len(q)):
            out[i] += q[i]
        return out

    def _poly_scale(self, p, a):
        return [a * x for x in p]

    def _poly_mul_x(self, p):
        return [0.0] + list(p)

    def _gegenbauer_poly_coeffs(self, k: int, alpha: float):
        # Returns coefficients [a0, a1, ..., ak] for C_k^{(alpha)}(x)
        if k == 0:
            return [1.0]
        if k == 1:
            return [0.0, 2.0 * alpha]

        Ckm2 = [1.0]
        Ckm1 = [0.0, 2.0 * alpha]
        for n in range(2, k + 1):
            term1 = self._poly_scale(self._poly_mul_x(Ckm1), 2.0 * (n + alpha - 1.0) / n)
            term2 = self._poly_scale(Ckm2, -(n + 2.0 * alpha - 2.0) / n)
            Ck = self._poly_add(term1, term2)
            Ckm2, Ckm1 = Ckm1, Ck
        return Ckm1

    def _build_polynomial_coefficients(self):
        # Build unnormalized phi(c) = sum_{m=0}^ell a_m c^m
        coeffs = [0.0] * (self.ell + 1)
        for k in range(1, self.ell + 1):
            weight = 1.0 + 2.0 * k / (self.d - 2)
            gk = self._gegenbauer_poly_coeffs(k, self.alpha)
            for m, val in enumerate(gk):
                coeffs[m] += weight * val

        # Normalize so that phi(1) = 1
        norm_const = sum(coeffs)
        coeffs = [x / norm_const for x in coeffs]

        d1 = [m * coeffs[m] for m in range(1, len(coeffs))]
        d2 = [m * (m - 1) * coeffs[m] for m in range(2, len(coeffs))]

        self.register_buffer("poly_coeffs", torch.tensor(coeffs, dtype=torch.float64))
        self.register_buffer("poly_coeffs_prime", torch.tensor(d1, dtype=torch.float64))
        self.register_buffer("poly_coeffs_second", torch.tensor(d2, dtype=torch.float64))

    def _eval_poly(self, c: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
        coeffs = coeffs.to(device=c.device, dtype=c.dtype)
        out = torch.zeros_like(c)
        for a in reversed(coeffs):
            out = out * c + a
        return out

    def phi(self, c: torch.Tensor) -> torch.Tensor:
        return self._eval_poly(c, self.poly_coeffs) # type: ignore

    def phi_prime(self, c: torch.Tensor) -> torch.Tensor:
        return self._eval_poly(c, self.poly_coeffs_prime)  # type: ignore

    def phi_second(self, c: torch.Tensor) -> torch.Tensor:
        return self._eval_poly(c, self.poly_coeffs_second)  # type: ignore

    def log_kernel(self, c: torch.Tensor):
        phi = torch.clamp(self.phi(c), min=self.eps)
        return torch.log(phi)

    def log_kernel_prime(self, c: torch.Tensor):
        phi = torch.clamp(self.phi(c), min=self.eps)
        return self.phi_prime(c) / phi

    def log_kernel_second(self, c: torch.Tensor):
        phi = torch.clamp(self.phi(c), min=self.eps)
        phi_p = self.phi_prime(c)
        phi_pp = self.phi_second(c)
        return phi_pp / phi - (phi_p / phi) ** 2

    def log_density(self, c: torch.Tensor):
        # Not a normalized density, but keeps same API
        return self.log_kernel(c)

    def _mmd_constant(self):
        """
        E[k(x,y)] under uniform.

        Since no l=0 term, expectation is 0.
        """
        return 0.0

    def _mmd_second_constant(self, num_points=2000):
        import torch
        
        c = torch.linspace(-1, 1, num_points, dtype=torch.float64)
        
        # density rho_d
        alpha = (self.d - 3) / 2
        rho = (1 - c**2)**alpha
        rho = rho / rho.sum()  # normalize discrete
        
        phi = self.phi(c)
        
        return float((phi**2 * rho).sum())



if __name__ == "__main__":
    import matplotlib.pyplot as plt

    d = 256
    ell_values = [1,2, 5, 10]
    kernels = [SobolevKernel(d=d, ell=ell) for ell in ell_values]

    c = torch.linspace(-1, 1, 500, dtype=torch.float64)

    plt.figure(figsize=(12, 8))

    # --------------------------------------------------
    # phi(c)
    # --------------------------------------------------
    plt.subplot(2, 2, 1)
    for ell, kernel in zip(ell_values, kernels):
        phi_vals = kernel.phi(c)
        plt.plot(c.numpy(), phi_vals.numpy(), label=f"ell={ell}")
    plt.legend()
    plt.title("phi(c)")

    # --------------------------------------------------
    # first derivative
    # --------------------------------------------------
    plt.subplot(2, 2, 3)
    for ell, kernel in zip(ell_values, kernels):
        d1 = kernel.phi_prime(c)
        plt.plot(c.numpy(), d1.numpy(), label=f"ell={ell}")
    plt.legend()
    plt.title("phi'(c)")

    # --------------------------------------------------
    # second derivative
    # --------------------------------------------------
    plt.subplot(2, 2, 4)
    for ell, kernel in zip(ell_values, kernels):
        d2 = kernel.phi_second(c)
        plt.plot(c.numpy(), d2.numpy(), label=f"ell={ell}")
    plt.legend()
    plt.title("phi''(c)")

    plt.tight_layout()
    plt.show()

    # --------------------------------------------------
    # Monte Carlo check: E[k(x,y)] ~ 0
    # --------------------------------------------------
    torch.manual_seed(0)

    N = 200_000
    x = torch.randn(N, d, dtype=torch.float64)
    x = x / x.norm(dim=-1, keepdim=True)

    y = torch.randn(N, d, dtype=torch.float64)
    y = y / y.norm(dim=-1, keepdim=True)

    c_mc = (x * y).sum(dim=-1)

    for ell, kernel in zip(ell_values, kernels):
        val = kernel.phi(c_mc).mean().item()
        print(f"ell={ell} | E[k(x,y)] ≈ {val:.6e} (should be ~0)")

        def __init__(self, kernel: SobolevKernel, grid_size: int = 4096):
            self.kernel = kernel
            self.grid = torch.linspace(-1.0, 1.0, grid_size, dtype=torch.float64)
            with torch.no_grad():
                self.values = kernel.phi(self.grid)

        def phi(self, c: torch.Tensor) -> torch.Tensor:
            grid = self.grid.to(device=c.device, dtype=c.dtype)
            values = self.values.to(device=c.device, dtype=c.dtype)

            flat = c.reshape(-1)
            idx = torch.bucketize(flat, grid)
            idx = idx.clamp(1, len(grid) - 1)

            i0 = idx - 1
            i1 = idx
            g0 = grid[i0]
            g1 = grid[i1]

            w1 = (flat - g0) / (g1 - g0)
            w0 = 1.0 - w1

            v0 = values[i0]
            v1 = values[i1]
            return (w0 * v0 + w1 * v1).view_as(c)
# %%
