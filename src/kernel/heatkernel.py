#%%
import torch
import torch.nn as nn

class HeatKernel(nn.Module):
    """
    Spectral heat kernel on S^{d-1} using an orthonormal Gegenbauer basis.

    This is mathematically equivalent to the original HeatKernel_old, but the
    spectral expansion is written with normalized polynomials

        psi_l(c) = C_l^{(alpha)}(c) / sqrt(C_l^{(alpha)}(1))

    and weights chosen so that phi(1)=1. This tends to be numerically more
    stable than summing raw Gegenbauer polynomials with large amplitudes.
    """

    def __init__(self, d: int, t: float, lmax: int = 512, grid_size: int = 4096):
        super().__init__()
        from scipy.special import eval_gegenbauer
        import numpy as np

        self.d = int(d)
        self.t = float(t)
        self.lmax = int(lmax)
        self.eps = 1e-15

        alpha = (d - 2) / 2
        l = np.arange(lmax + 1)
        lam = l * (l + d - 2)
        w = np.exp(-lam * t)

        # Normalization constants at c = 1.
        C1 = np.array([eval_gegenbauer(int(li), alpha, 1.0) for li in l], dtype=np.float64)

        # Orthonormalized zonal basis psi_l(c) = C_l(c) / sqrt(C_l(1)).
        # Then phi(c) = sum_l beta_l psi_l(c) with beta_l = w_l * sqrt(C_l(1)) / Z,
        # where Z = sum_l w_l C_l(1), so that phi(1) = 1.
        Z = np.sum(w * C1)
        self.normalization_constant = Z
        import math
        # log Z where Z = sum_l w_l C_l(1)
        self.log_Z = math.log(Z)

        # Since beta_0 = 1 / Z, normalization constant for density (with normalized measure)
        # is Z_t = beta_0 = 1 / Z, so log Z_t = -log Z
        self.log_normalization_constant = -self.log_Z
        beta = w * np.sqrt(C1) / Z

        self.alpha = alpha
        self.beta = torch.tensor(beta, dtype=torch.float64)
        self.C1 = torch.tensor(C1, dtype=torch.float64)

        self._build_interpolation_tables(grid_size)

    def forward(self, c: torch.Tensor):
        return self.phi(c)

    def phi(self, c: torch.Tensor):
        return self._interp_from_grid(c, self.phi_table)

    def phi_prime(self, c: torch.Tensor):
        return self._interp_from_grid(c, self.phi_prime_table)

    def phi_second(self, c: torch.Tensor):
        return self._interp_from_grid(c, self.phi_second_table)

    def _build_interpolation_tables(self, grid_size: int):
        """
        Precompute phi(c), log phi(c), and derivatives on a cosine grid using
        normalized Gegenbauer polynomials.
        """
        import numpy as np
        from scipy.special import eval_gegenbauer

        c_grid = np.linspace(-1.0, 1.0, grid_size)
        phi = np.zeros_like(c_grid, dtype=np.float64)
        phi_prime = np.zeros_like(c_grid, dtype=np.float64)
        phi_second = np.zeros_like(c_grid, dtype=np.float64)

        alpha = float(self.alpha)
        beta = self.beta.detach().cpu().numpy()
        C1 = self.C1.detach().cpu().numpy()

        for l in range(self.lmax + 1):
            scale = np.sqrt(C1[l])
            phi += beta[l] * eval_gegenbauer(l, alpha, c_grid) / scale

            if l >= 1:
                phi_prime += beta[l] * (2.0 * alpha) * eval_gegenbauer(l - 1, alpha + 1.0, c_grid) / scale

            if l >= 2:
                phi_second += beta[l] * (4.0 * alpha * (alpha + 1.0)) * eval_gegenbauer(l - 2, alpha + 2.0, c_grid) / scale

        phi        = np.maximum(phi, self.eps)
        phi_prime  = np.maximum(phi_prime, self.eps)
        phi_second = np.maximum(phi_second, self.eps)
        log_phi = np.log(phi)
        d1 = phi_prime / phi
        d2 = phi_second / phi - (phi_prime / phi) ** 2

        self.c_grid = torch.tensor(c_grid, dtype=torch.float64)
        self.phi_table = torch.tensor(phi[None, :], dtype=torch.float64)
        self.log_phi_table = torch.tensor(log_phi[None, :], dtype=torch.float64)
        self.log_phi_prime_table = torch.tensor(d1[None, :], dtype=torch.float64)
        self.log_phi_second_table = torch.tensor(d2[None, :], dtype=torch.float64)
        self.phi_prime_table = torch.tensor(phi_prime[None, :], dtype=torch.float64)
        self.phi_second_table = torch.tensor(phi_second[None, :], dtype=torch.float64)

    def _interp_from_grid(self, c: torch.Tensor, table: torch.Tensor):
        # grid = self.c_grid.to(c.device)
        # table = table.to(c.device)

        grid = self.c_grid.to(device=c.device, dtype=c.dtype)
        table = table.to(device=c.device, dtype=c.dtype)

        idx = torch.bucketize(c.reshape(-1), grid)
        idx = idx.clamp(1, len(grid) - 1)

        i0 = idx - 1
        i1 = idx

        g0 = grid[i0]
        g1 = grid[i1]

        w1 = (c.reshape(-1) - g0) / (g1 - g0)
        w0 = 1.0 - w1

        v0 = table[0, i0]
        v1 = table[0, i1]

        return (w0 * v0 + w1 * v1).view_as(c)

    def log_kernel(self, c: torch.Tensor):
        return self._interp_from_grid(c, self.log_phi_table)

    def log_kernel_prime(self, c: torch.Tensor):
        return self._interp_from_grid(c, self.log_phi_prime_table)

    def log_kernel_second(self, c: torch.Tensor):
        return self._interp_from_grid(c, self.log_phi_second_table)

    def log_density(self, c: torch.Tensor):
        """
        Log of normalized heat kernel density (with respect to uniform measure on the sphere).
        """
        return self.log_kernel(c) - self.log_normalization_constant

    def _mmd_constant(self):
        """
        Analytic value of E_{y,y'}[k(y,y')] for y,y' iid uniform on S^{d-1}.
        Only the l = 0 mode survives under spherical averaging.
        """
        return float(self.beta[0].item())

    def _mmd_second_constant(self):
        """
        Analytic value of E_{y,y'}[k(y,y')^2] for y,y' iid uniform on S^{d-1}.

        If
            phi(c) = sum_l beta_l psi_l(c)
        with orthonormal basis psi_l, then

            E[k(y,y')] = beta_0
            E[k(y,y')^2] = sum_l beta_l^2

        since the psi_l are orthonormal w.r.t. the spherical measure.
        """
        beta = self.beta
        return float((beta * beta).sum().item())

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np 
    d = 256
    t_values = [2/d, 3/d, 4/d, 5/d, 6/d]
    kernels = [HeatKernel(d=d, t=t, lmax=512, grid_size=4096) for t in t_values]
    c = torch.linspace(-1, 1, 500, dtype=torch.float64)

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    for t, kernel in zip(t_values, kernels):
        phi_vals = kernel.phi(c)
        plt.plot(c.numpy(), phi_vals.numpy(), label=f"t={int(round(t * d))}/d")
    # plot reference curve 4 - c^2 (dashed)
    plt.plot(c.numpy(), np.maximum(c.numpy(), 0.0), linestyle='--', label='max(c, 0)')
    # plt.yscale('log')
    plt.legend()
    plt.title("phi(c)")

    plt.subplot(2, 2, 2)
    for t, kernel in zip(t_values, kernels):
        log_phi = kernel.log_kernel(c)
        plt.plot(c.numpy(), log_phi.numpy(), label=f"t={int(round(t * d))}/d")
    plt.legend()
    plt.title("log phi(c)")

    plt.subplot(2, 2, 3)
    for t, kernel in zip(t_values, kernels):
        d1 = kernel.phi_prime(c)
        plt.plot(c.numpy(), d1.numpy(), label=f"t={int(round(t * d))}/d")
    plt.legend()
    plt.yscale('log')
    plt.title("Derivative of phi(c)")

    plt.subplot(2, 2, 4)
    for t, kernel in zip(t_values, kernels):
        d2 = kernel.phi_second(c)
        plt.plot(c.numpy(), d2.numpy(), label=f"t={int(round(t * d))}/d")
    plt.legend()
    plt.yscale('log')
    plt.title("Second derivative of phi(c)")

    plt.tight_layout()
    plt.show()

    # Plot the normalization constant for the different diffusion times
    norm_constants = [kernel.normalization_constant for kernel in kernels]
    log_norm_constants = [kernel.log_normalization_constant for kernel in kernels]

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot([t * d for t in t_values], norm_constants, marker='o')
    plt.xlabel("td")
    plt.ylabel("Z")
    plt.title("Normalization constant Z")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot([t * d for t in t_values], log_norm_constants, marker='o')
    plt.xlabel("td")
    plt.ylabel("log normalization constant")
    plt.title("log normalization constant")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # --------------------------------------------------
    # Test 2: Monte Carlo check for E_{y,y'}[h_t(y,y')]
    # --------------------------------------------------
    torch.manual_seed(0)

    N = 200_000  # increase if you want tighter error bars
    y = torch.randn(N, d, dtype=torch.float64)
    y = y / y.norm(dim=-1, keepdim=True)

    yp = torch.randn(N, d, dtype=torch.float64)
    yp = yp / yp.norm(dim=-1, keepdim=True)

    c_mc = (y * yp).sum(dim=-1)
    kernel_ref = kernels[1]  # t = 3/d
    k_mc = kernel_ref.phi(c_mc)  # k(y,y') = φ(y^T y')
    mc_est = k_mc.mean().item()

    exact = kernel_ref._mmd_constant()

    abs_err = abs(mc_est - exact)
    rel_err = abs_err / (abs(exact))

    print("\nMonte Carlo check for E[k(y,y')]")
    print(f"d={d}, t={t_values[1]:.4f}, N={N}")
    print(f"MC estimate : {mc_est:.10f}")
    print(f"Exact value : {exact:.10f}")
    print(f"Abs error   : {abs_err:.3e}")
    print(f"Rel error   : {rel_err:.3e}")

    # --------------------------------------------------
    # Inverse-Laplacian^2 kernel
    # --------------------------------------------------

# %%


