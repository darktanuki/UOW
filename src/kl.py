# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from src.test import generic_compare

class KLReg(nn.Module):
    """
    Kernel KDE estimate of KL(P || Uniform(S^{d-1}))

    Generic version that works with any spherical kernel providing
    `phi(c)` and `_mmd_constant()`

    The loss is shifted so that the uniform distribution ≈ 0 and
    normalized so that a Dirac distribution maps to 1.
    """

    def __init__(self, kernel):
        super().__init__()
        self.kernel = kernel

        logC, dirac_const = self._kl_constant()
        self.register_buffer("logC", torch.tensor(logC, dtype=torch.float32))
        self.register_buffer("dirac_const", torch.tensor(dirac_const, dtype=torch.float32))

    def _kl_constant(self):
        # constant C = E[k(y,y')] for y,y' uniform on the sphere
        C = float(self.kernel._mmd_constant())
        logC = math.log(C + 1e-12)

        # kernel value on the diagonal k(1)
        c1 = torch.tensor([1.0])
        k1 = float(self.kernel.phi(c1)[0].item())

        # normalization so that Dirac -> 1
        dirac_const = math.log(k1 + 1e-12) - logC
        return logC, dirac_const
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = F.normalize(z, dim=-1)
        B = z.size(0)
        sim = torch.einsum('bd,cd->bc', z, z)
        K = self.kernel.phi(sim)
        logK = torch.log(K + 1e-12)

        # leave‑one‑out KDE
        logK.fill_diagonal_(float("-inf"))
        log_p = torch.logsumexp(logK, dim=1) - math.log(B - 1)

        # KL surrogate
        kl = log_p.mean() - self.logC.to(z.device, z.dtype) # type: ignore

        return kl / self.dirac_const.to(z.device, z.dtype) # type: ignore

class KL(nn.Module):
    def __init__(self, params):
        """
        """
        super().__init__()  
        self.params = params
        self.lambd = getattr(self.params, 'lambda_reg', 1)
        self.dim = getattr(self.params, 'proj_out_dim', 256)
        self.temperature = getattr(self.params, 'temperature', 2/256)
        self.kernel_name = getattr(self.params, 'kernel_name', 'heat') # heat, sobolev
        self.ell = getattr(self.params, 'ell', 2) # for sobolev

        if self.kernel_name == 'heat':
            from src.kernel.heatkernel import HeatKernel
            kernel = HeatKernel(d=self.dim, t=self.temperature)
        elif self.kernel_name == 'sobolev':
            from src.kernel.sobolkernel import SobolevKernel
            kernel = SobolevKernel(d=self.dim, ell=self.ell)
        else:
            raise ValueError(f"Unknown kernel_name={self.kernel_name}")

        self.loss_reg = KLReg(kernel=kernel)
        self.loss_inv = lambda z1, z2 : (1- (z1 * z2).sum(dim=-1)).mean()  # (B,).mean()

    def forward(self, z1, z2, mode = 'train'):  
        metrics = {}     
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        Linv = self.loss_inv(z1,z2)
        Lreg = (self.loss_reg(z1) + self.loss_reg(z2) ) / 2 
        L = self.lambd * Linv + Lreg

        metrics[f'{mode}/Linv'] = Linv.item()
        metrics[f'{mode}/Lreg'] = Lreg.item()
        metrics[f'{mode}/L'] = L.item()    
        return L, metrics


if __name__ == "__main__":
    B=100
    d=256
    seed=3
    device="cpu"
    

    from src.kernel.heatkernel import HeatKernel

    title = f"KLReg comparison B={B} d={d}"

    model = {        "heat": KLReg(kernel=HeatKernel(d=d, t=2/256)),
        "heat2": KLReg(kernel=HeatKernel(d=d, t=1.8/256)),
    }
    
    generic_compare(model, B, d, seed, device, title=title, yscale=True)


# %%
