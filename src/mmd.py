#%%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F



class MMDReg(nn.Module):
    """
    """

    def __init__(self, kernel):
        super().__init__()
        self.kernel = kernel 
        self.register_buffer("mmd_constant", torch.tensor(float(self.kernel._mmd_constant()), dtype=torch.float32))
        self.register_buffer("dirac_const", torch.tensor(self._mmd_dirac_constant(), dtype=torch.float32))
    
    def _mmd_dirac_constant(self):
        # normalization so that MMD^2(Dirac, uniform) = 1
        c1 = torch.tensor([1.0])
        k1 = float(self.kernel.phi(c1)[0].item())
        dirac_const = k1 - float(self.kernel._mmd_constant())
        return dirac_const
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B, d = z.shape
        z = F.normalize(z, dim=-1)

        # Empirical term E[k(z,z')]
        sim = torch.einsum('bd,cd->bc', z, z)  # (B, B)
        A1 = self.kernel.phi(sim.reshape(-1)).mean()
        mmd2 = A1 - self.mmd_constant.to(device=z.device, dtype=z.dtype)
        return mmd2 / self.dirac_const 

class MMD(nn.Module):
    def __init__(self, params):
        """
        """
        super().__init__()  
        self.params = params
        self.lambd = getattr(self.params, 'lambda_reg', 1)
        self.dim = getattr(self.params, 'proj_out_dim', 256)
        self.temperature = getattr(self.params, 'temperature', 5/256)
        self.kernel_name = getattr(self.params, 'kernel_name', 'heat') # heat, sobolev
        self.ell = getattr(self.params, 'ell', 2) # heat, sobolev

        if self.kernel_name == 'heat':
            from src.kernel.heatkernel import HeatKernel
            kernel = HeatKernel(d=self.dim, t=self.temperature)
        elif self.kernel_name == 'sobolev':
            from src.kernel.sobolkernel import SobolevKernel
            kernel = SobolevKernel(d=self.dim, ell=self.ell)

        elif self.kernel_name == 'susreginduced':
            from src.kernel.susreginducedkernel import SUSRegInducedKernel
            kernel = SUSRegInducedKernel(d=self.dim)
        else:
            raise ValueError(f"Unknown kernel_name={self.kernel_name}")

        self.loss_reg = MMDReg(kernel=kernel)
        self.loss_inv = lambda z1, z2 : (1- (z1 * z2).sum(dim=-1)).mean()  # (B,).mean()

    def forward(self, z1, z2, mode = 'train'):  
        metrics = {}     
        B = z1.shape[0]
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
    seed=0
    device="cpu"
    from src.test import generic_compare
    from src.kernel.heatkernel import HeatKernel
    from src.kernel.sobolkernel import SobolevKernel

    title = f"MMDReg comparison B={B} d={d}"
    model = {
            "heat4": MMDReg(kernel=HeatKernel(d=d, t=4/d)),
            "heat5": MMDReg(kernel=HeatKernel(d=d, t=5/d)),  
            "heat6": MMDReg(kernel=HeatKernel(d=d, t=6/d)),  
            "sobolev2": MMDReg(kernel=SobolevKernel(d=d, ell=2)),
            }
    
    generic_compare(model, B, d, seed, device, title=title, yscale=True)

# %%
