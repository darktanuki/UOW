#%%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class KSDReg(nn.Module):
    """
    """
    def __init__(self, kernel):
        super().__init__()
        self.kernel = kernel

        self.d = int(kernel.d)
        self.register_buffer("dirac_const", torch.tensor(self._ksd_dirac_constant(), dtype=torch.float32))

    def _ksd_dirac_constant(self):
        c1 = torch.tensor([1.0])
        phi1 = self.kernel.phi(c1)[0]
        psi_prime_1 = self.kernel.log_kernel_prime(c1)[0]
        dirac_const = 1/2 * phi1 * (self.d - 1.0) * psi_prime_1
        dirac_const = float(dirac_const.item())
        return dirac_const

    def forward(self, z: torch.Tensor):
        B, d = z.shape
        z = F.normalize(z, dim=-1)

        sim = torch.einsum('bd,cd->bc', z, z)  # (B, B)
        c = sim.reshape(-1)

        phi_p  = self.kernel.phi_prime(c)
        phi_pp = self.kernel.phi_second(c)
        A = 0.5 * ((d-1) * c * phi_p + (c**2 - 1) * phi_pp)
        U = A.mean()
        return U / self.dirac_const 

class KSD(nn.Module):
    def __init__(self, params):
        """
        """
        super().__init__()  
        self.params = params
        self.lambd = getattr(self.params, 'lambda_reg', 9)
        self.dim = getattr(self.params, 'proj_out_dim', 256)
        self.temperature = getattr(self.params, 'temperature', 4/256)
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

        self.loss_reg = KSDReg(kernel=kernel)
        self.loss_inv = lambda z1, z2 : (1- (z1 * z2).sum(dim=-1)).mean()  # (B,).mean()

    def forward(self, z1, z2, mode = 'train'):  
        metrics = {}     
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        B, d = z1.shape

        Linv = self.loss_inv(z1,z2)
        Lreg = (self.loss_reg(z1) + self.loss_reg(z2) ) / 2 
        L = self.lambd * Linv + Lreg

        metrics[f'{mode}/Linv'] = Linv.item()
        metrics[f'{mode}/Lreg'] = Lreg.item()
        metrics[f'{mode}/L'] = L.item()    

        return L, metrics

if __name__ == "__main__":
    from src.test import generic_compare
    B=100
    d=256
    seed=1
    device="cpu"
    v_stat=False
    
    from src.kernel.heatkernel import HeatKernel
    from src.kernel.sobolkernel import SobolevKernel

    title = f"KSDReg comparison B={B} d={d}"
    model = {
            "heat4": KSDReg(kernel=HeatKernel(d=d, t=4.0/256)),
            "heat5": KSDReg(kernel=HeatKernel(d=d, t=5.0/256)),  
            "heat6": KSDReg(kernel=HeatKernel(d=d, t=6.0/256)),  
            "sobolev2": KSDReg(kernel=SobolevKernel(d=d, ell=2)),
            }

    generic_compare(model, B, d, seed, device, title=title, yscale=True)

# %%
