#%%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class SIGReg(torch.nn.Module):
    def __init__(self, num_projections=1024,  t_max=5, knots=51, seed=0):
        super().__init__()
        self.num_projections = num_projections
        t = torch.linspace(0, t_max, knots, dtype=torch.float32)
        dt = t_max / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)
        self._seed = seed
        g = torch.Generator()
        g.manual_seed(self._seed)
        self._generator = g

    def forward(self, proj):
        if self._generator.device != proj.device:
            self._generator = torch.Generator(device=proj.device).manual_seed(int(self._seed))

        #proj : (B,d)
        A = torch.randn(proj.size(-1), self.num_projections, device=proj.device, dtype=proj.dtype, generator=self._generator) #(d,nb_proj)
        A = A.div_(A.norm(p=2, dim=0)) #(d,nb_proj)
        x_t = (proj @ A).unsqueeze(-1) * self.t #(B, nb_proj, T)
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square() #(nb_proj, T)
        statistic = (err @ self.weights) #* proj.size(-2)
        return statistic.mean()
    
class LEJEPA(nn.Module):
    def __init__(self, lambd=0.05, nb_global_views=2):
        """
        """
        super().__init__()  
        self.lambd = lambd
        self.nb_global_views = nb_global_views
        self.loss_reg = SIGReg()
             
    def forward(self, zs, mode='train'):
        """
        """
        metrics = {}
        mu = torch.stack(zs[:self.nb_global_views], dim=0).mean(dim=0, keepdim=True) #(1, B, D)
        zs_ = torch.stack(zs, dim=0) #(num_views, B, D)
        Linv = 1/2 * (zs_ - mu).pow(2).mean() #(1,)
        # SIGReg is an empirical MMD-like statistic whose magnitude shrinks with
        # the batch size.  Match the scaling used by OWReg/SIGRegInduced so the
        # regularizer remains comparable across batch sizes.
        Lreg = torch.stack([self.loss_reg(z) for z in zs]).mean() * zs[0].size(0)
        L = (1 - self.lambd) * (Linv) + self.lambd * Lreg 
        metrics[f'{mode}/Linv'] = Linv.item()
        metrics[f'{mode}/Lreg'] = Lreg.item()
        metrics[f'{mode}/L'] = L.item()    
        return L, metrics
        



#%%
if __name__ == "__main__":
    B = 10_000
    X = torch.randn(B, 256)
    reg = SIGReg()
    print(reg(X))

# %%
