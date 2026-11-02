import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCE(nn.Module):
    def __init__(self, params):
        """
        """
        super().__init__()
        self.params = params
        self.t = float(getattr(self.params, 'temperature', 0.1))
                 
    def forward(self, z1, z2, mode='train'):
        """
        - z1: (B, D)
        - z2: (B, D) 
        """
        sim = torch.einsum('bd,cd->bc', z1, z2) / self.t # (B, B) 
        B = sim.size(0)
        labels = torch.arange(B, device=sim.device)
        L = F.cross_entropy(sim, labels)

        # metrics = {
        #     f'{mode}/L': L.item(),
        # }
        return L
    