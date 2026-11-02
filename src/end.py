"""EnD regularization loss for unsupervised two-view training."""

from numpy.ma import count
import torch
import torch.nn as nn
import torch.nn.functional as F


class EnDReg(nn.Module):
    r"""Bias-conditional disentangling term from EnD.

    Computes the EnD disentangling penalty :math:`|G_b-I|` averaged over
    each bias group, where :math:`G_b` is the cosine-similarity Gram matrix.
    Subtracting the identity makes the (constant-one) self-similarities zero,
    as in the EnD formulation.
    """


    def forward(self, z, biased_labels):
        """
        Computes the expected regularization loss across unique bias groups.
        
        Args:
            z (torch.Tensor): Latent feature vectors of shape [B, D].
            biased_labels (torch.Tensor): Bias labels for each sample of shape [B] or [B, C].
            
        Returns:
            torch.Tensor: The expected regularization loss.
        """
        z = F.normalize(z, dim=-1)
        B, D = z.shape

        if biased_labels.shape[0] != B:
            raise ValueError(
                f"biased_labels batch dimension must match z: "
                f"{tuple(biased_labels.shape)} vs {tuple(z.shape)}"
            )

        if biased_labels.ndim == 1:
            unique_biases, counts = torch.unique(biased_labels, return_counts=True)
            bias_masks = [biased_labels == b for b in unique_biases]
        elif biased_labels.ndim == 2:
            unique_biases, counts = torch.unique(biased_labels, dim=0, return_counts=True)
            bias_masks = [(biased_labels == b).all(dim=1) for b in unique_biases]
        else:
            raise ValueError(
                f"biased_labels must have shape [B] or [B, C], got {tuple(biased_labels.shape)}"
            )


        reg_losses_bias = torch.stack([self.reg(z[mask]) * (mask*1.0).mean() for mask in bias_masks]).sum() #(1,) ie \sum D(X|b)P(B=b)
        reg_losses_bias = reg_losses_bias * B # because the reg has to be invariant of the batch size. 
        return reg_losses_bias #(1,)

    def reg(self, z: torch.Tensor) -> torch.Tensor:
        B, d = z.shape
        z = F.normalize(z, dim=-1)
        count = z.shape[0]
        identity = torch.eye(count, dtype=z.dtype, device=z.device)
        sim = torch.einsum('bd,cd->bc', z, z)  # (B, B)
        L = (sim - identity).abs().mean()
        return L 


class EnDUnsupervised(nn.Module):
    """Two-view EnD loss: ``(1 - lambda) * Linv + lambda * Lreg``."""

    def __init__(self, params=None, lambd: float | None = None):
        super().__init__()
        legacy_lambda = getattr(params, "lambda_end", None)
        if lambd is not None:
            self.lambd = float(lambd)
        elif legacy_lambda is not None:
            self.lambd = float(legacy_lambda)
        else:
            self.lambd = float(getattr(params, "lambda_reg", 0.05))
        if not 0.0 <= self.lambd <= 1.0:
            raise ValueError(f"EnD regularization weight must be in [0, 1], got {self.lambd}.")

        self.nb_global_views = int(getattr(params, "nb_global_views", 2))
        self.reg = EnDReg()


    def forward(self, zs, biased_labels, mode='train'):
        zs = [F.normalize(z, dim=-1) for z in zs]
        # Invariance loss (Linv)
        mu = torch.stack(zs[:self.nb_global_views], dim=0).mean(dim=0, keepdim=True) #(1, B, D)
        zs_ = torch.stack(zs, dim=0) #(num_views, B, D)
        Linv = 1 - (zs_ * mu).sum(dim=-1).mean() #(1,)
        Lreg_ = torch.stack([self.reg(z, biased_labels) for z in zs]) #(len(zs),)
        Lreg = Lreg_.mean() #(1,)        

        L = (1-self.lambd) * Linv +  self.lambd * Lreg
        L_dict = {
            f"{mode}/Linv": Linv.item(),
            f"{mode}/Lreg": Lreg.item(),
            f"{mode}/L": L.item(),
        }    
        return L, L_dict
