from src.kl import KLReg
from src.ksd import KSDReg
from src.mmd import MMDReg
from src.infonce import InfoNCE
import torch.nn.functional as F

import torch
import torch.nn as nn


RAW_EMBEDDING_REG_NAMES = {'sigreg', 'sigreginduced'}


def _unit_interval(value, name):
    """Return a scalar loss weight, rejecting values outside [0, 1]."""
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")
    return value


def _alpha_from_params(params):
    """Resolve the canonical alpha parameter, accepting its legacy alias."""
    alpha = getattr(params, "alpha", None)
    if alpha is None:
        alpha = getattr(params, "lambda_uow_reg", None)
    return _unit_interval(0.5 if alpha is None else alpha, "alpha")


class UOWReg(torch.nn.Module):
    def __init__(self, params) -> None:
        super().__init__() 
        self.params = params
        
        self.dim = getattr(self.params, 'proj_out_dim', 256)
        self.temperature = getattr(self.params, 'temperature', 5/256)
        self.kernel_name = getattr(self.params, 'kernel_name', 'heat') # heat, sobolev, susreginduced
        self.reg_name    = getattr(self.params, 'reg_name', 'mmd') # mmd, ksd or kl 
        print(self.reg_name)

        self.ell = getattr(self.params, 'ell', 2) # sobolev

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
        
        self.kernel = kernel
        if self.reg_name == 'kl':
            self.reg = KLReg(self.kernel)
        elif self.reg_name == 'ksd':
            self.reg = KSDReg(self.kernel)
        elif self.reg_name == 'mmd':
            self.reg = MMDReg(self.kernel)
        elif self.reg_name == 'sigreg':
            from src.sigreg import SIGReg
            self.reg = SIGReg()
        elif self.reg_name == 'sigreginduced':
            from src.sigreginduced import SIGRegInduced
            self.reg = SIGRegInduced(d=self.dim)
        else:
            raise NotImplementedError(f"Regularization method {self.params.reg_name} not implemented.")
        
    def forward(self, z, biased_labels):
        """
        Computes the expected regularization loss across unique bias groups.
        
        Args:
            z (torch.Tensor): Latent feature vectors of shape [B, D].
            biased_labels (torch.Tensor): Bias labels for each sample of shape [B] or [B, C].
            
        Returns:
            torch.Tensor: The expected regularization loss.
        """
        if self.reg_name not in RAW_EMBEDDING_REG_NAMES:
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
        if self.reg_name in ['mmd', 'ksd', 'sigreg', 'sigreginduced']:
            reg_losses_bias = reg_losses_bias * B # because the reg has to be invariant of the batch size. 
        return reg_losses_bias #(1,)
    
class UOW(torch.nn.Module):
    def __init__(self, params) -> None:
        super().__init__() 
        self.params = params
        self.lambd = _unit_interval(getattr(self.params, 'lambda_reg', 0.5), 'lambda_reg')
        self.alpha = _alpha_from_params(self.params)
        self.nb_global_views = getattr(self.params, 'nb_global_views', 2)
        self.uow_reg = UOWReg(params)
        self.ow_reg = OWReg(params)
        # Preserve the public regularizer hook used by diagnostics/probing code.
        self.reg = self.uow_reg
    
    def forward(self, zs, biased_labels, mode='train'):
        # Invariance loss (Linv)
        mu = torch.stack(zs[:self.nb_global_views], dim=0).mean(dim=0, keepdim=True) #(1, B, D)
        zs_ = torch.stack(zs, dim=0) #(num_views, B, D)
        if self.uow_reg.reg_name in RAW_EMBEDDING_REG_NAMES:
            Linv = 1/2 * (zs_ - mu).pow(2).mean() #(1,)
        else:
            Linv = 1 - (zs_ * mu).sum(dim=-1).mean() #(1,)

        Luowreg = torch.stack([self.uow_reg(z, biased_labels) for z in zs]).mean()
        Lowreg = torch.stack([self.ow_reg(z) for z in zs]).mean()

        conditional_weight = self.lambd * (1.0 - self.alpha)
        global_weight = self.lambd * self.alpha
        L = (1.0 - self.lambd) * Linv + conditional_weight * Luowreg + global_weight * Lowreg
        L_dict = {
            f"{mode}/Linv": Linv.item(),
            f"{mode}/LUOWReg": Luowreg.item(),
            f"{mode}/LOWReg": Lowreg.item(),
            f"{mode}/Lreg": (conditional_weight * Luowreg + global_weight * Lowreg).item(),
            f"{mode}/L": L.item(),
        }    
        return L, L_dict
    
class OWReg(torch.nn.Module):
    def __init__(self, params) -> None:
        super().__init__() 
        self.params = params
        
        self.dim = getattr(self.params, 'proj_out_dim', 256)
        self.temperature = getattr(self.params, 'temperature', 5/256)
        self.kernel_name = getattr(self.params, 'kernel_name', 'heat') # heat, sobolev, susreginduced
        self.reg_name    = getattr(self.params, 'reg_name', 'mmd') # mmd, ksd or kl 
        print(self.reg_name)
        
        self.ell = getattr(self.params, 'ell', 2) # sobolev

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
        
        self.kernel = kernel
        if self.reg_name == 'kl':
            self.reg = KLReg(self.kernel)
        elif self.reg_name == 'ksd':
            self.reg = KSDReg(self.kernel)
        elif self.reg_name == 'mmd':
            self.reg = MMDReg(self.kernel)
        elif self.reg_name == 'sigreg':
            from src.sigreg import SIGReg
            self.reg = SIGReg()
        elif self.reg_name == 'sigreginduced':
            from src.sigreginduced import SIGRegInduced
            self.reg = SIGRegInduced(d=self.dim)
        else:
            raise NotImplementedError(f"Regularization method {self.params.reg_name} not implemented.")
                   
    def forward(self, z):
        """
        Computes the regularization loss for the entire batch. """
        B,d = z.shape
        if self.reg_name not in RAW_EMBEDDING_REG_NAMES:
            z = z / z.norm(p=2, dim=-1, keepdim=True)
        # B,D = z.shape
        reg_losses_bias = self.reg(z)
        if self.reg_name in ['mmd', 'ksd', 'sigreg', 'sigreginduced']:
             reg_losses_bias = reg_losses_bias * B # because the reg has to be invariant of the batch size. 

        return  reg_losses_bias #(1,)
    
class OW(torch.nn.Module):
    def __init__(self, params) -> None:
        super().__init__() 
        self.params = params
        self.lambd = _unit_interval(getattr(self.params, 'lambda_reg', 0.5), 'lambda_reg')
        self.nb_global_views = getattr(self.params, 'nb_global_views', 2)
        self.reg = OWReg(params)
    
    def forward(self, zs, biased_labels, mode='train'):
        # Invariance loss (Linv)
        mu = torch.stack(zs[:self.nb_global_views], dim=0).mean(dim=0, keepdim=True) #(1, B, D)
        zs_ = torch.stack(zs, dim=0) #(num_views, B, D)

        if self.reg.reg_name in RAW_EMBEDDING_REG_NAMES:
            Linv = 1/2 * (zs_ - mu).pow(2).mean() #(1,)
        else:
            Linv = 1 - (zs_ * mu).sum(dim=-1).mean() #(1,)

        Lreg_ = torch.stack([self.reg(z) for z in zs]) #(len(zs),)
        Lreg = Lreg_.mean() #(1,)    
        L = (1.0 - self.lambd) * Linv + self.lambd * Lreg
        L_dict = {
            f"{mode}/Linv": Linv.item(),
            f"{mode}/Lreg": Lreg.item(),
            f"{mode}/L": L.item(),
        }    
        return L, L_dict
 
class InfoNCEWrapper(torch.nn.Module):
    def __init__(self, params) -> None:
        super().__init__()
        self.params = params
        self.reg = InfoNCE(params)

    def forward(self, zs, biased_labels, mode='train'):

        if len(zs) != 2:
            raise ValueError(f"InfoNCEWrapper expects exactly 2 views in zs, got {len(zs)}.")

        z0, z1 = zs
        z0 = z0 / z0.norm(p=2, dim=1, keepdim=True)
        z1 = z1 / z1.norm(p=2, dim=1, keepdim=True)

        L = self.reg(z0, z1, mode=mode)
        L_dict = {
            f"{mode}/L": L.item(),
        }
        return L, L_dict
    
class FairSupConLoss(nn.Module):
    def __init__(self, temperature=0.1, contrast_mode='all',
                 base_temperature=0.1):
        super(FairSupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    
    def forward(self, features, labels,sensitive_labels, group_norm, method,mask=None):
        """
        Args:
            features: [B, 2, D].
            labels: [B] or torch.zeros(B)
            sensitive_labels: [B]
            mask: None

            group_norm=0 for FSCL daggued 
            group_norm=1 for FSCL+ 
            method= FSCL or FSCL*
        Returns:
            A loss scalar.
        """

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]

        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            sensitive_labels = sensitive_labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
            sensitive_mask = torch.eq(sensitive_labels, sensitive_labels.T).float().to(device)
        else:
            mask = mask.float().to(device) #type: ignore

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        
        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        sensitive_mask = sensitive_mask.repeat(anchor_count, contrast_count)
        n_sensitive_mask=(~sensitive_mask.bool()).float()
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )

        # compute log_prob
        if method=="FSCL":
            mask = mask * logits_mask
            logits_mask_fair=logits_mask*(~mask.bool()).float()*sensitive_mask
            exp_logits_fair = torch.exp(logits) * logits_mask_fair
            exp_logits_sum=exp_logits_fair.sum(1, keepdim=True)
            log_prob = logits - torch.log(exp_logits_sum+((exp_logits_sum==0)*1))

        elif method=="SupCon":
            mask = mask * logits_mask
            exp_logits = torch.exp(logits) * logits_mask
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        elif method=="FSCL*":
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
            mask = mask.repeat(anchor_count, contrast_count)
            mask=mask*logits_mask
                
            logits_mask_fair=logits_mask*sensitive_mask
            exp_logits_fair = torch.exp(logits) * logits_mask_fair
            exp_logits_sum=exp_logits_fair.sum(1, keepdim=True)
            log_prob = logits - torch.log(exp_logits_sum+((exp_logits_sum==0)*1))

        elif method=="SimCLR":
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
            mask = mask.repeat(anchor_count, contrast_count)
            mask=mask*logits_mask
            exp_logits = torch.exp(logits) * logits_mask
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

              
        # compute mean of log-likelihood over positive
        #apply group normalization
        if group_norm==1:
            mean_log_prob_pos = ((mask*log_prob)/((mask*sensitive_mask).sum(1))).sum(1)
           
        else:
            mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
    
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos

        #apply group normalization
        if group_norm==1:
            C=loss.size(0)/8
            norm=(1/(((mask*sensitive_mask).sum(1)+1).float()))
            loss=(loss*norm)*C
            
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
