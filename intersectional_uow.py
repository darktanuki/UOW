"""UOW variant dedicated to the CelebA Male x Young experiment."""

import torch
import torch.nn.functional as F

from uow import OWReg, RAW_EMBEDDING_REG_NAMES, UOWReg, _alpha_from_params, _unit_interval


GROUPING_MODES = {"male", "young", "marginal", "joint", "all"}


class IntersectionalUOWReg(UOWReg):
    """Apply UOW to marginal and/or joint Male x Young groups."""

    def __init__(self, params) -> None:
        super().__init__(params)
        self.grouping = getattr(params, "intersectional_grouping", "all")
        self.min_group_size = int(getattr(params, "min_group_size", 2))
        if self.grouping not in GROUPING_MODES:
            raise ValueError(
                f"Unknown intersectional_grouping={self.grouping!r}; "
                f"expected one of {sorted(GROUPING_MODES)}"
            )
        if self.min_group_size < 1:
            raise ValueError("min_group_size must be at least 1")

    @staticmethod
    def _decode(joint_labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if joint_labels.ndim != 1:
            raise ValueError(
                "IntersectionalUOWReg expects joint ids with shape [B], "
                f"got {tuple(joint_labels.shape)}"
            )
        joint_labels = joint_labels.long()
        if joint_labels.numel() and (joint_labels.min() < 0 or joint_labels.max() > 3):
            raise ValueError("joint bias ids must be in {0, 1, 2, 3}")
        return joint_labels // 2, joint_labels % 2

    def grouping_labels(self, joint_labels: torch.Tensor) -> list[torch.Tensor]:
        male, young = self._decode(joint_labels)
        if self.grouping == "male":
            return [male]
        if self.grouping == "young":
            return [young]
        if self.grouping == "marginal":
            return [male, young]
        if self.grouping == "all":
            return [male, young, joint_labels.long()]
        return [joint_labels.long()]

    def _conditional_regularization(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        batch_size = z.shape[0]
        losses = []
        for group in torch.unique(labels):
            mask = labels == group
            count = int(mask.sum().item())
            if count < self.min_group_size:
                continue
            # Match the prevalence-weighted expectation used by the base UOW loss.
            losses.append(self.reg(z[mask]) * (count / batch_size))
        if not losses:
            return z.sum() * 0.0
        return torch.stack(losses).sum()

    def forward(self, z: torch.Tensor, joint_labels: torch.Tensor) -> torch.Tensor:
        if z.ndim != 2:
            raise ValueError(f"z must have shape [B, D], got {tuple(z.shape)}")
        if joint_labels.shape[0] != z.shape[0]:
            raise ValueError("joint_labels batch dimension must match z")
        if self.reg_name not in RAW_EMBEDDING_REG_NAMES:
            z = F.normalize(z, dim=-1)

        conditional_terms = [
            self._conditional_regularization(z, labels)
            for labels in self.grouping_labels(joint_labels)
        ]
        conditional = torch.stack(conditional_terms).mean()
        if self.reg_name in {"mmd", "ksd", "sigreg", "sigreginduced"}:
            conditional = conditional * z.shape[0]
        return conditional


class IntersectionalUOW(torch.nn.Module):
    """Multi-view invariance plus the dedicated intersectional regularizer."""

    def __init__(self, params) -> None:
        super().__init__()
        self.lambd = _unit_interval(getattr(params, "lambda_reg", 0.5), "lambda_reg")
        self.alpha = _alpha_from_params(params)
        self.nb_global_views = getattr(params, "nb_global_views", 2)
        self.uow_reg = IntersectionalUOWReg(params)
        self.ow_reg = OWReg(params)
        self.reg = self.uow_reg

    def forward(self, zs, joint_labels, mode="train"):
        mu = torch.stack(zs[:self.nb_global_views], dim=0).mean(dim=0, keepdim=True)
        stacked = torch.stack(zs, dim=0)
        if self.uow_reg.reg_name in RAW_EMBEDDING_REG_NAMES:
            invariance = 0.5 * (stacked - mu).pow(2).mean()
        else:
            invariance = 1.0 - (stacked * mu).sum(dim=-1).mean()

        uow_regularization = torch.stack([self.uow_reg(z, joint_labels) for z in zs]).mean()
        ow_regularization = torch.stack([self.ow_reg(z) for z in zs]).mean()
        conditional_weight = self.lambd * (1.0 - self.alpha)
        global_weight = self.lambd * self.alpha
        loss = (
            (1.0 - self.lambd) * invariance
            + conditional_weight * uow_regularization
            + global_weight * ow_regularization
        )
        return loss, {
            f"{mode}/Linv": invariance.item(),
            f"{mode}/LUOWReg": uow_regularization.item(),
            f"{mode}/LOWReg": ow_regularization.item(),
            f"{mode}/Lreg": (
                conditional_weight * uow_regularization
                + global_weight * ow_regularization
            ).item(),
            f"{mode}/L": loss.item(),
        }
