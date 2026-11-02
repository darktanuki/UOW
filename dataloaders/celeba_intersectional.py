"""Dedicated CelebA loader for the Male x Young intersectional experiment.

The existing CelebA loader is intentionally left unchanged.  This module fixes
the downstream target to ``Attractive`` and exposes the two binary sensitive
attributes as the joint id ``2 * Male + Young`` (values 0, 1, 2, 3).
"""

from collections import Counter

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T

from dataloaders.celeba import CelebAAttributes, SimCLRTwoCropTransform


TARGET_ATTRIBUTE = "Attractive"
BIAS_ATTRIBUTES = ("Male", "Young")
NUM_INTERSECTIONAL_GROUPS = 4


def decode_intersectional_bias(joint_bias):
    """Decode ``2 * Male + Young`` into a final dimension ``[Male, Young]``."""
    joint_bias = torch.as_tensor(joint_bias, dtype=torch.long)
    male = torch.div(joint_bias, 2, rounding_mode="floor")
    young = torch.remainder(joint_bias, 2)
    return torch.stack((male, young), dim=-1)


def intersectional_group_name(joint_bias: int) -> str:
    male, young = decode_intersectional_bias(joint_bias).tolist()
    return f"Male={male}, Young={young}"


class CelebAIntersectionalAttributes(CelebAAttributes):
    """CelebA Attractive target with the joint sensitive label Male x Young."""

    def __init__(self, root, split="train", transform=None):
        super().__init__(
            root=root,
            split=split,
            transform=transform,
            target_attr=TARGET_ATTRIBUTE,
            bias_attr="&".join(BIAS_ATTRIBUTES),
        )


def intersectional_group_counts(dataset: CelebAIntersectionalAttributes) -> dict[int, int]:
    """Count joint groups without decoding image data."""
    male = dataset.ds[BIAS_ATTRIBUTES[0]]
    young = dataset.ds[BIAS_ATTRIBUTES[1]]
    return dict(sorted(Counter(2 * int(m == 1) + int(y == 1) for m, y in zip(male, young)).items()))


def get_celeba_intersectional_loaders(
    root,
    batch_size=256,
    num_workers=8,
    eval_mode=False,
):
    """Return loaders dedicated to the intersectional experiment."""
    eval_transform = T.Compose([
        T.Resize(128),
        T.CenterCrop(128),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_transform = eval_transform if eval_mode else SimCLRTwoCropTransform()

    train_ds = CelebAIntersectionalAttributes(root, split="train", transform=train_transform)
    val_ds = CelebAIntersectionalAttributes(root, split="val", transform=eval_transform)
    test_ds = CelebAIntersectionalAttributes(root, split="test", transform=eval_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=not eval_mode,
        num_workers=num_workers,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train, _, _ = get_celeba_intersectional_loaders("./data", batch_size=256, num_workers=0)
    counts = intersectional_group_counts(train.dataset)  # type: ignore[arg-type]
    for group, count in counts.items():
        print(f"{group}: {intersectional_group_name(group)} -> {count} samples")
    views, bias, target = next(iter(train))
    print("views:", [tuple(view.shape) for view in views])
    print("joint bias:", tuple(bias.shape), torch.unique(bias, return_counts=True))
    print("decoded [Male, Young]:", decode_intersectional_bias(bias[:8]))
    print("target:", tuple(target.shape))
