"""Decide whether a Lightning checkpoint needs more training."""

from pathlib import Path


def checkpoint_complete(path: Path, epochs: int) -> bool:
    """A last checkpoint is complete after the final configured epoch."""
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return int(checkpoint["epoch"]) >= epochs - 1
