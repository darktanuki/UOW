"""Single source of truth for the Colored MNIST evaluation suite."""

from dataclasses import dataclass


SEEDS = (0, 1, 2)
EPOCHS = 100
BATCH_SIZE = 256
NUM_WORKERS = 8
EVAL_NUM_WORKERS = 0
LEARNING_RATE = 0.25
WEIGHT_DECAY = 5e-4
PROJECTION_DIM = 64
TEMPERATURE = 4 / 64
LAMBDA_REG = 0.05
LINEAR_PROBE_C = 1.0
ROOT_DATA = "./data"


@dataclass(frozen=True)
class Method:
    key: str
    title: str
    target_distribution: str
    loss_name: str
    reg_name: str
    alpha: float | None = None


METHODS = (
    Method("ow_gaussian", "OW Baseline", r"Gaussian $\mathcal{N}(0, I)$", "ow", "sigreginduced"),
    Method("uow_gaussian", "UOWReg (Ours)", r"Gaussian $\mathcal{N}(0, I)$", "uow", "sigreginduced", 0.0),
    Method("ow_spherical", "OW Baseline", r"Spherical $\mathcal{U}(\mathbb{S}^{d-1})$", "ow", "mmd"),
    Method("uow_spherical", "UOWReg (Ours)", r"Spherical $\mathcal{U}(\mathbb{S}^{d-1})$", "uow", "mmd", 0.0),
)
