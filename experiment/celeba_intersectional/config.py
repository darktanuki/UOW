"""Configuration for the CelebA Male/Young intersectional experiment."""

from dataclasses import dataclass


SEEDS = (0, 1, 2)
ROOT_DATA = "./data"
EPOCHS = 30
BATCH_SIZE = 256
NUM_WORKERS = 8
EVAL_NUM_WORKERS = 0
LEARNING_RATE = 5e-2
WEIGHT_DECAY = 5e-4
PROJECTION_DIM = 64
TEMPERATURE = 4 / 64
LAMBDA_REG = 0.05
ALPHA = 0.5
LINEAR_PROBE_C = 0.01
MIN_GROUP_SIZE = 2


@dataclass(frozen=True)
class Method:
    key: str
    method: str
    conditioning: str
    loss_name: str
    grouping: str
    alpha: float | None


METHODS = (
    Method("ow", "OW", "Global only", "ow", "joint", None),
    Method("uow_male", "UOW", "Male only", "uow_intersectional", "male", ALPHA),
    Method("uow_separate", "UOW", "Male and Young separately", "uow_intersectional", "marginal", ALPHA),
    Method("uow_joint", "UOW", r"Male $\times$ Young jointly", "uow_intersectional", "joint", ALPHA),
)
