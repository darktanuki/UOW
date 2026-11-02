"""Single source of truth for the real-world benchmark suite."""

from dataclasses import dataclass


# Shared training parameters. Method-specific coefficients live in METHODS.
SEEDS = (0, 1, 2)
EPOCHS = 30
BATCH_SIZE = 256
LEARNING_RATE = 5e-2
WEIGHT_DECAY = 5e-4
PROJECTION_DIM = 64
TEMPERATURE = 4 / 64
LAMBDA_REG = 0.05
NUM_WORKERS = 8
EVAL_NUM_WORKERS = 0


@dataclass(frozen=True)
class Task:
    key: str
    dataset: str
    title: str
    root_data: str
    experiment: str | None
    probe_c: float


@dataclass(frozen=True)
class Method:
    key: str
    title: str
    section: str
    loss_name: str
    reg_name: str | None
    alpha: float | None


TASKS = (
    Task("celeba_attractive_male", "celeba", "CelebA — Attractive / Male", "./data", "T=a/S=m", 0.01),
    Task("celeba_bignose_male", "celeba", "CelebA — Big Nose / Male", "./data", "T=b/S=m", 0.001),
    Task("waterbirds", "birds", "Waterbirds — Bird type / Background", "./data/waterbird_complete95_forest2water2", None, 1.0))


# Gaussian OW/UOW use SIGReg-Induced. Spherical OW/UOW use the standard MMD.
# Only UOW consumes alpha; unused arguments are deliberately left unset.
METHODS = (
    Method("ce", "CE", "supervised", "crossentropy", None, None),
    Method("fscl_plus", r"FSCL$+$", "supervised_debiasing", "fscl+", None, None),
    Method("lejepa_gaussian", "LeJEPA (Gaussian)", "unsupervised", "lejepa", "sigreg", None),
    Method("ow_gaussian", "Baseline OW (Gaussian)", "unsupervised", "ow", "sigreginduced", None),
    Method("ow_spherical", "Baseline OW (Spherical)", "unsupervised", "ow", "mmd", None),
    Method("fscl_dagger", r"FSCL$^\dagger$", "ssl_debiasing", "fscl*", None, None),
    Method("end", "EnD", "ssl_debiasing", "endunsupervised", None, None),
    Method("uow_gaussian", "UOW (Gaussian)", "ssl_debiasing", "uow", "sigreginduced", 0.5),
    Method("uow_spherical", "UOW (Spherical)", "ssl_debiasing", "uow", "mmd", 0.5),
)

SECTION_TITLES = {
    "supervised": "Supervised (Requires labels during training)",
    "supervised_debiasing": "Supervised Debiasing (Requires labels during training)",
    "unsupervised": "Fully Unsupervised",
    "ssl_debiasing": "Self-Supervised Debiasing (Requires bias labels only)",
}
