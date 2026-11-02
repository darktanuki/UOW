"""Configuration for the CelebA Attractive/Male lambda-alpha ablation."""


TARGET_EXPERIMENT = "T=a/S=m"
SEED = 0

LAMBDAS = (
    ("1e-2", 1e-2, r"10^{-2}"),
    ("2.5e-2", 2.5e-2, r"2.5\times10^{-2}"),
    ("5e-2", 5e-2, r"5\times10^{-2}"),
    ("7.5e-2", 7.5e-2, r"7.5\times10^{-2}"),
    ("1e-1", 1e-1, r"10^{-1}"),
)
ALPHAS = (("0", 0.0), ("0.25", 0.25), ("0.5", 0.5), ("0.75", 0.75), ("1", 1.0))

ROOT_DATA = "./data"
REG_NAME = "mmd"
KERNEL_NAME = "heat"
EPOCHS = 30
BATCH_SIZE = 256
NUM_WORKERS = 8
EVAL_NUM_WORKERS = 0
LEARNING_RATE = 5e-2
WEIGHT_DECAY = 5e-4
PROJECTION_DIM = 64
TEMPERATURE = 4 / 64
LINEAR_PROBE_C = 0.01
