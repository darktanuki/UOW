"""Configuration for the controlled Synthetic Engraving experiments."""


SEEDS = (0, 1, 2)
M_VALUES = (2, 4, 8, 16, 32, 128)

# Ablation: seed 0 and m=2.
LAMBDAS = (("1/4", 1 / 4), ("1/3", 1 / 3), ("1/2", 1 / 2))
ALPHAS = (("0", 0.0), ("0.25", 0.25), ("0.5", 0.5), ("0.75", 0.75), ("1", 1.0))

# Main OW/UOW comparison.
FIXED_LAMBDA_LABEL = "1/3"
FIXED_LAMBDA = 1 / 3
FIXED_ALPHA_LABEL = "0.25"
FIXED_ALPHA = 0.25

# Shared optimization and geometry parameters.
REG_NAME = "kl"
KERNEL_NAME = "heat"
PROJECTION_DIM = 256
TEMPERATURE = 2 / 256
EPOCHS = 40
BATCH_SIZE = 128
NUM_WORKERS = 8
EVAL_NUM_WORKERS = 0
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 5e-4
EARLY_STOPPING_PATIENCE = 8
