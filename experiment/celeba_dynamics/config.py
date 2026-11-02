"""Configuration reproducing the CelebA scratch UOW dynamics figure."""


SEEDS = (0, 1, 2, 3)
ROOT_DATA = "./data"
EPOCHS = 200
BATCH_SIZE = 256
NUM_WORKERS = 8
LEARNING_RATE = 5e-2
WEIGHT_DECAY = 5e-4
PROJECTION_DIM = 64
TEMPERATURE = 4 / 64
LAMBDA_REG = 0.05
ALPHA = 0.0
