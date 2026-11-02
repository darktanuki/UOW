# CelebA scratch UOW dynamics

This suite produces `celeba_scratch_uow_alpha0p5_early_stopping_dynamics` with
CelebA Attractive/Male, spherical UOW, `alpha=0.5`, `lambda_reg=0.05`, a
randomly initialized ResNet-18, 200 epochs, and seeds 0, 1, and 2.
Training keeps 8 loader workers. With `--skip-existing`, finished validation
histories are reused and partial training resumes from `last.ckpt`.

```bash
# Inspect the four commands.
uv run python experiment/celeba_dynamics/run_all.py --device 0 --dry-run

# Run or resume all seeds, then regenerate every dynamics plot.
uv run python experiment/celeba_dynamics/run_all.py --device 0 --skip-existing

# Regenerate plots from existing metrics.
uv run python experiment/celeba_dynamics/run_all.py --only-plot
```

The requested PNG/PDF and the three companion dynamics figures are written in
this directory. Per-seed logs, validation histories, and checkpoints live under
`artifacts/runs/`.

To reproduce the original `alpha=0` early-stopping figure directly from a
directory of completed multi-seed Lightning runs:

```bash
uv run --no-sync python experiment/celeba_dynamics/plot_early_stopping_from_logs.py \
  --runs-dir ~/Downloads/results_cbc/celeba_scratch_uow_alpha0
```

The script discovers `seed_*/training_logs/version_*/metrics.csv`, selects the
most complete log for each seed, and writes the aggregated PDF and PNG to this
directory. Use `--transition-epoch` to move the dashed early-stopping marker.
