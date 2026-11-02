# CelebA Attractive/Male lambda-alpha ablation

This suite evaluates UOW on CelebA with target `Attractive`, sensitive
attribute `Male`, spherical MMD regularization, and the complete grid:

- lambda: `1e-2`, `2.5e-2`, `5e-2`, `7.5e-2`, `1e-1`
- alpha: `0`, `0.25`, `0.5`, `0.75`, `1`
- seed: `0` only

Training uses 8 loader workers and frozen evaluation uses 0. With
`--skip-existing`, the runner reuses saved metrics or a completed checkpoint;
it resumes a partial checkpoint from its last epoch.

From the repository root:

```bash
# Inspect all 25 train/eval pairs.
uv run python experiment/celeba_ablation/run_all.py --device 0 --dry-run

# Test one grid point.
uv run python experiment/celeba_ablation/run_all.py --device 0 --lambdas 5e-2 --alphas 0.5

# Execute or resume the complete seed-0 grid.
uv run python experiment/celeba_ablation/run_all.py --device 0 --skip-existing

# Rebuild the EO and accuracy tables.
uv run python experiment/celeba_ablation/run_all.py --only-tables

# Plot the accuracy--EO Pareto frontier from an existing results.csv.
uv run --no-sync python experiment/celeba_ablation/plot_pareto.py \
  --input ~/Downloads/experiment_cbc/celeba_ablation/results.csv
```

Outputs are stored in this directory and under `artifacts/`.
