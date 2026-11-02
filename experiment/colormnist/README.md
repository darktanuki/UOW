# Colored MNIST

This suite trains OW and UOW with Gaussian and spherical target distributions
over seeds 0, 1, and 2, then evaluates every frozen encoder with both linear
probing and k-NN. UOW uses `alpha=0.0` in both target geometries.

Shared parameters are defined in `config.py`. The suite uses 100 epochs, batch
size 256, learning rate 0.25, projection dimension 64, temperature `4/64`, and
`lambda_reg=0.05`.
Training uses 8 loader workers and frozen evaluation uses 0. With
`--skip-existing`, the runner reuses saved metrics or a completed checkpoint;
it resumes a partial checkpoint from its last epoch.

From the repository root:

```bash
# Inspect the 12 train/eval command pairs.
uv run python experiment/colormnist/run_all.py --device 0 --dry-run

# Execute or resume the complete suite.
uv run python experiment/colormnist/run_all.py --device 0 --skip-existing

# Test one run first.
uv run python experiment/colormnist/run_all.py --device 0 --methods uow_gaussian --seeds 0

# Rebuild both tables from results.csv.
uv run python experiment/colormnist/run_all.py --only-tables
```

Outputs are kept here: `results.csv`, `table_knn.tex`,
`table_linear_probe.tex`, and the `artifacts/` directory.
