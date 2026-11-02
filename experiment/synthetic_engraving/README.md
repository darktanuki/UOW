# Controlled Evaluation on Synthetic Engraving

This suite has two phases with shared settings: KL regularization, heat kernel,
projection dimension 256, and temperature `2/256`.

1. `ablation`: seed 0, `m=2`, three lambda values × five alpha values.
2. `comparison`: OW versus UOW with `lambda=1/3`, UOW `alpha=0.25`, six
   values of `m`, and seeds 0, 1, and 2.

The UOW point `(m=2, lambda=1/3, alpha=0.25, seed=0)` belongs to both phases
and is trained only once. Thus `--phase all` launches 50 unique runs rather
than duplicating this exact configuration.

From the repository root:

```bash
# Inspect every command without training.
uv run python experiment/synthetic_engraving/run_all.py --device 0 --dry-run

# Run the ablation first.
uv run python experiment/synthetic_engraving/run_all.py --device 0 --phase ablation --skip-existing

# Then run the OW/UOW comparison (the shared point will be reused).
uv run python experiment/synthetic_engraving/run_all.py --device 0 --phase comparison --skip-existing

# Or run/resume the complete suite.
uv run python experiment/synthetic_engraving/run_all.py --device 0 --phase all --skip-existing

# Rebuild both LaTeX tables from the CSV.
uv run python experiment/synthetic_engraving/run_all.py --only-tables
```

Training uses 8 data-loader workers; the final test uses 0. With
`--skip-existing`, a completed result is skipped, and an existing best
checkpoint is reused for a missing test instead of retraining. Test metrics
are also saved beside the checkpoint in `test_metrics.csv`.

Each training command also evaluates the best checkpoint. `results.csv` stores
the best validation `mAP/emb`, its epoch, both test mAP values, paths, and run
duration. Percentages are stored on a 0–100 scale.

Generated artifacts stay in this directory:

- `results.csv`
- `table_ablation.tex`
- `table_comparison.tex`
- `artifacts/logs/`
- `artifacts/checkpoints/`

The experiment-specific trainer is `train.py` in this directory.

## ViT-S/8 sweep over m at 1e-4

Run UOW for every configured value of `m` and seeds 0, 1, and 2 with pretrained
ViT-S/8, `lambda=1/3`, `alpha=0.25`, and learning rate `1e-4`:

```bash
uv run python experiment/synthetic_engraving/run_all.py --backbone vit_s8 --lr 1e-4 --device 0 --phase comparison --losses uow --skip-existing
```

The `m=2`, seed-0 UOW point (`lambda=1/3`, `alpha=0.25`) is imported from the
completed `temp_vit_s8_lr_sweep` run at `1e-4` when its CSV, checkpoint, and
test metrics are available locally. This avoids training that point again.
Use `--phase all` only when the 15-point lambda/alpha ablation is also wanted;
its other seed-0 configurations have different hyperparameters and will train.
Use `--dry-run` to inspect the commands first. The ViT outputs are kept in
`experiment/synthetic_engraving/vit_s8_lr_0.0001/` (results, tables, logs, and
checkpoints). Other training settings match the ConvNeXt suite. If using a
machine without a GPU, omit `--device 0` and use `--device auto`.

## Temporary ViT-S/8 learning-rate sweep

`train.py` accepts `--backbone dinov3_convnext` (default) or `--backbone vit_s8`.
The ViT-S/8 uses the pretrained `facebook/dino-vits8` weights.
Both backbones use the same 256-dimensional projection head. Their checkpoint
names are distinct when training directly with `train.py`.

To run the temporary ViT-S/8 sweep sequentially with `m=2`, seed 0, UOW,
`lambda=1/3`, and `alpha=0.25`:

```bash
uv run python experiment/synthetic_engraving/temp_vit_s8_lr_sweep.py --device 0
```

The script tries learning rates `1e-4`, `5e-4`, `1e-3`, `5e-3`, `1e-2`, and
`5e-2`, then prints a small table of the best validation `mAP|c` on backbone
embeddings. Runs and a resumable `results.csv` are saved under
`experiment/synthetic_engraving/temp_vit_s8_lr_sweep/`. Use `--skip-existing`
to reuse completed runs, `--dry-run` to inspect commands, or `--only-table` to
reprint the table. Omit `--device 0` when running on CPU.
