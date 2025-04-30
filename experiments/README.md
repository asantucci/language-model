# LLM Sweep and Rerun Research Framework

This repository implements a research-grade hyperparameter sweep, analysis, rerun, and artifact management system for language model pretraining and fine-tuning experiments.

It provides tooling to:
- Launch hyperparameter sweeps over learning rate, warmup percentage, and other training hyperparameters
- Analyze sweep results using both early loss trajectory and final loss
- Select and relaunch best configurations for full training
- Perform focused resampling around promising hyperparameter regions
- Track all generated configs and model artifacts cleanly

## Overall Workflow
1. **Launch Sweep**
    - Run random or structured sweeps with `sweeps/sweep_pretrain_lr_warmup.py`.
    - Configs loaded dynamically from modular YAML files in the top-level directory's `config/` folder.

2. **Analyze Sweep Results**
    - Run `analysis/sweep_analysis.py` to:
        - Compute early loss slopes and composite scores.
        - Rank sweep runs.
        - Output top-k best hyperparameter configurations.

3. **Rerun Best Hyperparameters**
    - Run `rerun/rerun_best_hyperparams.py`:
        - Restart full training from scratch.
        - Use best hyperparameters from analysis.
        - Save all rerun configs to `rerun/rerun_configs/` for reproducibility.

4. **Resample Focused Sweeps (Optional)**
    - Run `sweeps/sweep_resample_focused.py`:
        - Focus additional sweeps around best hyperparameter regions.

5. **Upload Artifacts**
    - Run `rerun/upload_rerun_configs.py`:
        - Upload rerun configs as versioned WandB Artifacts.
        - (Optional) Upload best model checkpoints too.

---

### Example Flow
```
# 1. Launch initial sweep
python sweeps/sweep_pretrain_lr_warmup.py

# 2. Analyze sweep and select best configs
python analysis/sweep_analysis.py \
  --wandb-csv path/to/wandb_export.csv \
  --output-yaml rerun/rerun_configs/top5_best_configs.yaml

# 3. Relaunch top configs for full training
python rerun/rerun_best_hyperparams.py \
  --best-yaml rerun/rerun_configs/top5_best_configs.yaml

# 4. (Optional) Focused resweep
python sweeps/sweep_resample_focused.py \
  --best-yaml rerun/rerun_configs/top5_best_configs.yaml

# 5. Upload rerun configs to WandB Artifacts
python rerun/upload_rerun_configs.py \
  --config-dir rerun/rerun_configs \
  --artifact-name top5_rerun_configs_v1 \
  --project my-wandb-project
```
## Notes
  - Training runs can optionally resume from saved checkpoints, but by default, reruns restart from scratch for full reproducibility.

  - Sweep runs use early stopping heuristics via TrainingLossMonitor to avoid wasting compute on bad configurations.

  - Modular YAML configs allow easy swapping between model sizes, training settings, and sweep parameters.

  - All experiments are designed to be auditable and reproducible via artifact tracking.

### Future Improvements
  - Auto-upload best model checkpoints as artifacts

  - Smarter adaptive focused resweeps based on convergence rates

  - Full multi-phase sweep orchestration controller
  
  - Fine-tuning best models after pretraining reruns