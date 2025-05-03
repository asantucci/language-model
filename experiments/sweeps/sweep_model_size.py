import yaml
import uuid
from pathlib import Path
import subprocess
import shutil

BASE_TRAIN_CONFIG = Path("config/train/base_pretrain.yaml")
EARLY_STOPPING_CONFIG = Path("config/early_stopping/stop.yaml")
TEMP_CONFIG_DIR = Path("experiments/temp_configs")
TRAIN_SCRIPT = "train/pretrain.py"

TEMP_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

def run_sweep():
    config_dir = Path("experiments/configs")
    for i, config_path in enumerate(sorted(config_dir.glob("*.yaml"))):
        run_id = f"run_{i:03d}_{uuid.uuid4().hex[:6]}"
        with open(BASE_TRAIN_CONFIG) as f:
            train_cfg = yaml.safe_load(f)

        # Inject unique wandb run name
        train_cfg["wandb"]["wandb_run_name"] = run_id
        train_cfg["wandb"]["wandb_project"] = "deepseek-tiny-stories-sweep_model_size"

        # Write modified config to temp dir
        temp_cfg_path = TEMP_CONFIG_DIR / f"{run_id}_train.yaml"
        with open(temp_cfg_path, "w") as f:
            yaml.dump(train_cfg, f)

        log_path = Path("experiments/run_logs") / f"{run_id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"[{run_id}] Launching sweep job with {config_path.name}")
        with open(log_path, "w") as log_file:
            subprocess.run([
                "python3", TRAIN_SCRIPT,
                "--model-config", str(config_path),
                "--train-config", str(temp_cfg_path),
                "--early-stopping", "true",
                "--early-stopping-config", str(EARLY_STOPPING_CONFIG),
            ], stdout=log_file, stderr=subprocess.STDOUT)

if __name__ == "__main__":
    run_sweep()
