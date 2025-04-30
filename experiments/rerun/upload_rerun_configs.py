"""
python rerun/upload_rerun_configs.py \
  --config-dir rerun/rerun_configs \
  --artifact-name top5_rerun_configs_v1 \
  --project deepseek_sweep
"""
import os
import wandb
import argparse

def upload_rerun_configs(rerun_config_dir, artifact_name, project_name="my-wandb-project"):
    """
    Uploads all rerun configuration files as a WandB Artifact.

    Args:
        rerun_config_dir (str): Directory containing rerun YAML configs.
        artifact_name (str): Name to assign to the artifact.
        project_name (str): WandB project to log into.
    """
    # Initialize a temporary run just for uploading
    run = wandb.init(project=project_name, job_type="upload_rerun_configs")

    artifact = wandb.Artifact(artifact_name, type="rerun-configs")

    # Add all .yaml files in the rerun config directory
    for file_name in os.listdir(rerun_config_dir):
        if file_name.endswith(".yaml"):
            file_path = os.path.join(rerun_config_dir, file_name)
            artifact.add_file(file_path)

    run.log_artifact(artifact)
    run.finish()

    print(f"✅ Uploaded all configs in {rerun_config_dir} as artifact '{artifact_name}'.")

def main():
    parser = argparse.ArgumentParser(description="Upload rerun configs as WandB Artifact.")
    parser.add_argument("--config-dir", type=str, required=True, help="Path to rerun configs directory.")
    parser.add_argument("--artifact-name", type=str, required=True, help="Name to assign to the artifact.")
    parser.add_argument("--project", type=str, required=True, help="WandB project name.")
    args = parser.parse_args()

    upload_rerun_configs(args.config_dir, args.artifact_name, args.project)

if __name__ == "__main__":
    main()
