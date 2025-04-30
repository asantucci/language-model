import wandb
import pandas as pd

def parse_lr_and_warmup(run_name: str):
    """
    Parses a run_name of the form 'lr_<lr>_warmup_<warmup>' and returns floats.

    Args:
        run_name (str): Name string formatted as 'lr_<value>_warmup_<value>'

    Returns:
        Tuple[float, float]: (learning_rate, pct_warmup)
    """
    try:
        parts = run_name.strip().split("_")
        lr = float(parts[1])
        warmup = float(parts[3])
        return lr, warmup
    except (IndexError, ValueError):
        return None, None  # Defensive fallback

def fetch_wandb_runs_with_history(entity_project: str, metric="Train Loss", filter_finished=True):
    """
    Fetches config, summary, and full training history for each run in a WandB project.

    Args:
        entity_project (str): Format 'entity/project'
        metric (str): Metric name to fetch from training logs
        filter_finished (bool): If True, only include finished runs

    Returns:
        (pd.DataFrame, pd.DataFrame):
            - runs_df: summary/config metadata per run
            - history_df: long-format training history per step
    """
    api = wandb.Api()
    runs = api.runs(entity_project)

    summary_list, config_list, name_list = [], [], []
    history_rows = []

    for run in runs:
        if filter_finished and run.state != "finished":
            continue

        print(f"Processing run: {run.name}")

        # Run-level metadata
        config = {k: v for k, v in run.config.items() if not k.startswith("_")}
        summary = run.summary._json_dict
        run_name = run.name

        summary_list.append(summary)
        config_list.append(config)
        name_list.append(run_name)

        # Step-level log history
        try:
            history_df = run.history(keys=["Step", metric], pandas=True)
        except Exception as e:
            print(f"Failed to fetch history for run {run_name}: {e}")
            continue

        lr, warmup = parse_lr_and_warmup(run_name)
        for _, row in history_df.iterrows():
            if pd.isna(row.get(metric)):
                continue
            history_rows.append({
                "run_name": run_name,
                "learning_rate": lr,
                "warmup": warmup,
                "Step": row["Step"],
                metric: row[metric],
            })

    runs_df = pd.DataFrame({
        "summary": summary_list,
        "config": config_list,
        "name": name_list
    })

    history_df = pd.DataFrame(history_rows)

    print(f"Collected metadata for {len(runs_df)} runs")
    print(f"Collected {len(history_df)} training log steps")
    return runs_df, history_df

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download WandB sweep metadata + training histories.")
    parser.add_argument("--project", type=str, required=True, help="WandB project, e.g. 'entity/project-name'")
    parser.add_argument("--metric", type=str, default="Train Loss", help="Which metric to extract from training logs")
    parser.add_argument("--metadata-out", type=str, default="runs_metadata.csv")
    parser.add_argument("--history-out", type=str, default="training_history_long.csv")
    args = parser.parse_args()

    runs_df, history_df = fetch_wandb_runs_with_history(args.project, args.metric)
    runs_df.to_csv(args.metadata_out, index=False)
    history_df.to_csv(args.history_out, index=False)

    print(f"Saved run metadata to {args.metadata_out}")
    print(f"Saved training history to {args.history_out}")
