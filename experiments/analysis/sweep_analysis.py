"""
Example usage:
uv run experiments/analysis/sweep_analysis.py \
    --wandb-csv=/tmp/training_history_long.csv \
    --output-yaml=/tmp/best_configs
"""
import pandas as pd
import numpy as np
import os
import argparse
import yaml

def compute_early_slope(df, max_step=2000):
    """
    Computes early loss slope (rate of decrease) over first max_step steps.

    Args:
        df (DataFrame): WandB export dataframe for a single run.
        max_step (int): Max step to consider for early behavior.

    Returns:
        float: Slope (negative is good: faster loss decrease).
    """
    early_df = df[df['Step'] <= max_step]
    if early_df.empty:
        return 0.0

    x = early_df['Step'].values
    y = early_df['Train Loss'].values

    A = np.vstack([x, np.ones_like(x)]).T
    slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    return slope

def compute_composite_score(early_slope, final_loss, w1=1.0, w2=1.0):
    """
    Computes a composite score combining early loss decrease and final loss.

    Args:
        early_slope (float): Early loss slope (negative = good).
        final_loss (float): Final loss at last step.
        w1 (float): Weight for early slope.
        w2 (float): Weight for final loss.

    Returns:
        float: Composite score (higher is better).
    """
    return (-w1 * early_slope) - (w2 * final_loss)

def analyze_sweep(wandb_csv_path, output_yaml_path, top_k=5, early_max_step=2000):
    """
    Analyzes sweep results and outputs top-k hyperparameter configs.

    Args:
        wandb_csv_path (str): Path to WandB training loss data output from `fetch_wandb_runs.py`.
        output_yaml_path (str): Where to save best configs YAML.
        top_k (int): Number of best configs to select.
        early_max_step (int): Maximum step for early slope calculation.
    """
    df = pd.read_csv(wandb_csv_path)

    runs = []
    for run_id in df['run_name'].unique():
        run_df = df[df['run_name'] == run_id]

        early_slope = compute_early_slope(run_df, max_step=early_max_step)
        final_loss = run_df['Train Loss'].iloc[-1]
        score = compute_composite_score(early_slope, final_loss)

        # Assume learning_rate and pct_warmup were logged
        example_row = run_df.iloc[0]
        run_info = {
            "run_name": run_id,
            "learning_rate": float(example_row["learning_rate"]),
            "warmup": float(example_row["warmup"]),
            "final_loss": float(final_loss),
            "early_slope": float(early_slope),
            "score": float(score),
        }
        runs.append(run_info)

    # Rank by score descending
    runs = sorted(runs, key=lambda x: x['score'], reverse=True)

    print(f"Top {top_k} Runs:")
    for run in runs[:top_k]:
        print(f"  {run['run_name']}: score={run['score']:.4f}")

    # Output top configs to YAML
    best_points = [{"learning_rate": r["learning_rate"], "pct_warmup": r["warmup"]} for r in runs[:top_k]]
    
    os.makedirs(os.path.dirname(output_yaml_path), exist_ok=True)
    with open(output_yaml_path, "w") as f:
        yaml.dump(best_points, f)

    print(f"\nSaved top-{top_k} best hyperparameters to {output_yaml_path}")

def main():
    parser = argparse.ArgumentParser(description="Analyze sweep results and select top hyperparameter configs.")
    parser.add_argument("--wandb-csv", type=str, required=True, help="Path to exported wandb CSV.")
    parser.add_argument("--output-yaml", type=str, required=True, help="Path to output YAML of best configs.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top configs to select.")
    parser.add_argument("--early-max-step", type=int, default=2000, help="Steps to use for early slope computation.")
    args = parser.parse_args()

    analyze_sweep(args.wandb_csv, args.output_yaml, top_k=args.top_k, early_max_step=args.early_max_step)

if __name__ == "__main__":
    main()
