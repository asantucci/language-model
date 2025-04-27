import argparse

from train.shared import train_loop

def main():
    parser = argparse.ArgumentParser(description="DeepSeek Supervised Fine-Tuning (SFT) Script")

    # Dataset and model parameters
    parser.add_argument("--dataset", type=str, default="fineweb_edu")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)

    # Optimization parameters
    parser.add_argument("--max-train-steps", type=int, default=30_000)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--min-learning-rate", type=float, default=5e-5)
    parser.add_argument("--decay-lr", type=bool, default=True)
    parser.add_argument("--scheduler-type", type=str, default="cosine", choices=["cosine", "linear"])
    parser.add_argument("--adamw-beta1", type=float, default=0.9)
    parser.add_argument("--adamw-beta2", type=float, default=0.95)
    parser.add_argument("--adamw-weight-decay", type=float, default=0.1)
    parser.add_argument("--adamw-use-fused", type=bool, default=True)
    parser.add_argument("--use-eight-bit-optimizer", type=bool, default=True)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    # Evaluation and checkpointing
    parser.add_argument("--eval-iters", type=int, default=5)
    parser.add_argument("--eval-interval", type=int, default=1000)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=1_000)
    parser.add_argument("--out-dir", type=str, default="./checkpoints")
    parser.add_argument("--checkpoint-path", type=str, default="sft_ckpt")
    parser.add_argument("--resume", type=str, default="")

    # WandB logging
    parser.add_argument("--wandb-log", type=bool, default=True)
    parser.add_argument("--wandb-project", type=str, default="deepseek-sft")
    parser.add_argument("--wandb-run-name", type=str, default="run")

    args = parser.parse_args()
    train_loop(args, mode="sft")

if __name__ == "__main__":
    main()
