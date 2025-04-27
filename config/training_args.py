from dataclasses import dataclass

@dataclass
class TrainingArgs:
    device: str
    dtype: str
    wandb_log: bool
    wandb_project: str
    wandb_run_name: str
    log_interval: int
    save_interval: int
    eval_interval: int
    eval_iters: int
    checkpoint_path: str
    out_dir: str
    model_config_path: str
    resume: str
    seq_len: int
    batch_size: int
    max_train_steps: int
    gradient_accumulation_steps: int
    grad_clip: float
    hf_dataset_name: str
    hf_dataset_dir: str
    adamw_weight_decay: float
    adamw_beta1: float
    adamw_beta2: float
    adamw_use_fused: bool
    use_eight_bit_optimizer: bool
    learning_rate: float
    min_learning_rate: float
    decay_lr: bool
    lr_decay_iters: 1
    scheduler_type: str
    pretrained_tokenizer_name: str
    pretrained_tokenizer_max_length: str
    generate_interval: int
    
