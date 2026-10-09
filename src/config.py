"""全局配置：路径、模型、训练超参。"""
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    swdd_depressed: str = str(ROOT / "data" / "swdd" / "depressed.jsonl")
    swdd_control: str = str(ROOT / "data" / "swdd" / "control.jsonl")
    reddit_csv: str = str(ROOT / "data" / "reddit" / "reddit_depression_suicidewatch.csv")
    model_path: str = str(ROOT / "qwen_model" / "Qwen" / "Qwen2.5-7B-Instruct")
    swdd_users_per_class: int = 800
    swdd_posts_per_user: int = 15
    reddit_train_size: int = 6000
    reddit_test_size: int = 1000
    val_ratio: float = 0.1
    test_ratio: float = 0.15
    max_length: int = 512
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    num_train_epochs: int = 3
    learning_rate: float = 1e-4
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    few_shot_k: int = 6
    max_new_tokens: int = 8
    seed: int = 42
    output_dir: str = str(ROOT / "outputs")


CFG = Config()
