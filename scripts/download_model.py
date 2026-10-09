"""从 ModelScope 下载 Qwen2.5-7B-Instruct 到本地 qwen_model/ 目录。"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.config import CFG


def main():
    from modelscope import snapshot_download
    target = Path(CFG.model_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f">>> Downloading Qwen/Qwen2.5-7B-Instruct -> {target}")
    path = snapshot_download("Qwen/Qwen2.5-7B-Instruct", local_dir=str(target))
    print(f">>> Done: {path}")


if __name__ == "__main__":
    main()
