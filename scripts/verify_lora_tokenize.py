# -*- coding: utf-8 -*-
"""验证 LoRA 训练样本的 tokenize：三分类答案保留 + loss masking 正确。"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from transformers import AutoTokenizer
from scripts.run_lora import tokenize_for_training
from src.data_utils import load_swdd, make_splits
from src.config import CFG
from src.prompts import ANSWER_WORDS

tok = AutoTokenizer.from_pretrained(CFG.model_path, trust_remote_code=True)
df = load_swdd()
tr, _, _ = make_splits(df)
samples = pd.concat([tr[tr["label"] == c].head(3) for c in [0, 1, 2]]).reset_index(drop=True)
ds = tokenize_for_training(samples, tok, "SWDD", CFG)

ok = True
for i in range(len(samples)):
    ex = ds[i]
    ids = ex["input_ids"]
    labels = ex["labels"]
    supervised = [t for t in labels if t != -100]
    supervised_text = tok.decode(supervised)
    n_masked = sum(1 for t in labels if t == -100)
    lab = int(samples.iloc[i]["label"])
    ans = ANSWER_WORDS["SWDD"][lab]
    pass_ans = ans in supervised_text
    ok = ok and pass_ans
    print(f"样本{i} (标签={lab}, 答案={ans}): 总长={len(ids)}, mask={n_masked}, "
          f"监督token={len(supervised)}, 监督段={supervised_text!r} {'OK' if pass_ans else 'FAIL'}")

print("\n" + ("全部通过" if ok else "存在失败样本"))
sys.exit(0 if ok else 1)
