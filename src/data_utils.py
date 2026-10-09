"""数据加载与预处理：把两个来源不同的数据集统一成一个 DataFrame。"""
import json
import re
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from .config import CFG

_HTML_TAG = re.compile(r"<[^>]+>")
_URL = re.compile(r"https?://\S+|www\.\S+")
_MULTI_SPACE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = _HTML_TAG.sub(" ", text)
    text = _URL.sub(" ", text)
    text = text.replace("转发微博", " ")
    text = _MULTI_SPACE.sub(" ", text)
    return text.strip()


def _extract_user_text(obj, cfg):
    tweets = obj.get("tweets", [])
    posts = []
    if isinstance(tweets, list):
        for tw in tweets:
            if isinstance(tw, dict):
                t = clean_text(tw.get("text", "") or tw.get("content", ""))
                if len(t) > 5:
                    posts.append(t)
            if len(posts) >= cfg.swdd_posts_per_user:
                break
    return " ; ".join(posts)


def load_swdd(cfg=CFG) -> pd.DataFrame:
    cap = cfg.swdd_users_per_class
    rows = []
    n0 = 0
    with open(cfg.swdd_control, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if n0 >= cap:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            user_text = _extract_user_text(obj, cfg)
            if len(user_text) > 20:
                rows.append({"text": user_text, "label": 0, "dataset": "SWDD"})
                n0 += 1
    n1, n2 = 0, 0
    with open(cfg.swdd_depressed, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if n1 >= cap and n2 >= cap:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            suicidal = bool(obj.get("label", {}).get("symptoms", {}).get("suicidal_ideation", False))
            lab = 2 if suicidal else 1
            if (lab == 1 and n1 >= cap) or (lab == 2 and n2 >= cap):
                continue
            user_text = _extract_user_text(obj, cfg)
            if len(user_text) > 20:
                rows.append({"text": user_text, "label": lab, "dataset": "SWDD"})
                if lab == 1:
                    n1 += 1
                else:
                    n2 += 1
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)
    print(f"[SWDD] 三分类 total={len(df)}")
    print("  各类样本数:", df["label"].value_counts().sort_index().to_dict())
    return df


def load_reddit(cfg=CFG) -> pd.DataFrame:
    df = pd.read_csv(cfg.reddit_csv)
    df = df.dropna(subset=["text", "label"]).copy()
    df["label"] = df["label"].map({"depression": 0, "SuicideWatch": 1})
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    df["text"] = df["text"].astype(str).map(clean_text)
    df = df[df["text"].str.len() > 20].copy()
    df["dataset"] = "Reddit"
    if len(df) > cfg.reddit_train_size + cfg.reddit_test_size:
        df = df.sample(n=cfg.reddit_train_size + cfg.reddit_test_size,
                       random_state=cfg.seed).reset_index(drop=True)
    print(f"[Reddit] total={len(df)}")
    return df


def make_splits(df: pd.DataFrame, cfg=CFG):
    train_df, temp_df = train_test_split(
        df, test_size=(cfg.val_ratio + cfg.test_ratio),
        random_state=cfg.seed, stratify=df["label"],
    )
    relative_test = cfg.test_ratio / (cfg.val_ratio + cfg.test_ratio)
    val_df, test_df = train_test_split(
        temp_df, test_size=relative_test,
        random_state=cfg.seed, stratify=temp_df["label"],
    )
    print(f"  train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


if __name__ == "__main__":
    swdd = load_swdd()
    red = load_reddit()
    print(swdd["label"].value_counts().sort_index())
    print(red["label"].value_counts().sort_index())
