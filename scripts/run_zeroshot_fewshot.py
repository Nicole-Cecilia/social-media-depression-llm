"""Zero-shot 与 Few-shot 基线评估。"""
import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import CFG
from src.data_utils import load_swdd, load_reddit, make_splits
from src.prompts import build_zero_shot_messages, build_few_shot_messages, parse_label, _SUICIDE_WORDS
from src.model_utils import load_model_and_tokenizer, chat_generate
from src.evaluate import evaluate, save_results


def pick_few_shot_examples(train_df, k, seed, dataset_name="SWDD"):
    import pandas as pd
    classes = sorted(train_df["label"].unique())
    k_each = max(1, k // len(classes))
    parts = []
    for c in classes:
        sub = train_df[train_df["label"] == c]
        if dataset_name == "SWDD" and c in (1, 2):
            has_sui = sub["text"].apply(lambda t: any(w in t for w in _SUICIDE_WORDS))
            if c == 1:
                pool = sub[~has_sui] if (~has_sui).sum() >= k_each else sub
            else:
                pool = sub[has_sui] if has_sui.sum() >= k_each else sub
        else:
            pool = sub
        parts.append(pool.sample(n=min(k_each, len(pool)), random_state=seed))
    ex = pd.concat(parts).sample(frac=1.0, random_state=seed)
    return [(r["text"][:150], int(r["label"])) for _, r in ex.iterrows()]


def run_method(test_df, dataset_name, tokenizer, model, method, few_shot_examples=None, limit=None):
    y_true, y_pred = [], []
    n = len(test_df) if limit is None else min(limit, len(test_df))
    for i in range(n):
        row = test_df.iloc[i]
        text = row["text"][:CFG.max_length]
        if method == "zero":
            msgs = build_zero_shot_messages(text, dataset_name)
        else:
            msgs = build_few_shot_messages(text, few_shot_examples, dataset_name)
        raw = chat_generate(msgs, tokenizer, model, max_new_tokens=CFG.max_new_tokens)
        pred = parse_label(raw, dataset_name)
        y_true.append(int(row["label"]))
        y_pred.append(pred)
        if (i + 1) % 20 == 0:
            print(f"    [{method}] {i+1}/{n} done")
    return y_true, y_pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["swdd", "reddit", "both"], default="both")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    datasets = {}
    if args.dataset in ("swdd", "both"):
        swdd = load_swdd()
        tr, va, te = make_splits(swdd)
        datasets["SWDD"] = (tr, te)
    if args.dataset in ("reddit", "both"):
        red = load_reddit()
        tr, va, te = make_splits(red)
        datasets["Reddit"] = (tr, te)
    tokenizer, model = load_model_and_tokenizer()
    rows = []
    for ds_name, (train_df, test_df) in datasets.items():
        print(f"\n===== {ds_name} =====")
        examples = pick_few_shot_examples(train_df, CFG.few_shot_k, CFG.seed, ds_name)
        print("--- Zero-shot ---")
        yt, yp = run_method(test_df, ds_name, tokenizer, model, "zero", limit=args.limit)
        rows.append(evaluate(yt, yp, "Zero-shot", ds_name))
        print("--- Few-shot ---")
        yt, yp = run_method(test_df, ds_name, tokenizer, model, "few", few_shot_examples=examples, limit=args.limit)
        rows.append(evaluate(yt, yp, "Few-shot", ds_name))
    out_name = ("zeroshot_fewshot_results_smoke.xlsx" if args.limit
                else "zeroshot_fewshot_results.xlsx")
    save_results(rows, str(ROOT / "outputs" / "results" / out_name))


if __name__ == "__main__":
    main()
