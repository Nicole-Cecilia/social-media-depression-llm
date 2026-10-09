"""LoRA 微调脚本（8G 显存适配版）。"""
import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq

from src.config import CFG
from src.data_utils import load_swdd, load_reddit, make_splits
from src.model_utils import load_model_and_tokenizer, chat_generate
from src.prompts import ANSWER_WORDS, build_zero_shot_messages, parse_label
from src.evaluate import evaluate


def tokenize_for_training(df, tokenizer, dataset_name, cfg=CFG):
    def build(rec):
        ans = ANSWER_WORDS[dataset_name][int(rec["label"])]
        reserve = 160
        user_ids = tokenizer(
            rec["text"], add_special_tokens=False,
            truncation=True, max_length=cfg.max_length - reserve,
        )["input_ids"]
        text_trimmed = tokenizer.decode(user_ids)
        prompt_msgs = build_zero_shot_messages(text_trimmed, dataset_name)
        prompt_text = tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True)
        full_msgs = prompt_msgs + [{"role": "assistant", "content": ans}]
        full_text = tokenizer.apply_chat_template(
            full_msgs, tokenize=False, add_generation_prompt=False)
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
        if tokenizer.eos_token_id is not None and full_ids[-1] != tokenizer.eos_token_id:
            full_ids = full_ids + [tokenizer.eos_token_id]
        labels = [-100] * min(len(prompt_ids), len(full_ids)) + \
                 full_ids[min(len(prompt_ids), len(full_ids)):]
        return {"input_ids": full_ids, "attention_mask": [1]*len(full_ids), "labels": labels}
    ds = Dataset.from_pandas(df[["text", "label"]])
    return ds.map(build, remove_columns=ds.column_names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["swdd", "reddit"], required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--eval-limit", type=int, default=None)
    ap.add_argument("--epochs", type=float, default=None)
    ap.add_argument("--rank", type=int, default=CFG.lora_rank)
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()
    if (args.limit or args.eval_limit) and not args.tag:
        args.tag = "smoke"

    if args.dataset == "swdd":
        df = load_swdd(); ds_name = "SWDD"
    else:
        df = load_reddit(); ds_name = "Reddit"
    train_df, val_df, test_df = make_splits(df)
    if args.dataset == "reddit":
        train_df = train_df.sample(n=min(2000, len(train_df)), random_state=CFG.seed).reset_index(drop=True)
        default_epochs = 2
    else:
        default_epochs = 3
    epochs = int(args.epochs) if args.epochs else default_epochs
    if args.limit:
        train_df = train_df.head(args.limit).reset_index(drop=True)
    if args.eval_limit:
        test_df = test_df.head(args.eval_limit).reset_index(drop=True)
    print(f"训练样本={len(train_df)}  epochs={epochs}  rank={args.rank}  评估样本={len(test_df)}")

    tokenizer, model = load_model_and_tokenizer()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        r=args.rank, lora_alpha=args.rank * 2, lora_dropout=CFG.lora_dropout,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    train_ds = tokenize_for_training(train_df, tokenizer, ds_name)
    suffix = f"_{args.tag}" if args.tag else ""
    out_dir = str(ROOT / "outputs" / f"lora_{args.dataset}{suffix}")
    targs = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=CFG.batch_size,
        gradient_accumulation_steps=CFG.gradient_accumulation_steps,
        num_train_epochs=epochs,
        learning_rate=CFG.learning_rate,
        logging_steps=10,
        save_strategy="no",
        fp16=True,
        report_to="none",
        seed=CFG.seed,
        optim="paged_adamw_8bit",
    )
    trainer = Trainer(
        model=model, args=targs, train_dataset=train_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )
    print(">>> Start LoRA training ...")
    trainer.train()
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f">>> Adapter saved to {out_dir}")

    print(">>> Evaluating LoRA on test set ...")
    model.eval()
    y_true, y_pred = [], []
    for i in range(len(test_df)):
        row = test_df.iloc[i]
        msgs = build_zero_shot_messages(row["text"][:CFG.max_length], ds_name)
        raw = chat_generate(msgs, tokenizer, model, max_new_tokens=8)
        y_true.append(int(row["label"]))
        y_pred.append(parse_label(raw, ds_name))
        if (i+1) % 50 == 0:
            print(f"    eval {i+1}/{len(test_df)}")
    method_name = f"LoRA-{args.tag}" if args.tag else "LoRA"
    res = evaluate(y_true, y_pred, method_name, ds_name)
    import json
    with open(Path(out_dir) / "eval_result.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    if not args.tag:
        from src.evaluate import save_results
        save_results([res], str(ROOT / "outputs" / "results" / "lora_results.xlsx"))


if __name__ == "__main__":
    main()
