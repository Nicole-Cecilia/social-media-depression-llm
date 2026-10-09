"""模型与 tokenizer 加载（4-bit 量化，适配 8G 显存）。"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from .config import CFG


def load_model_and_tokenizer(model_path: str = None):
    path = model_path or CFG.model_path
    print(f">>> Loading tokenizer from {path}")
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(">>> Loading model in 4-bit quantization ...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f">>> Model ready. GPU mem: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    return tokenizer, model


def chat_generate(messages, tokenizer, model, max_new_tokens=None) -> str:
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens or 8,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
