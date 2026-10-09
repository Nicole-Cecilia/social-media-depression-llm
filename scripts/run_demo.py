"""Gradio 交互 Demo：粘贴一段文本，给出抑郁风险分级 + 证据句 + 关怀指引。"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import gradio as gr
import torch
from peft import PeftModel

from src.config import CFG
from src.model_utils import load_model_and_tokenizer, chat_generate
from src.prompts import build_zero_shot_messages, parse_label

RISK_META = {
    0: {"name": "正常", "color": "#2e7d32", "emoji": "✅",
        "advice": "当前文本未显示明显心理困扰。继续保持规律作息和社交连接。"},
    1: {"name": "抑郁倾向", "color": "#f57c00", "emoji": "🟡",
        "advice": "文本中出现了持续情绪低落的迹象。建议：① 允许自己休息；② 和信任的人聊聊；③ 若持续两周以上，建议去心理科聊聊。"},
    2: {"name": "高风险（伴自杀意念）", "color": "#c62828", "emoji": "🔴",
        "advice": "检测到可能的自杀相关表述。24小时心理援助热线：010-82951332、400-161-9995；如有危险请立即拨打120或前往急诊。"},
}


def load_model():
    tokenizer, model = load_model_and_tokenizer()
    adapter_path = ROOT / "outputs" / "lora_swdd"
    if adapter_path.exists():
        print(f">>> Loading LoRA adapter from {adapter_path}")
        model = PeftModel.from_pretrained(model, str(adapter_path))
        model.eval()
    return tokenizer, model


def classify(text, tokenizer, model):
    msgs = build_zero_shot_messages(text[:CFG.max_length], "SWDD")
    raw = chat_generate(msgs, tokenizer, model, max_new_tokens=8)
    return parse_label(raw, "SWDD"), raw


def extract_evidence(text, risk_label, tokenizer, model):
    risk_name = RISK_META[risk_label]["name"]
    prompt = (
        f"用户写了以下社交媒体文本，我们判断其心理风险等级为「{risk_name}」。\n"
        f"请从原文中逐句挑出最能支持这个判断的句子（3到8句），"
        f"按原文出现顺序列出，每句一行，不要改写、不要解释、不要编号。\n"
        f"如果没有明显支持句，直接回答\"无\"。\n\n原文：{text[:CFG.max_length]}"
    )
    msgs = [
        {"role": "system", "content": "你是严谨的文本分析助手，只做摘录。"},
        {"role": "user", "content": prompt},
    ]
    raw = chat_generate(msgs, tokenizer, model, max_new_tokens=256)
    lines = [l.strip("0123456789.、-• ") for l in raw.split("\n") if l.strip()]
    lines = [l for l in lines if l and l != "无"]
    return lines[:8]


def predict(text, tokenizer, model):
    if not text or not text.strip():
        return "## 请先输入文本", "", ""
    risk, raw = classify(text, tokenizer, model)
    meta = RISK_META[risk]
    evidence = extract_evidence(text, risk, tokenizer, model)
    header = (f"## {meta['emoji']} 风险等级：{meta['name']}\n"
              f"<span style='color:{meta['color']}'>（模型输出：{raw}）</span>")
    if evidence:
        ev_html = "### 支持判断的关键证据句：\n"
        for i, s in enumerate(evidence, 1):
            ev_html += f"{i}. <span style='background:#fff3e0;padding:2px 6px;border-radius:4px'>{s}</span>\n"
    else:
        ev_html = "### 支持判断的关键证据句：\n（基于整体文本氛围判断）"
    advice = f"### 关怀指引\n{meta['advice']}"
    return header, ev_html, advice


def build_demo():
    tokenizer, model = load_model()
    with gr.Blocks(title="社交媒体抑郁风险检测") as demo:
        gr.Markdown("# 🧠 社交媒体抑郁风险检测\nQwen2.5-7B-Instruct + LoRA")
        with gr.Row():
            with gr.Column(scale=1):
                text = gr.Textbox(label="粘贴微博/小红书/朋友圈文本", lines=8,
                                  placeholder="最近好累，什么都不想做……")
                btn = gr.Button("开始检测", variant="primary")
            with gr.Column(scale=1):
                out_risk = gr.Markdown()
                out_evidence = gr.Markdown()
                out_advice = gr.Markdown()
        btn.click(fn=lambda t: predict(t, tokenizer, model), inputs=text,
                  outputs=[out_risk, out_evidence, out_advice])
        gr.Examples(
            examples=[
                ["今天和朋友去吃了火锅，特别开心！明天还要早起上班，晚安~"],
                ["最近一个月都提不起劲，失眠到凌晨三四点，对什么都没兴趣，觉得自己很没用。"],
                ["我真的撑不下去了，不想再活了，想找个地方安静地结束这一切。"],
                ["连续加班三周，每天都在崩溃边缘，觉得活着没什么意思，但还得咬牙坚持。"],
            ],
            inputs=text,
            label="试几个示例（点击自动填入）",
        )
    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(server_name="127.0.0.1", server_port=7860)
