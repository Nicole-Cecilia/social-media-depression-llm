"""Prompt 模板与输出解析。"""

SYSTEM = {
    "SWDD": "你是一个社交媒体心理健康风险分级助手。你只根据用户发布的微博文本判断其心理风险等级，不要输出任何解释。",
    "Reddit": "You are a mental-health classification assistant. You classify posts based only on their text. Output ONLY the label word, no explanation.",
}

INSTRUCTION = {
    "SWDD": (
        "判断下面这段微博文本来自哪一类用户，共三个风险等级：\n"
        "- 正常：没有明显心理问题，内容是积极或平淡的日常生活。\n"
        "- 抑郁：有持续情绪低落、兴趣减退、失眠、疲惫、自责、痛苦等抑郁表现，"
        "但通篇没有出现明确的自杀意念表述。\n"
        "- 高风险：文本中明确出现了自杀、轻生、想死、不想活了、结束生命等直接的自杀意念或计划。\n"
        "判别规则（按优先级）：\n"
        "1. 如果文本明确出现\"自杀\"\"轻生\"\"想死\"\"不想活\"\"结束生命\"等字眼，"
        "即使整体读起来只是情绪低落，也必须判\"高风险\"——自杀意念不能漏报。\n"
        "2. 如果文本只是情绪低落、疲惫、深夜emo、说\"活着没意思\"\"好痛苦\"但没有出现上面那些自杀字眼，"
        "判\"抑郁\"，不要因为氛围压抑就升级为高风险。\n"
        "只回答一个词：正常、抑郁 或 高风险。"
    ),
    "Reddit": (
        "Classify the following Reddit post. "
        "Answer with ONLY one word: depression or SuicideWatch."
    ),
}

ANSWER_WORDS = {
    "SWDD": {0: "正常", 1: "抑郁", 2: "高风险"},
    "Reddit": {0: "depression", 1: "SuicideWatch"},
}

_SUICIDE_WORDS = ["高风险", "自杀", "轻生", "想死", "不想活", "活不下去",
                  "了断", "寻死", "活着没", "不想再活", "结束生命", "自尽"]
_NEG_SUICIDE = ["没有自杀", "没自杀", "无自杀", "未自杀", "不自杀",
                "不存在自杀", "未表达自杀", "未提及自杀", "没有轻生", "无轻生",
                "没有想自杀", "无自杀意念", "没有自杀意念", "无自杀倾向",
                "没有自杀倾向", "无自杀念头", "没有自杀念头"]


def build_zero_shot_messages(text: str, dataset: str):
    return [
        {"role": "system", "content": SYSTEM[dataset]},
        {"role": "user", "content": f"{INSTRUCTION[dataset]}\n\n文本: {text}" if dataset == "SWDD"
                                    else f"{INSTRUCTION[dataset]}\n\nPost: {text}"},
    ]


def build_few_shot_messages(text: str, examples, dataset: str):
    shots = []
    for i, (ex_text, ex_label) in enumerate(examples, 1):
        ans = ANSWER_WORDS[dataset][ex_label]
        if dataset == "SWDD":
            shots.append(f"例子{i}:\n文本: {ex_text}\n回答: {ans}")
        else:
            shots.append(f"Example {i}:\nPost: {ex_text}\nAnswer: {ans}")
    block = "\n\n".join(shots)
    if dataset == "SWDD":
        user_content = f"{INSTRUCTION[dataset]}\n\n{block}\n\n现在判断:\n文本: {text}"
    else:
        user_content = f"{INSTRUCTION[dataset]}\n\n{block}\n\nNow classify:\nPost: {text}"
    return [
        {"role": "system", "content": SYSTEM[dataset]},
        {"role": "user", "content": user_content},
    ]


def parse_label(raw_output: str, dataset: str):
    out = raw_output.strip()
    low = out.lower()
    if dataset == "SWDD":
        check = out
        for neg in _NEG_SUICIDE:
            check = check.replace(neg, "")
        if any(w in check for w in _SUICIDE_WORDS):
            return 2
        if any(w in out for w in ["不是抑郁", "没有抑郁", "无抑郁", "不属于抑郁"]):
            return 0
        if "抑郁" in out:
            return 1
        if any(w in out for w in ["正常", "健康", "无风险", "低风险", "否"]):
            return 0
        return 0
    else:
        sw = "suicidewatch" in low or "suicide" in low or "kill myself" in low
        dep = "depression" in low or "depressed" in low or "depressing" in low
        if sw and dep:
            return 1 if low.find("suicide") < low.find("depress") else 0
        if sw:
            return 1
        return 0


def n_classes(dataset: str) -> int:
    return len(ANSWER_WORDS[dataset])
