"""合并所有实验结果，生成对比看板图。"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

df1 = pd.read_excel(ROOT / "outputs" / "results" / "zeroshot_fewshot_results.xlsx")
df2 = pd.read_excel(ROOT / "outputs" / "results" / "lora_results.xlsx")
df = pd.concat([df1, df2], ignore_index=True)
print(df)
out_xlsx = ROOT / "outputs" / "results" / "all_results.xlsx"
df.to_excel(out_xlsx, index=False)
print(f"[saved] {out_xlsx}")

metrics = ["Accuracy", "Precision", "Recall", "F1"]
datasets = ["SWDD", "Reddit"]
methods = ["Zero-shot", "Few-shot", "LoRA"]
colors = ["#4C72B0", "#55A868", "#C44E52"]

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
for ax, ds in zip(axes, datasets):
    sub = df[df["Dataset"] == ds].set_index("Method").reindex(methods)
    x = range(len(metrics))
    width = 0.25
    for i, method in enumerate(methods):
        vals = [sub.loc[method, m] for m in metrics]
        bars = ax.bar([p + i*width for p in x], vals, width, label=method, color=colors[i])
        for b, v in zip(bars, vals):
            ax.text(b.get_x()+b.get_width()/2, v+0.01, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks([p + width for p in x])
    ax.set_xticklabels(metrics)
    ax.set_title(f"{ds} (N={int(sub['N'].iloc[0])})")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
axes[0].set_ylabel("Score")
axes[1].legend(loc="lower right")
fig.suptitle("Qwen2.5-7B-Instruct 抑郁风险检测对比", fontsize=12)
plt.tight_layout()
out_png = ROOT / "outputs" / "results" / "Experiment_Dashboard.png"
plt.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"[saved] {out_png}")
