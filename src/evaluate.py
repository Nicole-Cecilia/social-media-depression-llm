"""统一评估：Accuracy / Precision / Recall / F1。"""
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def evaluate(y_true, y_pred, method: str, dataset: str, verbose: bool = True):
    labels_present = sorted(set(y_true) | set(y_pred))
    n_classes = max(labels_present) + 1
    all_labels = list(range(n_classes))
    average = "binary" if n_classes == 2 else "macro"
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    res = {
        "Dataset": dataset, "Method": method,
        "Accuracy": round(acc, 4), "Precision": round(p, 4),
        "Recall": round(r, 4), "F1": round(f1, 4), "N": len(y_true),
    }
    if verbose:
        print(f"[{dataset} | {method}] average={average}, N={len(y_true)}")
        print(res)
        print("混淆矩阵（行=真实, 列=预测）：")
        for i, row in enumerate(cm):
            print(f"  {i}: {list(row)}")
        if n_classes > 2:
            pc, rc, fc, _ = precision_recall_fscore_support(
                y_true, y_pred, labels=all_labels, zero_division=0)
            for j in all_labels:
                print(f"  类{j}: P={pc[j]:.3f} R={rc[j]:.3f} F1={fc[j]:.3f}")
    res["_confusion_matrix"] = cm.tolist()
    return res


def strip_internal(res: dict) -> dict:
    return {k: v for k, v in res.items() if not k.startswith("_")}


def save_results(rows, out_path: str, merge: bool = True):
    df_new = pd.DataFrame([strip_internal(r) for r in rows])
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if merge and path.exists():
        df_old = pd.read_excel(path)
        df = pd.concat([df_old, df_new], ignore_index=True)
        df = df.drop_duplicates(subset=["Dataset", "Method"], keep="last")
        df = df.sort_values(["Dataset", "Method"]).reset_index(drop=True)
    else:
        df = df_new
    df.to_excel(path, index=False)
    print(f"[saved] {path}")
    print(df.to_string(index=False))
    return df
