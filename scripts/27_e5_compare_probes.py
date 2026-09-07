"""E5b：把自訓 DINO（一或多個 `results/e5_embeddings_*.npz`）的指紋可分性，
跟 E1 已經量過的 `dinov2_pretrained`（ImageNet SSL 預訓練）、`dinov2_random`
（隨機初始化、完全沒訓練過）放在同一組 probe 方法論下比較。

判準（計畫 §6 E5b）：自訓的指紋可分性是否高於現成 dinov2_pretrained？
- 高於 → SSL 目標函數在這批資料上訓練時，確實把指紋當成捷徑放大了。
- 介於 dinov2_random 與 dinov2_pretrained 之間、或低於兩者 → 自訓沒有主動放大
  指紋，E1b 觀察到的高可分性主要是 ImageNet 預訓練或架構本身帶來的，不是
  「在 REAL-Colon 上訓練」這件事本身造成的。

用跟 15_e1_probes.py 完全一樣的 probe 方法論（video_id：held-out frame，
StratifiedShuffleSplit；cohort/brand/polyp：held-out 影片，GroupShuffleSplit），
才能公平比較。
"""

import glob

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from config import RESULTS_DIR

N_REPEATS = 10


def probe_held_out_frame(X, y, n_repeats=N_REPEATS):
    accs = []
    for seed in range(n_repeats):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        train_idx, test_idx = next(sss.split(X, y))
        scaler = StandardScaler().fit(X[train_idx])
        clf = LogisticRegression(max_iter=2000)
        clf.fit(scaler.transform(X[train_idx]), y[train_idx])
        pred = clf.predict(scaler.transform(X[test_idx]))
        accs.append((pred == y[test_idx]).mean())
    accs = np.array(accs)
    return {"mean_accuracy": float(accs.mean()), "std_accuracy": float(accs.std())}


def probe_held_out_video(X, y, groups, n_repeats=N_REPEATS):
    accs = []
    for seed in range(n_repeats):
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        train_idx, test_idx = next(gss.split(X, y, groups=groups))
        if len(set(y[train_idx])) < 2:
            continue
        scaler = StandardScaler().fit(X[train_idx])
        clf = LogisticRegression(max_iter=2000)
        clf.fit(scaler.transform(X[train_idx]), y[train_idx])
        pred = clf.predict(scaler.transform(X[test_idx]))
        accs.append((pred == y[test_idx]).mean())
    if not accs:
        return {"mean_accuracy": float("nan"), "std_accuracy": float("nan"), "n_valid_repeats": 0}
    accs = np.array(accs)
    return {"mean_accuracy": float(accs.mean()), "std_accuracy": float(accs.std()), "n_valid_repeats": len(accs)}


def load_backbone_embeddings():
    """回傳 {backbone_name: (embeddings, frame_id, video_id, cohort, brand, polyp)}。
    先讀 E1 既有的 dinov2_pretrained/dinov2_random 當基準，再掃描所有
    `e5_embeddings_*.npz` 加進來。"""
    backbones = {}

    e1 = np.load(RESULTS_DIR / "e1_embeddings.npz", allow_pickle=True)
    common = {
        "frame_id": e1["frame_id"], "video_id": e1["video_id"],
        "cohort": e1["cohort"], "endoscope_brand": e1["endoscope_brand"],
        "polyp_label": e1["polyp_label"],
    }
    for name in ["dinov2_pretrained", "dinov2_random"]:
        backbones[name] = {"embeddings": e1[f"emb_{name}"], **common}

    for path in sorted(glob.glob(str(RESULTS_DIR / "e5_embeddings_*.npz"))):
        data = np.load(path, allow_pickle=True)
        tag = [k for k in data.files if k.startswith("emb_")][0][len("emb_"):]
        backbones[tag] = {
            "embeddings": data[f"emb_{tag}"],
            "frame_id": data["frame_id"], "video_id": data["video_id"],
            "cohort": data["cohort"], "endoscope_brand": data["endoscope_brand"],
            "polyp_label": data["polyp_label"],
        }
    return backbones


def main():
    backbones = load_backbone_embeddings()
    print(f"比較的 backbone：{list(backbones.keys())}")

    rows = []
    for name, d in backbones.items():
        video_id_int = pd.factorize(d["video_id"])[0]
        cohort_int = pd.factorize(d["cohort"])[0]
        brand_int = pd.factorize(d["endoscope_brand"])[0]

        video_id_result = probe_held_out_frame(d["embeddings"], video_id_int)
        cohort_result = probe_held_out_video(d["embeddings"], cohort_int, d["video_id"])
        brand_result = probe_held_out_video(d["embeddings"], brand_int, d["video_id"])
        polyp_result = probe_held_out_video(d["embeddings"], d["polyp_label"], d["video_id"])

        rows.append({
            "backbone": name,
            "video_id_acc": video_id_result["mean_accuracy"],
            "video_id_std": video_id_result["std_accuracy"],
            "cohort_acc": cohort_result["mean_accuracy"],
            "brand_acc": brand_result["mean_accuracy"],
            "polyp_acc_heldout_video": polyp_result["mean_accuracy"],
        })

    summary = pd.DataFrame(rows).set_index("backbone")
    n_videos = len(set(next(iter(backbones.values()))["video_id"]))
    chance_video_id = 1.0 / n_videos

    lines = ["# E5b：自訓 DINO vs 現成 DINOv2 的指紋可分性對照\n"]
    lines.append(
        f"Video ID（{n_videos}-way，held-out frame，chance={chance_video_id:.3f}）、"
        "cohort/brand/polyp（held-out 影片，10 折平均）。\n"
    )
    lines.append(summary.round(4).to_markdown())
    lines.append("")

    if "dinov2_pretrained" in summary.index:
        pretrained_acc = summary.loc["dinov2_pretrained", "video_id_acc"]
        random_acc = summary.loc["dinov2_random", "video_id_acc"]
        lines.append("## 解讀（E5b 判準：自訓的指紋可分性是否高於現成 dinov2_pretrained？）\n")
        for name in summary.index:
            if name in ("dinov2_pretrained", "dinov2_random"):
                continue
            acc = summary.loc[name, "video_id_acc"]
            if acc > pretrained_acc:
                verdict = (
                    f"**{name}**：{acc:.3f} > dinov2_pretrained（{pretrained_acc:.3f}）——"
                    "自訓的指紋可分性高於現成 ImageNet 預訓練版本，代表 SSL 目標函數"
                    "在 REAL-Colon 這批資料上訓練時，確實把採集指紋當成捷徑主動放大了，"
                    "不只是架構歸納偏見或 ImageNet 預訓練帶來的附帶編碼。"
                )
            elif acc > random_acc:
                verdict = (
                    f"**{name}**：介於 dinov2_random（{random_acc:.3f}）與 "
                    f"dinov2_pretrained（{pretrained_acc:.3f}）之間（{acc:.3f}）——"
                    "自訓確實比完全沒訓練過學到更多指紋，但沒有超過 ImageNet 預訓練版本，"
                    "代表訓練本身有放大效果，但這批 REAL-Colon 訓練語料規模/訓練時長"
                    "可能還不足以讓自訓版本超車。"
                )
            else:
                verdict = (
                    f"**{name}**：{acc:.3f} ≤ dinov2_random（{random_acc:.3f}）——"
                    "自訓沒有讓指紋可分性超過隨機初始化的下限，可能是訓練還沒收斂"
                    "（epoch/資料量不夠），需要檢查 train_log.csv 的 loss 曲線。"
                )
            lines.append(f"{verdict}\n")

    out_path = RESULTS_DIR / "e5_compare_report.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}")
    print(summary)


if __name__ == "__main__":
    main()
