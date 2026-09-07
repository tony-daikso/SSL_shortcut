"""E1a/b/d：用三種 backbone 的 embedding 訓練 linear probe。

**切分策略依任務性質不同**（呼應計畫原始 Phase 0 spec 的設計）：
- **video ID**（5-way）：同影片內 held-out frame（分層隨機切分），因為要留一支
  完全沒看過的影片出來測「這是哪支影片」沒有意義——video ID 任務本來就要求訓練時
  看過所有影片。這是 pilot 規模下**唯一有足夠統計效力**的任務（每支影片 100 張，
  train/test 都有全部 5 支影片）。
- **cohort / endoscope_brand / polyp_label**：GroupShuffleSplit 依 video_id 分組
  held-out 影片。**在 5 支影片的 pilot 規模下這幾個任務統計效力很有限**（跟 E0e
  遇到的問題一樣：單折、有時整個類別在訓練集裡缺席），這裡跑出來的數字只做流程
  驗證，不是正式結論。

E1d（病理 baseline）：polyp_label 用跟 cohort/brand 一樣的 GroupShuffleSplit 切法，
放在同一個 embedding 上比較，這樣「指紋可分性」才有一個對照基準可以比較，不是
孤立數字。
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from config import RESULTS_DIR

N_REPEATS = 10


def _fit_eval(X_train, y_train, X_test, y_test):
    scaler = StandardScaler().fit(X_train)
    clf = LogisticRegression(max_iter=2000)
    clf.fit(scaler.transform(X_train), y_train)
    pred = clf.predict(scaler.transform(X_test))
    return (pred == y_test).mean()


def probe_held_out_frame(X, y, n_repeats=N_REPEATS):
    """同影片內 held-out frame，分層隨機切分（video ID 任務用）。"""
    accs = []
    for seed in range(n_repeats):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        train_idx, test_idx = next(sss.split(X, y))
        accs.append(_fit_eval(X[train_idx], y[train_idx], X[test_idx], y[test_idx]))
    accs = np.array(accs)
    return {"mean_accuracy": float(accs.mean()), "std_accuracy": float(accs.std())}


def probe_held_out_video(X, y, groups, n_repeats=N_REPEATS):
    """GroupShuffleSplit 依 video_id 分組（cohort/brand/polyp 任務用）。"""
    accs = []
    for seed in range(n_repeats):
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        train_idx, test_idx = next(gss.split(X, y, groups=groups))
        if len(set(y[train_idx])) < 2:
            continue
        accs.append(_fit_eval(X[train_idx], y[train_idx], X[test_idx], y[test_idx]))
    if not accs:
        return {"mean_accuracy": float("nan"), "std_accuracy": float("nan"), "n_valid_repeats": 0}
    accs = np.array(accs)
    return {"mean_accuracy": float(accs.mean()), "std_accuracy": float(accs.std()), "n_valid_repeats": len(accs)}


def majority_baseline(y):
    vals, counts = np.unique(y, return_counts=True)
    return counts.max() / len(y)


def main():
    data = np.load(RESULTS_DIR / "e1_embeddings.npz", allow_pickle=True)
    video_id = data["video_id"]
    cohort = data["cohort"]
    brand = data["endoscope_brand"]
    polyp = data["polyp_label"]

    backbone_names = ["dinov2_pretrained", "dinov2_random", "imagenet_supervised"]

    n_videos = len(set(video_id))
    n_frames = len(video_id)
    scale_tag = "pilot" if n_videos < 60 else "full"

    lines = [f"# E1a/b/d：指紋可解碼性 probe 結果（{scale_tag}，{n_videos} 支影片，{n_frames} 張影格）\n"]

    # ---- E1a: video ID (最有統計效力的任務，用 dinov2_pretrained) ----
    lines.append(f"## E1a：Video ID（{n_videos}-way，held-out frame，dinov2_pretrained）\n")
    video_id_int = pd.factorize(video_id)[0]
    result = probe_held_out_frame(data["emb_dinov2_pretrained"], video_id_int)
    chance = 1.0 / n_videos
    lines.append(
        f"accuracy = {result['mean_accuracy']:.3f} ± {result['std_accuracy']:.3f}"
        f"（chance = {chance:.3f}，majority baseline = {majority_baseline(video_id_int):.3f}）\n"
    )
    lines.append(
        f"**解讀**：{n_videos} 支影片中隨機抽 20% 的 frame 當測試集，其餘 80%（含全部"
        f"{n_videos} 支影片的其他 frame）當訓練集。"
        + ("這是唯一在 pilot 規模下有足夠統計效力的任務。\n" if scale_tag == "pilot"
           else "在全部 60 支影片、6000 張影格的規模下，這是正式結論（非 pilot 流程驗證）。\n")
    )

    # ---- E1a: cohort / brand（train/test 依 video 分組）----
    lines.append("## E1a：Cohort / Endoscope Brand（held-out 影片，dinov2_pretrained）\n")
    cohort_int = pd.factorize(cohort)[0]
    brand_int = pd.factorize(brand)[0]
    cohort_result = probe_held_out_video(data["emb_dinov2_pretrained"], cohort_int, video_id)
    brand_result = probe_held_out_video(data["emb_dinov2_pretrained"], brand_int, video_id)
    lines.append(
        f"Cohort（{len(set(cohort))}-way）：accuracy = {cohort_result['mean_accuracy']:.3f} ± "
        f"{cohort_result.get('std_accuracy', float('nan')):.3f}"
        f"（chance = {1/len(set(cohort)):.3f}，majority = {majority_baseline(cohort_int):.3f}，"
        f"有效折數 = {cohort_result.get('n_valid_repeats', N_REPEATS)}/{N_REPEATS}）\n\n"
        f"Endoscope brand（2-way）：accuracy = {brand_result['mean_accuracy']:.3f} ± "
        f"{brand_result.get('std_accuracy', float('nan')):.3f}"
        f"（majority = {majority_baseline(brand_int):.3f}，"
        f"有效折數 = {brand_result.get('n_valid_repeats', N_REPEATS)}/{N_REPEATS}）\n\n"
        + ("**跟 E0e 一樣的規模限制**：5 支影片切 train/test 時容易讓某個類別整個缺席"
           "（有效折數若明顯 <10，代表很多次重複都被跳過），這兩個數字暫時只做流程"
           "驗證，不是正式結論。\n" if scale_tag == "pilot" else
           f"**全資料集規模**：{n_videos} 支影片、每個 cohort/brand 都有多支影片代表，"
           "有效折數 10/10 代表每次 held-out 切分訓練集都完整涵蓋所有類別，可當正式"
           "結論。\n")
    )

    # ---- E1b: backbone 對照 ----
    lines.append("## E1b：Backbone 對照（video ID 任務，統計效力最足夠的任務）\n")
    backbone_rows = {}
    for name in backbone_names:
        backbone_rows[name] = probe_held_out_frame(data[f"emb_{name}"], video_id_int)
    summary = pd.DataFrame(backbone_rows).T
    summary["chance"] = chance
    lines.append(summary.to_markdown())
    lines.append(
        "\n**解讀**：`dinov2_random`（隨機初始化、完全沒訓練過）如果 accuracy 遠低於 "
        "`dinov2_pretrained`，代表 video ID 指紋不是光靠架構的歸納偏見就能讀出來，"
        "確實需要某種訓練（不論是 SSL 還是監督式）才學得到。`imagenet_supervised` "
        "是 ResNet-50，跟 DINOv2 的 ViT 架構不同、輸出維度也不同（2048 vs 384），"
        "只能看方向、不是嚴格對照。\n"
    )

    # ---- E1d: 病理 baseline ----
    lines.append("## E1d：病理（polyp_label）baseline，同一個 embedding\n")
    polyp_result_frame = probe_held_out_frame(data["emb_dinov2_pretrained"], polyp)
    polyp_result_video = probe_held_out_video(data["emb_dinov2_pretrained"], polyp, video_id)
    lines.append(
        f"Held-out frame 切法（跟 video ID 用同一種切分，比較基準）：accuracy = "
        f"{polyp_result_frame['mean_accuracy']:.3f} ± {polyp_result_frame['std_accuracy']:.3f}"
        f"（majority baseline = {majority_baseline(polyp):.3f}，抽樣集中在影片中間 1/3，"
        "polyp frame 偏少，majority baseline 天生就高）。\n\n"
        f"Held-out 影片切法（跟 cohort/brand 用同一種切分）：accuracy = "
        f"{polyp_result_video['mean_accuracy']:.3f} ± "
        f"{polyp_result_video.get('std_accuracy', float('nan')):.3f}"
        f"（majority = {majority_baseline(polyp):.3f}，"
        f"有效折數 = {polyp_result_video.get('n_valid_repeats', N_REPEATS)}/{N_REPEATS}）。\n\n"
        "**這是指紋可分性的對照組**：如果指紋（video ID/cohort）的可分性明顯高於"
        "病理可分性，才能說 embedding 把容量更多分配在採集特徵而不是病理特徵上。\n"
    )

    out_path = RESULTS_DIR / "e1_probe_report.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}")
    print(f"\nVideo ID (dinov2_pretrained): {result}")
    print(f"Cohort: {cohort_result}")
    print(f"Brand: {brand_result}")
    print(f"Backbone comparison:\n{summary}")
    print(f"Polyp (held-out frame): {polyp_result_frame}")
    print(f"Polyp (held-out video): {polyp_result_video}")


if __name__ == "__main__":
    main()
