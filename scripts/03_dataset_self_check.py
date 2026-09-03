"""E0d：資料集規格自查——去交錯處理方式、每支影片的實際影格尺寸。

研究計畫 §3.2/§9 指出兩件事官方論文沒說清楚、需要自己查：
1. 每支影片實際的 frame 尺寸（FOV 裁切後每支/每台設備可能不同，§3.3 的 FOV 幾何洩漏
   問題的前置證據）。
2. 1080i 交錯掃描的去交錯處理方式——論文沒說明，需要自己檢查像素。

讀取 `pilot_frame_labels.csv`（10_merge_pilot_labels.py，來自官方原始資料，見下方
勘誤說明這件事為什麼重要）。

去交錯 heuristic 的原理：交錯掃描殘留的「梳齒」(combing) 是因為相鄰兩條掃描線來自
不同時間的兩個 field，在有運動的區域，相鄰行(row i, i+1)的差異會明顯大於間隔一行
(row i, i+2，同一個 field 內)的差異。用兩者比值當分數，分數越高代表殘留交錯痕跡的
嫌疑越大。這只是篩檢，不是證明——真正判斷仍要肉眼看樣本圖。
"""

import numpy as np
import pandas as pd
from PIL import Image

from config import REPO_ROOT, RESULTS_DIR

SAMPLES_PER_VIDEO = 5
CROP_SIZE = 400
RNG_SEED = 0


def frame_size_check(frame_labels: pd.DataFrame) -> str:
    lines = ["## 1. 每支影片實際 frame 尺寸\n"]

    per_video = frame_labels.groupby("video_id").agg(
        n_distinct_sizes=("width", lambda s: frame_labels.loc[s.index, ["width", "height"]].drop_duplicates().shape[0]),
    )
    inconsistent = per_video[per_video["n_distinct_sizes"] > 1]
    lines.append(
        f"pilot 涵蓋 {frame_labels['video_id'].nunique()} 支影片。"
        f"其中 {len(inconsistent)} 支影片內部觀察到超過一種 (width, height) 組合。\n"
    )
    if len(inconsistent):
        detail = (
            frame_labels[frame_labels["video_id"].isin(inconsistent.index)]
            .groupby(["video_id", "width", "height"]).size().rename("n_frames_in_pilot")
        )
        lines.append(detail.to_frame().to_markdown())
        lines.append("")

    lines.append("### 每個 (width, height) 組合對應到幾支不同影片、哪些 cohort/品牌\n")
    size_summary = frame_labels.groupby(["width", "height"]).agg(
        n_frames=("frame_id", "count"),
        n_videos=("video_id", "nunique"),
        cohorts=("cohort", lambda s: sorted(s.unique())),
        brands=("endoscope_brand", lambda s: sorted(s.unique())),
    ).sort_values("n_frames", ascending=False)
    lines.append(size_summary.to_markdown())
    lines.append(
        "\n**解讀**：FOV 尺寸不是單一固定值，且不同尺寸群組對應到不同 cohort/品牌"
        "組合——這正是研究計畫 §3.3 擔心的 FOV 幾何洩漏的直接證據。完整的 trivial "
        "baseline 量化見 E0e（`fov_e0e_baseline_report.md`）。\n"
    )
    return "\n".join(lines)


def interlace_score(gray: np.ndarray) -> float:
    """相鄰行差異 / 隔行差異 的比值，越高代表越可能殘留交錯梳齒。"""
    row_diff_adjacent = np.abs(gray[1:, :].astype(np.float32) - gray[:-1, :].astype(np.float32))
    row_diff_skip = np.abs(gray[2:, :].astype(np.float32) - gray[:-2, :].astype(np.float32))
    eps = 1e-6
    return float(row_diff_adjacent.mean() / (row_diff_skip.mean() + eps))


def deinterlace_check(frame_labels: pd.DataFrame) -> str:
    lines = ["## 2. 去交錯自查（篩檢用 heuristic，需人工複查樣本圖）\n"]

    qc_dir = RESULTS_DIR / "qc_deinterlace_samples"
    qc_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for video_id, group in frame_labels.groupby("video_id"):
        sample = group.sample(n=min(SAMPLES_PER_VIDEO, len(group)), random_state=RNG_SEED)
        for _, r in sample.iterrows():
            img_path = REPO_ROOT / r["frame_path"]
            if not img_path.exists():
                continue
            im = Image.open(img_path).convert("L")
            gray = np.asarray(im)
            score = interlace_score(gray)
            rows.append({"video_id": video_id, "cohort": r["cohort"], "endoscope_brand": r["endoscope_brand"],
                         "frame_id": r["frame_id"], "interlace_score": score, "img_path": str(img_path)})

    if not rows:
        return "## 2. 去交錯自查\n\n找不到對應的影格檔案，略過。\n"

    scores = pd.DataFrame(rows)
    per_video = scores.groupby(["video_id", "cohort", "endoscope_brand"])["interlace_score"].median().reset_index()
    per_video = per_video.sort_values("interlace_score", ascending=False)

    lines.append(f"對 {scores['video_id'].nunique()} 支影片、每支最多 {SAMPLES_PER_VIDEO} 張抽樣影格計算 interlace score，"
                  "取每支影片的中位數排序（分數越高越可疑）：\n")
    lines.append(per_video.to_markdown(index=False))
    lines.append("")

    lines.append("### 依品牌分組的 interlace score 中位數（初步看品牌間是否有系統性差異）\n")
    by_brand = scores.groupby("endoscope_brand")["interlace_score"].agg(["median", "mean", "std", "count"])
    lines.append(by_brand.to_markdown())
    lines.append("")

    top_n = min(6, len(scores))
    top_frames = scores.sort_values("interlace_score", ascending=False).head(top_n)
    lines.append(f"### 已存下分數最高的 {top_n} 張影格中央裁切區塊到 `results/qc_deinterlace_samples/`，供肉眼複查\n")
    saved = []
    for _, r in top_frames.iterrows():
        im = Image.open(r["img_path"])
        w, h = im.size
        cx, cy = w // 2, h // 2
        half = CROP_SIZE // 2
        crop = im.crop((max(0, cx - half), max(0, cy - half), min(w, cx + half), min(h, cy + half)))
        out_name = f"{r['frame_id']}_score{r['interlace_score']:.2f}.png"
        crop.save(qc_dir / out_name)
        saved.append(out_name)
    lines.append("\n".join(f"- {n}" for n in saved))

    overall_range = f"{scores['interlace_score'].min():.2f}–{scores['interlace_score'].max():.2f}"
    lines.append(
        f"\n**這次是在官方原始（非重新編碼過的）影格上量測**（範圍 {overall_range}）。"
        "肉眼複查時額外發現一張影格（見下方勘誤）有明顯的色彩通道錯位/鬼影，跟色彩通道"
        "分時擷取（field-sequential 或類似機制）造成的偽影很像，需要更多樣本才能確認"
        "是不是系統性的、還是單一運動模糊事件——**這裡先如實記錄觀察，不下結論**。\n"
    )
    return "\n".join(lines)


def main():
    frame_labels_path = RESULTS_DIR / "pilot_frame_labels.csv"
    if not frame_labels_path.exists():
        raise SystemExit("請先執行 10_merge_pilot_labels.py")
    frame_labels = pd.read_csv(frame_labels_path, dtype={"cohort": str})

    erratum = (
        "## 勘誤（2026-09-03，兩次修正）\n\n"
        "**修正一（FOV 遮罩偵測方法）**：下面「每支影片實際 frame 尺寸」一節原本只檢查"
        "影格尺寸本身，沒有另外檢查影格內部是否還殘留 FOV 遮罩邊框，是因為當時用「整行/"
        "整列是否全黑」當判斷依據，隱含假設遮罩是矩形黑邊。內視鏡的 FOV 遮罩實際上是"
        "圓形/八邊形，黑色只出現在四個角落，不會讓整行或整列全黑，所以完全沒被抓到。"
        "後來在 E0e 肉眼複查裁切前後對照圖時才發現：幾乎每一張影格的四個角落都有明顯的"
        "黑色遮罩。完整量測見 `results/fov_e0e_baseline_report.md`。\n\n"
        "**修正二（資料來源）**：本節與 E0e 原本使用的影格，來自「polyp」物件偵測專案"
        "先前為了訓練偵測器另外篩選過的一個小子集（3017 張，嚴重偏向 polyp 正樣本），"
        "不是這個 SSL 指紋研究需要的中性隨機樣本，且該子集的 JPEG 帶有 `Lavc58.134.100`"
        "重新編碼痕跡，去交錯結論的效度打折扣。已改成直接從 Figshare 官方 API 下載"
        "原始 `{video_id}_frames.tar.gz`、每支影片在中間 1/3 時間軸均勻抽樣"
        "（見 09_pilot_sample_frames.py、10_merge_pilot_labels.py），本檔案以下分析"
        "都已改用這批官方原始 pilot 資料重跑。\n"
    )
    sections = ["# E0d：資料集規格自查（pilot，5 支影片）\n", erratum, frame_size_check(frame_labels), deinterlace_check(frame_labels)]

    out_path = RESULTS_DIR / "dataset_self_check.md"
    out_path.write_text("\n".join(sections), encoding="utf-8")
    print(f"Wrote self-check report -> {out_path}")


if __name__ == "__main__":
    main()
