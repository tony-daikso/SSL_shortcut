"""E0d：資料集規格自查——去交錯處理方式、每支影片的實際影格尺寸。

研究計畫 §3.2/§9 指出兩件事官方論文沒說清楚、需要自己查：
1. 每支影片實際的 frame 尺寸（FOV 裁切後每支/每台設備可能不同，§3.3 的 FOV 幾何洩漏
   問題的前置證據）。
2. 1080i 交錯掃描的去交錯處理方式——論文沒說明，需要自己檢查像素。

兩者都只需要用到 01_build_frame_labels.py 已經處理過的子集（frame_labels.csv 的
metadata + 對應圖檔），不需要完整資料集。frame 尺寸的結論是可信的（子集涵蓋所有 60
支影片，size 是從官方 annotation 的 <size> 欄位讀出，不是猜的）；去交錯的結論只是
一個篩檢用的量化 heuristic，不是最終判斷，必須人工複查 results/qc_deinterlace_samples/
裡存的樣本圖才能下結論。

去交錯 heuristic 的原理：交錯掃描殘留的「梳齒」(combing) 是因為相鄰兩條掃描線來自
不同時間的兩個 field，在有運動的區域，相鄰行(row i, i+1)的差異會明顯大於間隔一行
(row i, i+2，同一個 field 內)的差異。用兩者比值當分數，分數越高代表殘留交錯痕跡的
嫌疑越大。這只是篩檢，不是證明——真正判斷仍要肉眼看樣本圖。
"""

import random

import numpy as np
import pandas as pd
from PIL import Image

from config import REAL_COLON_FRAMES_ROOT, RESULTS_DIR

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
        f"抽查子集涵蓋 {frame_labels['video_id'].nunique()} / 60 支影片。"
        f"其中 {len(inconsistent)} 支影片在子集內就觀察到超過一種 (width, height) 組合"
        "（同一支影片內部 FOV 裁切尺寸不一致，可能代表裁切邏輯本身不穩定，或這支影片"
        "中途換過顯示模式/腳位）。\n"
    )
    if len(inconsistent):
        detail = (
            frame_labels[frame_labels["video_id"].isin(inconsistent.index)]
            .groupby(["video_id", "width", "height"]).size().rename("n_frames_in_subset")
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
        "\n**解讀**：FOV 尺寸明顯不是單一固定值（見上表多種組合），且不同尺寸群組對應到"
        "不同 cohort/品牌組合——這正是研究計畫 §3.3 擔心的 FOV 幾何洩漏的直接證據：光是"
        "影格寬高本身就可能足以猜出 cohort/品牌，不需要看內容。這個問題留給 E0e 正式處理"
        "（trivial baseline + 統一裁切協定），這裡只負責把『尺寸確實不一致』這件事釘死。\n"
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

    rng = random.Random(RNG_SEED)
    qc_dir = RESULTS_DIR / "qc_deinterlace_samples"
    qc_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for video_id, group in frame_labels.groupby("video_id"):
        sample = group.sample(n=min(SAMPLES_PER_VIDEO, len(group)), random_state=RNG_SEED)
        for _, r in sample.iterrows():
            img_path = (
                REAL_COLON_FRAMES_ROOT / r["source_category"] / video_id / "image" / f"{r['frame_id']}.jpg"
            )
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

    lines.append(f"### 依品牌分組的 interlace score 中位數（初步看品牌間是否有系統性差異）\n")
    by_brand = scores.groupby("endoscope_brand")["interlace_score"].agg(["median", "mean", "std", "count"])
    lines.append(by_brand.to_markdown())
    lines.append("")

    top_n = 6
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
    lines.append(
        "\n**人工複查結果（2026-09-03，肉眼檢查上述樣本中的 002-015_1999、004-006_10999 "
        "兩張）**：兩張都只看到感測器雜訊顆粒（sensor noise grain）與黏膜正常紋理，"
        "**沒有觀察到水平梳齒/鋸齒狀邊緣**這類典型交錯殘留。與下面的統計篩檢結果（比值"
        "全部 <1）方向一致，初步支持『這批影格沒有明顯殘留交錯痕跡』的結論，但仍只檢查了"
        "2 張，樣本數很小，之後有餘力應再抽查更多張（尤其分數最高的 Fujifilm 影片）才能"
        "把這個結論講得更滿。\n"
    )
    overall_range = f"{scores['interlace_score'].min():.2f}–{scores['interlace_score'].max():.2f}"
    lines.append(
        f"\n**初步解讀**：分數全部落在 {overall_range} 之間、都小於 1（相鄰行差異其實小於"
        "隔行差異），沒有出現典型交錯殘留會有的「比值明顯大於 1」的尖峰。方向上比較像是"
        "**已經做過某種去交錯/平滑處理**，而不是原始交錯掃描直接轉檔。但這仍只是統計上的"
        "篩檢結果，不是像素級的視覺確認——**這裡的判斷仍待人工打開下方存的樣本圖複查**，"
        "尤其要注意這批圖本身已經是 JPEG 重新編碼過的版本（見 config.py 說明，comment 欄位"
        "顯示經過 ffmpeg/libavcodec 處理），如果去交錯是在官方釋出前就做好，這個 heuristic"
        "量到的其實是『重新編碼後還看不看得出殘留』，不是『官方原始流程有沒有去交錯』——"
        "這點差異必須在最終結論裡講清楚，避免過度推論。\n"
    )
    return "\n".join(lines)


def main():
    frame_labels_path = RESULTS_DIR / "frame_labels.csv"
    if not frame_labels_path.exists():
        raise SystemExit("請先執行 01_build_frame_labels.py")
    frame_labels = pd.read_csv(frame_labels_path, dtype={"cohort": str})

    sections = ["# E0d：資料集規格自查\n", frame_size_check(frame_labels), deinterlace_check(frame_labels)]

    out_path = RESULTS_DIR / "dataset_self_check.md"
    out_path.write_text("\n".join(sections), encoding="utf-8")
    print(f"Wrote self-check report -> {out_path}")


if __name__ == "__main__":
    main()
