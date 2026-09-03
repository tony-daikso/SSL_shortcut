"""E0e-1 + E0e-2：FOV 幾何洩漏檢查、trivial baseline。

## 幾何特徵的兩個來源

1. **影格像素尺寸**（width, height, aspect_ratio, area）：直接來自 frame_labels.csv
   的 annotation `<size>` 欄位，不需要讀圖。
2. **FOV 遮罩角落殘留**（corner_black_fraction）：需要實際讀取影像才能算，見
   `fov_protocol.corner_black_fraction()`。這個特徵是修正過先前一個錯誤才加進來的——
   E0d（`03_dataset_self_check.py`）原本用「整行/整列是否全黑」檢查有沒有殘留 FOV
   遮罩，結論是「幾乎沒有」，但這個檢查方法有漏洞：內視鏡的 FOV 遮罩是圓形/八邊形，
   黑色只出現在四個角落，不會讓整行或整列全黑，所以完全沒被那個檢查抓到。肉眼複查
   E0e 的裁切前後對照圖時才發現幾乎每一張影格角落都有明顯黑色遮罩，用 20% 大小的
   角落框在 200 張隨機抽樣影格上量測，平均有 ~12%（中位數更高，見下方 per-frame
   實際統計）的像素是黑的，100% 的抽樣影格都有這個現象——比單純的尺寸差異更普遍、
   更系統性。E0d 的報告已經加上勘誤，這裡是正式把它當一個特徵處理。

## Trivial baseline

只用上述幾何特徵（完全不看影像內容/紋理）訓練 logistic regression，在**完全沒看過的
影片**（官方 test split）上測 cohort / endoscope_brand 猜得準不準。這就是 E1
（DINOv2 embedding 的 probe）之後必須超越的下限——如果 E1 的 probe 準確率跟這裡的
trivial baseline 差不多，代表 probe 學到的可能只是幾何，不是真正的內容/紋理指紋。
"""

import pandas as pd

from config import RESULTS_DIR, REAL_COLON_FRAMES_ROOT
from fov_protocol import corner_black_fraction
from geometry_common import GEOMETRY_FEATURES, add_geometry_features, train_test_split_baseline, leave_one_video_out_baseline
from PIL import Image

EXTENDED_FEATURES = GEOMETRY_FEATURES + ["corner_black_fraction"]


def add_corner_black_fraction(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    values = []
    for _, r in df.iterrows():
        img_path = REAL_COLON_FRAMES_ROOT / r["source_category"] / r["video_id"] / "image" / f"{r['frame_id']}.jpg"
        if not img_path.exists():
            values.append(None)
            continue
        values.append(corner_black_fraction(Image.open(img_path)))
    df["corner_black_fraction"] = values
    return df.dropna(subset=["corner_black_fraction"])


def per_video_geometry_table(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["video_id", "cohort", "endoscope_brand"]).agg(
        n_frames=("frame_id", "count"),
        width=("width", "median"), height=("height", "median"),
        aspect_ratio=("aspect_ratio", "median"), log_area=("log_area", "median"),
        corner_black_fraction=("corner_black_fraction", "median"),
        n_distinct_wh=("width", lambda s: df.loc[s.index, ["width", "height"]].drop_duplicates().shape[0]),
    ).reset_index()
    return g


def main():
    frame_labels = pd.read_csv(RESULTS_DIR / "frame_labels.csv", dtype={"cohort": str})
    frame_labels = add_geometry_features(frame_labels)
    print("正在讀取全部影格計算 corner_black_fraction（需要實際打開圖檔）...")
    frame_labels = add_corner_black_fraction(frame_labels)

    per_video = per_video_geometry_table(frame_labels)
    per_video_path = RESULTS_DIR / "fov_geometry.csv"
    per_video.to_csv(per_video_path, index=False)

    lines = ["# E0e-1/E0e-2：FOV 幾何洩漏檢查 + trivial baseline\n"]

    corner_by_cohort = frame_labels.groupby(["cohort", "endoscope_brand"])["corner_black_fraction"].agg(
        ["count", "median", "mean", "std"]
    )
    lines.append(
        "## FOV 遮罩角落殘留（corner_black_fraction，20% 角落框內黑色像素比例）\n\n"
        "**勘誤**：E0d 原本的檢查方法（整行/整列全黑）沒有偵測到這個訊號，實際上"
        f"{len(frame_labels)} 張影格裡角落殘留是普遍存在的，且依 cohort/品牌有明顯差異：\n"
    )
    lines.append(corner_by_cohort.to_markdown())
    lines.append("")

    lines.append(f"每支影片的幾何摘要已存到 `{per_video_path.name}`（{len(per_video)} 支影片）。\n")

    lines.append("## E0e-2a：cohort（4-way）trivial baseline，train→test（held-out 影片）\n")
    lines.append("### 只用影格尺寸（width/height/aspect_ratio/log_area）\n")
    cohort_size_only = train_test_split_baseline(frame_labels, "cohort", GEOMETRY_FEATURES)
    lines.append(pd.Series(cohort_size_only).to_frame("value").to_markdown())
    lines.append("\n### 加上 corner_black_fraction\n")
    cohort_result = train_test_split_baseline(frame_labels, "cohort", EXTENDED_FEATURES)
    lines.append(pd.Series(cohort_result).to_frame("value").to_markdown())
    lines.append("")

    lines.append("## E0e-2b：endoscope_brand（2-way）trivial baseline，train→test（held-out 影片）\n")
    brand_result = train_test_split_baseline(frame_labels, "endoscope_brand", EXTENDED_FEATURES)
    lines.append(pd.Series(brand_result).to_frame("value").to_markdown())
    lines.append(
        "\n**注意**：brand 和 cohort 高度共線（只有 cohort 002 同時有兩種品牌，見 E0a 的"
        "報表），這裡的高準確率有可能只是在猜『這是不是 cohort 002 的影片』，不是真的"
        "學到品牌的幾何差異。下面用 cohort 002 內部做 within-cohort control 拆解這個問題。\n"
    )

    lines.append("## E0e-2c：endoscope_brand within-cohort-002 control（leave-one-video-out）\n")
    cohort002 = frame_labels[frame_labels["cohort"] == "002"]
    within_result = leave_one_video_out_baseline(cohort002, "endoscope_brand", EXTENDED_FEATURES)
    lines.append(pd.Series(within_result).to_frame("value").to_markdown())
    lines.append(
        "\n**解讀**：只用 cohort 002 內部的影片（同時有 Olympus 和 Fujifilm），輪流留一支"
        "出來測試。若準確率仍明顯高於 majority baseline，代表光是幾何（尺寸+角落遮罩）"
        "就能分辨品牌，不是靠 cohort 這個共線變因取巧。\n"
    )

    out_path = RESULTS_DIR / "fov_e0e_baseline_report.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}")
    print()
    print("cohort (size only):", cohort_size_only)
    print("cohort (+ corner):", cohort_result)
    print("brand (global):", brand_result)
    print("brand (within cohort 002):", within_result)


if __name__ == "__main__":
    main()
