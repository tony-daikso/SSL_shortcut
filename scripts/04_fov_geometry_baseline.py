"""E0e-1 + E0e-2：FOV 幾何洩漏檢查、trivial baseline。

讀取 `pilot_frame_labels.csv`（10_merge_pilot_labels.py 的輸出，來自官方原始資料
隨機抽樣，非任何專案篩選過的子集）。corner_black_fraction 已經在抽樣當下
（09_pilot_sample_frames.py）算好，這裡不需要重新讀圖。

## 幾何特徵的兩個來源

1. **影格像素尺寸**（width, height, aspect_ratio, area）：直接來自抽樣時實際讀到的
   影格尺寸。
2. **FOV 遮罩角落殘留**（corner_black_fraction）：抽樣當下用
   `fov_protocol.corner_black_fraction()` 算好，量化四個角落框裡有多少比例是黑的。

## Trivial baseline

只用上述幾何特徵（完全不看影像內容/紋理）訓練 logistic regression，在**完全沒看過的
影片**（GroupShuffleSplit 依 video_id 分組，見 geometry_common.py）上測 cohort /
endoscope_brand 猜得準不準。這就是 E1（DINOv2 embedding 的 probe）之後必須超越的
下限——如果 E1 的 probe 準確率跟這裡的 trivial baseline 差不多，代表 probe 學到的
可能只是幾何，不是真正的內容/紋理指紋。

**這是 pilot 版本**（目前只有 5 支影片），train/test 各自能分到的影片數很小，準確率
數字本身統計效力有限，主要目的是驗證流程正確；擴大到全部 60 支後才是正式的 E0e 結論。
"""

import pandas as pd

from config import RESULTS_DIR
from geometry_common import GEOMETRY_FEATURES, add_geometry_features, train_test_split_baseline

EXTENDED_FEATURES = GEOMETRY_FEATURES + ["corner_black_fraction"]


def per_video_geometry_table(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["video_id", "cohort", "endoscope_brand"]).agg(
        n_frames=("frame_id", "count"),
        width=("width", "median"), height=("height", "median"),
        aspect_ratio=("aspect_ratio", "median"), log_area=("log_area", "median"),
        corner_black_fraction=("corner_black_fraction", "median"),
    ).reset_index()
    return g


def main():
    frame_labels = pd.read_csv(RESULTS_DIR / "pilot_frame_labels.csv", dtype={"cohort": str})
    frame_labels = add_geometry_features(frame_labels)

    per_video = per_video_geometry_table(frame_labels)
    per_video_path = RESULTS_DIR / "fov_geometry.csv"
    per_video.to_csv(per_video_path, index=False)

    lines = ["# E0e-1/E0e-2：FOV 幾何洩漏檢查 + trivial baseline（pilot，5 支影片）\n"]

    corner_by_cohort = frame_labels.groupby(["cohort", "endoscope_brand"])["corner_black_fraction"].agg(
        ["count", "median", "mean", "std"]
    )
    lines.append("## FOV 遮罩角落殘留（corner_black_fraction，20% 角落框內黑色像素比例）\n")
    lines.append(corner_by_cohort.to_markdown())
    lines.append("")

    lines.append(f"每支影片的幾何摘要已存到 `{per_video_path.name}`（{len(per_video)} 支影片）。\n")
    lines.append(per_video.to_markdown(index=False))
    lines.append("")

    lines.append("## E0e-2a：cohort（多類別）trivial baseline，held-out 影片\n")
    lines.append("### 只用影格尺寸（width/height/aspect_ratio/log_area）\n")
    cohort_size_only = train_test_split_baseline(frame_labels, "cohort", GEOMETRY_FEATURES)
    lines.append(pd.Series(cohort_size_only).to_frame("value").to_markdown())
    lines.append("\n### 加上 corner_black_fraction\n")
    cohort_result = train_test_split_baseline(frame_labels, "cohort", EXTENDED_FEATURES)
    lines.append(pd.Series(cohort_result).to_frame("value").to_markdown())
    lines.append("")

    lines.append("## E0e-2b：endoscope_brand trivial baseline，held-out 影片\n")
    brand_result = train_test_split_baseline(frame_labels, "endoscope_brand", EXTENDED_FEATURES)
    lines.append(pd.Series(brand_result).to_frame("value").to_markdown())
    lines.append(
        "\n**注意**：pilot 裡只有 cohort 002 同時涵蓋兩種品牌（002-004 Olympus、"
        "002-006 Fujifilm），其餘 3 支影片（001-001、003-001、004-003）全是 "
        "Olympus，brand 和 cohort 高度共線，這個數字暫時不能拿來下結論，等擴大到全部"
        "60 支、cohort 002 內有足夠 Olympus+Fujifilm 影片時才有意義。\n"
    )

    out_path = RESULTS_DIR / "fov_e0e_baseline_report.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}")
    print()
    print("cohort (size only):", cohort_size_only)
    print("cohort (+ corner):", cohort_result)
    print("brand (global):", brand_result)


if __name__ == "__main__":
    main()
