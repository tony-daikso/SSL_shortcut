"""合併 pilot 抽樣結果，產生取代舊版 frame_labels.csv 的 pilot_frame_labels.csv。

三個來源接在一起：
- `pilot_sampled_frames.csv`（09_pilot_sample_frames.py）：像素導出的欄位
  （width/height/corner_black_fraction/frame_path），來自官方原始 frames.tar.gz。
- `video_manifest.csv`（00_build_video_manifest.py）：來源標籤（cohort/endoscope_brand）。
- `full_annotation_labels.csv`（07_download_all_annotations.py）：病理標籤
  （polyp_label/n_bbox），依 video_id + frame_index 對應到同一張影格，官方逐格標註，
  不是抽樣估計。

輸出的 `pilot_frame_labels.csv` 是 03/04/05/06 這幾支 E0d/E0e 腳本目前該讀取的資料，
取代已經證實有問題的舊版 `frame_labels.csv`（來自「polyp」專案的偏差子集，見
config.py 的修正記錄）。
"""

import pandas as pd

from config import RESULTS_DIR


def main():
    sampled = pd.read_csv(RESULTS_DIR / "pilot_sampled_frames.csv")
    video_manifest = pd.read_csv(RESULTS_DIR / "video_manifest.csv", dtype={"cohort": str})
    full_annotations = pd.read_csv(RESULTS_DIR / "full_annotation_labels.csv", dtype={"cohort": str})

    merged = sampled.merge(
        video_manifest[["video_id", "cohort", "endoscope_brand"]], on="video_id", how="left"
    )
    merged = merged.merge(
        full_annotations[["video_id", "frame_index", "n_bbox", "polyp_label"]],
        on=["video_id", "frame_index"], how="left",
    )

    missing_annotation = merged["polyp_label"].isna().sum()
    if missing_annotation:
        print(f"警告：{missing_annotation} 張抽樣影格在 full_annotation_labels.csv 裡找不到"
              "對應的 frame_index，polyp_label 會是 NaN，需要檢查抽樣邏輯或 annotation 覆蓋率。")

    merged["frame_id"] = merged["video_id"] + "_" + merged["frame_index"].astype(str)

    out_path = RESULTS_DIR / "pilot_frame_labels.csv"
    merged.to_csv(out_path, index=False)

    print(f"Wrote {len(merged)} 列 -> {out_path}")
    print(merged.groupby(["cohort", "endoscope_brand"]).agg(
        n_frames=("frame_id", "count"), n_videos=("video_id", "nunique"),
        n_polyp=("polyp_label", "sum"),
    ))


if __name__ == "__main__":
    main()
