"""E0a + E0b（video 層級）：建立影片層級的 manifest。

輸出 results/video_manifest.csv，一列一支影片，欄位涵蓋：
- 來源標籤：cohort（=study_id，從 unique_video_name 前三碼解析）、endoscope_brand、video_id
- metadata 標籤：fps、num_frames、num_lesions、bbps、age、sex
- split：依官方規則（cohort 內 VVV 001-010 train / 011-012 val / 013-015 test）指派，
  切分單位是「整支影片」，不會有同一支影片的 frame 同時出現在兩個 split 裡
  （對應 E0a「按影片切，不按影格切」的要求）。

不需要任何影格像素資料，只讀 video_info.csv。
"""

import pandas as pd

from config import VIDEO_INFO_CSV, RESULTS_DIR, official_split_for_video


def main():
    df = pd.read_csv(VIDEO_INFO_CSV)
    df = df.rename(columns={"unique_video_name": "video_id"})
    df["cohort"] = df["video_id"].str.split("-").str[0]
    df["split"] = df["video_id"].apply(official_split_for_video)

    cols = [
        "video_id", "cohort", "endoscope_brand", "split",
        "fps", "num_frames", "num_lesions", "bbps", "age", "sex",
    ]
    df = df[cols].sort_values(["cohort", "video_id"]).reset_index(drop=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "video_manifest.csv"
    df.to_csv(out_path, index=False)

    print(f"Wrote {len(df)} videos -> {out_path}")
    print()
    print("Split x cohort 分布（確認每個 cohort 都是 10/2/3 支）：")
    print(df.pivot_table(index="cohort", columns="split", values="video_id", aggfunc="count", fill_value=0))
    print()
    print("Brand x cohort 分布（確認 §3.1 的 brand/cohort 共線問題）：")
    print(df.pivot_table(index="cohort", columns="endoscope_brand", values="video_id", aggfunc="count", fill_value=0))
    print()
    no_lesion = df[df["num_lesions"] == 0]
    print(f"完全無 polyp 的影片數：{len(no_lesion)}（研究計畫 §3.1 記載為 14 支）")
    print(no_lesion["video_id"].tolist())


if __name__ == "__main__":
    main()
