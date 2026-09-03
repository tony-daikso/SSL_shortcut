"""E0a + E0b（video 層級）：建立影片層級的 manifest。

輸出 results/video_manifest.csv，一列一支影片，欄位涵蓋：
- 來源標籤：cohort（=study_id，從 unique_video_name 前三碼解析）、endoscope_brand、video_id
- metadata 標籤：fps、num_frames、num_lesions、bbps、age、sex

**不含 split 欄位**（修正記錄見 config.py）：官方 001-010/011-012/013-015 是息肉偵測
benchmark 的切分慣例，不是這個 SSL 指紋研究的預設切分。E0a 真正的要求只是「切分單位
是整支影片，不按影格切」，這件事由需要 train/test 分組的分析（例如 E0e 的 trivial
baseline，見 geometry_common.py）在當下用 GroupShuffleSplit（依 video_id 分組）自己
切，不預先綁定某個特定慣例。

不需要任何影格像素資料，只讀 video_info.csv。
"""

import pandas as pd

from config import VIDEO_INFO_CSV, RESULTS_DIR


def main():
    df = pd.read_csv(VIDEO_INFO_CSV)
    df = df.rename(columns={"unique_video_name": "video_id"})
    df["cohort"] = df["video_id"].str.split("-").str[0]

    cols = [
        "video_id", "cohort", "endoscope_brand",
        "fps", "num_frames", "num_lesions", "bbps", "age", "sex",
    ]
    df = df[cols].sort_values(["cohort", "video_id"]).reset_index(drop=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "video_manifest.csv"
    df.to_csv(out_path, index=False)

    print(f"Wrote {len(df)} videos -> {out_path}")
    print()
    print("Brand x cohort 分布（確認 §3.1 的 brand/cohort 共線問題）：")
    print(df.pivot_table(index="cohort", columns="endoscope_brand", values="video_id", aggfunc="count", fill_value=0))
    print()
    no_lesion = df[df["num_lesions"] == 0]
    print(f"完全無 polyp 的影片數：{len(no_lesion)}（研究計畫 §3.1 記載為 14 支）")
    print(no_lesion["video_id"].tolist())


if __name__ == "__main__":
    main()
