"""Path configuration for E0 scripts.

REAL-Colon 官方原始 60 支影片的 tar（frames）與 annotation 壓縮包本身沒有放進這個
repo（太大，見研究計畫 §3.5 對算力/儲存的討論）。目前唯一在這台機器上已經解壓出來、
可直接讀取的 per-frame VOC annotation，是「polyp」物件偵測專案先前為了訓練 YOLO
從官方 annotation 中萃取出來的一個子集（見 REAL_COLON_FRAMES_ROOT）。

這個子集**不是**官方完整的逐格標註：
- all_polyp/：只包含目視有 lesion bbox 的影格，且經過額外篩選（每支影片只有個位數到
  數十張，遠少於官方總計 351,264 個 bbox），不是「該影片所有帶 bbox 的影格」。
- no_polyp/：每支影片等間隔抽樣出的一小批負樣本影格（數十張），不是官方 87.6% 的
  全部負影格。

因此本專案用它來源生的 frame-level 統計（frame_labels.csv、confound_report 裡標示為
「子集」的部分）只能當作**探索性/建置管線用**的數字，不能取代日後下載官方完整
annotation 後重跑的版本。E0d 的自查結論也建立在這個前提上。

若之後下載到官方完整資料，只需要把 REAL_COLON_FRAMES_ROOT 指到新的路徑（維持
`{video_id}/label/*.xml` 的目錄結構，或者調整 01_build_frame_labels.py 裡的
glob pattern），其餘腳本不需要改動。
"""

from pathlib import Path

# 官方 video_info.csv / lesion_info.csv 的本地副本（已複製進本 repo，見 data/raw_refs/）
VIDEO_INFO_CSV = Path(__file__).resolve().parent.parent / "data" / "raw_refs" / "video_info.csv"
LESION_INFO_CSV = Path(__file__).resolve().parent.parent / "data" / "raw_refs" / "lesion_info.csv"

# 外部路徑：polyp 專案先前從官方 annotation 萃取出的子集（見上方說明，非本 repo 管理）
REAL_COLON_FRAMES_ROOT = Path(
    "/Users/tony.tu/Desktop/戴承智慧/polyp/Real_colon_data/all_data_png"
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# 官方切分規則（研究計畫 §3.1）：每個 cohort 的 VVV 編號 001-010 train、011-012 val、013-015 test
def official_split_for_video(video_id: str) -> str:
    vvv = int(video_id.split("-")[1])
    if 1 <= vvv <= 10:
        return "train"
    if 11 <= vvv <= 12:
        return "val"
    if 13 <= vvv <= 15:
        return "test"
    raise ValueError(f"unexpected video numbering: {video_id}")
