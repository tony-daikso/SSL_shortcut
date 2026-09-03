"""Path configuration for E0 scripts.

## 修正記錄（2026-09-03）

最初的版本用「polyp」物件偵測專案先前從官方 annotation 萃取出的一個子集（見已刪除的
REAL_COLON_FRAMES_ROOT）當 frame 資料來源，並且用官方 001-010/011-012/013-015 這個
切分規則當 E0a 的 train/val/test。使用者指出兩者都不對：
- 那個子集嚴重偏向 polyp 正樣本、負樣本抽樣又稀疏不均，跟這個 SSL 指紋研究需要的
  中性隨機樣本完全不符（研究計畫 §4 說得很清楚：SSL 預訓練與指紋軸分析都要用
  REAL-Colon 本來的樣子，2.7M 影格、未經處理）。
- 官方 001-010/011-012/013-015 切分是**息肉偵測 benchmark 的慣例**（另一個專案在用），
  跟這裡「按影片切、不按影格切」的要求（避免同影片高度相關的 frame 洩漏）是兩件不同
  的事——E0a 只要求切分單位是整支影片，沒有要求套用那個特定的切分方案。

修正後：
- Frame 像素資料改成直接從 Figshare 官方 API 下載 `{video_id}_frames.tar.gz`，每支
  影片自己在中間 1/3 時間軸做均勻抽樣（見 09_pilot_sample_frames.py），不依賴任何
  其他專案處理過的子集。
- Annotation（病理標籤）改成下載官方 60 支 `{video_id}_annotations.tar.gz`（見
  07_download_all_annotations.py）——這批很小、可以整批下載，且實測是逐格標註
  （每一格都有一個 XML），比任何抽樣子集都更完整、更權威。
- 不再有一個全域「官方切分」的 split 欄位。任何需要 train/test 分組的分析
  （例如 E0e 的 trivial baseline）自己在函式內部用 GroupShuffleSplit（依 video_id
  分組）現場切，避免不小心把某個特定用途（息肉偵測 benchmark）的慣例誤用成整個
  研究計畫的預設切分。
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# 官方 video_info.csv / lesion_info.csv 的本地副本（已複製進本 repo，見 data/raw_refs/）
VIDEO_INFO_CSV = REPO_ROOT / "data" / "raw_refs" / "video_info.csv"
LESION_INFO_CSV = REPO_ROOT / "data" / "raw_refs" / "lesion_info.csv"

RESULTS_DIR = REPO_ROOT / "results"
