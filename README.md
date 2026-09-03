# SSL_shortcut

研究計畫：內視鏡自監督學習中的採集捷徑（REAL-Colon）。完整計畫見 Notion：
https://app.notion.com/p/3d08ebc0eef38055834bc7ab97b3f024

本 repo 目前涵蓋計畫 §6 的 **E0：資料準備** 與 **E0e：FOV 幾何洩漏檢查**。之後的
E1（指紋可解碼性）等會是獨立的後續階段。

## 跟 SSL_research（既有 Phase 0 repo）的關係

`SSL_research` 是同一個大方向下、更早開始的 Phase 0 可行性測試（frozen DINOv2 + linear
probe，已有初步 GO 結論）。這個 repo 是刻意**獨立重新建立**的：不 import 它的程式碼。
唯一的例外是 004-003 這支影片直接讀取 `SSL_research/videos/004-003_frames.tar`（它先前
下載/解壓過的殘留檔），省下一次重新下載官方 9GB 原始檔，純粹讀取不修改/不刪除。

## 資料來源（重要，看懂這個才看得懂 scripts 在幹嘛）

- `data/raw_refs/`：官方 `video_info.csv`、`lesion_info.csv`、`dataset_description.md` 的
  本地副本、以及 `figshare_files.json`（Figshare API 檔案清單快取）。60 支影片、132 顆
  病灶的完整記錄，**權威、完整**。
- `results/full_annotation_labels.csv`：下載官方全部 60 支 `{video_id}_annotations.tar.gz`
  （每支 KB 等級，全部加總數十 MB，安全整批下載，見 `07_download_all_annotations.py`）
  解析出的**完整逐格**病理標籤——2,757,723 列，覆蓋率 100%（每一格都有一個 annotation
  XML，只是沒有病灶時 `<object>` 是空的），加權平均 polyp 比例 12.41%，跟官方論文的
  「87.6% 影格無標註」完全吻合。這是全資料集層級（不只 pilot）都可信的權威資料。
- `data/sampled_frames/{video_id}/`：pilot 影格像素資料，來自 Figshare 官方原始
  `{video_id}_frames.tar.gz`（每支 7-16GB，60 支加總遠超本機可用空間，只能一支一支
  下載、抽完立刻刪除），每支影片在**中間 1/3 時間軸**隨機抽 100 張（見
  `09_pilot_sample_frames.py`）。**目前只有 5 支影片的 pilot**（001-001、002-004、
  002-006、003-001、004-003，涵蓋 4 個 cohort、兩種品牌），驗證流程正確；是否擴大到
  全部 60 支待決定。

### 修正記錄（2026-09-03）

最初的版本誤用「polyp」物件偵測專案先前篩選過的一個子集當 frame 資料來源、且套用
官方 001-010/011-012/013-015 切分當 E0a 的 train/test。使用者指出這兩者都不對：
- 那個子集嚴重偏向 polyp 正樣本、負樣本抽樣稀疏不均，跟這個 SSL 指紋研究需要的中性
  隨機樣本不符（研究計畫 §4：SSL 預訓練與指紋軸分析都要用 REAL-Colon 本來的樣子）。
- 官方切分是息肉偵測 benchmark 的慣例，不是這個研究的切分需求——E0a 只要求「按影片切、
  不按影格切」，不要求套用那個特定慣例。

修正後：frame 像素改成直接向 Figshare 官方 API 下載、自己隨機抽樣；病理標籤改成下載
官方完整 annotation；train/test 分組改成 `geometry_common.py` 用 `GroupShuffleSplit`
依 video_id 動態切，不綁定任何特定慣例。詳細說明見 `scripts/config.py`、
`scripts/09_pilot_sample_frames.py` 開頭的 docstring。

## 執行順序

```bash
pip install -r requirements.txt
cd scripts
python3 00_build_video_manifest.py       # E0a + E0b(video 層級)：來源/metadata 標籤
python3 07_download_all_annotations.py   # E0b(病理標籤)：下載官方全部 60 支逐格 annotation
python3 02_confound_report.py            # E0c：per-cohort 盛行率/大小/型態/histology 分布
python3 09_pilot_sample_frames.py        # E0a(frame 像素)：pilot 影片抽樣（平行下載）
python3 10_merge_pilot_labels.py         # 合併三個來源成 pilot_frame_labels.csv
python3 03_dataset_self_check.py         # E0d：影格尺寸自查 + 去交錯 heuristic 篩檢
python3 04_fov_geometry_baseline.py      # E0e-1/2：FOV 幾何特徵 + trivial baseline（裁切前）
python3 05_calibrate_crop_margin.py      # 校準統一裁切協定的邊距參數
python3 06_verify_unify_crop.py          # E0e-3/4：套用統一裁切協定 + 驗證 baseline 掉到 chance
```

輸出都在 `results/`：
- `video_manifest.csv`：60 支影片的來源/metadata 標籤（不含 split，見上方修正記錄）
- `full_annotation_labels.csv` / `annotation_coverage.csv`：官方完整逐格病理標籤
- `confound_report.md`：E0c 報表（Part A 病灶層級、Part B frame 層級，兩者都是官方完整資料）
- `pilot_sampled_frames.csv` / `pilot_frame_labels.csv`：pilot 5 支影片、500 張影格的
  像素導出特徵 + 三組標籤
- `dataset_self_check.md`：E0d 報表
- `qc_deinterlace_samples/`：去交錯自查挑出的可疑影格裁切圖，供肉眼複查
- `fov_geometry.csv` / `fov_e0e_baseline_report.md`：E0e-1/2，每支影片的幾何特徵 +
  trivial baseline 準確率（含品牌專屬 UI 疊字的質性觀察）
- `fov_crop_margin_calibration.md`：裁切邊距（`fov_protocol.INSET_FRACTION`）的校準過程
- `fov_e0e4_verification.md`：E0e-3/4，套用統一裁切協定後重跑驗證的結果
- `qc_unify_crop_samples/`：裁切前後對照圖，供肉眼複查

## 兩個值得注意的發現

1. **FOV 角落遮罩殘留**：E0d 一開始用「整行/整列是否全黑」檢查，結論是「幾乎沒有」，
   但這個方法有漏洞——內視鏡遮罩是圓形/八邊形，黑色只出現在四個角落，不會讓整行/整列
   全黑。肉眼複查裁切前後對照圖才發現幾乎每張影格角落都有明顯黑色遮罩，比單純的影格
   尺寸差異更普遍、更系統性，已在 `dataset_self_check.md` 加上勘誤。
2. **品牌專屬的螢幕燒錄 UI 疊字**：Olympus 影格右上角有「Near Focus」徽章；Fujifilm
   影格左上角有完整的日期/時間戳記 + 右上角拍攝設定資訊——這是比任何幾何統計量都更
   直接可讀的採集指紋。統一裁切協定實測都能一併裁掉。詳見
   `fov_e0e_baseline_report.md`。

完整結論與交接記錄見 [docs/E0_summary.md](docs/E0_summary.md)。
