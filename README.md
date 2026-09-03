# SSL_shortcut

研究計畫：內視鏡自監督學習中的採集捷徑（REAL-Colon）。完整計畫見 Notion：
https://app.notion.com/p/3d08ebc0eef38055834bc7ab97b3f024

本 repo 目前只涵蓋計畫 §6 的 **E0：資料準備** 這一步。之後的 E0e（FOV 幾何洩漏檢查）、
E1（指紋可解碼性）等會是獨立的後續階段。

## 跟 SSL_research（既有 Phase 0 repo）的關係

`SSL_research` 是同一個大方向下、更早開始的 Phase 0 可行性測試（frozen DINOv2 + linear
probe，已有初步 GO 結論）。這個 repo 是刻意**獨立重新建立**的：不 import 它的程式碼，
也不共用它下載/處理過的影片。兩者都是同一個研究計畫底下的產出，但各自維護。

## 資料依賴（重要，看懂這個才看得懂 scripts 在幹嘛）

- `data/raw_refs/`：官方 `video_info.csv`、`lesion_info.csv`、`dataset_description.md` 的
  本地副本（複製自 `polyp/Real_colon_data/`，這是公開資料集本身的 metadata，不是程式碼）。
  60 支影片、132 顆病灶的完整記錄，**權威、完整**。
- `scripts/config.py` 裡的 `REAL_COLON_FRAMES_ROOT` 指向一個**外部路徑**
  （`polyp/Real_colon_data/all_data_png/`）：這是「polyp」物件偵測專案之前從官方
  annotation 裡萃取出來的一個小子集（46 支影片的少量 polyp 影格 + 60 支影片的少量抽樣
  負樣本影格，總共 3017 張，1.1GB），**不是**官方完整的逐格標註（官方有 2,757,723 張
  影格、351,264 個 bbox）。這個子集足以拿來建置/測試 E0 的 pipeline 並看出方向性的
  結論，但任何由它算出來的 frame 層級統計都在對應報告裡明確標成「探索性/子集」，
  不能對外當成最終數字。
  若之後下載到官方完整資料，只要把這個路徑換掉，其餘程式不用改。
  詳細說明見 `scripts/config.py` 開頭的 docstring。

## 執行順序

```bash
pip install -r requirements.txt
cd scripts
python3 00_build_video_manifest.py     # E0a + E0b(video 層級)：官方切分 + 來源/metadata 標籤
python3 01_build_frame_labels.py       # E0b(frame 層級)：polyp/no-polyp 標籤 + 實際影格尺寸
python3 02_confound_report.py          # E0c：per-cohort 盛行率/大小/型態/histology 分布
python3 03_dataset_self_check.py       # E0d：影格尺寸自查 + 去交錯 heuristic 篩檢
```

輸出都在 `results/`：
- `video_manifest.csv`：60 支影片的來源/metadata 標籤 + split
- `frame_labels.csv`：3017 張抽樣影格的三組標籤
- `confound_report.md`：E0c 報表
- `dataset_self_check.md`：E0d 報表
- `qc_deinterlace_samples/`：去交錯自查挑出的可疑影格裁切圖，供肉眼複查

完整結論與交接記錄見 [docs/E0_summary.md](docs/E0_summary.md)。
