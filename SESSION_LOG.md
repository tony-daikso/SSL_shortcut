# SESSION_LOG

這份檔案是按時間順序的工作紀錄，記錄「每次工作階段做了什麼、發現什麼、留下
什麼待辦」——比較細節、可信的科學結論在 `docs/E*_summary.md`，這裡是給接手的人
（包含之後的自己）快速抓到「現在整體進度到哪、下一步該接什麼」的地方。

---

## 2026-09-03（第一階段）：E0 資料準備，發現並修正資料來源錯誤

- 一開始誤用「polyp」物件偵測專案先前篩選過的偏差子集當 frame 資料來源（嚴重
  偏向 polyp 正樣本），且套用官方息肉偵測 benchmark 的 001-010/011-012/013-015
  切分規則當 E0a 的 train/test。
- **使用者指出這兩者都不對**：frame 抽樣應該直接向 Figshare 官方 API 下載、
  自己隨機抽樣；train/test 切分不該套用跟這個 SSL 指紋研究無關的官方慣例。
- 全部重做：改用 `GroupShuffleSplit` 依 video_id 動態切分（`geometry_common.py`），
  frame 像素改成直接下載官方原始 `{video_id}_frames.tar.gz`、每支影片中間
  1/3 時間軸隨機抽 100 張。
- 下載官方全部 60 支 `{video_id}_annotations.tar.gz`，發現 annotation 其實是
  逐格的（`full_annotation_labels.csv`，2,757,723 列，覆蓋率 100%），比原本
  預期的「需要抽樣估計」還要好。
- 完成 E0b/E0c（confound 報表：cohort 002 病灶明顯偏小、以 HP 為主，其他 cohort
  以 AD 為主）。
- 完成 5 支影片 pilot 規模的 E0d/E0e：發現 FOV 角落遮罩殘留是普遍問題（原本用
  「整行/整列全黑」檢查誤判成幾乎不存在）、品牌專屬 UI 疊字（Olympus「Near
  Focus」徽章、Fujifilm 時間戳記+拍攝設定）。統一裁切協定（15% inset）驗證
  有效，corner_black_fraction 中位數降到 0。
- 完成 E0.5 合成指紋校準（三項判準全數通過），過程中抓到一個關鍵 bug：
  `_class_seed()` 用 Python 內建 `hash()` 受 `PYTHONHASHSEED` 隨機化影響，
  改用 `hashlib.md5` 解決。

## 2026-09-04：E1 pilot 流程驗證（5 支影片，500 張影格）

- 完成 E1a/b/c/d 的 pilot 規模流程驗證：video ID 91.0%（chance 20%），但
  **隨機初始化的 DINOv2 就有 78.9%**——揭示相當一部分「指紋可解碼性」是 ViT
  架構歸納偏見造成的附帶編碼，不是 SSL 目標函數主動學來的。
- cohort/brand 任務在 5 支影片規模下統計效力不足（單折、常缺類別），只做流程
  驗證，不是正式結論。
- 產出 `docs/E0_summary.md`、`docs/E1_summary.md`，上傳到 Notion 子頁面。

## 資源危機與擴大到 60 支影片（跨越數天）

- 決定把 frame 抽樣從 5 支影片擴大到全部 60 支，改用 ThreadPoolExecutor 平行
  下載（診斷發現下載慢是台灣→新加坡→美國的高延遲路徑造成的頻寬-延遲乘積限制，
  不是 Figshare 或本地網路問題）。
- 發現本機也有 REAL-Colon 的 NAS 備份（945GB zip，STORED 未壓縮格式），改用
  NAS 當主要資料來源，比 Figshare 快很多。
- 過程中一度發生嚴重系統資源危機（swap 接近打滿、process 卡在 uninterruptible
  sleep、macOS 觸發 spindump），緊急停止並把 `MAX_PARALLEL_DOWNLOADS` 從 4
  調降到 2，之後全程加上系統負載監控。
- Repo 推上 GitHub（`git@github.com:tony-daikso/SSL_shortcut.git`），用 Git
  LFS 處理大檔案（`full_annotation_labels.csv`、`*.npz`）。
- 全部 60 支影片（6000 張影格）最終下載/抽樣完成。

## 2026-09-07（本次工作階段）：E0/E1 擴大到全部 60 支影片 + E3 + E4 + E5 程式碼

**上半場：把 pilot 結論升級成全資料集正式結論**
- 確認 60/60 支影片抽樣完成，重跑 E0d/E0e/E1 全部流程。
- 修正好幾支腳本裡寫死的 pilot 規模假設（單一 train/test 切分、針對 5-way
  任務校準的絕對 PC 準確率門檻、報告裡殘留的「5 支影片」文字）——改成能感知
  資料規模、10 折重複切分的版本。
- **核心發現**：video ID（47.5%，chance 1.7%）、cohort（62.8%）、brand
  （95.9%）都遠超 chance/majority；polyp_label（89.0%/85.5%）幾乎貼著
  majority baseline——支持「acquisition shortcut」確實存在。
- 隨機初始化 DINOv2 在 60-way 任務下仍有 28.8%（chance 1.7%，約 17 倍）。
- git 整理：把可重新產生的原始影格圖片（~2GB）、逐支影片抽樣 checkpoint 改成
  gitignore，不再直接進 git；清掉舊 pilot 留下的過期 QC 樣本圖。
- 更新 `docs/E0_summary.md`、`docs/E1_summary.md`、README、Notion 頁面。

**中場：E3（augmentation 劑量反應）**
- 使用者在 E2（跨 cohort 下游後果，需要新建物件偵測訓練流程）跟 E3（複用現有
  frozen embedding 架構）之間選了 E3。
- 建了 jitter 強度 sweep 管線（5 支新腳本），沿用 E0.5 的合成指紋 + E1 的
  frozen DINOv2 架構，另外手刻一個輕量 SSIM（避免引入 scikit-image 依賴）。
- **結果**：color_shift（色彩類指紋）隨 jitter 強度降到接近/低於 chance
  （47.8%→22.9%）；pattern_noise（感測器雜訊類指紋）只溫和下降、留下殘餘地板
  （45.2%→37.9%）；真實 polyp_label 幾乎零代價（89.0%→88.5%）。這是計畫 §7
  定義的「殘餘地板」最佳結果形態。

**下半場：E4（雜訊白化）驗證迴路**
- 針對 E3 留下的殘餘地板（pattern_noise），設計「雜訊重合成/白化」：估計每張
  圖的高頻殘差，按強度比例替換成每張圖各自獨立抽樣的新雜訊。
- **結果**：pattern_noise 隨白化強度單調降到 chance 以下（43.0%→24.1%），
  color_shift 維持平穩（證明修法針對性、非巧合），polyp_label 幾乎零代價
  （88.6%→87.7%）。**驗證迴路完全成立**——雙重驗證了核心機制假說。

**收尾：E5（自訓 DINO）程式碼**
- 使用者要求先把訓練程式碼寫好，之後在遠端 GPU server 執行；本機先用小規模
  資料跑 smoke test 確認能正常運作。
- 因為 timm 裝不上（本機 pip 被系統環境擋住，遠端也不確定有沒有裝），自己刻了
  一個最小 ViT-S/16（含位置編碼插值，支援 DINO multi-crop 需要的不同解析度
  輸入），只依賴既有的 torch/torchvision，不需要額外套件。
- 完整實作 DINOHead、MultiCropWrapper、DINOLoss、multi-crop augmentation（含
  E5c 用的 E4 白化變體）、資料集載入（套用 unify_crop）、訓練主程式、
  embedding 抽取、跟 E1 既有結果比較的探針腳本。
- 本機 smoke test（180 張影格、1 epoch）確認完整流程無誤：loss 從理論正確的
  初始值 ln(8192)≈9.0 開始，訓練/存檔/抽 embedding/比較全部跑通，smoke test
  暫存輸出已清除、沒有進 git。
- **E5a/b/c 尚未有正式結論**——需要在遠端 GPU 上完成正式訓練後補上，操作說明見
  `docs/E5_notes.md`。

### 目前狀態總覽

| 階段 | 狀態 | 結論文件 |
|---|---|---|
| E0/E0d/E0e | ✅ 全部 60 支影片正式結論 | `docs/E0_summary.md` |
| E0.5 | ✅ 三項判準全數通過（pilot 規模，設計上不需擴大） | `results/e05_report.md` |
| E1 | ✅ 全部 60 支影片正式結論 | `docs/E1_summary.md` |
| E2 | ⏸️ 未開始（使用者選了先做 E3） | — |
| E3 | ✅ 全部 60 支影片正式結論 | `docs/E3_summary.md` |
| E4 | ✅ 全部 60 支影片正式結論，驗證迴路成立 | `docs/E4_summary.md` |
| E5 | 🔧 程式碼完成、本機 smoke test 通過，待遠端 GPU 正式訓練 | `docs/E5_notes.md` |

### 待辦 / 下一步

1. **E5 正式訓練**：使用者要在遠端 GPU server（單卡 L40S 48GB）上執行，需要
   先把 `data/sampled_frames/`（本機有、gitignore 掉了）傳過去或在遠端重新
   產生，再依實際 GPU 規格調整 `--epochs`/`--batch-size`/`--n-local-crops`。
2. E5 訓練完成後：跑 `26_e5_extract_embeddings.py` + `27_e5_compare_probes.py`，
   補上 `docs/E5_summary.md`（比照其他 E*_summary.md 的格式），更新 README
   跟 Notion。
3. 依計畫順序，E5 之後可以考慮 E2（跨 cohort 下游後果，需要新建物件偵測訓練
   流程）或 E6（既有捷徑移除方法統一比較）。
