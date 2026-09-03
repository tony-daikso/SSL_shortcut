# E0：資料準備——完成摘要（修正版）

對應研究計畫（Notion）§6「E0：資料準備」與「E0e：FOV 幾何洩漏檢查」，日期 2026-09-03。

**這份摘要取代同日稍早的版本**：原始版本用了「polyp」物件偵測專案篩選過的偏差子集
當 frame 資料來源、且誤用官方息肉偵測 benchmark 的切分規則當 E0a 的 train/test，
使用者指出後已全部重做。詳細修正過程見 `README.md` 與 `scripts/config.py`。

## E0a：影格抽樣與切分——✅ 完成（修正版）

- **切分**：不再套用官方 001-010/011-012/013-015 慣例（那是息肉偵測 benchmark 的
  切分，跟這個 SSL 指紋研究無關）。改由 `geometry_common.py` 用 `GroupShuffleSplit`
  依 video_id 動態分組，需要 train/test 的分析當下自己切，維持「按影片切、不按影格
  切」這個唯一的硬性要求。
- **frame 抽樣**：直接向 Figshare 官方 API 下載原始 `{video_id}_frames.tar.gz`，
  每支影片只從**中間 1/3 時間軸**隨機抽 100 張（排除前後各 1/3），不依賴任何其他
  專案處理過的子集。目前完成 **5 支影片的 pilot**（001-001、002-004、002-006、
  003-001、004-003，涵蓋 4 個 cohort、兩種品牌，共 500 張），是否擴大到全部 60 支
  待決定。

## E0b：三組標籤——✅ 完成（video 層級與病理標籤都是官方完整資料）

- **來源標籤**：cohort、endoscope_brand、video_id——`video_manifest.csv`，官方完整
  覆蓋 60 支影片。
- **病理標籤**：polyp/no-polyp，由 bbox 是否存在導出——`full_annotation_labels.csv`，
  下載官方全部 60 支 annotation 後解析，**2,757,723 列，覆蓋率 100%**（每一格都有
  annotation XML），不是抽樣估計。這比最初設想的「需要抽樣估計」的規劃還要好。
- **metadata 標籤**：fps、num_frames、num_lesions、bbps（video 層級，官方完整）；
  實際影格寬高、corner_black_fraction（frame 層級，來自 pilot 抽樣，5 支影片 500 張）。

## E0c：Confound 報表——✅ 完成（權威資料）

`results/confound_report.md`。Part A（病灶層級，132 顆病灶完整記錄）：
- 影片層級盛行率：cohort 001 60%、002 73.3%、003 93.3%、004 80%。
- 病灶大小：cohort 002 明顯偏小（mean 1.71mm vs. 其他 3.8–4.6mm）。
- Histology：cohort 002 以 HP 為主，其他 cohort 以 AD 為主——cohort 與病理型態高度
  相關，是 leave-one-cohort-out（E2）最大的 confound 風險來源之一。
- 資料品質瑕疵：`site` 欄位拼字不一致（`caecum`/`cecum`、`transverse`/`trasnverse`），
  如實記錄未擅自合併。

Part B（frame 層級，**2026-09-03 升級為官方完整資料**，2,757,723 張影格）：
- 加權平均 polyp frame 比例 12.41%，跟官方論文「87.6% 無標註」完全吻合，驗證解析
  邏輯正確。
- 各 cohort frame 層級盛行率：001 9.41%、002 16.35%、003 10.70%、004 15.30%——
  **跟病灶層級的盛行率排序不完全一致**（cohort 002 病灶顆數多但 frame 盛行率不是
  最高），代表「病灶顆數」與「病灶在時間軸上出現的長度」是兩種不同的量測，E2c 做
  confound 對照時要留意用哪一種。

## E0d：規格自查——✅ 完成（在官方原始資料上重新驗證）

`results/dataset_self_check.md`：
1. **影格尺寸不是單一固定值**，且與 cohort/品牌強烈對應，直接印證 §3.3 FOV 幾何
   洩漏疑慮（完整量化見 E0e）。
2. **去交錯**：在官方原始（非重新編碼過）影格上重新量測，59→5 支影片的 interlace
   score 全部落在 0.52–0.58 之間、都 <1，沒有典型交錯殘留的訊號。肉眼複查時額外
   發現一張影格有色彩通道錯位/鬼影，記錄下來但不下結論（樣本太小，需要更多樣本）。

## E0e：FOV 幾何洩漏檢查——✅ 完成（pilot 規模，5 支影片）

**E0e-1（幾何特徵）**：兩個獨立來源——
- 影格像素尺寸（width/height/aspect_ratio/area）。
- **角落 FOV 遮罩殘留**（corner_black_fraction）：E0d 原本用「整行/整列全黑」檢查
  誤判成幾乎不存在（那個方法偵測不到圓形/八邊形遮罩的角落殘留），肉眼複查裁切前後
  對照圖才發現普遍存在，200→500 張影格上重新量測，中位數依 cohort/品牌落在
  0.57–0.78 之間。
- **質性發現**：Olympus 影格右上角有「Near Focus」UI 徽章；Fujifilm 影格左上角有
  完整日期/時間戳記（例如 `9/05/2021 0:48:34`）+ 右上角拍攝設定——比任何像素統計都
  更直接可讀的品牌指紋。統一裁切協定實測都能一併裁掉。

**E0e-2（trivial baseline，裁切前，held-out 影片）**：
- Cohort：純尺寸 77.3%→pilot 縮小到 5 支後單折結果不穩定（0% 或 100%，取決於哪支
  影片被留出來測試，詳見 `fov_e0e_baseline_report.md` 的具體解釋）。
- Endoscope brand：pilot 只有 1 支 Fujifilm 影片，train/test 分組時容易讓某一邊
  完全沒有 Fujifilm 樣本，分類器訓練不起來——**這是 pilot 規模限制，不是方法問題**，
  正式結論要等擴大到全部 60 支才有意義。

**E0e-3（統一裁切協定）**：中央方形裁切 → 再裁掉校準得出的 15% 邊距 → resize 到
224x224。邊距校準過程見 `fov_crop_margin_calibration.md`：inset 從 0 加到 0.05 就讓
平均角落殘留從 ~14% 驟降到 ~0.4%，選比最小可行值稍保守的 0.15。

**E0e-4（驗證，裁切後）**：分類器層級的驗證在 5 支影片規模下退化（訓練折常缺類別，
accuracy 不是 0/1 的退化值就是 nan），**改用 corner_black_fraction 中位數是否降到
0 當主要判準**——裁切後 500 張影格的中位數 = 0.00000（裁切前 0.57–0.78），
**閘門判定：✅ 通過**。分類器層級的正式驗證留給全部 60 支影片跑完後。

## E0.5：合成指紋校準——✅ 完成，三項判準全數通過

`results/e05_report.md`。用 pilot 的 500 張真實影格當底圖，注入兩種頻域上相反的
人工合成指紋（陽性對照，答案已知），跑 frozen DINOv2（ViT-S/14）+ linear probe：

- **pattern_noise**（低階/高頻）：每個合成類別一張固定、無色（同時加到 R/G/B 三個
  通道）的雜訊紋理，模擬感測器 PRNU。
- **color_shift**（高階/低頻）：每個合成類別一個固定的色相旋轉角度，模擬白平衡/
  色彩處理的系統性色偏，刻意跟 color jitter 的 hue 參數對齊，讓機制比較公平。

**E0.5a（probe 抓得到嗎）**：pattern_noise 45.2%、color_shift 47.8%（chance 25%，
clean 對照組 24.8%，符合預期）——✅ 通過。

**E0.5b（兩種訊號的 probe 靈敏度是否對等）**：差距 0.026（<0.10 門檻）——✅ 通過。
中間過程踩過一個坑：兩者原始設計（彩色雜訊 + 任意方向 RGB 偏移）在同一振幅下，
color_shift 比 pattern_noise 好測超過 0.13，因為 ViT 的 14x14 patch embedding 對
逐像素獨立雜訊有類似平均池化的抑制效果，對整張圖一致的偏移完全不會削弱——這正是
E0.5b 設計要抓的「probe 頻率響應不對等」問題，靠加大雜訊振幅（做成無色）校準回來。

**E0.5c（機制驗證：jitter 該壓的壓下去、不該壓的別亂壓）**：pattern_noise 套用
color jitter 後只掉 0.041（幾乎不受影響）；color_shift 掉 0.208（趨近 chance）——
✅ 通過，『一條降一條平』的機制性預測在合成資料上成立。中間也踩過坑：一開始
color_shift 用任意方向 RGB 向量時，jitter 後只掉 ~3%（因為 jitter 的
brightness/contrast/saturation/hue 不保證能抵銷任意方向的向量），改成色相旋轉、
直接對齊 hue 這個自由度後才有乾淨的因果關係。

**一個更關鍵的 bug**：校準過程中一度發現數字在「沒改任何參數」的情況下重跑就大幅
跳動，追查後發現是 `_class_seed()` 用了 Python 內建 `hash()` 對字串取雜湊，而
字串 hash 預設受 `PYTHONHASHSEED` 隨機化影響，每次重新啟動 Python process
都會變、完全不可重現。改用 `hashlib.md5` 解決，兩次獨立執行已驗證產出的影像位元組
對位元組相同。**這個 bug 如果沒抓到，後面所有校準都是在追一個會自己亂動的目標。**

## 已知限制（誠實記錄）

1. **frame 像素/幾何資料目前只有 5 支影片的 pilot**（500 張），是否擴大到全部
   60 支待決定；病理標籤（annotation）已經是全資料集官方完整資料，不受此限制。
2. **E0e-2/E0e-4 的分類器數字在 pilot 規模下統計效力很有限**（單折、有時整個
   類別在訓練集裡缺席），已在報告裡逐一標註哪些數字是「流程驗證用」、哪些是
   「不受樣本數限制的直接證據」（corner_black_fraction 中位數）。
3. E0c 的 confound 報表目前只做到「呈現分布」，還沒有做真正的 confound 控制或
   matching（計畫裡的 E2c 才會做）。
4. 官方 Fujifilm 影格上仍留有看起來像真實檢查時間的燒錄時間戳，論文聲稱的去識別化
   流程顯然沒處理到這個欄位——這是官方資料集本身的性質，記錄下來供日後注意。

## 下一步

E0（含 E0e、E0.5）在 pilot 規模上已完成、流程已驗證正確，E0.5 三項判準全數通過，
代表 frozen DINOv2 + linear probe 這個量測工具本身是可信的，可以拿去用在真實資料
上。下一步是：(a) 決定是否把 frame 抽樣擴大到全部 60 支影片以取得統計上站得住腳的
E0e/E1 結論，或 (b) 直接進入計畫 §5/§6 的 **E1：L1 指紋可解碼性**（frozen DINOv2 +
linear probe，用真實 cohort/品牌/video ID 當標籤），並拿 pilot 量到的 trivial
baseline（純尺寸/角落殘留）當下限比較基準。
