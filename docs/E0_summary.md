# E0：資料準備——完成摘要

對應研究計畫（Notion）§6「E0：資料準備」，日期 2026-09-03。

## E0a：影格抽樣與切分——✅ 完成

`results/video_manifest.csv` 依照計畫 §3.1 記載的官方規則（每個 cohort 內 `VVV` 編號
001–010 train / 011–012 val / 013–015 test）指派 60 支影片的 split，切分單位是整支
影片，不會有同一支影片的 frame 同時出現在兩個 split。已驗證每個 cohort 都是
10/2/3 支，跟官方切分完全吻合。

## E0b：三組標籤——✅ 完成（video 層級權威，frame 層級為子集）

- **來源標籤**：cohort（001–004）、endoscope_brand、video_id ——`video_manifest.csv`
  （來自官方 `video_info.csv`，完整覆蓋 60 支影片）。
- **metadata 標籤**：fps、num_frames、num_lesions、bbps、age、sex（video 層級，官方
  完整）；實際影格寬高（frame 層級，來自子集裡 3017 張影格的 annotation `<size>` 欄位，
  非假設值）。
- **病理標籤**：polyp/no-polyp，由 bbox 是否存在導出——`frame_labels.csv`，3017 張
  抽樣影格（60 支影片全部涵蓋），632 張 polyp、2385 張 no-polyp。
  **這是目前唯一沒有做到「官方完整逐格」的部分**：可用的 annotation 只有「polyp」
  專案先前萃取的一個子集，不是全部 2,757,723 張影格。詳見 README 與
  `scripts/config.py` 的說明。

## E0c：Confound 報表——✅ 完成

`results/confound_report.md`。Part A（官方完整資料，可信）：
- 影片層級盛行率：cohort 001 60%、002 73.3%、003 93.3%、004 80%——**跟計畫 §3.4
  提到的「刻意讓平均病灶數接近整體平均」不完全一致**，cohort 001 明顯偏低，值得在
  E2c 做 confound 對照時留意。
- 病灶大小：cohort 002 明顯偏小（mean 1.71mm vs. 其他 cohort 3.8–4.6mm）。
- Histology：cohort 002 以 HP（增生性瘉肉）為主（31/38），其他 cohort 以 AD
  （腺瘤）為主——**cohort 與病理型態高度相關，是後續 leave-one-cohort-out（E2）
  最大的 confound 風險來源之一**。
- 意外發現：132 顆病灶裡有 19 顆的 `histology_class` 是 `NO POLYP`（切除後判讀不是
  真正息肉），且分布不均（002、003 較多）——分析大小/型態時已排除，但列成獨立表格。
- 資料品質瑕疵：官方 `site` 欄位有拼字不一致（`caecum`/`cecum`、
  `transverse`/`trasnverse`），已如實記錄，未擅自合併。

Part B（子集，僅供方向參考）：各 cohort 在子集裡的 frame 級 polyp 比例，明確標註
不能代表官方完整資料集的 87.6% 無標註比例。

## E0d：規格自查——✅ 完成（含一項新發現）

`results/dataset_self_check.md`：

1. **影格尺寸並非單一固定值**：觀察到至少 11 種不同的 (width, height) 組合
   （例如 (1352,1080) 主要在 cohort 001/003、(1248,959) 只出現在 cohort 002 的
   Fujifilm、(1162,1007) 只出現在 cohort 004），且尺寸與 cohort/品牌強烈對應。
   **這直接證實了計畫 §3.3 擔心的 FOV 幾何洩漏**——光是影格寬高就可能猜出
   cohort/品牌，不需要看內容。這是 E0e（下一步，FOV 幾何洩漏檢查）最直接的立論基礎。
2. **去交錯**：用「相鄰行差異 / 隔行差異」比值當篩檢分數，59 支影片的分數全部落在
   0.53–0.59 之間、都小於 1（沒有典型交錯殘留會有的比值 >1 尖峰）。人工複查 2 張
   分數最高的樣本圖（見 `results/qc_deinterlace_samples/`）也沒看到梳齒狀邊緣，方向上
   支持「這批影格沒有明顯殘留交錯」，但样本數小，且這批圖本身已經過 JPEG 重新編碼，
   跟官方原始流程是否去交錯是兩件事，不能直接劃等號——完整結論待官方原始資料可取得
   後用更大樣本重驗。

## 已知限制（誠實記錄，不要在後續步驟忽略）

1. **frame 層級標籤只涵蓋 3017 張影格的子集**，不是官方完整逐格標註。E1 起若要用
   frame-level 的 polyp 標籤訓練或評估任何東西，需要先解決官方完整 annotation 的
   下載/儲存問題（60 個 `_annotation` 壓縮包，官方論文未給出總大小，需另外確認）。
2. **去交錯結論的樣本數很小**（僅肉眼複查 2 張），且建立在已被重新編碼過的影像上。
3. E0c 的 confound 報表目前只做到「呈現分布」，還沒有做真正的 confound 控制或
   matching（計畫裡的 E2c 才會做）。

## 下一步

依計畫 §5/§6，緊接著的是 **E0e：FOV 幾何洩漏檢查**（trivial baseline：只用影格尺寸
猜 cohort/品牌）——E0d 已經觀察到的尺寸差異，正好是 E0e 要正式量化的起點。
