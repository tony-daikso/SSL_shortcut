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

## E0e：FOV 幾何洩漏檢查——✅ 完成

`results/fov_e0e_baseline_report.md`（裁切前）+ `results/fov_crop_margin_calibration.md`
（校準過程）+ `results/fov_e0e4_verification.md`（裁切後驗證）。

**E0e-1（幾何特徵）**：除了 E0d 已知的影格尺寸差異，肉眼複查裁切前後對照圖時發現一個
E0d 沒抓到的更嚴重問題——**幾乎每張影格角落都有明顯的黑色 FOV 遮罩殘留**（圓形/八邊形
遮罩，E0d 原本用「整行/整列是否全黑」檢查，方法上完全無法偵測角落殘留）。200 張隨機
抽樣影格 100% 都有這個現象，20% 大小角落框內平均 ~11-12% 像素是黑的，且依 cohort/
品牌分布不均。`dataset_self_check.md` 已加上勘誤。

**E0e-2（trivial baseline，裁切前，train→held-out test 影片）**：
- Cohort（4-way）：**77.3%**（size only）／**78.3%**（size + corner_black_fraction）
  vs. majority baseline 42.6%、chance 25%——遠高於兩者。
- Endoscope brand（全域）：**100%** vs. majority baseline 93.3%。
- Endoscope brand（cohort 002 內部乾淨對照，leave-one-video-out）：**100%**
  vs. majority baseline 57.7%——排除 cohort 共線因素後，光憑幾何仍能完美分辨品牌。

**這是 E1（DINOv2 embedding probe）之後必須超越的下限**：若 E1 的 probe 準確率跟這裡
的 trivial baseline 差不多，代表 probe 學到的可能只是幾何，不是真正的內容/紋理指紋。

**E0e-3（統一裁切協定）**：中央方形裁切 → 再裁掉校準得出的 15% 邊距
（`fov_protocol.INSET_FRACTION = 0.15`）→ resize 到 224x224。邊距參數的校準過程見
`fov_crop_margin_calibration.md`：inset 從 0 加到 0.05 就讓平均角落殘留從 ~11.4% 驟降
到 ~0.32%，之後緩慢下降，選比最小可行值稍保守的 0.15。

**E0e-4（驗證，裁切後）**：
- Cohort baseline：42.4%（majority baseline 42.6%，幾乎完全等於猜多數類別）。
- Brand within-cohort-002 baseline：57.7%（跟 majority baseline 完全相等）。
- **閘門判定：✅ 通過**——裁切後兩個 baseline 都掉到 majority baseline 附近，統一裁切
  協定確實把幾何線索壓到接近無法利用的程度。肉眼複查裁切前後對照圖
  （`qc_unify_crop_samples/`）確認裁切後看不到黑色遮罩，且黏膜主體內容仍清楚可辨。

## 下一步

E0（含 E0e）已完成。依計畫 §5/§6，下一步是 **E1：L1 指紋可解碼性**（frozen DINOv2 +
linear probe），且必須拿這裡量到的 trivial baseline（cohort 78.3%、brand 100%）當
下限比較基準——probe 的結論只有在明顯超過這個下限、或至少能證明訊號來源不是幾何時
才站得住腳。
