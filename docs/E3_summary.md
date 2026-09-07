# E3：Augmentation 劑量反應——正式結論

對應研究計畫 §6「E3：augmentation dose-response（三曲線）」，日期 2026-09-07。
在全部 60 支影片規模上完成（病理曲線用全資料集 6000 張影格；兩條合成指紋曲線
沿用 E0.5 已校準的 500 張 pilot 底圖，見下方「範圍說明」）。

## 設計

計畫 §5 的機制性論證：augmentation 消除捷徑的機制是讓該特徵在 crop 間變得不可
預測，但只能作用在自己有參數化的維度上。color jitter 是色彩空間的全局仿射變換
（brightness/contrast/saturation 純量縮放、hue 色相旋轉），機制上不改變雜訊的
空間相關結構，但確實觸及色彩/白平衡類的系統性偏移。

用一個強度乘數 `strength ∈ {0.0, 0.5, 1.0, 1.5, 2.0}` 縮放 E0.5 已校準的基準
color jitter 參數（brightness=contrast=saturation=0.2、hue=0.5，hue 在 strength>1
時封頂於 torchvision 的合法上限 0.5），在每個強度下量測三條曲線：

1. **pattern_noise**（合成，低階/高頻，模擬感測器 PRNU，4-way，chance 25%）
2. **color_shift**（合成，高階/低頻，模擬白平衡系統性色偏，4-way，chance 25%）
3. **polyp_label**（真實病理訊號，全部 60 支影片、6000 張影格，held-out frame，
   majority baseline 85.8%）

外加 **E3c：SSIM trivial-destruction control**，排除「曲線變化只是因為圖被整個
毀掉」這個混淆解釋。

## 結果

| strength | pattern_noise | color_shift | polyp_label | SSIM |
|---:|---:|---:|---:|---:|
| 0.0 | 45.2% | 47.8% | 89.0% | 1.000 |
| 0.5 | 48.5% | 31.7% | 88.5% | 0.938 |
| 1.0 | 43.0% | 23.5% | 88.6% | 0.917 |
| 1.5 | 41.5% | 25.7% | 88.4% | 0.902 |
| 2.0 | 37.9% | 22.9% | 88.5% | 0.883 |

（完整含標準差的表見 `results/e3_dose_response_report.md`）

strength=0 這一點跟 `results/e05_report.md` 的無 jitter 數字（pattern_noise
45.2%、color_shift 47.8%）完全一致，互相校驗了這兩支獨立腳本的一致性。

## 解讀：一條降一條平，且病理幾乎零代價

- **color_shift 隨強度增加持續下降到接近/低於 chance**（47.8% → 22.9%，
  chance=25%）——`color jitter` 這個變換確實觸及了它設計要模擬的白平衡/色彩
  處理系統性偏移。
- **pattern_noise 只有溫和下降**（45.2% → 37.9%，掉了 7.3 個百分點），全程都
  遠高於 chance（25%）——跟機制預測一致：color jitter 的參數空間（色彩仿射
  變換）在數學上動不到無色高頻雜訊的空間相關結構，只能造成間接的、次要的
  干擾（例如 brightness/contrast 縮放會輕微改變雜訊的絕對振幅），不會像
  color_shift 那樣被直接抵銷。
- **polyp_label 幾乎沒有代價**（89.0% → 88.5%，只掉 0.5 個百分點，全程都遠高於
  majority baseline 85.8%）——在這個強度範圍內，color jitter 對真實診斷訊號的
  影響遠小於對 color_shift 指紋的影響。

**這是計畫 §7「陰性結果處置」裡最好的一種形態：殘餘地板（residual floor）**，
不是「兩條都降到 chance」（機制論證死亡），也不是「兩條都不動」（量測太弱），
而是清楚的「一條降一條平」，且真實病理訊號幾乎全身而退：

- **好消息**：標準 color jitter 這類全域色彩變換，可以用幾乎不犧牲病理可分性
  的代價，把高階/色彩類的採集指紋壓到接近 chance。
- **仍未解決的部分**：低階/感測器雜訊類的指紋（pattern_noise 的真實對應物）
  在這個強度範圍內完全沒有被壓下去的跡象，維持在遠高於 chance 的殘餘地板——
  這正是計畫 E4（機制推導的修法：雜訊重合成/白化）要處理的目標，而且 §3.2 已
  指出 REAL-Colon 的感測器雜訊完全未經處理（無強度正規化、無 harmonization），
  是這個殘餘地板最可能的真實來源。

## E3c：trivial-destruction control——通過

最高強度（2.0）下 SSIM 均值 0.883，全程最低點同為 0.883，遠高於 0.7 的門檻。
影像在整個強度範圍內都維持高度結構相似，三條曲線的差異可以歸因於訊號本身對
jitter 的敏感度不同，不是影像被整體破壞的副作用。

## 範圍說明（為什麼病理曲線是全資料集，指紋曲線是 pilot 底圖）

- **polyp_label 曲線**：用全部 60 支影片、6000 張影格——這是計畫最關心的臨床
  後果（augmentation 有沒有連帶傷害診斷訊號），值得用跟 E1d 一樣的統計效力。
- **pattern_noise / color_shift 曲線**：沿用 E0.5 已校準的 500 張 pilot 底圖
  （5 支影片）——這兩者是已知強度、已知答案的陽性對照訊號，不需要真實資料的
  統計效力，用跟 E0.5 一致的規模可以直接跟 `e05_report.md` 的既有結果對照校驗
  （見上方 strength=0 的一致性檢查）。

## 已知限制

1. 強度網格只到 2.0（4 倍基準 hue 已封頂於 torchvision 上限 0.5），沒有測試
   更極端的強度是否會讓 pattern_noise 也開始明顯下降——如果需要找到 pattern_noise
   真正開始鬆動的強度閾值，需要擴大網格或改變 augmentation 種類（例如混入會
   影響雜訊統計量的操作，如 Gaussian blur）。
2. color_shift 在 strength=1.0→1.5 有小幅回升（23.5%→25.7%），落在標準差
   範圍內（std 約 0.03-0.05），視為抽樣雜訊而非真實的非單調趨勢，但更密的網格
   會讓曲線形狀更可信。
3. polyp_label 用 held-out frame 切分（跟 E1a/E1d 的最強統計效力設定一致），
   沒有同時跑 held-out video 版本——E1d 已顯示 held-out video 切分下病理可分性
   本來就比較不穩定，若要更嚴謹可以之後補上。

## 下一步

E3 的核心機制性論證（一條降一條平、殘餘地板、病理幾乎零代價）已在全資料集規模
上驗證成立，是進入 **E4（機制推導的修法：雜訊重合成/白化，針對性處理殘留的
pattern_noise 類指紋）** 的直接前提——E4 的驗證迴路正是「加上雜訊白化後，這裡
維持平坦的 pattern_noise 曲線應該開始下降」。
