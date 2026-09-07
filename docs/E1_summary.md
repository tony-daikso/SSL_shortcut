# E1：L1 指紋可解碼性——全資料集（60 支影片）正式結論

對應研究計畫 §6「E1：L1 指紋可解碼性」，日期 2026-09-07。**這是在全部 60 支
影片、6000 張影格（每支影片中間 1/3 時間軸隨機抽 100 張）上跑的正式結果**，
取代 2026-09-04 的 5 支影片 pilot 流程驗證（該版本見 git 歷史，數字僅供流程
比對，不再是有效結論）。

## E1a：三個指紋任務

- Video ID（60-way，held-out frame）：accuracy = **47.5% ± 1.0%**
  （chance = 1.7%）——比 pilot 更嚴格的任務（60-way vs 5-way），依然遠超
  chance 近 28 倍。
- Cohort（4-way，held-out 影片）：accuracy = **62.8% ± 4.8%**（chance =
  majority = 25%，有效折數 10/10）——60 支影片下每個 cohort 都有 15
  支左右代表，不再受 pilot 的「留一支出來剛好把整個 cohort 排除」問題影響，
  是穩固結論。
- Endoscope brand（2-way，held-out 影片）：accuracy = **95.9% ± 1.3%**
  （majority = 88.3%，有效折數 10/10）。

## E1b：Backbone 對照——本輪最重要的發現（在全資料集下更明顯）

<table>
<tr><td>Backbone</td><td>Video ID accuracy（60-way）</td></tr>
<tr><td>dinov2_pretrained（frozen SSL 預訓練）</td><td>47.5%</td></tr>
<tr><td><b>dinov2_random（隨機初始化，完全沒訓練過）</b></td><td><b>28.8%</b></td></tr>
<tr><td>imagenet_supervised（ResNet-50）</td><td>44.4%</td></tr>
</table>

（chance = 1.7%）

**隨機初始化的 DINOv2 在 60-way 任務下依然把 video ID 猜到 28.8%**——是
chance 的近 17 倍。這比 pilot（5-way，隨機基線 78.9% vs chance 20%，約 4
倍）在**相對倍數上更極端**：分類任務變難（60-way）之後，如果指紋只是巧合的
低階統計量，隨機基線應該會被稀釋到接近 chance；但實際上隨機基線相對 chance
的優勢反而擴大，說明「未訓練 ViT 的架構歸納偏見能讀出的指紋」不是 pilot
規模下的偶然結果，是穩固存在、且在更難的任務下依然有效的訊號。

同時，pretrained 相對 random 的**絕對**增量在全資料集下也擴大了（18.7 個
百分點，pilot 是 12.1 個百分點），代表 SSL 預訓練確實有放大指紋可解碼性
的效果——但隨機基線本身的量級（17 倍 chance）證明相當一部分「video ID
可解碼」的來源仍然是架構本身，不是 SSL 目標函數從頭學來的。E1 的論述維持
pilot 階段的結論：務必把隨機基線當真正的下限，不能只看 pretrained 的絕對
數字。

## E1c：維度分析

- 累積到 90% 變異量需要 126 個 PC（總共 384 維）。
- Video ID 需要約 **128 個 PC** 才能達到觀察到的峰值準確率（42.9%，出現在
  k=384，全維度）的 90% 以上——跟 pilot（5-way 任務只需 32 個 PC）相比，
  60-way 任務需要動用多出約 4 倍的主成分才追上峰值，這是分類數變多、需要
  編碼更多資訊的預期規模效應，不是訊號變弱。
- **k=384 全維度不再出現 pilot 觀察到的過擬合假象**：pilot（樣本數 500 接近
  特徵數 384）時全維度 accuracy 反而低於 k=128 峰值；全資料集（樣本數 6000
  遠大於特徵數 384）下，全維度準確率（42.9%）本身就是峰值，印證 pilot 那個
  現象確實是小樣本下的 PCA+正則化交互作用假象，不是真的維度詛咒。
- 單一 PC 層級：前 20 個 PC 裡沒有任何一個對 video_id 的單變量可分性明顯
  高於 chance 的 3 倍（60-way chance 只有 1.7%，門檻 5.0%，全部落在
  2.1%-3.9% 之間）——這跟 pilot（PC4、PC5 明顯突出）不同，但不代表訊號消失：
  對照累積曲線，video_id 訊號在 60-way 下需要合併約 128 個 PC 才追上峰值，
  代表訊號**分散**在遠多於前 20 個主成分裡，用「前 20 個 PC 逐一單看」這個
  尺度已經看不出來，是分類數增加造成的規模效應，需要用累積曲線而非單 PC
  可分性判讀。
- Polyp 訊號同樣沒有任何前 20 個 PC 的單變量準確率明顯高於 majority
  baseline（跟 pilot 一致）——病理訊號本來就需要多維度組合才看得出來。
- 兩者（指紋、病理）在前 20 個 PC 裡都沒有找到「明顯突出」的維度，所以無法
  用這個尺度判斷兩者是否糾纏；這個問法本身在全資料集規模下不適用，需要用
  累積曲線的整體形狀比較（見上）。

## E1d：病理 baseline（指紋可分性的對照組）——核心對比，全資料集下更清楚

- Polyp_label，跟 video ID 用同一種切分（held-out frame）：**89.0% ± 0.6%**
  （majority baseline 85.8%，只贏過 majority 3.2 個百分點）。
- Polyp_label，跟 cohort/brand 用同一種切分（held-out 影片）：**85.5% ± 3.2%**
  （majority baseline 85.8%，**平均低於 majority baseline**，10/10 折都跑
  得出來——換影片測試時病理可分性沒有超過「一律猜多數類別」的笨方法）。

**這是全資料計畫最重要的一組對比**：同一個 embedding 空間裡，
video ID（47.5%，chance 1.7%）、cohort（62.8%，chance 25%）、brand
（95.9%，majority 88.3%）全部遠遠甩開各自的 chance/majority baseline；
polyp（89.0%／85.5%）幾乎貼著甚至低於 majority baseline。同一個切分方式
下直接比：video ID 遠超 chance ≫ polyp 幾乎持平 majority；held-out 影片
切分下：brand 大勝 majority（+7.6pp）≫ polyp 反而輸 majority（-0.3pp）。
**在全部 60 支影片的規模下，這個對比穩固成立，是本研究最核心的證據**：
frozen DINOv2 embedding 把容量分配在採集特徵（video ID/brand/cohort）
上遠勝於病理特徵（polyp），支持「acquisition shortcut」確實存在的假設。

## 已知限制

1. **backbone 對照不是嚴格控制實驗**：imagenet_supervised 用 ResNet-50
   （CNN、2048 維），跟 DINOv2（ViT、384 維）架構家族不同，只能看方向。
2. PCA 是在全部 6000 張影格上一次 fit（train+test 都在裡面）才切 train/test
   訓練 probe，嚴格來說有微小的資訊外洩（PCA 基底看過測試集），但由於 PCA
   是無監督降維、不看標籤，實務上外洩程度很小；在樣本數 6000 遠大於特徵數
   384 的規模下，這個外洩對數字的影響應該比 pilot 階段更小，但正式結論若要
   進一步收斂應改成只在訓練集上 fit PCA。
3. Polyp_label 的抽樣範圍限定在每支影片「中間 1/3 時間軸」，跟官方逐格
   annotation（100% 覆蓋率）的整體盛行率（12.41%）不完全一致，polyp
   majority baseline（85.8% 對應 polyp 盛行率約 14.2%）是這個抽樣窗口下
   的局部盛行率，不是全影片的代表值——但這不影響 E1d 的核心對比（同一批
   frame 上比較 video_id/cohort/brand 與 polyp 的可分性差距），因為兩者用
   的是同一個抽樣。

## 下一步

E1 在全部 60 支影片規模下的結論已經穩固：**指紋可分性（尤其是 video
ID/brand）遠超病理可分性，且相當一部分指紋來自 ViT 架構本身的歸納偏見而非
SSL 訓練**。原始計畫 §6 E5（在 REAL-Colon 上自訓 DINO，而非只用官方
ImageNet-pretrained 權重）是回答「SSL 目標函數本身有沒有主動放大這個指紋」
的關鍵後續實驗——現在 E1 的統計基礎已經足夠扎實，可以作為 E5 訓練後比較的
正式基準線（不再是 pilot 數字）。
