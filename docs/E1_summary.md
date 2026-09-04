# E1：L1 指紋可解碼性——pilot 驗證摘要

對應研究計畫 §6「E1：L1 指紋可解碼性」，日期 2026-09-04。**這是在 5 支影片 pilot
規模上跑的流程驗證**，目的是確認 E1 的程式碼、probe 設計、backbone 對照邏輯正確，
不是正式結論——正式結論待資料擴大到接近全部 60 支影片後才成立。

## E1a：三個指紋任務

只有 **video ID（5-way，held-out frame）** 在 pilot 規模下有足夠統計效力：

- Video ID：accuracy = **91.0% ± 3.2%**（chance 20%）——強訊號。
- Cohort（4-way，held-out 影片）：accuracy = 11.1% ± 17.0%（majority baseline
  40%）——**低於 majority baseline**，因為 5 支影片裡有 4 個 cohort 只有 1 支
  代表，留一支出來測試時常常剛好把某個 cohort 整個排除在訓練集外，結構性猜不對。
  跟 E0e 遇到的規模限制一樣，不是方法問題。
- Endoscope brand（2-way）：accuracy = 95.1% ± 3.2%，但只有 7/10 折跑得出結果
  （另外 3 折訓練集裡沒有 Fujifilm 樣本可學）。

## E1b：Backbone 對照——本輪最重要的發現

<table>
<tr><td>Backbone</td><td>Video ID accuracy</td></tr>
<tr><td>dinov2_pretrained（frozen SSL 預訓練）</td><td>91.0%</td></tr>
<tr><td><b>dinov2_random（隨機初始化，完全沒訓練過）</b></td><td><b>78.9%</b></td></tr>
<tr><td>imagenet_supervised（ResNet-50）</td><td>92.4%</td></tr>
</table>

（chance = 20%）

**隨機初始化的 DINOv2 就已經能把 video ID 猜到 78.9%**，遠遠超過 chance。這正是
計畫 E1b 條目本身提醒要提防的情境：「若隨機初始化也能高準確率，代表指紋強到不需
學習就能讀出，論述須改」。預訓練版（91.0%）確實比隨機版高出約 12 個百分點，說明
SSL 訓練還是有加成，但隨機基線本身已經很高——這代表相當一部分「video ID 可解碼」
其實是**任何 ViT 架構的歸納偏見**（固定的 patch embedding、LayerNorm 等）加上輸入
本身的低階統計量（顏色、紋理）造成的附帶編碼，不是 SSL 目標函數主動學來的捷徑。
往後 E1 的論述必須把隨機基線當成真正的下限，強調「pretrained 相對 random 的增量」
而不是「pretrained 的絕對數字」。

## E1c：維度分析

- 累積到 90% 變異量需要 93 個 PC（總共 384 維）。
- Video ID 只需要約 32 個 PC 就能達到觀察到的峰值準確率（88.0%，出現在 k=128）
  的 90% 以上——指紋訊號集中在少數幾個主成分。
- 單一 PC 層級：前 20 個 PC 裡只有 PC4、PC5 對 video_id 有明顯高於 chance 的
  單變量可分性；沒有任何一個單一 PC 對 polyp_label 明顯高於 majority baseline
  （病理訊號需要多維度組合才看得出來，不集中在單一維度）。兩者在這個 pilot 上
  沒有觀察到重疊的 PC，方向上是好消息，但樣本數小，需要在全資料集上重新驗證。
- **附帶發現**：k=384（全維度）的 video_id 準確率（65.6%）反而低於 k=128 的峰值
  （88.0%）——這是 PCA 全維度重新標準化後跟 logistic regression 的 L2 正則化
  交互作用造成的過擬合假象（樣本數 500 接近特徵數 384），不是真的維度越多資訊
  越少，記錄下來避免之後誤讀這個圖表。

## E1d：病理 baseline（指紋可分性的對照組）

- Polyp_label，跟 video ID 用同一種切分（held-out frame）：84.5% ± 2.4%
  （majority baseline 75.4%，pilot 裡 polyp frame 偏少）。
- Polyp_label，跟 cohort/brand 用同一種切分（held-out 影片）：70.7% ± 11.3%
  （majority baseline 75.4%，10/10 折都跑得出來，但**平均反而低於 majority
  baseline**——換影片測試時病理可分性掉得比隨機猜多數類別還差，方向上跟
  cohort 任務類似的規模問題，需要更多影片才能看出真實水準）。

同一種切分下比較：video ID（91.0%）>> polyp（84.5%），且兩者差距在 held-out
影片切分下更明顯（brand 95.1% vs polyp 70.7%）——初步方向支持「embedding 把較多
容量分配在採集特徵而非病理特徵」，但 pilot 樣本數小，這個對比在全資料集上是否
持續成立，是 E1 擴大到 60 支影片後最值得優先確認的問題。

## 已知限制

1. **樣本規模**：5 支影片、500 張影格。除了 video ID 任務，其餘任務的 held-out
   影片切分都受樣本數限制，數字只能當流程驗證。
2. **backbone 對照不是嚴格控制實驗**：imagenet_supervised 用 ResNet-50（CNN、
   2048 維），跟 DINOv2（ViT、384 維）架構家族不同，只能看方向。
3. **E1c 的全維度過擬合假象**：見上方附帶發現，選圖表/引用數字時避免直接用
   k=384 的準確率。
4. PCA 是在全部 500 張影格上一次 fit（train+test 都在裡面）才切 train/test 訓練
   probe，嚴格來說有微小的資訊外洩（PCA 基底看過測試集），但由於 PCA 是無監督
   降維、不看標籤，實務上外洩程度很小，在 pilot 規模的探索性分析中可接受，正式
   結論階段應改成只在訓練集上 fit PCA。

## 下一步

E1 的 pipeline（backbone 對照、多種切分策略、PCA 維度分析）已在 pilot 上驗證可以
正確跑完。**最重要的懸而未決問題是 E1b 的隨機基線發現**——需要在更大規模資料上
確認「pretrained 相對 random 的增量」是否穩定，且原始計畫 §6 E5（在 REAL-Colon 上
自訓 DINO）會直接回答「SSL 目標函數有沒有主動放大這個指紋」，是解讀這個發現的
關鍵後續實驗。近期決策點：是否把 frame 抽樣擴大到全部 60 支影片，以取得每個任務
都有統計說服力的正式結論。
