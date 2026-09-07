# E1a/b/d：指紋可解碼性 probe 結果（full，60 支影片，6000 張影格）

## E1a：Video ID（60-way，held-out frame，dinov2_pretrained）

accuracy = 0.475 ± 0.010（chance = 0.017，majority baseline = 0.017）

**解讀**：60 支影片中隨機抽 20% 的 frame 當測試集，其餘 80%（含全部60 支影片的其他 frame）當訓練集。在全部 60 支影片、6000 張影格的規模下，這是正式結論（非 pilot 流程驗證）。

## E1a：Cohort / Endoscope Brand（held-out 影片，dinov2_pretrained）

Cohort（4-way）：accuracy = 0.628 ± 0.048（chance = 0.250，majority = 0.250，有效折數 = 10/10）

Endoscope brand（2-way）：accuracy = 0.959 ± 0.013（majority = 0.883，有效折數 = 10/10）

**全資料集規模**：60 支影片、每個 cohort/brand 都有多支影片代表，有效折數 10/10 代表每次 held-out 切分訓練集都完整涵蓋所有類別，可當正式結論。

## E1b：Backbone 對照（video ID 任務，統計效力最足夠的任務）

|                     |   mean_accuracy |   std_accuracy |    chance |
|:--------------------|----------------:|---------------:|----------:|
| dinov2_pretrained   |        0.47525  |     0.0102405  | 0.0166667 |
| dinov2_random       |        0.288417 |     0.00984921 | 0.0166667 |
| imagenet_supervised |        0.44425  |     0.0108362  | 0.0166667 |

**解讀**：`dinov2_random`（隨機初始化、完全沒訓練過）如果 accuracy 遠低於 `dinov2_pretrained`，代表 video ID 指紋不是光靠架構的歸納偏見就能讀出來，確實需要某種訓練（不論是 SSL 還是監督式）才學得到。`imagenet_supervised` 是 ResNet-50，跟 DINOv2 的 ViT 架構不同、輸出維度也不同（2048 vs 384），只能看方向、不是嚴格對照。

## E1d：病理（polyp_label）baseline，同一個 embedding

Held-out frame 切法（跟 video ID 用同一種切分，比較基準）：accuracy = 0.890 ± 0.006（majority baseline = 0.858，抽樣集中在影片中間 1/3，polyp frame 偏少，majority baseline 天生就高）。

Held-out 影片切法（跟 cohort/brand 用同一種切分）：accuracy = 0.855 ± 0.032（majority = 0.858，有效折數 = 10/10）。

**這是指紋可分性的對照組**：如果指紋（video ID/cohort）的可分性明顯高於病理可分性，才能說 embedding 把容量更多分配在採集特徵而不是病理特徵上。
