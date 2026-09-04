# E1a/b/d：指紋可解碼性 probe 結果（pilot，5 支影片，500 張影格）

## E1a：Video ID（5-way，held-out frame，dinov2_pretrained）

accuracy = 0.910 ± 0.032（chance = 0.200，majority baseline = 0.200）

**解讀**：5 支影片中隨機抽 20% 的 frame 當測試集，其餘 80%（含全部 5 支影片的其他 frame）當訓練集。這是唯一在 pilot 規模下有足夠統計效力的任務。

## E1a：Cohort / Endoscope Brand（held-out 影片，dinov2_pretrained）

Cohort（4-way）：accuracy = 0.111 ± 0.170（chance = 0.250，majority = 0.400，有效折數 = 10/10）

Endoscope brand（2-way）：accuracy = 0.951 ± 0.032（majority = 0.800，有效折數 = 7/10）

**跟 E0e 一樣的規模限制**：5 支影片切 train/test 時容易讓某個類別整個缺席（有效折數若明顯 <10，代表很多次重複都被跳過），這兩個數字暫時只做流程驗證，不是正式結論。

## E1b：Backbone 對照（video ID 任務，統計效力最足夠的任務）

|                     |   mean_accuracy |   std_accuracy |   chance |
|:--------------------|----------------:|---------------:|---------:|
| dinov2_pretrained   |           0.91  |      0.0319374 |      0.2 |
| dinov2_random       |           0.789 |      0.0326956 |      0.2 |
| imagenet_supervised |           0.924 |      0.0253772 |      0.2 |

**解讀**：`dinov2_random`（隨機初始化、完全沒訓練過）如果 accuracy 遠低於 `dinov2_pretrained`，代表 video ID 指紋不是光靠架構的歸納偏見就能讀出來，確實需要某種訓練（不論是 SSL 還是監督式）才學得到。`imagenet_supervised` 是 ResNet-50，跟 DINOv2 的 ViT 架構不同、輸出維度也不同（2048 vs 384），只能看方向、不是嚴格對照。

## E1d：病理（polyp_label）baseline，同一個 embedding

Held-out frame 切法（跟 video ID 用同一種切分，比較基準）：accuracy = 0.845 ± 0.024（majority baseline = 0.754，pilot 裡 polyp frame 偏少，majority baseline 天生就高）。

Held-out 影片切法（跟 cohort/brand 用同一種切分）：accuracy = 0.707 ± 0.113（majority = 0.754，有效折數 = 10/10）。

**這是指紋可分性的對照組**：如果指紋（video ID/cohort）的可分性明顯高於病理可分性，才能說 embedding 把容量更多分配在採集特徵而不是病理特徵上。
