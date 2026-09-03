# E0e-1/E0e-2：FOV 幾何洩漏檢查 + trivial baseline（pilot，5 支影片）

## FOV 遮罩角落殘留（corner_black_fraction，20% 角落框內黑色像素比例）

|                     |   count |   median |     mean |        std |
|:--------------------|--------:|---------:|---------:|-----------:|
| ('001', 'Olympus')  |     100 | 0.571591 | 0.571619 | 0.00116578 |
| ('002', 'Fujifilm') |     100 | 0.775364 | 0.775177 | 0.00148083 |
| ('002', 'Olympus')  |     100 | 0.681908 | 0.688318 | 0.0195368  |
| ('003', 'Olympus')  |     100 | 0.684773 | 0.692556 | 0.0242315  |
| ('004', 'Olympus')  |     100 | 0.588818 | 0.595617 | 0.0153704  |

每支影片的幾何摘要已存到 `fov_geometry.csv`（5 支影片）。

| video_id   |   cohort | endoscope_brand   |   n_frames |   width |   height |   aspect_ratio |   log_area |   corner_black_fraction |
|:-----------|---------:|:------------------|-----------:|--------:|---------:|---------------:|-----------:|------------------------:|
| 001-001    |      001 | Olympus           |        100 |    1352 |     1080 |        1.25185 |    14.1941 |                0.571591 |
| 002-004    |      002 | Olympus           |        100 |    1350 |     1080 |        1.25    |    14.1926 |                0.681908 |
| 002-006    |      002 | Fujifilm          |        100 |    1248 |      959 |        1.30136 |    13.9952 |                0.775364 |
| 003-001    |      003 | Olympus           |        100 |    1352 |     1080 |        1.25185 |    14.1941 |                0.684773 |
| 004-003    |      004 | Olympus           |        100 |    1244 |     1080 |        1.15185 |    14.1108 |                0.588818 |

## 質性觀察：品牌專屬的螢幕燒錄 UI 疊字（肉眼複查 `qc_unify_crop_samples/` 時發現）

官方原始影格上的裝置 UI 疊字不是像素統計，而是**可直接讀出品牌是誰的文字**，比任何幾何統計量都更直接的採集指紋：
- **Olympus**（001-001、002-004、003-001、004-003）：右上角「Near Focus」字樣徽章，偶爾右下角有小圖示，沒有時間戳記。
- **Fujifilm**（002-006）：左上角完整的日期/時間戳記（例如 `9/05/2021 0:48:34`，隨檢查進行持續跳動，格式明顯是原始錄影裝置燒錄的時間，不是後製加上去識別化用的假時間戳）+ 右上角拍攝設定資訊（例如 `*1/100`、`Lv+5`、`AUTO`）。

兩種疊字位置、內容都不同，但都落在角落區域，統一裁切協定（見 E0e-3/4）實測都能一併裁掉。這裡如實記錄一個資料治理觀察：Fujifilm 影格上還留著看起來像真實檢查時間的燒錄時間戳，論文聲稱的去識別化流程顯然沒有處理到這個欄位——這是官方公開釋出的資料集本身的性質，不是這個 repo 造成的，但值得記錄，日後若要引用/展示這些影格範例時要注意這一點。

## E0e-2a：cohort（多類別）trivial baseline，held-out 影片

### 只用影格尺寸（width/height/aspect_ratio/log_area）

|                   |   value |
|:------------------|--------:|
| n_train_frames    |     400 |
| n_test_frames     |     100 |
| n_train_videos    |       4 |
| n_test_videos     |       1 |
| accuracy          |       0 |
| majority_baseline |       0 |
| uniform_chance    |       1 |

### 加上 corner_black_fraction

|                   |   value |
|:------------------|--------:|
| n_train_frames    |     400 |
| n_test_frames     |     100 |
| n_train_videos    |       4 |
| n_test_videos     |       1 |
| accuracy          |       1 |
| majority_baseline |       0 |
| uniform_chance    |       1 |

**只用尺寸時 accuracy=0、加上 corner_black_fraction 後變成 1**，不是隨機雜訊：這一折被留出來測試的剛好是 002-006（Fujifilm），它的 (width,height)=(1248,959) 跟訓練集裡任何一支影片都不完全相同，純尺寸模型猜錯；但它的 corner_black_fraction（≈0.775）明顯偏高，跟同為 cohort 002/003 的影片（≈0.68-0.69，也偏高）同一側，跟 cohort 001/004（≈0.57-0.59，偏低）不同側，這個額外的軸剛好幫分類器猜對。**5 支影片只留 1 支測試，單一折的結果本來就不穩定**，這裡只是把這次具體發生的原因講清楚，不是說 corner_black_fraction 已被證實比尺寸更有鑑別力——正式結論要等擴大到全部 60 支、有多折平均之後才算數。

## E0e-2b：endoscope_brand trivial baseline，held-out 影片

跳過：This solver needs samples of at least 2 classes in the data, but the data contains only one class: 'Olympus'

只有 5 支 pilot 影片、其中只有 1 支是 Fujifilm（002-006），GroupShuffleSplit 隨機切分時很容易讓 train 或 test 其中一邊完全沒有Fujifilm 影片，分類器訓練不起來。

**注意**：pilot 裡只有 cohort 002 同時涵蓋兩種品牌（002-004 Olympus、002-006 Fujifilm），其餘 3 支影片（001-001、003-001、004-003）全是 Olympus，brand 和 cohort 高度共線，這個數字暫時不能拿來下結論，等擴大到全部60 支、cohort 002 內有足夠 Olympus+Fujifilm 影片時才有意義。
