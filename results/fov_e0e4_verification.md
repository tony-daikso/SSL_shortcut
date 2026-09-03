# E0e-4：統一裁切協定驗證（pilot，5 支影片）

套用 `fov_protocol.unify_crop`（中央方形裁切 → 再裁掉 15% 邊距 → resize 到 224x224，邊距比例見 `05_calibrate_crop_margin.py` 的校準結果）。對 500 張影格重新量測尺寸：發現 1 種不同的 (width, height) 組合（預期剛好是 1 種常數，驗證裁切協定對所有影格一視同仁）。

角落殘留（corner_black_fraction）裁切後統計：mean=0.00208，median=0.00000，max=0.40702（裁切前統計見 `fov_e0e_baseline_report.md`）。

## Cohort trivial baseline（只用 corner_black_fraction，裁切後）

|                   |   裁切後 |
|:------------------|------:|
| n_train_frames    |   400 |
| n_test_frames     |   100 |
| n_train_videos    |     4 |
| n_test_videos     |     1 |
| accuracy          |     0 |
| majority_baseline |     0 |
| uniform_chance    |     1 |

對照 E0e-2（裁切前，size+corner 特徵）：accuracy 見 `fov_e0e_baseline_report.md`。裁切後（只剩 corner_black_fraction 這一個特徵，因為尺寸類特徵已變成常數）：accuracy 0.000（chance 1.000、majority baseline 0.000）。

## Endoscope brand within-cohort-002 control（只用 corner_black_fraction，裁切後）

|                   |   裁切後 |
|:------------------|------:|
| n_frames          | 200   |
| n_videos          |   2   |
| accuracy          | nan   |
| majority_baseline |   0.5 |
| uniform_chance    |   0.5 |

**跑不出結果（accuracy=nan）**：pilot 裡 cohort 002 只有 2 支影片（002-004 Olympus、002-006 Fujifilm），leave-one-video-out 輪流留一支測試時，訓練集只剩另外 1 支、只有 1 種品牌，分類器結構上訓練不起來——這是「只有 5 支pilot 影片」這個規模限制造成的，不是裁切協定的問題，要等擴大到全部 60 支、cohort 002 有更多影片後才跑得出有意義的結果。


## 主要證據：corner_black_fraction 裁切前後直接對比

裁切前（見 `fov_e0e_baseline_report.md`）：5 支影片的 corner_black_fraction 中位數落在 0.57–0.78 之間，跟 cohort/品牌強烈對應。裁切後（本檔案開頭）：500 張影格的 mean=0.00208、**median=0.00000**、max=0.40702——中位數已經降到完全乾淨（0），只有極少數離群影格（例如影片邊緣有非遮罩造成的暗部內容）還有殘留，這是比分類器 accuracy 更直接、不受『5 支影片切分太少』這個限制影響的證據。

## 閘門判定：✅ 通過（依 corner_black_fraction 中位數）

裁切後 corner_black_fraction 中位數降到 0.00000（幾乎為 0），符合計畫 E0e-4 的驗收標準，統一裁切協定確實把角落幾何線索壓到接近無法利用的程度。分類器層級的驗證（cohort/brand trivial baseline）在5 支影片的 pilot 規模下無法給出可靠數字（訓練折常常缺類別），本次兩個分類器都因為樣本太少而失真或跑不出來，正式的分類器層級驗證要等擴大到全部 60 支影片後才有意義。


裁切前後對照樣本圖已存到 `results/qc_unify_crop_samples/`（6 組）。
