# E0e-4：統一裁切協定驗證

套用 `fov_protocol.unify_crop`（中央方形裁切 → 再裁掉 15% 邊距 → resize 到 224x224，邊距比例見 `05_calibrate_crop_margin.py` 的校準結果）。對 2975 張影格重新量測尺寸：發現 1 種不同的 (width, height) 組合（預期剛好是 1 種常數，驗證裁切協定對所有影格一視同仁）。

角落殘留（corner_black_fraction）裁切後統計：mean=0.00116，median=0.00000，max=0.51791（裁切前 mean 約 0.12，見 `fov_e0e_baseline_report.md`）。

## Cohort trivial baseline（只用 corner_black_fraction，裁切前 vs 裁切後）

|                   |         裁切後 |
|:------------------|------------:|
| n_train_frames    | 1975        |
| n_test_frames     |  653        |
| n_train_videos    |   40        |
| n_test_videos     |   12        |
| accuracy          |    0.424196 |
| majority_baseline |    0.425727 |
| uniform_chance    |    0.25     |

對照 E0e-2（裁切前，size+corner 特徵）：accuracy 見 `fov_e0e_baseline_report.md`。裁切後（只剩 corner_black_fraction 這一個特徵，因為尺寸類特徵已變成常數）：accuracy 0.424（chance 0.250、majority baseline 0.426）。

## Endoscope brand within-cohort-002 control（只用 corner_black_fraction，裁切前 vs 裁切後）

|                   |        裁切後 |
|:------------------|-----------:|
| n_frames          | 610        |
| n_videos          |  15        |
| accuracy          |   0.577049 |
| majority_baseline |   0.577049 |
| uniform_chance    |   0.5      |

對照 E0e-2c（裁切前）：accuracy 1.000（majority baseline 0.577）。裁切後：accuracy 0.577（majority baseline 0.577）。

## 閘門判定：✅ 通過

裁切後兩個 trivial baseline 的準確率都落在 majority baseline 附近（誤差 <1%），符合計畫 E0e-4 的驗收標準，統一裁切協定確實把角落幾何線索壓到接近無法利用的程度。


裁切前後對照樣本圖已存到 `results/qc_unify_crop_samples/`（6 組）。
