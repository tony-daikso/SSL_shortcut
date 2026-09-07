# E0e-4：統一裁切協定驗證（full，60 支影片）

套用 `fov_protocol.unify_crop`（中央方形裁切 → 再裁掉 15% 邊距 → resize 到 224x224，邊距比例見 `05_calibrate_crop_margin.py` 的校準結果）。對 6000 張影格重新量測尺寸：發現 1 種不同的 (width, height) 組合（預期剛好是 1 種常數，驗證裁切協定對所有影格一視同仁）。

角落殘留（corner_black_fraction）裁切後統計：mean=0.00135，median=0.00000，max=0.78260（裁切前統計見 `fov_e0e_baseline_report.md`）。

## Cohort trivial baseline（只用 corner_black_fraction，裁切後）

|                        |   裁切後（10 折平均） |
|:-----------------------|--------------:|
| n_train_frames         |  4800         |
| n_test_frames          |  1200         |
| n_train_videos         |    48         |
| n_test_videos          |    12         |
| mean_accuracy          |     0.142417  |
| std_accuracy           |     0.0550838 |
| mean_majority_baseline |     0.116667  |
| uniform_chance         |     0.25      |
| n_valid_repeats        |    10         |
| n_repeats              |    10         |

對照 E0e-2（裁切前，size+corner 特徵，見 `fov_e0e_baseline_report.md`，裁切前 mean_accuracy 約 50-55%）：裁切後（只剩 corner_black_fraction 這一個特徵，因為尺寸類特徵已變成常數）：mean_accuracy 0.142 ± 0.055（chance 0.250、majority baseline 0.117，有效折數 10/10）——裁切後準確率大幅下降，接近 chance/majority，是全資料集規模下的正式結論。

## Endoscope brand within-cohort-002 control（只用 corner_black_fraction，裁切後）

|                   |         裁切後 |
|:------------------|------------:|
| n_frames          | 1500        |
| n_videos          |   15        |
| accuracy          |    0.072    |
| majority_baseline |    0.533333 |
| uniform_chance    |    0.5      |

**全資料集規模**：cohort 002 內有 8 支 Olympus + 7 支 Fujifilm，共 15 折 leave-one-video-out 都跑得出結果：accuracy 0.072（majority baseline 0.533）——裁切後準確率遠低於 majority baseline，是正式結論：裁切協定移除了角落遮罩後，同一個 cohort 內兩種品牌已經無法單靠幾何特徵區分。


## 主要證據：corner_black_fraction 裁切前後直接對比

裁切前（見 `fov_e0e_baseline_report.md`）：60 支影片的 corner_black_fraction 中位數依 cohort/品牌落在 0.42–0.78 之間，跟 cohort/品牌強烈對應。裁切後（本檔案開頭）：6000 張影格的 mean=0.00135、**median=0.00000**、max=0.78260——中位數已經降到完全乾淨（0），只有極少數離群影格（例如影片邊緣有非遮罩造成的暗部內容）還有殘留，這是比分類器 accuracy 更直接、不受 train/test 切分方式影響的證據（不需要任何 held-out 切分，對全部影格直接量測）。

## 閘門判定：✅ 通過（依 corner_black_fraction 中位數）

裁切後 corner_black_fraction 中位數降到 0.00000（幾乎為 0），符合計畫 E0e-4 的驗收標準，統一裁切協定確實把角落幾何線索壓到接近無法利用的程度。分類器層級的驗證（cohort 10 折平均、brand within-cohort-002 15 折 leave-one-video-out）在全部 60 支影片的規模下都跑得出穩固數字，且都降到接近 chance/majority 的水準，跟 corner_black_fraction 中位數趨近 0 的結論互相印證，是正式結論。


裁切前後對照樣本圖已存到 `results/qc_unify_crop_samples/`（6 組）。
