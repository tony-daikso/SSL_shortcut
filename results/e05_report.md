# E0.5：合成指紋校準結果

4-way 分類，chance level = 0.250。GroupShuffleSplit 依 video_id 分組，10 次不同 random seed 取平均 ± 標準差。

## 各變體的 probe 準確率

|                      |   mean_accuracy |   std_accuracy |   n_repeats |   chance |
|:---------------------|----------------:|---------------:|------------:|---------:|
| clean                |           0.248 |      0.0769155 |          10 |     0.25 |
| pattern_noise        |           0.452 |      0.0446766 |          10 |     0.25 |
| color_shift          |           0.478 |      0.0763937 |          10 |     0.25 |
| pattern_noise_jitter |           0.411 |      0.0459239 |          10 |     0.25 |
| color_shift_jitter   |           0.27  |      0.0531037 |          10 |     0.25 |

## E0.5a：probe 抓不抓得到注入的合成指紋

pattern_noise（無 jitter）：0.452 ± 0.045；color_shift（無 jitter）：0.478 ± 0.076；對照 clean（不該有任何可分性，因為合成類別跟內容無關）：0.248 ± 0.077。

**✅ 通過**：兩種注入的合成指紋都遠高於 chance，probe 抓得到，可以繼續 E0.5b/c。

## E0.5b：probe 容量匹配（等強度下兩種訊號的靈敏度是否對等）

兩者 accuracy 差距：0.026（pattern_noise 0.452 vs color_shift 0.478）。

**✅ 通過（差距 <0.10，視為大致對等）**

## E0.5c：機制驗證（color jitter 應該壓下 color_shift、不該壓下 pattern_noise）

pattern_noise：0.452 → 0.411（jitter 後掉了 0.041，預期應該幾乎不掉）。

color_shift：0.478 → 0.270（jitter 後掉了 0.208，預期應該明顯掉、趨近 chance）。

**✅ 通過**：『一條降一條平』的機制性預測在合成資料上成立，代表 E3 之後在真實資料上做同樣的 dose-response 分析，這個機制性論證的邏輯本身是站得住腳的。
