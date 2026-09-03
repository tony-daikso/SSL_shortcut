# E0e-1/E0e-2：FOV 幾何洩漏檢查 + trivial baseline

## FOV 遮罩角落殘留（corner_black_fraction，20% 角落框內黑色像素比例）

**勘誤**：E0d 原本的檢查方法（整行/整列全黑）沒有偵測到這個訊號，實際上2975 張影格裡角落殘留是普遍存在的，且依 cohort/品牌有明顯差異：

|                     |   count |   median |     mean |        std |
|:--------------------|--------:|---------:|---------:|-----------:|
| ('001', 'Olympus')  |     543 | 0.676438 | 0.637946 | 0.0601257  |
| ('002', 'Fujifilm') |     258 | 0.772802 | 0.768808 | 0.00833684 |
| ('002', 'Olympus')  |     352 | 0.681763 | 0.685531 | 0.0142371  |
| ('003', 'Olympus')  |    1250 | 0.68182  | 0.644807 | 0.0540168  |
| ('004', 'Olympus')  |     572 | 0.436177 | 0.490722 | 0.0844606  |

每支影片的幾何摘要已存到 `fov_geometry.csv`（59 支影片）。

## E0e-2a：cohort（4-way）trivial baseline，train→test（held-out 影片）

### 只用影格尺寸（width/height/aspect_ratio/log_area）

|                   |       value |
|:------------------|------------:|
| n_train_frames    | 1975        |
| n_test_frames     |  653        |
| n_train_videos    |   40        |
| n_test_videos     |   12        |
| accuracy          |    0.773354 |
| majority_baseline |    0.425727 |
| uniform_chance    |    0.25     |

### 加上 corner_black_fraction

|                   |       value |
|:------------------|------------:|
| n_train_frames    | 1975        |
| n_test_frames     |  653        |
| n_train_videos    |   40        |
| n_test_videos     |   12        |
| accuracy          |    0.782542 |
| majority_baseline |    0.425727 |
| uniform_chance    |    0.25     |

## E0e-2b：endoscope_brand（2-way）trivial baseline，train→test（held-out 影片）

|                   |       value |
|:------------------|------------:|
| n_train_frames    | 1975        |
| n_test_frames     |  653        |
| n_train_videos    |   40        |
| n_test_videos     |   12        |
| accuracy          |    1        |
| majority_baseline |    0.932619 |
| uniform_chance    |    0.5      |

**注意**：brand 和 cohort 高度共線（只有 cohort 002 同時有兩種品牌，見 E0a 的報表），這裡的高準確率有可能只是在猜『這是不是 cohort 002 的影片』，不是真的學到品牌的幾何差異。下面用 cohort 002 內部做 within-cohort control 拆解這個問題。

## E0e-2c：endoscope_brand within-cohort-002 control（leave-one-video-out）

|                   |      value |
|:------------------|-----------:|
| n_frames          | 610        |
| n_videos          |  15        |
| accuracy          |   1        |
| majority_baseline |   0.577049 |
| uniform_chance    |   0.5      |

**解讀**：只用 cohort 002 內部的影片（同時有 Olympus 和 Fujifilm），輪流留一支出來測試。若準確率仍明顯高於 majority baseline，代表光是幾何（尺寸+角落遮罩）就能分辨品牌，不是靠 cohort 這個共線變因取巧。
