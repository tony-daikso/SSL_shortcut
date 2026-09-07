# E5：在 REAL-Colon 上自訓 DINO——操作說明（程式碼已就緒，等待遠端 GPU 執行）

對應研究計畫 §6「E5：在 REAL-Colon 上自訓 DINO」，日期 2026-09-07。**這份文件
是操作說明，不是結果摘要**——訓練本身需要在有 GPU 的遠端 server 上跑（計畫 §1
標注的硬體是單卡 L40S 48GB），這裡先把完整程式碼寫好、在本機（Mac，MPS/CPU）
用極小規模的資料跑過 smoke test 確認整條 pipeline 沒有 bug，正式訓練與 E5b/E5c
的結論留給遠端跑完後補上。

## 為什麼從零自己刻 ViT，不用 timm/torch.hub

DINO 的 multi-crop 訓練需要同一個 backbone 同時吃 224px 的 global crop 跟
96px 的 local crop，需要位置編碼能動態插值到不同解析度。timm 有支援，但這台
本機機器裝不上（pip 被 Homebrew 的 externally-managed-environment 擋住），且
不確定遠端 server 上有沒有裝。改成用 `scripts/e5_vision_transformer.py` 自己
刻一份最小 ViT-S/16 實作（只依賴 torch，含位置編碼插值），跟現有
`requirements.txt` 完全相容，不需要在遠端額外裝任何套件。

## 架構與程式碼

- `scripts/e5_vision_transformer.py`：ViT-S/16（embed_dim=384, depth=12,
  num_heads=6，跟 DINO/DINOv2 論文的 ViT-S 標準配置一致），含
  `interpolate_pos_encoding` 支援任意輸入解析度。
- `scripts/e5_dino_model.py`：`DINOHead`（MLP + L2 normalize + 權重正規化最後一
  層）、`MultiCropWrapper`（依解析度分組批次過 backbone）、`DINOLoss`
  （teacher centering + 溫度銳化的 cross-entropy）。照官方 DINO 架構設計。
- `scripts/e5_dino_augmentations.py`：`DINOMultiCropAugmentation`——2 個 global
  crop（224px）+ N 個 local crop（96px），標準 color jitter/grayscale/blur/
  solarize 配方；`add_noise_whitening=True` 時額外疊加 **E4a 驗證過**的雜訊白化
  （`e4_noise_whitening.whiten_noise`，強度沿用 E4 報告裡代價最小、效果最明確
  的 strength=1.0），對應 E5c。
- `scripts/e5_dino_dataset.py`：讀 `results/pilot_frame_labels.csv`（目前是
  全部 60 支影片、每支影片中間 1/3 時間軸抽的 100 張，見 09_pilot_sample_frames.py
  的既有抽樣），套用跟 E0e/E1 一致的 `unify_crop`（先切掉 FOV 幾何線索，避免
  自訓模型連這個已知捷徑都直接學進去）。
- `scripts/25_e5_train_dino.py`：訓練主程式，從**隨機初始化**開始訓練（不是
  微調 ImageNet 預訓練權重）——這樣才能乾淨測「SSL 目標函數本身有沒有在這批
  資料上主動放大指紋」，直接對應 E1b 的 `dinov2_random`/`dinov2_pretrained`
  兩個既有對照組。
- `scripts/26_e5_extract_embeddings.py`：從一個訓練完的 checkpoint（teacher
  backbone）抽 embedding，用跟 E1 完全一樣的 6000 張真實影格 + 前處理，確保
  直接可比。
- `scripts/27_e5_compare_probes.py`：把自訓 embedding 跟 E1 既有的
  `dinov2_pretrained`/`dinov2_random` 放進同一組 probe 方法論比較，自動產生
  E5b 判準的解讀文字。

## 本機 smoke test（已完成，確認整條 pipeline 可正常運作）

用極小規模（每支影片抽 3 張、共 180 張、1 個 epoch、batch size 4、2 個 local
crop）在這台 Mac 上驗證：

```bash
cd scripts
# 標準 augmentation（對應 E5a/b）
python3 25_e5_train_dino.py --max-frames-per-video 3 --epochs 1 --batch-size 4 \
    --n-local-crops 2 --output-dir ../results/e5_smoke_test

# 加上 E4 雜訊白化（對應 E5c）
python3 25_e5_train_dino.py --max-frames-per-video 3 --epochs 1 --batch-size 4 \
    --n-local-crops 2 --add-noise-whitening --output-dir ../results/e5_smoke_test_whitened

# 抽 embedding + 跟 E1 既有結果比較
python3 26_e5_extract_embeddings.py --checkpoint ../results/e5_smoke_test/checkpoint.pt --tag dino_smoke_test
python3 27_e5_compare_probes.py
```

結果：loss 從初始值 ~9.0（= ln(out_dim=8192)，未訓練時均勻分布的理論值，證明
初始化正確）開始，訓練/儲存 checkpoint/抽 embedding/比較 probe 全部跑完沒有
崩潰。自訓 45 步（180 張圖）的 video_id 可分性（26.0%）略低於 dinov2_random
（28.8%）——這是**預期中的結果**：訓練量遠遠不足以收斂，這裡只是驗證程式碼
正確性，不代表任何 E5b 的正式結論。smoke test 的暫存輸出已經清掉，沒有留在
repo 裡。

## 在遠端 GPU 上正式執行

```bash
# E5a/b：標準 augmentation，正式規模訓練
python3 25_e5_train_dino.py --epochs 100 --batch-size 64 --n-local-crops 8 \
    --output-dir ../results/e5_dino_standard

# E5c：加上 E4 驗證過的雜訊白化
python3 25_e5_train_dino.py --epochs 100 --batch-size 64 --n-local-crops 8 \
    --add-noise-whitening --output-dir ../results/e5_dino_whitened

# 各自抽 embedding 並跟 E1 既有結果比較
python3 26_e5_extract_embeddings.py --checkpoint ../results/e5_dino_standard/checkpoint.pt --tag dino_selftrained_standard
python3 26_e5_extract_embeddings.py --checkpoint ../results/e5_dino_whitened/checkpoint.pt --tag dino_selftrained_whitened
python3 27_e5_compare_probes.py
```

`--epochs`/`--batch-size`/`--n-local-crops` 需要依實際 GPU 記憶體與可訓練時間
調整——上面只是示意值，不是校準過的建議值（DINO 原始論文的 ImageNet 規模訓練
用 batch size 1024、100+ epoch、8+ 張 GPU，REAL-Colon 規模小很多，且只有單卡
L40S，需要視情況大幅縮小或拉長訓練時間，`train_log.csv` 的 loss 曲線是否收斂
是判斷訓練夠不夠的第一手依據）。

## 已知限制／待確認事項

1. **訓練語料目前限定在既有的 6000 張抽樣影格**（每支影片中間 1/3 時間軸抽
   100 張）。計畫 E5a 原文建議「每支影片均勻抽樣，不用全部影格」，這批既有
   資料已經是抽樣後的結果，可以直接當訓練語料；但如果要更貼近論文原意（涵蓋
   整支影片的時間軸，不只中間 1/3），需要在遠端重新抽樣、產生新的
   frame_labels manifest，再用 `--frame-labels-csv` 指向新檔案，不需要改
   `25_e5_train_dino.py` 本身。
2. **`out_dim=8192` 是猜測值，不是校準過的超參數**——DINO 官方在 ImageNet
   規模（~1.2M 張、1000 類）用 65536；REAL-Colon 規模小很多（6000 張、
   60 支影片、4 個 cohort），縮小 out_dim 是合理方向，但沒有做敏感度分析。
3. **teacher_temp 目前是固定值（0.04），沒有實作官方常見的 warmup schedule**
   （官方在 ImageNet 規模訓練前 30 epoch 把 teacher_temp 從 0.04 warmup 到
   0.07，避免訓練初期崩潰）。REAL-Colon 規模小、訓練 epoch 數可能也遠少於
   ImageNet 規模的設定，固定溫度是否需要 warmup 需要實際訓練時觀察 loss 曲線
   判斷；如果訓練不穩定（loss 早期跳動或崩潰到單一值），第一個該調的參數是這個。
4. **E5c 的雜訊白化強度直接沿用 E4 離線分析驗證過的 strength=1.0**，沒有在
   「訓練中當 augmentation 用」這個新情境下重新驗證/調整——E4 驗證的是「對
   已經訓練好的 frozen 模型抽 embedding 時套用白化」，跟「訓練過程中每個 crop
   都套用白化」是不同的使用情境，理論上應該只會更強化雜訊白化的效果（模型
   訓練時就學不到固定雜訊指紋），但沒有實證。
5. 這份文件寫成時只完成程式碼與 smoke test，**E5a/b/c 的正式結論都還沒有**，
   需要在遠端 GPU 完成正式訓練後另外補上（比照 `docs/E1_summary.md`/
   `E3_summary.md`/`E4_summary.md` 的格式）。
