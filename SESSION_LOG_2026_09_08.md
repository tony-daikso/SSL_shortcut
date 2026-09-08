# SESSION_LOG_2026_09_08

## 2026-09-08：E5 正式訓練啟動（遠端 L40S GPU server）

延續 `docs/E5_notes.md` 留下的待辦——E5 程式碼跟本機 smoke test 都已完成，這次
在遠端有 GPU 的 server 上正式跑 E5a/b（標準 augmentation）+ E5c（雜訊白化）。

### 環境確認與資料路徑修正

- `git pull` 確認 repo 是最新版；`requirements.txt` 列的套件在 conda env
  `workenv` 裡都已裝好（torch 2.6.0+cu124），不需要另外安裝。
- 影格資料實際放在 `/datadrive/SSL_shortcut/data/sampled_frames/`，但
  `pilot_frame_labels.csv` 裡的 `frame_path` 是相對 repo root 的路徑
  （`data/sampled_frames/...`）——建了一個 symlink
  `data/sampled_frames → /datadrive/SSL_shortcut/data/sampled_frames` 解決。
- 用 5 張影格的極小規模跑了一次 GPU smoke test，確認整條 pipeline 在這台機器
  的 CUDA 環境下能正常跑通（沿用本機 Mac smoke test 已驗證過的邏輯）。

### GPU 資源狀況與參數校準

- 這張 L40S 48GB 是**共用的**：另有一個 `model_polyp` 專案的 finetune job
  （`semi_code/run_semi.py --stages finetune --init ssl`）常態占用約 27.5GB，
  實際可用記憶體只剩約 17.9GB，且會隨時間浮動（後來觀察過一次降到 10.8GB
  used、也觀察過兩個 job 疊加衝到 36.5GB used 只剩 8.8GB 空閒的高點）。
- `docs/E5_notes.md` 裡遠端執行範例值（`--batch-size 64`）只是示意值，實測跟
  這台機器實際狀況校準：
  - `--num-workers 0`（腳本預設）在這個 FUSE 掛載的資料路徑上是 I/O bound，
    單步接近 1.6s，不能用；改用 `--num-workers 8` 後單步降到約 0.3–0.4s。
  - `--batch-size 32 --n-local-crops 8`：實測尖峰記憶體約 9GB，緩衝充足。
  - `--batch-size 64 --n-local-crops 8`：實測尖峰記憶體約 16.7GB，只剩約
    1.2GB 緩衝——在共用 GPU、且另一個 job 用量會浮動的情況下風險太高，放棄。
  - 最終採用：`--epochs 100 --batch-size 32 --n-local-crops 8 --num-workers 8
    --log-every 20`。全資料集（6000 張）一個 epoch 約 187 step，穩定速度約
    0.32–0.4s/step，單組（standard 或 whitened）估計約 1.5–2 小時，兩組合計
    約 3.5–4 小時。

### 持久化儲存修正（重要）

- 一開始把訓練輸出寫到 `results/`（repo 在 `/root/Desktop/SSL_shortcut`），
  後來使用者提醒：這個 server 是 container，`/root/Desktop` 掛在 docker
  overlay 層（`mount` 確認是 `overlay on /`），**server 關掉/重啟內容會消失**；
  只有 `/datadrive`（獨立 FUSE 掛載）是持久化的。
- 立刻停掉當時剛啟動的訓練，把 `results/` 整個搬到
  `/datadrive/SSL_shortcut/results/`，repo 裡的 `results/` 改成 symlink 指過去
  （跟 `data/sampled_frames` 用同一招）。之後 checkpoint、train_log.csv、
  embedding、比較報告都會自動落在持久化儲存，不會因為 server 重開而消失。
- 訓練 stdout log 也直接寫進 `/datadrive/SSL_shortcut/results/e5_train_run.log`
  （非正式研究產出，純粹方便回頭查訓練過程）。

### 執行方式

- 寫了一個 driver script 依序跑：`25_e5_train_dino.py`（standard）→ 同腳本
  加 `--add-noise-whitening`（whitened）→ `26_e5_extract_embeddings.py`
  （兩個 tag：`dino_selftrained_standard`/`dino_selftrained_whitened`）→
  `27_e5_compare_probes.py`，全部背景執行（nohup），並設一個背景 monitor
  每 10 分鐘回報一次進度（目前 epoch/step/loss/lr、GPU 總用量），出錯（OOM/
  loss 發散/崩潰）會立即通知，不用等下一個週期。

### 訓練會產生的檔案（都在 `/datadrive/SSL_shortcut/results/`）

- `e5_dino_standard/checkpoint.pt`、`e5_dino_whitened/checkpoint.pt` +
  各自的 `train_log.csv`
- `e5_embeddings_dino_selftrained_standard.npz`、
  `e5_embeddings_dino_selftrained_whitened.npz`
- `e5_compare_report.md`（跟 E1 既有的 `dinov2_pretrained`/`dinov2_random`
  對照組比較）

### 訓練動態（截至本次工作階段記錄時）

- Loss 從理論初始值 ln(8192)≈9.0 開始，epoch 0–5 平均 8.95→8.81 緩降，
  epoch 6–11 加速到 8.77→8.50，epoch 12–20 降到 8.0 附近，個別 step 已見
  7.5–7.7，收斂趨勢正常、無發散跡象。
- **這次訓練沒有 validation/test set**——`25_e5_train_dino.py` 用全部 6000
  張影格直接訓練，log 裡的 loss 是純 training loss，不算 validation loss。
  這是 SSL 預訓練的正常做法（無標籤，loss 本身只是訓練動態指標，不是泛化
  分數）；真正的下游評估在訓練完之後的 `26`/`27` 腳本，那裡才會用到按
  `video_id` 分組的 train/test split（給 probe classifier 用，不是給 DINO
  backbone 本身）。

### 資料洩漏疑慮的討論（已記錄進 `docs/E5_notes.md` 已知限制第 6 點）

使用者提出：如果下游任務也用到這些影片，會不會有 data leakage？討論結論：

- **跟同時在跑的 `model_polyp` finetune job 無關**——查過
  `model_polyp/semi_code/config.yaml`，那是完全獨立的專案（YOLO26 + MAE 風格
  SSL、不同資料來源 `CG_data/.../ssl_pretrain/images`、backbone 是 7 月就
  存好的舊 checkpoint），跟這次 E5 訓練沒有交集。
- **但 SSL_shortcut 專案自己內部確實有個值得記錄的疑慮**：E5 backbone 在無
  標籤的 SSL 階段已經看過全部 60 支影片的畫面，`27_e5_compare_probes.py`
  按 `video_id` 切的 train/test 只切給 probe classifier，不是給 backbone——
  所以 backbone 對任何「按這批影片切分」的下游評估都不是真正未見過
  （pixel-level 曝光，非 label 洩漏）。
- 若下游換一個完全不重疊的 dataset，這個特定洩漏管道會消失，但要先確認真的
  沒有來源重疊（REAL-Colon 是公開資料集），且「沒洩漏」不代表遷移效果會好
  （domain shift 是另一個獨立問題）。
- **決定：這次訓練繼續跑，不改成 held-out 版本。** 理由是 E5a/b 要回答的
  問題本來就是「SSL 在這批資料上訓練會不會放大指紋」，對照組是完全沒看過
  REAL-Colon 的 `dinov2_pretrained`——backbone 看過全部 60 支影片是這個比較
  設計上需要的極端，不是方法論漏洞。更嚴謹的 held-out 版本留給未來、若真的
  要拿這個 backbone 做下游泛化評估時，當一個新的獨立實驗（暫定 E5d）再做，
  不取代現在這組。此結論已寫進 `docs/E5_notes.md` 已知限制第 6 點。

### 目前狀態（本次工作階段結束時）

- E5a/b（standard）訓練仍在背景執行中，尚未跑到 whitened 階段，也還沒有
  `docs/E5_summary.md` 的正式結論。
- 下一步：等 standard + whitened 兩組訓練、embedding 抽取、probe 比較全部
  跑完後，補上 `docs/E5_summary.md`（比照其他 `E*_summary.md` 格式），更新
  `SESSION_LOG.md` 的狀態總覽表跟 README/Notion。
