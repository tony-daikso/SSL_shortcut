"""E0a（修正版）：從官方 Figshare 原始資料下載影片、隨機抽 frame，取代先前誤用
「polyp」專案偵測器訓練子集的做法。

抽樣協定（使用者確認，2026-09-03 由 200 張均勻抽樣改成 100 張隨機抽樣）：
- 每支影片抽 100 張。
- **只從影片中間 1/3 抽**（排除前 1/3、後 1/3），在
  [num_frames * 1/3, num_frames * 2/3] 這個範圍內用均勻分布隨機抽 n 個索引位置
  （不是百分位數等距選點，兩者的差異：等距選點保證涵蓋整個範圍且彼此間距固定，
  隨機抽樣則允許間距不均、可能出現局部群聚，但不會有「每隔固定張數就抽一張」這種
  人為週期性）。
- 按影片處理、抽完立刻刪除下載的 tar（frames.tar.gz 每支 7-11GB，60 支加總
  500-600GB，本機剩餘空間放不下，見 figshare_client.py 的說明）。

2026-09-03 先用 5 支影片（PILOT_VIDEO_IDS 原本的值）驗證流程正確；2026-09-04
使用者確認擴大到**全部 60 支**，PILOT_VIDEO_IDS 改成從 video_manifest.csv 讀取
全部影片。已經處理過的影片（`data/sampled_frames/{video_id}/` 底下已經有滿
N_SAMPLES_PER_VIDEO 張的）會跳過下載，直接從既有檔案重建結果列，避免重複下載
浪費頻寬與時間（見 `already_sampled()`）。

004-003 這支影片的 frames.tar 已經在 SSL_research/videos/ 裡（另一個專案先前下載、
解壓過的殘留檔），這裡直接讀取、不重新下載，也不刪除它（不是這個 repo 建立的檔案，
不擅自清理）。其餘 pilot 影片從 Figshare 全新下載。

## 平行下載（2026-09-03 加入）

實測單一 TCP 連線下載 Figshare（AWS S3 後端）只有 ~6-7.5MB/s：traceroute 顯示這是
台灣→新加坡→美國的跨太平洋路徑，RTT ~217ms，單一連線的吞吐量受頻寬-延遲乘積限制
（≈ 視窗大小 / RTT），跟區網本身是 Gigabit、延遲 <1ms 無關，也不是 Figshare 主動
限速。解法是同時開多條連線（各自有獨立的 TCP 視窗），用 ThreadPoolExecutor 平行
下載＋處理多支影片（I/O bound，GIL 不是瓶頸，不需要 multiprocessing）。

## 改用 NAS 本地副本（2026-09-04 加入）

使用者發現公司 NAS（`/Volumes/homes/大腸公開資料集-Real Colon/22202866.zip`）上
已經有完整 60 支影片的官方原始資料（同一個 Figshare article 22202866 打包成一個
945GB 的 zip，內部每個 `{video_id}_frames.tar.gz` 用 STORED（不壓縮）方式存放，
可以直接當一般檔案 random-access 讀取，不需要先解壓整個 zip）。實測從 NAS 單一
連線讀取就有 ~103MB/s，比 Figshare 4 條平行連線加總（~20-23MB/s）快了將近 5 倍，
且不佔用對外頻寬。`get_or_download_tar()` 因此改成優先順序：
1. `EXISTING_TARS`（其他專案先前留下的本機殘留檔，見下）
2. NAS zip（`extract_from_nas_zip()`，用 `zipfile` 抽取單一 member 到本機暫存）
3. Figshare 官方 API 下載（`download_file()`，NAS 不可用時的備援）

這個切換是在跑到 34/60 支的時候才發現 NAS 有這份資料、臨時中止原本的 Figshare
下載改過來的——當下犧牲了幾支影片已經下載一半的進度（改用 NAS 重新抓更快，划算），
已經抽樣完成（`data/sampled_frames/{video_id}/` 有滿 100 張）的影片不受影響，
`already_sampled()` 會直接跳過重抓。
"""

import random
import shutil
import tarfile
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from PIL import Image

from config import RESULTS_DIR
from figshare_client import get_file_index, download_file
from fov_protocol import corner_black_fraction

NAS_ZIP_PATH = Path("/Volumes/homes/大腸公開資料集-Real Colon/22202866.zip")

N_SAMPLES_PER_VIDEO = 100
SAMPLE_RANGE = (1 / 3, 2 / 3)  # 只從影片中間 1/3 抽樣
SAMPLE_SEED = 0
MAX_PARALLEL_DOWNLOADS = 2
# 校準記錄（2026-09-04）：切換成 NAS 來源後一度用 4，實測跑到 cohort 003/004（部分
# 影片 20-44GB）時，4 支同時複製+抽取讓可用記憶體降到 ~55MB、暫存磁碟一度只剩
# 38GB（4 個大檔案同時停在 /tmp 裡等抽取），有塞爆硬碟的風險。改成 2，犧牲一些
# 聚合吞吐量換取穩定度——反正 NAS 單線速度已經比 Figshare 4 線聚合快，不需要靠
# 高並行度才能贏。


def all_video_ids() -> list[str]:
    manifest = pd.read_csv(RESULTS_DIR / "video_manifest.csv", dtype={"cohort": str})
    return manifest["video_id"].tolist()

TMP_DIR = Path("/tmp/e0_frames_dl")
SAMPLED_FRAMES_DIR = Path(__file__).resolve().parent.parent / "data" / "sampled_frames"

# 已經存在本機的殘留檔（來自另一個專案，先前下載+解壓過），直接讀取不重新下載
EXISTING_TARS = {
    "004-003": Path("/Users/tony.tu/Desktop/戴承智慧/SSL_research/videos/004-003_frames.tar"),
}


def pick_sample_indices(available_indices: list[int], n: int, sample_range: tuple[float, float]) -> list[int]:
    """available_indices 是 tar 裡實際存在的 frame index（已排序）。在
    [len*sample_range[0], len*sample_range[1]) 這個子範圍內，隨機抽 n 個索引位置
    （固定 SAMPLE_SEED 保證可重現）。"""
    lo = int(len(available_indices) * sample_range[0])
    hi = int(len(available_indices) * sample_range[1])
    window = available_indices[lo:hi]
    if len(window) <= n:
        return window
    rng = random.Random(SAMPLE_SEED)
    return sorted(rng.sample(window, n))


def extract_from_nas_zip(video_id: str) -> Path | None:
    """從 NAS 的 zip 裡把單一 member（STORED，不壓縮）抽到本機暫存，回傳路徑；
    NAS 不可用/檔案不存在時回傳 None，讓呼叫端退回 Figshare 下載。"""
    if not NAS_ZIP_PATH.exists():
        return None
    key = f"{video_id}_frames.tar.gz"
    try:
        with zipfile.ZipFile(NAS_ZIP_PATH) as zf:
            if key not in zf.namelist():
                return None
            TMP_DIR.mkdir(parents=True, exist_ok=True)
            print(f"[{video_id}] 從 NAS 抽取 {key}...", flush=True)
            extracted = zf.extract(key, path=TMP_DIR)
            print(f"[{video_id}] NAS 抽取完成", flush=True)
            return Path(extracted)
    except OSError as e:
        print(f"[{video_id}] NAS 讀取失敗（{e}），改用 Figshare 下載", flush=True)
        return None


def get_or_download_tar(video_id: str, file_index: dict) -> tuple[Path, bool]:
    """回傳 (tar_path, should_delete_after)。優先順序見檔案開頭說明：既有殘留檔 →
    NAS zip → Figshare 下載。"""
    if video_id in EXISTING_TARS and EXISTING_TARS[video_id].exists():
        print(f"[{video_id}] 沿用既有殘留檔：{EXISTING_TARS[video_id]}", flush=True)
        return EXISTING_TARS[video_id], False

    nas_path = extract_from_nas_zip(video_id)
    if nas_path is not None:
        return nas_path, True

    key = f"{video_id}_frames.tar.gz"
    if key not in file_index:
        raise RuntimeError(f"figshare 上找不到 {key}")

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    gz_path = TMP_DIR / key
    size_gb = file_index[key]["size"] / 1e9
    print(f"[{video_id}] NAS 沒有，改從 Figshare 下載 {key}（{size_gb:.1f} GB）...", flush=True)
    download_file(file_index[key]["download_url"], gz_path)
    print(f"[{video_id}] 下載完成", flush=True)
    return gz_path, True


def open_tar(tar_path: Path):
    if tar_path.suffix == ".gz" or tar_path.suffixes[-2:] == [".tar", ".gz"]:
        return tarfile.open(tar_path, "r:gz")
    return tarfile.open(tar_path, "r")


def rebuild_from_existing(video_id: str) -> pd.DataFrame:
    """這支影片先前已經抽樣過、影格檔案還在本機，直接重新掃描本機檔案重建結果，
    不重新下載。corner_black_fraction 重新計算（檔案本身很小，成本可忽略）。"""
    out_dir = SAMPLED_FRAMES_DIR / video_id
    rows = []
    for frame_path in sorted(out_dir.glob(f"{video_id}_*.jpg")):
        idx = int(frame_path.stem.split("_")[-1])
        im = Image.open(frame_path)
        width, height = im.size
        cbf = corner_black_fraction(im)
        rows.append({
            "video_id": video_id, "frame_index": idx,
            "width": width, "height": height, "corner_black_fraction": cbf,
            "frame_path": str(frame_path.relative_to(SAMPLED_FRAMES_DIR.parent.parent)),
        })
    print(f"[{video_id}] 沿用既有已抽樣的 {len(rows)} 張影格，跳過下載", flush=True)
    return pd.DataFrame(rows)


def already_sampled(video_id: str) -> bool:
    out_dir = SAMPLED_FRAMES_DIR / video_id
    if not out_dir.exists():
        return False
    return len(list(out_dir.glob(f"{video_id}_*.jpg"))) >= N_SAMPLES_PER_VIDEO


def process_video(video_id: str, file_index: dict) -> pd.DataFrame:
    if already_sampled(video_id):
        return rebuild_from_existing(video_id)

    tar_path, should_delete = get_or_download_tar(video_id, file_index)

    print(f"[{video_id}] 掃描 tar 內的影格清單...", flush=True)
    with open_tar(tar_path) as tar:
        members = {m.name: m for m in tar.getmembers() if m.name.endswith(".jpg")}
        # 檔名格式 {video_id}/{video_id}_{frame_index}.jpg 或 {video_id}_{frame_index}.jpg，
        # 用最後一個底線後的數字當 frame_index，兩種都能處理。
        def frame_index_of(name):
            stem = Path(name).stem
            return int(float(stem.split("_")[-1]))

        indexed = sorted(members.items(), key=lambda kv: frame_index_of(kv[0]))
        available_indices = [frame_index_of(name) for name, _ in indexed]
        name_by_index = {frame_index_of(name): name for name, _ in indexed}

        print(f"[{video_id}] tar 內共 {len(indexed)} 張影格，範圍 {available_indices[0]}-{available_indices[-1]}", flush=True)
        chosen = pick_sample_indices(available_indices, N_SAMPLES_PER_VIDEO, SAMPLE_RANGE)
        print(f"[{video_id}] 從中間 1/3 抽出 {len(chosen)} 張", flush=True)

        out_dir = SAMPLED_FRAMES_DIR / video_id
        out_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        for idx in chosen:
            member = members[name_by_index[idx]]
            data = tar.extractfile(member).read()
            frame_path = out_dir / f"{video_id}_{idx}.jpg"
            frame_path.write_bytes(data)

            im = Image.open(frame_path)
            width, height = im.size
            cbf = corner_black_fraction(im)
            rows.append({
                "video_id": video_id, "frame_index": idx,
                "width": width, "height": height, "corner_black_fraction": cbf,
                "frame_path": str(frame_path.relative_to(SAMPLED_FRAMES_DIR.parent.parent)),
            })

    if should_delete:
        print(f"[{video_id}] 刪除下載的 tar：{tar_path}", flush=True)
        tar_path.unlink()

    df = pd.DataFrame(rows)
    # 每支影片處理完立刻存一份，即使中途被中斷也不會丟掉已完成影片的進度
    df.to_csv(RESULTS_DIR / f"pilot_sampled_frames_{video_id}.csv", index=False)
    print(f"[{video_id}] 完成，{len(df)} 張", flush=True)
    return df


def main():
    video_ids = all_video_ids()
    file_index = get_file_index()

    results = {}
    errors = {}
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_DOWNLOADS) as pool:
        futures = {pool.submit(process_video, vid, file_index): vid for vid in video_ids}
        for future in as_completed(futures):
            video_id = futures[future]
            try:
                results[video_id] = future.result()
            except Exception as e:
                print(f"[{video_id}] 失敗：{e}", flush=True)
                errors[video_id] = e

    if errors:
        print(f"\n{len(errors)} 支影片失敗：{list(errors.keys())}")

    result = pd.concat([results[v] for v in video_ids if v in results], ignore_index=True)
    out_path = RESULTS_DIR / "pilot_sampled_frames.csv"
    result.to_csv(out_path, index=False)

    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)

    print(f"\nWrote {len(result)} 列 -> {out_path}")
    print(result.groupby("video_id").size())


if __name__ == "__main__":
    main()
