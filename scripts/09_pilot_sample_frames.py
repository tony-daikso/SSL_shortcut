"""E0a（修正版）：從官方 Figshare 原始資料下載影片、隨機抽 frame，取代先前誤用
「polyp」專案偵測器訓練子集的做法。

抽樣協定（使用者確認）：
- 每支影片抽 200 張。
- **只從影片中間 1/3 抽**（排除前 1/3、後 1/3），用百分位數在
  [num_frames * 1/3, num_frames * 2/3] 這個範圍內均勻選點，而不是像原始 Phase 0
  spec 那樣涵蓋全片 0-100%。
- 按影片處理、抽完立刻刪除下載的 tar（frames.tar.gz 每支 7-11GB，60 支加總
  500-600GB，本機剩餘空間放不下，見 figshare_client.py 的說明）。

這支腳本是 pilot：只處理 PILOT_VIDEO_IDS 這幾支（涵蓋 4 個 cohort、兩種品牌），
驗證流程正確後，再決定要不要擴大到全部 60 支（見 10_full_sample_frames.py，等
pilot 驗證通過後才寫/執行）。

004-003 這支影片的 frames.tar 已經在 SSL_research/videos/ 裡（另一個專案先前下載、
解壓過的殘留檔），這裡直接讀取、不重新下載，也不刪除它（不是這個 repo 建立的檔案，
不擅自清理）。其餘 pilot 影片從 Figshare 全新下載。
"""

import shutil
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from config import RESULTS_DIR
from figshare_client import get_file_index, download_file
from fov_protocol import corner_black_fraction

PILOT_VIDEO_IDS = ["001-001", "002-004", "002-006", "003-001", "004-003"]
N_SAMPLES_PER_VIDEO = 200
SAMPLE_RANGE = (1 / 3, 2 / 3)  # 只從影片中間 1/3 抽樣

TMP_DIR = Path("/tmp/e0_frames_dl")
SAMPLED_FRAMES_DIR = Path(__file__).resolve().parent.parent / "data" / "sampled_frames"

# 已經存在本機的殘留檔（來自另一個專案，先前下載+解壓過），直接讀取不重新下載
EXISTING_TARS = {
    "004-003": Path("/Users/tony.tu/Desktop/戴承智慧/SSL_research/videos/004-003_frames.tar"),
}


def pick_sample_indices(available_indices: list[int], n: int, sample_range: tuple[float, float]) -> list[int]:
    """available_indices 是 tar 裡實際存在的 frame index（已排序）。在
    [len*sample_range[0], len*sample_range[1]) 這個子範圍內，用百分位數均勻選 n 個
    索引位置（不是均勻選 frame index 數值，因為兩者在有缺格的情況下不等價）。"""
    lo = int(len(available_indices) * sample_range[0])
    hi = int(len(available_indices) * sample_range[1])
    window = available_indices[lo:hi]
    if len(window) <= n:
        return window
    positions = np.linspace(0, len(window) - 1, n).round().astype(int)
    positions = sorted(set(positions.tolist()))
    return [window[p] for p in positions]


def get_or_download_tar(video_id: str, file_index: dict) -> tuple[Path, bool]:
    """回傳 (tar_path, should_delete_after)。"""
    if video_id in EXISTING_TARS and EXISTING_TARS[video_id].exists():
        print(f"  沿用既有殘留檔：{EXISTING_TARS[video_id]}")
        return EXISTING_TARS[video_id], False

    key = f"{video_id}_frames.tar.gz"
    if key not in file_index:
        raise RuntimeError(f"figshare 上找不到 {key}")

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    gz_path = TMP_DIR / key
    size_gb = file_index[key]["size"] / 1e9
    print(f"  下載 {key}（{size_gb:.1f} GB）...")
    download_file(file_index[key]["download_url"], gz_path)
    return gz_path, True


def open_tar(tar_path: Path):
    if tar_path.suffix == ".gz" or tar_path.suffixes[-2:] == [".tar", ".gz"]:
        return tarfile.open(tar_path, "r:gz")
    return tarfile.open(tar_path, "r")


def process_video(video_id: str, file_index: dict) -> pd.DataFrame:
    tar_path, should_delete = get_or_download_tar(video_id, file_index)

    print("  掃描 tar 內的影格清單...")
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

        print(f"  tar 內共 {len(indexed)} 張影格，範圍 {available_indices[0]}-{available_indices[-1]}")
        chosen = pick_sample_indices(available_indices, N_SAMPLES_PER_VIDEO, SAMPLE_RANGE)
        print(f"  從中間 1/3 抽出 {len(chosen)} 張")

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
        print(f"  刪除下載的 tar：{tar_path}")
        tar_path.unlink()

    return pd.DataFrame(rows)


def main():
    file_index = get_file_index()

    all_rows = []
    for video_id in PILOT_VIDEO_IDS:
        print(f"=== {video_id} ===")
        df = process_video(video_id, file_index)
        all_rows.append(df)

    result = pd.concat(all_rows, ignore_index=True)
    out_path = RESULTS_DIR / "pilot_sampled_frames.csv"
    result.to_csv(out_path, index=False)

    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)

    print(f"\nWrote {len(result)} 列 -> {out_path}")
    print(result.groupby("video_id").size())


if __name__ == "__main__":
    main()
