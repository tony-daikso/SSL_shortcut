"""跟 Figshare API 互動的小工具：REAL-Colon 官方資料集是 figshare article 22202866
（Biffi et al., Sci Data 2024，見 data/raw_refs/dataset_description.md）。

每支影片對應兩個檔案：
- `{video_id}_frames.tar.gz`：全部影格，壓縮後約 7-11GB/支，60 支加總 500-600GB，
  不可能整批下載保留，要一支一支下載、抽完樣就刪除（見 09_pilot_sample_frames.py）。
- `{video_id}_annotations.tar.gz`：每支影片全部影格的 per-frame VOC annotation
  （不是只有含 bbox 的影格才有 XML——實測 004-003 這支影片的 annotation XML 數量
  剛好等於它的 num_frames，代表**每一格都有一個 XML**，只是沒有 lesion 時
  `<object>` 欄位是空的）。每支只有 KB 等級，60 支加總數十 MB，可以整批下載。
"""

import json
from pathlib import Path

import requests

ARTICLE_URL = "https://api.figshare.com/v2/articles/22202866"
CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "raw_refs" / "figshare_files.json"


def get_file_index(force_refresh: bool = False) -> dict:
    """回傳 {filename: {"download_url":..., "size":...}}，本地快取避免每次都打 API。"""
    if CACHE_PATH.exists() and not force_refresh:
        return json.loads(CACHE_PATH.read_text())

    resp = requests.get(ARTICLE_URL, timeout=30)
    resp.raise_for_status()
    files = resp.json()["files"]
    index = {f["name"]: {"download_url": f["download_url"], "size": f["size"]} for f in files}

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(index, indent=2))
    return index


def download_file(url: str, dest_path: Path, chunk_size: int = 1 << 20) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
