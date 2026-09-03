"""下載官方 60 支 `{video_id}_annotations.tar.gz`（每支 KB 等級，全部加總數十 MB，
安全整批下載），解析每一張 XML，建立**完整、逐格**的 polyp/no-polyp 標籤表。

這是修正先前錯誤的核心一步：先前 E0b/E0c 用的是「polyp」專案為了訓練偵測器另外
篩選過的一個小子集（3017 張，嚴重偏向 polyp 正樣本），不是這個 SSL 指紋研究需要的
中性樣本。官方 annotation 本身在 (至少部分) 影片上是每一格都有一個 XML（用
004-003 驗證過：annotation XML 數量剛好等於 video_info.csv 的 num_frames），
`<object>` 欄位有沒有出現才是 polyp/no-polyp 的真正判斷依據——所以下載官方
annotation 本身（不需要影格像素）就能建立完整、無偏的逐格病理標籤與 confound 統計。

輸出 `results/full_annotation_labels.csv`：一列一個「有 annotation 的 frame」，欄位
video_id, cohort, frame_index, width, height, n_bbox, polyp_label。同時輸出
`results/annotation_coverage.csv` 記錄每支影片「annotation XML 數量 / num_frames」，
如果不是 100% 涵蓋，如實記錄，不假設。

下載下來的 tar.gz 解析完就刪除，不留在 repo 裡（60 支雖然小，但沒必要囤積原始檔，
之後要重新下載也很快）。
"""

import shutil
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

from config import VIDEO_INFO_CSV, RESULTS_DIR
from figshare_client import get_file_index, download_file

TMP_DIR = Path("/tmp/e0_annotations_dl")


def parse_xml_bytes(data: bytes):
    root = ET.fromstring(data)
    size = root.find("size")
    width = int(size.findtext("width")) if size is not None else None
    height = int(size.findtext("height")) if size is not None else None
    n_bbox = len(root.findall("object"))
    return width, height, n_bbox


def process_video(video_id: str, file_index: dict) -> pd.DataFrame:
    key = f"{video_id}_annotations.tar.gz"
    if key not in file_index:
        print(f"  警告：figshare 上找不到 {key}，略過")
        return pd.DataFrame()

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    tar_path = TMP_DIR / key
    download_file(file_index[key]["download_url"], tar_path)

    rows = []
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            if not member.name.endswith(".xml"):
                continue
            # 官方 001-012 這支影片的檔名裡 frame index 多一個奇怪的 `.0` 尾巴
            # （例如 001-012_40104.0.xml）——這是官方原始資料就有的，不是任何本地
            # 處理造成的（「polyp」專案的子集也有一樣的現象，繼承自同一個來源）。
            # 用 float() 再轉 int 容忍它。
            stem = Path(member.name).stem
            frame_index = int(float(stem.split("_")[-1]))
            data = tar.extractfile(member).read()
            width, height, n_bbox = parse_xml_bytes(data)
            rows.append({
                "video_id": video_id, "frame_index": frame_index,
                "width": width, "height": height,
                "n_bbox": n_bbox, "polyp_label": int(n_bbox > 0),
            })

    tar_path.unlink()
    return pd.DataFrame(rows)


def main():
    video_manifest = pd.read_csv(VIDEO_INFO_CSV).rename(columns={"unique_video_name": "video_id"})
    video_manifest["cohort"] = video_manifest["video_id"].str.split("-").str[0]

    file_index = get_file_index()

    all_rows = []
    coverage_rows = []
    for _, v in video_manifest.iterrows():
        video_id = v["video_id"]
        print(f"處理 {video_id} annotation ...")
        df = process_video(video_id, file_index)
        if df.empty:
            continue
        df["cohort"] = v["cohort"]
        all_rows.append(df)
        coverage_rows.append({
            "video_id": video_id, "cohort": v["cohort"],
            "num_frames_official": v["num_frames"],
            "n_annotation_xml": len(df),
            "coverage_pct": round(len(df) / v["num_frames"] * 100, 2),
            "n_polyp_frames": int(df["polyp_label"].sum()),
        })

    full = pd.concat(all_rows, ignore_index=True)
    full_path = RESULTS_DIR / "full_annotation_labels.csv"
    full.to_csv(full_path, index=False)

    coverage = pd.DataFrame(coverage_rows)
    coverage_path = RESULTS_DIR / "annotation_coverage.csv"
    coverage.to_csv(coverage_path, index=False)

    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)

    print(f"\nWrote {len(full)} 列 frame-level annotation -> {full_path}")
    print(f"Wrote coverage table -> {coverage_path}")
    print("\nCoverage 摘要（是否每支影片都是逐格標註）：")
    print(coverage["coverage_pct"].describe())
    print("\n依 cohort 的整體 frame-level polyp 盛行率（官方完整資料，非子集）：")
    by_cohort = full.groupby("cohort").agg(n_frames=("frame_index", "count"), n_polyp=("polyp_label", "sum"))
    by_cohort["polyp_pct"] = (by_cohort["n_polyp"] / by_cohort["n_frames"] * 100).round(2)
    print(by_cohort)


if __name__ == "__main__":
    main()
