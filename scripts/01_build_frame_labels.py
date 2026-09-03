"""E0b（frame 層級）：從可取得的 per-frame VOC annotation 萃取三組標籤。

資料來源與其限制見 config.py 開頭的說明——這是 polyp 專案先前萃取的一個子集
（46 支影片有 all_polyp、60 支影片都有 no_polyp 抽樣負樣本），不是官方完整逐格標註。

每一列一個 frame，欄位：
- 病理標籤：polyp_label（1/0，由該 frame 的 XML 是否含 <object> 導出）、n_bbox
- 來源標籤：video_id、cohort、endoscope_brand（合併自 video_manifest）
- metadata 標籤：實際 frame 寬高（來自 XML 的 <size>，不是假設值）、frame_index
  （從檔名解析）、split（合併自 video_manifest，按影片繼承，不會洩漏）
- source_category：all_polyp / no_polyp，標記這個 frame 是從哪個子資料夾萃取出來的
  （用來提醒：no_polyp 資料夾是等間隔抽樣，all_polyp 資料夾本身也是被篩選過的子集，
  兩者合併統計出的「盛行率」不能代表官方完整資料集，見 02_confound_report.py 的說明）
"""

import re
import xml.etree.ElementTree as ET

import pandas as pd

from config import REAL_COLON_FRAMES_ROOT, RESULTS_DIR

# 001-012 這支影片的檔名多了一個奇怪的 `.0` 尾巴（例如 001-012_10999.0.xml），
# 猜測是產生這批子集時、上游某段程式碼把 frame index 當成 float 處理的殘留，
# 用 (?:\.0)? 容忍它，不影響 frame_index 的解析結果。
FRAME_ID_RE = re.compile(r"^(\d{3}-\d{3})_(\d+)(?:\.0)?$")


def parse_xml(xml_path):
    root = ET.parse(xml_path).getroot()
    filename = root.findtext("filename", default="")
    size = root.find("size")
    width = int(size.findtext("width")) if size is not None else None
    height = int(size.findtext("height")) if size is not None else None
    depth = int(size.findtext("depth")) if size is not None else None
    objects = root.findall("object")
    return filename, width, height, depth, len(objects)


def main():
    if not REAL_COLON_FRAMES_ROOT.exists():
        raise SystemExit(
            f"找不到 {REAL_COLON_FRAMES_ROOT}，這個路徑是外部（polyp 專案）資料，"
            "不在本 repo 管理範圍內，請確認該路徑是否還存在，或修改 config.py。"
        )

    video_manifest = pd.read_csv(RESULTS_DIR / "video_manifest.csv")
    video_lookup = video_manifest.set_index("video_id")[["cohort", "endoscope_brand", "split"]]

    rows = []
    for category in ("all_polyp", "no_polyp"):
        category_dir = REAL_COLON_FRAMES_ROOT / category
        if not category_dir.exists():
            continue
        for video_dir in sorted(category_dir.iterdir()):
            if not video_dir.is_dir():
                continue
            video_id = video_dir.name
            label_dir = video_dir / "label"
            if not label_dir.exists():
                continue
            for xml_path in sorted(label_dir.glob("*.xml")):
                stem = xml_path.stem
                m = FRAME_ID_RE.match(stem)
                if not m:
                    print(f"跳過無法解析檔名的 XML：{xml_path}")
                    continue
                video_from_name, frame_index = m.group(1), int(m.group(2))
                if video_from_name != video_id:
                    print(f"警告：{xml_path} 檔名前綴與所在資料夾 {video_id} 不一致")
                filename, width, height, depth, n_bbox = parse_xml(xml_path)
                if video_id not in video_lookup.index:
                    print(f"警告：{video_id} 不在 video_manifest 裡，略過 {xml_path}")
                    continue
                meta = video_lookup.loc[video_id]
                rows.append({
                    "frame_id": f"{video_id}_{frame_index}",
                    "video_id": video_id,
                    "cohort": meta["cohort"],
                    "endoscope_brand": meta["endoscope_brand"],
                    "split": meta["split"],
                    "frame_index": frame_index,
                    "width": width,
                    "height": height,
                    "depth": depth,
                    "polyp_label": int(n_bbox > 0),
                    "n_bbox": n_bbox,
                    "source_category": category,
                })

    df = pd.DataFrame(rows).sort_values(["video_id", "frame_index"]).reset_index(drop=True)
    out_path = RESULTS_DIR / "frame_labels.csv"
    df.to_csv(out_path, index=False)

    print(f"Wrote {len(df)} frames -> {out_path}")
    print()
    print("polyp_label 分布（依 source_category）：")
    print(df.pivot_table(index="source_category", columns="polyp_label", values="frame_id", aggfunc="count", fill_value=0))
    print()
    print("frame 尺寸分布（每個不同 (width, height) 組合出現次數，前 20）：")
    print(df.groupby(["width", "height"]).size().sort_values(ascending=False).head(20))
    print()
    n_videos_covered = df["video_id"].nunique()
    print(f"涵蓋影片數：{n_videos_covered} / 60")
    print(f"警告提醒：這是子集資料，不是官方完整逐格標註，見 config.py 開頭說明。")


if __name__ == "__main__":
    main()
