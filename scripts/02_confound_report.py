"""E0c：Confound 報表——每個 cohort 的 polyp 盛行率、大小、型態、histology 分布。

分兩部分，可信度不同：

Part A（權威、完整）：完全來自官方 video_info.csv + lesion_info.csv，涵蓋全部 60 支
影片、132 顆病灶，沒有任何抽樣或篩選，可放心引用。
- 影片層級盛行率：每個 cohort 有幾支影片至少有一顆病灶
- 病灶層級統計：每個 cohort 的病灶數、大小(size [mm])分布、site 分布、histology_class 分布
- 特別處理：lesion_info.csv 裡有些列的 histology_class 是 "NO POLYP"
  （切除後病理判讀不是息肉），這些列在計算「病灶大小/型態」統計時應排除或單獨列出，
  不能跟真正的息肉混在一起平均，否則會低估病灶嚴重度、高估某些 cohort 的病灶數。

Part B（探索性、子集）：來自 01_build_frame_labels.py 產出的 frame_labels.csv，也就是
polyp 專案先前萃取的子集（見 config.py 說明）。這部分算出的「frame 層級盛行率」不能
代表官方完整資料集，只能當作方向性參考，報告中會明確標註。
"""

import pandas as pd

from config import LESION_INFO_CSV, RESULTS_DIR


def part_a(video_manifest: pd.DataFrame, lesion_info: pd.DataFrame) -> str:
    lines = ["## Part A：官方完整資料（video_info.csv + lesion_info.csv，132 顆病灶，60 支影片）\n"]

    # 影片層級盛行率
    prevalence = (
        video_manifest.assign(has_lesion=video_manifest["num_lesions"] > 0)
        .groupby("cohort")
        .agg(n_videos=("video_id", "count"), n_with_lesion=("has_lesion", "sum"),
             total_lesions=("num_lesions", "sum"))
    )
    prevalence["video_prevalence_pct"] = (prevalence["n_with_lesion"] / prevalence["n_videos"] * 100).round(1)
    prevalence["mean_lesions_per_video"] = (prevalence["total_lesions"] / prevalence["n_videos"]).round(2)
    lines.append("### 影片層級盛行率\n")
    lines.append(prevalence.to_markdown())
    lines.append("")

    lesion = lesion_info.copy()
    lesion["cohort"] = lesion["unique_video_name"].str.split("-").str[0]

    n_no_polyp_histology = (lesion["histology_class"] == "NO POLYP").sum()
    lines.append(
        f"\n**注意**：lesion_info.csv 132 列中有 {n_no_polyp_histology} 列的 "
        "histology_class 是 `NO POLYP`（切除後病理判讀不是真正息肉）。"
        "以下大小/site/histology 分布統計時會排除這些列（列在單獨一節）。\n"
    )

    true_lesion = lesion[lesion["histology_class"] != "NO POLYP"]

    lines.append("### 病灶大小分布（size [mm]，排除 histology_class=NO POLYP 後）\n")
    size_stats = true_lesion.groupby("cohort")["size [mm]"].agg(["count", "mean", "median", "std", "min", "max"]).round(2)
    lines.append(size_stats.to_markdown())
    lines.append("")

    lines.append("### 解剖部位（site）分布\n")
    site_dist = pd.crosstab(true_lesion["cohort"], true_lesion["site"])
    lines.append(site_dist.to_markdown())
    lines.append(
        "\n**資料品質註記**：官方 `lesion_info.csv` 的 `site` 欄位本身有拼字不一致——"
        "同時存在 `caecum`/`cecum`、`transverse`/`trasnverse` 兩種拼法，指的應該是同一個"
        "解剖部位。這裡刻意原樣呈現（不擅自合併官方資料），但後續若要用 site 做分析"
        "（例如 E2c 的 confound 對照），需要先建一份拼字正規化對照表，否則會把同一個部位"
        "誤判成兩個類別。\n"
    )

    lines.append("### Histology class 分布（排除 NO POLYP 後）\n")
    hist_dist = pd.crosstab(true_lesion["cohort"], true_lesion["histology_class"])
    lines.append(hist_dist.to_markdown())
    lines.append("")

    lines.append("### 「切除後判讀不是息肉」（NO POLYP）的列數，依 cohort\n")
    no_polyp_by_cohort = lesion[lesion["histology_class"] == "NO POLYP"].groupby("cohort").size()
    lines.append(no_polyp_by_cohort.to_frame("n_no_polyp_histology").to_markdown())
    lines.append("")

    return "\n".join(lines)


def part_b(frame_labels: pd.DataFrame) -> str:
    lines = [
        "## Part B：探索性——frame 層級盛行率（子集資料，非官方完整逐格標註）\n",
        "**這一節的數字來自 polyp 專案先前萃取的子集**（46 支影片的 all_polyp 資料夾，"
        "每支只挑了個位數到數十張含 bbox 的 frame；60 支影片的 no_polyp 資料夾，每支等間隔"
        "抽樣幾十張負樣本），**不是**對官方 2,757,723 張影格、351,264 個 bbox 的完整統計。"
        "下面的『frame 盛行率』只能看方向、不能拿來對外主張具體數字。\n",
    ]

    by_cohort = frame_labels.groupby("cohort").agg(
        n_frames=("frame_id", "count"),
        n_polyp_frames=("polyp_label", "sum"),
        n_videos=("video_id", "nunique"),
    )
    by_cohort["polyp_frame_pct_in_subset"] = (by_cohort["n_polyp_frames"] / by_cohort["n_frames"] * 100).round(1)
    lines.append(by_cohort.to_markdown())
    lines.append("")
    lines.append(
        "（`polyp_frame_pct_in_subset` 偏高是因為 no_polyp 是刻意固定張數抽樣，"
        "跟該影片實際總長度無關，所以這個比例被抽樣設計本身決定，不代表官方資料集裡"
        "『87.6% 影格無標註』這個比例在各 cohort 之間有沒有差異。)"
    )
    lines.append("")
    return "\n".join(lines)


def main():
    # cohort 欄位（"001".."004"）若不強制指定 dtype，pandas 讀 csv 時會把它當數字，
    # 吃掉開頭的 0（"001" -> 1），跟 lesion_info 這邊用字串 split 出來的 "001" 對不起來。
    video_manifest = pd.read_csv(RESULTS_DIR / "video_manifest.csv", dtype={"cohort": str})
    lesion_info = pd.read_csv(LESION_INFO_CSV)
    frame_labels_path = RESULTS_DIR / "frame_labels.csv"

    sections = ["# E0c：Confound 報表\n"]
    sections.append(part_a(video_manifest, lesion_info))

    if frame_labels_path.exists():
        frame_labels = pd.read_csv(frame_labels_path, dtype={"cohort": str})
        sections.append(part_b(frame_labels))
    else:
        sections.append("## Part B：略過（尚未執行 01_build_frame_labels.py）\n")

    out_path = RESULTS_DIR / "confound_report.md"
    out_path.write_text("\n".join(sections), encoding="utf-8")
    print(f"Wrote confound report -> {out_path}")


if __name__ == "__main__":
    main()
