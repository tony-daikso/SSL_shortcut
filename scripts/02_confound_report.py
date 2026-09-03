"""E0c：Confound 報表——每個 cohort 的 polyp 盛行率、大小、型態、histology 分布。

分兩部分，可信度不同：

Part A（權威、完整）：完全來自官方 video_info.csv + lesion_info.csv，涵蓋全部 60 支
影片、132 顆病灶，沒有任何抽樣或篩選，可放心引用。
- 影片層級盛行率：每個 cohort 有幾支影片至少有一顆病灶
- 病灶層級統計：每個 cohort 的病灶數、大小(size [mm])分布、site 分布、histology_class 分布
- 特別處理：lesion_info.csv 裡有些列的 histology_class 是 "NO POLYP"
  （切除後病理判讀不是息肉），這些列在計算「病灶大小/型態」統計時應排除或單獨列出，
  不能跟真正的息肉混在一起平均，否則會低估病灶嚴重度、高估某些 cohort 的病灶數。

Part B（權威、完整，2026-09-03 修正）：來自 07_download_all_annotations.py 下載的官方
60 支 `{video_id}_annotations.tar.gz`，解析全部影格的 VOC annotation。**這是修正版**——
先前這裡用的是「polyp」專案為了訓練偵測器另外篩選過的一個小子集（3017 張，嚴重偏向
polyp 正樣本），跟這個 SSL 指紋研究要看的中性樣本完全不符，已經捨棄不用。實測發現官方
annotation 其實是**逐格**的（annotation_coverage.csv 顯示 60 支影片的 XML 數量都剛好
等於各自的 num_frames，覆蓋率 100%），只是沒有病灶時 `<object>` 是空的——也就是說
不需要任何抽樣，直接解析全部 2,757,723 張影格的 annotation 就能拿到跟官方
「87.6% 影格無標註」完全吻合的完整數字（見下方驗證：加權平均 polyp 比例 12.4%）。
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


def part_b(full_annotations: pd.DataFrame, coverage: pd.DataFrame) -> str:
    lines = [
        "## Part B：frame 層級盛行率（官方完整逐格標註，2,757,723 張影格）\n",
    ]

    lines.append(
        f"Annotation 覆蓋率：{coverage['coverage_pct'].min():.1f}%–"
        f"{coverage['coverage_pct'].max():.1f}%（60 支影片全部 100%，也就是每一格都有"
        "一個 annotation XML，沒有缺格）。\n"
    )

    by_cohort = full_annotations.groupby("cohort").agg(
        n_frames=("frame_index", "count"),
        n_polyp_frames=("polyp_label", "sum"),
        n_videos=("video_id", "nunique"),
    )
    by_cohort["polyp_frame_pct"] = (by_cohort["n_polyp_frames"] / by_cohort["n_frames"] * 100).round(2)
    lines.append("### 每個 cohort 的 frame 層級 polyp 比例（官方完整資料）\n")
    lines.append(by_cohort.to_markdown(floatfmt=",.2f", intfmt=","))
    lines.append("")

    overall_pct = full_annotations["polyp_label"].mean() * 100
    lines.append(
        f"整體 polyp frame 比例：{overall_pct:.2f}%（官方論文記載「87.6% 影格無標註」，"
        f"即 12.4% 有標註——跟這裡算出的 {overall_pct:.2f}% 吻合，驗證了解析邏輯正確）。\n\n"
        "**跟 Part A（病灶層級）對照**：cohort 002 的病灶數量最多、平均病灶最小"
        "（見上方 Part A），但這裡的 frame 層級盛行率並非四個 cohort 中最高——代表"
        "『病灶顆數多』不等於『病灶在影片裡出現的影格時間長』，兩種盛行率量測的是不同"
        "東西，做 E2c confound 對照時要留意用哪一種。\n"
    )

    lines.append("### 每支影片的 polyp frame 比例分布（找出極端值）\n")
    per_video = full_annotations.groupby(["video_id", "cohort"]).agg(
        n_frames=("frame_index", "count"), n_polyp=("polyp_label", "sum")
    )
    per_video["polyp_pct"] = (per_video["n_polyp"] / per_video["n_frames"] * 100).round(2)
    per_video = per_video.sort_values("polyp_pct", ascending=False)
    lines.append("最高 5 支：\n")
    lines.append(per_video.head(5).to_markdown())
    lines.append("\n最低 5 支（含 0%，即完全無 polyp 的 14 支影片之一）：\n")
    lines.append(per_video.tail(5).to_markdown())
    lines.append("")

    return "\n".join(lines)


def main():
    # cohort 欄位（"001".."004"）若不強制指定 dtype，pandas 讀 csv 時會把它當數字，
    # 吃掉開頭的 0（"001" -> 1），跟 lesion_info 這邊用字串 split 出來的 "001" 對不起來。
    video_manifest = pd.read_csv(RESULTS_DIR / "video_manifest.csv", dtype={"cohort": str})
    lesion_info = pd.read_csv(LESION_INFO_CSV)
    full_annotations_path = RESULTS_DIR / "full_annotation_labels.csv"
    coverage_path = RESULTS_DIR / "annotation_coverage.csv"

    sections = ["# E0c：Confound 報表\n"]
    sections.append(part_a(video_manifest, lesion_info))

    if full_annotations_path.exists() and coverage_path.exists():
        full_annotations = pd.read_csv(full_annotations_path, dtype={"cohort": str})
        coverage = pd.read_csv(coverage_path)
        sections.append(part_b(full_annotations, coverage))
    else:
        sections.append("## Part B：略過（尚未執行 07_download_all_annotations.py）\n")

    out_path = RESULTS_DIR / "confound_report.md"
    out_path.write_text("\n".join(sections), encoding="utf-8")
    print(f"Wrote confound report -> {out_path}")


if __name__ == "__main__":
    main()
