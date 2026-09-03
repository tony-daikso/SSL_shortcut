"""E0e-4：驗證統一裁切協定有效——套用 fov_protocol.unify_crop 後重跑 E0e-2 的 trivial
baseline，確認幾何特徵真的無法再預測 cohort/brand。

讀取 `pilot_frame_labels.csv`（官方原始資料，見 10_merge_pilot_labels.py）。對每張
影格實際讀圖、裁切、resize，重新量測寬高（理論上裁完後全部影格的 width/height 會是
同一個常數，這裡仍然實際跑一次而不是純推論，順便檢查有沒有讀檔/裁切邏輯的錯誤），
並且重新計算 corner_black_fraction——這是真正需要驗證的部分，因為 width/height/
aspect_ratio/area 裁完後必然變成常數，分類器在定義上就用不上，唯一可能還殘留訊號的
是角落遮罩有沒有真的被裁乾淨（見 05_calibrate_crop_margin.py 校準 INSET_FRACTION 的
過程）。同時存幾組裁切前後的對照圖，方便肉眼確認沒有把黏膜主體切掉太多。

**這是 pilot 版本**（5 支影片），brand within-cohort-002 control 目前只有
002-004（Olympus）、002-006（Fujifilm）兩支影片可用，leave-one-video-out 只有兩折，
統計效力有限，主要目的是驗證流程正確。
"""

import pandas as pd
from PIL import Image

from config import REPO_ROOT, RESULTS_DIR
from fov_protocol import TARGET_SIZE, INSET_FRACTION, unify_crop, corner_black_fraction
from geometry_common import add_geometry_features, train_test_split_baseline, leave_one_video_out_baseline

N_QC_SAMPLES = 6
# 裁切後 width/height/aspect_ratio/area 是常數，對分類器沒有意義，真正要驗證的是
# corner_black_fraction 有沒有被裁乾淨。
POST_CROP_FEATURES = ["corner_black_fraction"]


def main():
    frame_labels = pd.read_csv(RESULTS_DIR / "pilot_frame_labels.csv", dtype={"cohort": str})

    qc_dir = RESULTS_DIR / "qc_unify_crop_samples"
    qc_dir.mkdir(parents=True, exist_ok=True)
    qc_rows = frame_labels.sample(n=N_QC_SAMPLES, random_state=0)

    widths, heights, corner_fracs = [], [], []
    for _, r in frame_labels.iterrows():
        img_path = REPO_ROOT / r["frame_path"]
        if not img_path.exists():
            widths.append(None)
            heights.append(None)
            corner_fracs.append(None)
            continue
        im = Image.open(img_path)
        cropped = unify_crop(im)
        widths.append(cropped.size[0])
        heights.append(cropped.size[1])
        corner_fracs.append(corner_black_fraction(cropped))
        if r["frame_id"] in qc_rows["frame_id"].values:
            im.save(qc_dir / f"{r['frame_id']}_before_{im.size[0]}x{im.size[1]}.jpg")
            cropped.save(qc_dir / f"{r['frame_id']}_after_{TARGET_SIZE}x{TARGET_SIZE}.jpg")

    post = frame_labels.copy()
    post["width"] = widths
    post["height"] = heights
    post["corner_black_fraction"] = corner_fracs
    post = post.dropna(subset=["width", "height", "corner_black_fraction"])
    post = add_geometry_features(post)

    lines = ["# E0e-4：統一裁切協定驗證（pilot，5 支影片）\n"]
    lines.append(
        f"套用 `fov_protocol.unify_crop`（中央方形裁切 → 再裁掉 {INSET_FRACTION:.0%} 邊距 "
        f"→ resize 到 {TARGET_SIZE}x{TARGET_SIZE}，邊距比例見 `05_calibrate_crop_margin.py` "
        f"的校準結果）。對 {len(post)} 張影格重新量測尺寸：發現 "
        f"{post[['width','height']].drop_duplicates().shape[0]} 種不同的 (width, height) "
        "組合（預期剛好是 1 種常數，驗證裁切協定對所有影格一視同仁）。\n"
    )
    lines.append(
        f"角落殘留（corner_black_fraction）裁切後統計：mean={post['corner_black_fraction'].mean():.5f}，"
        f"median={post['corner_black_fraction'].median():.5f}，"
        f"max={post['corner_black_fraction'].max():.5f}（裁切前統計見 "
        "`fov_e0e_baseline_report.md`）。\n"
    )

    lines.append("## Cohort trivial baseline（只用 corner_black_fraction，裁切後）\n")
    cohort_after = train_test_split_baseline(post, "cohort", POST_CROP_FEATURES)
    lines.append(pd.Series(cohort_after).to_frame("裁切後").to_markdown())
    lines.append(
        "\n對照 E0e-2（裁切前，size+corner 特徵）：accuracy 見 `fov_e0e_baseline_report.md`。"
        f"裁切後（只剩 corner_black_fraction 這一個特徵，因為尺寸類特徵已變成常數）："
        f"accuracy {cohort_after['accuracy']:.3f}（chance {cohort_after['uniform_chance']:.3f}、"
        f"majority baseline {cohort_after['majority_baseline']:.3f}）。\n"
    )

    lines.append("## Endoscope brand within-cohort-002 control（只用 corner_black_fraction，裁切後）\n")
    cohort002_after = post[post["cohort"] == "002"]
    brand_after = leave_one_video_out_baseline(cohort002_after, "endoscope_brand", POST_CROP_FEATURES)
    lines.append(pd.Series(brand_after).to_frame("裁切後").to_markdown())
    lines.append(
        "\n對照見 `fov_e0e_baseline_report.md`。裁切後："
        f"accuracy {brand_after['accuracy']:.3f}（majority baseline "
        f"{brand_after['majority_baseline']:.3f}）。**pilot 裡 cohort 002 只有 2 支影片"
        "（各一種品牌），leave-one-video-out 只有兩折，數字僅供流程驗證，不是正式結論。**\n"
    )

    # 判準用 majority_baseline 而非 uniform_chance：分類器學不到任何東西時，理論上就是
    # 永遠猜多數類別，準確率等於 majority_baseline，這才是正確的比較基準（跟 Phase 0
    # 對 device brand 任務的判讀方式一致，見 SSL_research 的 SESSION_LOG）。容忍 1% 誤差，
    # 因為 corner_black_fraction 是連續值，殘留的極小量測雜訊可能讓分類器學到一點雜訊。
    tol = 0.01
    gate_pass = (
        abs(cohort_after["accuracy"] - cohort_after["majority_baseline"]) < tol
        and abs(brand_after["accuracy"] - brand_after["majority_baseline"]) < tol
    )
    lines.append(
        f"## 閘門判定：{'✅ 通過' if gate_pass else '❌ 未通過'}\n\n"
        + ("裁切後兩個 trivial baseline 的準確率都落在 majority baseline 附近（誤差 <1%），"
           "符合計畫 E0e-4 的驗收標準，統一裁切協定確實把角落幾何線索壓到接近無法利用"
           "的程度。（pilot 樣本數小，正式結論待全部 60 支影片跑完。）\n"
           if gate_pass else
           "裁切後仍有明顯高於 majority baseline 的殘留訊號，代表裁切協定不夠、還有"
           "其他幾何線索沒被移除，需要回頭檢查（例如加大 INSET_FRACTION 或改進裁切"
           "邏輯）。\n")
    )

    lines.append(f"\n裁切前後對照樣本圖已存到 `results/qc_unify_crop_samples/`（{N_QC_SAMPLES} 組）。\n")

    out_path = RESULTS_DIR / "fov_e0e4_verification.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}")
    print("cohort after crop:", cohort_after)
    print("brand within-cohort-002 after crop:", brand_after)
    print("gate pass:", gate_pass)


if __name__ == "__main__":
    main()
