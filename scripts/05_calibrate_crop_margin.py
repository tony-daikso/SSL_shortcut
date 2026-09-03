"""校準 fov_protocol.INSET_FRACTION：中央方形裁切之後，還要往內裁掉多少邊距，才能
把角落 FOV 遮罩殘留壓到接近乾淨。背景與發現過程見 fov_protocol.py 開頭的修正記錄。

這支腳本本身不是 E0e 的正式產物，是產生「為什麼 INSET_FRACTION 選這個值」這個答案的
過程，跑一次記錄下來即可，之後校準結果直接寫死在 fov_protocol.py 裡使用。讀取的是
`pilot_frame_labels.csv`（官方原始資料，見 10_merge_pilot_labels.py），不是任何專案
篩選過的子集。
"""

import random

import numpy as np
import pandas as pd
from PIL import Image

from config import RESULTS_DIR, REPO_ROOT
from fov_protocol import corner_black_fraction

CANDIDATE_INSETS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
N_SAMPLE = 150


def unify_square_crop_with_inset(image, inset_fraction):
    w, h = image.size
    side = min(w, h)
    left, top = (w - side) // 2, (h - side) // 2
    sq = image.crop((left, top, left + side, top + side))
    inset = int(side * inset_fraction)
    if inset > 0:
        sq = sq.crop((inset, inset, side - inset, side - inset))
    return sq


def main():
    frame_labels = pd.read_csv(RESULTS_DIR / "pilot_frame_labels.csv", dtype={"cohort": str})
    sample = frame_labels.sample(n=min(N_SAMPLE, len(frame_labels)), random_state=0)

    images = []
    for _, r in sample.iterrows():
        p = REPO_ROOT / r["frame_path"]
        if p.exists():
            images.append(Image.open(p))

    rows = []
    for inset in CANDIDATE_INSETS:
        cbf = np.array([corner_black_fraction(unify_square_crop_with_inset(im, inset)) for im in images])
        rows.append({
            "inset_fraction": inset, "n_frames": len(cbf),
            "mean_corner_black": cbf.mean(), "max_corner_black": cbf.max(),
            "frac_frames_with_residual_gt_0.5pct": float((cbf > 0.005).mean()),
        })

    df = pd.DataFrame(rows)
    out_path = RESULTS_DIR / "fov_crop_margin_calibration.md"
    lines = [
        "# 裁切邊距校準（fov_protocol.INSET_FRACTION 的依據）\n",
        f"對 {len(images)} 張隨機抽樣影格，在「中央方形裁切」之後測試不同 inset，量測角落"
        "黑色像素殘留比例：\n",
        df.to_markdown(index=False),
        "\n**選擇 0.15**（見 `fov_protocol.INSET_FRACTION`）：inset 從 0 加到 0.05 就讓"
        "平均殘留驟降，之後緩慢下降、殘留影格比例在小範圍內震盪（推測是少數影格本身"
        "邊緣就有暗部內容，不是遮罩，inset 再大也無法消除）。選比最小可行值稍微保守"
        "一點的 0.15，同時避免為了追殺極少數離群影格犧牲太多可用內容。",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(df.to_string(index=False))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
