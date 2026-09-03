"""E0.5a/b/c：用 frozen DINOv2 embedding 訓練 linear probe，檢驗合成指紋校準結果。

判準（研究計畫 §6 E0.5）：
- E0.5a：probe 能不能抓到注入的合成指紋（pattern_noise、color_shift 各自，
  沒有套用 color jitter 的版本）。抓不到 → probe 太弱，不該拿去跑真實資料。
- E0.5b：等強度注入下，pattern_noise probe 跟 color_shift probe 的靈敏度
  （accuracy）是否對等。不對等 → 之後 E3「一條降一條平」的結果可能只是兩個
  probe 靈敏度本來就不同造成的假象，不是真的機制差異。
- E0.5c：套用 color jitter 後，color_shift 應該被壓下去（accuracy 掉到接近
  chance）、pattern_noise 應該幾乎不受影響（accuracy 維持）。不成立 → 機制論證
  在合成資料上就站不住腳。

用 video_id 分組的 GroupShuffleSplit 切 train/test（避免同一支來源影片的底圖同時
出現在兩邊），4-way 分類，chance = 25%。
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from config import RESULTS_DIR

N_REPEATS = 10


def probe_accuracy(X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_repeats: int = N_REPEATS) -> dict:
    accs = []
    for seed in range(n_repeats):
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        train_idx, test_idx = next(gss.split(X, y, groups=groups))
        scaler = StandardScaler().fit(X[train_idx])
        clf = LogisticRegression(max_iter=2000)
        clf.fit(scaler.transform(X[train_idx]), y[train_idx])
        pred = clf.predict(scaler.transform(X[test_idx]))
        accs.append((pred == y[test_idx]).mean())
    accs = np.array(accs)
    return {"mean_accuracy": float(accs.mean()), "std_accuracy": float(accs.std()), "n_repeats": n_repeats}


def main():
    data = np.load(RESULTS_DIR / "e05_embeddings.npz", allow_pickle=True)
    df = pd.DataFrame({
        "frame_id": data["frame_id"], "video_id": data["video_id"],
        "synthetic_class": data["synthetic_class"], "variant": data["variant"],
    })
    embeddings = data["embeddings"]
    chance = 1.0 / df["synthetic_class"].nunique()

    lines = ["# E0.5：合成指紋校準結果\n"]
    lines.append(f"4-way 分類，chance level = {chance:.3f}。GroupShuffleSplit 依 video_id 分組，"
                 f"{N_REPEATS} 次不同 random seed 取平均 ± 標準差。\n")

    results = {}
    for variant in ["clean", "pattern_noise", "color_shift", "pattern_noise_jitter", "color_shift_jitter"]:
        mask = df["variant"] == variant
        X = embeddings[mask.values]
        y = df.loc[mask, "synthetic_class"].values
        groups = df.loc[mask, "video_id"].values
        results[variant] = probe_accuracy(X, y, groups)

    summary = pd.DataFrame(results).T
    summary["chance"] = chance
    lines.append("## 各變體的 probe 準確率\n")
    lines.append(summary.to_markdown())
    lines.append("")

    lines.append("## E0.5a：probe 抓不抓得到注入的合成指紋\n")
    a_pass = results["pattern_noise"]["mean_accuracy"] > chance + 0.15 and results["color_shift"]["mean_accuracy"] > chance + 0.15
    lines.append(
        f"pattern_noise（無 jitter）：{results['pattern_noise']['mean_accuracy']:.3f} ± "
        f"{results['pattern_noise']['std_accuracy']:.3f}；"
        f"color_shift（無 jitter）：{results['color_shift']['mean_accuracy']:.3f} ± "
        f"{results['color_shift']['std_accuracy']:.3f}；"
        f"對照 clean（不該有任何可分性，因為合成類別跟內容無關）："
        f"{results['clean']['mean_accuracy']:.3f} ± {results['clean']['std_accuracy']:.3f}。\n\n"
        f"**{'✅ 通過' if a_pass else '❌ 未通過'}**："
        + ("兩種注入的合成指紋都遠高於 chance，probe 抓得到，可以繼續 E0.5b/c。\n"
           if a_pass else "至少一種訊號 probe 抓不到，代表 probe 太弱或注入強度不夠，"
           "不該拿這個 probe 設定去跑真實資料。\n")
    )

    lines.append("## E0.5b：probe 容量匹配（等強度下兩種訊號的靈敏度是否對等）\n")
    diff = abs(results["pattern_noise"]["mean_accuracy"] - results["color_shift"]["mean_accuracy"])
    b_pass = diff < 0.10
    lines.append(
        f"兩者 accuracy 差距：{diff:.3f}（pattern_noise {results['pattern_noise']['mean_accuracy']:.3f} vs "
        f"color_shift {results['color_shift']['mean_accuracy']:.3f}）。\n\n"
        f"**{'✅ 通過（差距 <0.10，視為大致對等）' if b_pass else '❌ 未通過（差距過大，需要重新校準注入強度）'}**\n"
    )

    lines.append("## E0.5c：機制驗證（color jitter 應該壓下 color_shift、不該壓下 pattern_noise）\n")
    pn_drop = results["pattern_noise"]["mean_accuracy"] - results["pattern_noise_jitter"]["mean_accuracy"]
    cs_drop = results["color_shift"]["mean_accuracy"] - results["color_shift_jitter"]["mean_accuracy"]
    c_pass = (cs_drop > 0.15) and (pn_drop < 0.10)
    lines.append(
        f"pattern_noise：{results['pattern_noise']['mean_accuracy']:.3f} → "
        f"{results['pattern_noise_jitter']['mean_accuracy']:.3f}（jitter 後掉了 {pn_drop:.3f}，"
        "預期應該幾乎不掉）。\n\n"
        f"color_shift：{results['color_shift']['mean_accuracy']:.3f} → "
        f"{results['color_shift_jitter']['mean_accuracy']:.3f}（jitter 後掉了 {cs_drop:.3f}，"
        "預期應該明顯掉、趨近 chance）。\n\n"
        f"**{'✅ 通過' if c_pass else '❌ 未通過'}**："
        + ("『一條降一條平』的機制性預測在合成資料上成立，代表 E3 之後在真實資料上做"
           "同樣的 dose-response 分析，這個機制性論證的邏輯本身是站得住腳的。\n"
           if c_pass else
           "機制性預測沒有在合成資料上重現，需要檢查是不是注入強度、jitter 強度、或"
           "probe 本身有問題，再往下走 E3 會缺乏立論基礎。\n")
    )

    out_path = RESULTS_DIR / "e05_report.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}")
    print(summary)
    print("E0.5a pass:", a_pass, "| E0.5b pass:", b_pass, "| E0.5c pass:", c_pass)


if __name__ == "__main__":
    main()
