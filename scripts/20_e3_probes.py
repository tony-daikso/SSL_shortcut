"""E3：整合三條曲線 + SSIM 對照，產生 dose-response 報告。

三條曲線（跨 e3_augmentation_common.JITTER_STRENGTHS 5 個強度）：
- pattern_noise（低階/高頻合成指紋，模擬感測器 PRNU）：機制預測幾乎不受 jitter 影響。
- color_shift（高階/低頻合成指紋，模擬白平衡系統性色偏）：機制預測被壓到接近 chance。
- polyp_label（真實病理訊號，全部 60 支影片規模）：這是「augmentation 有沒有連帶
  傷害到診斷訊號」的直接量測，計畫最關心的臨床後果。

判讀依據計畫 §7「陰性結果的處置」的四種形態分類。
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from config import RESULTS_DIR

N_REPEATS = 10


def probe_by_group(X, y, groups, n_repeats=N_REPEATS):
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
    return float(accs.mean()), float(accs.std())


def probe_held_out_frame(X, y, n_repeats=N_REPEATS):
    accs = []
    for seed in range(n_repeats):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        train_idx, test_idx = next(sss.split(X, y))
        scaler = StandardScaler().fit(X[train_idx])
        clf = LogisticRegression(max_iter=2000)
        clf.fit(scaler.transform(X[train_idx]), y[train_idx])
        pred = clf.predict(scaler.transform(X[test_idx]))
        accs.append((pred == y[test_idx]).mean())
    accs = np.array(accs)
    return float(accs.mean()), float(accs.std())


def main():
    synth = np.load(RESULTS_DIR / "e3_synthetic_embeddings.npz", allow_pickle=True)
    real = np.load(RESULTS_DIR / "e3_real_embeddings.npz", allow_pickle=True)
    ssim_df = pd.read_csv(RESULTS_DIR / "e3_ssim_control.csv")

    synth_df = pd.DataFrame({
        "video_id": synth["video_id"], "synthetic_class": synth["synthetic_class"],
        "mechanism": synth["mechanism"], "strength": synth["strength"],
    })
    synth_emb = synth["embeddings"]

    real_df = pd.DataFrame({
        "video_id": real["video_id"], "polyp_label": real["polyp_label"], "strength": real["strength"],
    })
    real_emb = real["embeddings"]

    strengths = sorted(set(synth_df["strength"].tolist()))
    rows = []
    for strength in strengths:
        row = {"strength": strength}
        for mechanism in ["pattern_noise", "color_shift"]:
            mask = (synth_df["mechanism"] == mechanism) & (synth_df["strength"] == strength)
            X = synth_emb[mask.values]
            y = synth_df.loc[mask, "synthetic_class"].values
            groups = synth_df.loc[mask, "video_id"].values
            mean_acc, std_acc = probe_by_group(X, y, groups)
            row[f"{mechanism}_acc"] = mean_acc
            row[f"{mechanism}_std"] = std_acc

        mask = real_df["strength"] == strength
        X = real_emb[mask.values]
        y = real_df.loc[mask, "polyp_label"].values
        mean_acc, std_acc = probe_held_out_frame(X, y)
        row["polyp_acc"] = mean_acc
        row["polyp_std"] = std_acc

        ssim_row = ssim_df[ssim_df["strength"] == strength]["ssim"]
        row["ssim_mean"] = float(ssim_row.mean())

        rows.append(row)

    curve = pd.DataFrame(rows)

    chance_4way = 0.25
    polyp_majority = pd.Series(real_df["polyp_label"]).value_counts(normalize=True).max()

    lines = ["# E3：Augmentation 劑量反應（全部 60 支影片 + E0.5 合成指紋，正式結論）\n"]
    lines.append(
        "三條曲線隨 color jitter 強度變化：pattern_noise（合成，低階/高頻，4-way，"
        f"chance={chance_4way:.2f}）、color_shift（合成，高階/低頻，4-way，"
        f"chance={chance_4way:.2f}）、polyp_label（真實，全部 60 支影片，held-out "
        f"frame，majority={polyp_majority:.3f}）。strength=0 對應無 jitter（跟 "
        "`results/e05_report.md` 的無 jitter 數字互為校驗）。\n"
    )
    lines.append(curve.round(4).to_markdown(index=False))
    lines.append("")

    pn_start, pn_end = curve["pattern_noise_acc"].iloc[0], curve["pattern_noise_acc"].iloc[-1]
    cs_start, cs_end = curve["color_shift_acc"].iloc[0], curve["color_shift_acc"].iloc[-1]
    polyp_start, polyp_end = curve["polyp_acc"].iloc[0], curve["polyp_acc"].iloc[-1]
    pn_drop = pn_start - pn_end
    cs_drop = cs_start - cs_end
    polyp_drop = polyp_start - polyp_end
    ssim_min = curve["ssim_mean"].min()

    lines.append("## 解讀\n")
    lines.append(
        f"**pattern_noise**：{pn_start:.3f} → {pn_end:.3f}（掉了 {pn_drop:.3f}）。\n\n"
        f"**color_shift**：{cs_start:.3f} → {cs_end:.3f}（掉了 {cs_drop:.3f}）。\n\n"
        f"**polyp_label**：{polyp_start:.3f} → {polyp_end:.3f}（掉了 {polyp_drop:.3f}，"
        f"majority baseline {polyp_majority:.3f}）。\n\n"
        f"**SSIM 對照（E3c，trivial-destruction control）**：最高強度下 SSIM 均值 "
        f"{curve['ssim_mean'].iloc[-1]:.3f}（全程最低點 {ssim_min:.3f}）。\n"
    )

    cs_drops_to_chance = cs_end < chance_4way + 0.10
    pn_stays_flat = pn_drop < 0.10
    ssim_high = ssim_min > 0.7

    if pn_stays_flat and cs_drops_to_chance:
        pattern_verdict = (
            "**『一條降一條平』的機制性預測在真實強度 sweep 上成立**：pattern_noise "
            "幾乎不受影響、color_shift 隨強度增加持續掉到接近 chance。跟 E0.5c 在單一"
            "強度下的結論一致，這裡進一步證明這個機制在整個劑量範圍內都穩固，不是"
            "單一校準點的巧合。"
        )
    elif not pn_stays_flat and not cs_drops_to_chance:
        pattern_verdict = "兩條曲線都沒有明顯變化——如果 SSIM 也顯示影像幾乎沒被破壞，代表這個強度範圍還不夠大，需要往上加大 strength 網格；如果 SSIM 也掉很多，需要先懷疑量測管線（回頭檢查 E0.5）。"
    elif not pn_stays_flat and cs_drops_to_chance:
        pattern_verdict = "兩條曲線都降到接近 chance——機制論證被推翻，代表 color jitter 這類全域色彩變換其實同時觸及了低階與高階指紋，跟原本『jitter 參數空間動不到雜訊空間相關結構』的假設不符，需要在報告裡誠實記錄這個修正。"
    else:
        pattern_verdict = "pattern_noise 反而比 color_shift 掉得更多，方向與機制預測相反，需要檢查是否強度網格/probe 設定有問題。"
    lines.append(f"\n{pattern_verdict}\n")

    if polyp_drop < 0.05:
        polyp_verdict = (
            f"**病理訊號幾乎沒有代價**（掉了 {polyp_drop:.3f}）：在這個強度範圍內，"
            "color jitter 對 polyp 可分性的影響遠小於對 color_shift 指紋的影響——"
            "跟計畫 §7 的『唯一真正死亡條件』（指紋降到 chance 而病理幾乎不掉）方向"
            "一致，代表標準 augmentation 有機會在不犧牲太多診斷訊號的前提下，至少"
            "壓制掉高階色彩類的採集指紋。"
        )
    elif polyp_drop < cs_drop:
        polyp_verdict = (
            f"**病理訊號也有代價，但小於 color_shift 指紋的掉幅**（病理掉 "
            f"{polyp_drop:.3f} vs color_shift 掉 {cs_drop:.3f}）：代表存在一個 "
            "trade-off——augmentation 強度越大，指紋壓得越乾淨，但診斷訊號也跟著"
            "流失一部分，需要量化這個權衡邊界（在什麼強度下指紋降到可接受水準、"
            "病理代價還沒超過門檻）。"
        )
    else:
        polyp_verdict = (
            f"**病理訊號的掉幅（{polyp_drop:.3f}）不小於甚至超過 color_shift 指紋"
            f"的掉幅（{cs_drop:.3f}）**：代表這個強度範圍內，color jitter 對病理"
            "訊號的傷害跟對高階指紋的壓制至少一樣大，機制性論證的『安全空間』"
            "很窄甚至不存在，需要更謹慎地選擇實務上可用的 jitter 強度。"
        )
    lines.append(f"{polyp_verdict}\n")

    lines.append("## E3c：trivial-destruction control\n")
    lines.append(
        f"{'✅ 排除' if ssim_high else '⚠️ 無法完全排除'}：最高強度下 SSIM 最低值為 "
        f"{ssim_min:.3f}（門檻 0.7）。"
        + ("影像在整個強度範圍內都維持高度結構相似，上面三條曲線的變化差異可以"
           "歸因於訊號本身對 jitter 的敏感度不同，不是影像被整體破壞。\n"
           if ssim_high else
           "SSIM 在高強度下明顯偏低，代表影像結構已經有一定程度被破壞，曲線的掉幅"
           "需要謹慎解讀，不能完全排除是「圖爛了」而非「機制性選擇壓制」。\n")
    )

    out_path = RESULTS_DIR / "e3_dose_response_report.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}")
    print(curve)


if __name__ == "__main__":
    main()
