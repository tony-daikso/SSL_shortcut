"""E0e 共用工具：幾何特徵定義 + trivial baseline 分類器。

「FOV 幾何」在這個資料集上有兩個獨立來源：影格本身的像素尺寸（width/height/
aspect_ratio/area）、以及角落 FOV 遮罩殘留（corner_black_fraction，見
fov_protocol.py——這個是修正過一次的，早期用「整行/整列全黑」檢查誤判成幾乎不存在，
實際上普遍存在）。

train_test_split_baseline 不依賴預先算好的 split 欄位，而是每次呼叫時用
GroupShuffleSplit（依 video_id 分組）現場切 train/test，避免不小心誤用某個特定用途
（例如息肉偵測 benchmark）的官方切分慣例（見 config.py 的修正記錄）。
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

GEOMETRY_FEATURES = ["width", "height", "aspect_ratio", "log_area"]


def add_geometry_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["aspect_ratio"] = df["width"] / df["height"]
    df["log_area"] = np.log(df["width"].astype(float) * df["height"].astype(float))
    return df


def _fit_predict(X_train, y_train, X_test):
    scaler = StandardScaler().fit(X_train)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(scaler.transform(X_train), y_train)
    return clf.predict(scaler.transform(X_test))


def train_test_split_baseline(
    df: pd.DataFrame, label_col: str, feature_cols=GEOMETRY_FEATURES,
    test_size: float = 0.2, random_state: int = 0,
) -> dict:
    """依 video_id 分組隨機切 train/test（GroupShuffleSplit，固定 random_state 可重現），
    在完全沒看過的影片上評估——不是套用官方切分慣例，見本檔開頭說明。"""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(df, groups=df["video_id"]))
    train, test = df.iloc[train_idx], df.iloc[test_idx]

    X_train, y_train = train[feature_cols].values, train[label_col].values
    X_test, y_test = test[feature_cols].values, test[label_col].values

    pred = _fit_predict(X_train, y_train, X_test)
    acc = float((pred == y_test).mean())

    majority_class = pd.Series(y_train).mode().iloc[0]
    majority_acc = float((y_test == majority_class).mean())
    chance = 1.0 / test[label_col].nunique()

    return {
        "n_train_frames": len(train), "n_test_frames": len(test),
        "n_train_videos": train["video_id"].nunique(), "n_test_videos": test["video_id"].nunique(),
        "accuracy": acc, "majority_baseline": majority_acc, "uniform_chance": chance,
    }


def train_test_split_baseline_repeated(
    df: pd.DataFrame, label_col: str, feature_cols=GEOMETRY_FEATURES,
    test_size: float = 0.2, n_repeats: int = 10,
) -> dict:
    """`train_test_split_baseline` 的多次重複版：不同 random_state 各切一次
    train/test 取平均 ± 標準差，避免單一 split 剛好抽到極端影片組合造成的
    偶然結果。資料量夠大（例如全部 60 支影片）時應該用這個而不是單一 split。"""
    accs, majority_accs = [], []
    n_train_videos = n_test_videos = n_train_frames = n_test_frames = None
    valid_repeats = 0
    for seed in range(n_repeats):
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(gss.split(df, groups=df["video_id"]))
        train, test = df.iloc[train_idx], df.iloc[test_idx]
        if train[label_col].nunique() < 2:
            continue

        X_train, y_train = train[feature_cols].values, train[label_col].values
        X_test, y_test = test[feature_cols].values, test[label_col].values
        pred = _fit_predict(X_train, y_train, X_test)
        accs.append(float((pred == y_test).mean()))

        majority_class = pd.Series(y_train).mode().iloc[0]
        majority_accs.append(float((y_test == majority_class).mean()))
        n_train_videos, n_test_videos = train["video_id"].nunique(), test["video_id"].nunique()
        n_train_frames, n_test_frames = len(train), len(test)
        valid_repeats += 1

    chance = 1.0 / df[label_col].nunique()
    return {
        "n_train_frames": n_train_frames, "n_test_frames": n_test_frames,
        "n_train_videos": n_train_videos, "n_test_videos": n_test_videos,
        "mean_accuracy": float(np.mean(accs)) if accs else float("nan"),
        "std_accuracy": float(np.std(accs)) if accs else float("nan"),
        "mean_majority_baseline": float(np.mean(majority_accs)) if majority_accs else float("nan"),
        "uniform_chance": chance, "n_valid_repeats": valid_repeats, "n_repeats": n_repeats,
    }


def leave_one_video_out_baseline(df: pd.DataFrame, label_col: str, feature_cols=GEOMETRY_FEATURES) -> dict:
    """留一支影片出來測試，輪流跑完所有影片（適合資料量小、想榨乾樣本數的情境，例如
    cohort 002 內部的 brand within-cohort control）。"""
    logo = LeaveOneGroupOut()
    X = df[feature_cols].values
    y = df[label_col].values
    groups = df["video_id"].values

    correct = 0
    total = 0
    for train_idx, test_idx in logo.split(X, y, groups):
        if len(set(y[train_idx])) < 2:
            continue  # 訓練集裡只剩一個類別，無法訓練有意義的分類器，跳過這一折
        pred = _fit_predict(X[train_idx], y[train_idx], X[test_idx])
        correct += (pred == y[test_idx]).sum()
        total += len(test_idx)

    majority_class = pd.Series(y).mode().iloc[0]
    majority_acc = float((y == majority_class).mean())

    return {
        "n_frames": len(df), "n_videos": df["video_id"].nunique(),
        "accuracy": correct / total if total else float("nan"),
        "majority_baseline": majority_acc, "uniform_chance": 1.0 / pd.Series(y).nunique(),
    }
