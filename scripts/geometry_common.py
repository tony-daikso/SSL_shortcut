"""E0e 共用工具：幾何特徵定義 + trivial baseline 分類器。

E0d 已經發現：這批本地子集影格幾乎沒有殘留的黑色 FOV 遮罩邊框（content bbox 幾乎
等於整張影格，見 04_fov_geometry_baseline.py 開頭 docstring 的驗證），所以「FOV 幾何」
在這批資料上實際上就是**影格本身的像素尺寸**：不需要额外偵測遮罩形狀，寬高/長寬比/
面積就是全部的幾何訊號來源。這個結論本身也記錄在 results/fov_geometry.csv 裡。
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
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


def train_test_split_baseline(df: pd.DataFrame, label_col: str, feature_cols=GEOMETRY_FEATURES) -> dict:
    """用官方 train split 訓練、官方 test split（完全沒看過的影片）評估。"""
    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]
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
