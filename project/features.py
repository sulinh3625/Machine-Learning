from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis, skew

from .constants import FEATURE_COLUMNS


def correlation_feature(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_centered = left - left.mean(axis=1, keepdims=True)
    right_centered = right - right.mean(axis=1, keepdims=True)
    numerator = (left_centered * right_centered).sum(axis=1)
    denominator = np.sqrt((left_centered**2).sum(axis=1) * (right_centered**2).sum(axis=1)) + 1e-8
    return numerator / denominator


def extract_statistical_features(X: np.ndarray) -> tuple[np.ndarray, list[str]]:
    mean = X.mean(axis=1)
    std = X.std(axis=1)
    minimum = X.min(axis=1)
    maximum = X.max(axis=1)
    median = np.median(X, axis=1)
    energy = np.mean(np.square(X), axis=1)
    abs_mean = np.mean(np.abs(X), axis=1)
    iqr = np.percentile(X, 75, axis=1) - np.percentile(X, 25, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        skewness = np.nan_to_num(skew(X, axis=1, bias=False), nan=0.0, posinf=0.0, neginf=0.0)
        kurt = np.nan_to_num(
            kurtosis(X, axis=1, fisher=True, bias=False), nan=0.0, posinf=0.0, neginf=0.0
        )

    back_magnitude = np.linalg.norm(X[:, :, :3], axis=2)
    thigh_magnitude = np.linalg.norm(X[:, :, 3:], axis=2)
    same_axis_corr = np.column_stack(
        [
            correlation_feature(X[:, :, 0], X[:, :, 3]),
            correlation_feature(X[:, :, 1], X[:, :, 4]),
            correlation_feature(X[:, :, 2], X[:, :, 5]),
        ]
    )
    magnitude_features = np.column_stack(
        [
            back_magnitude.mean(axis=1),
            back_magnitude.std(axis=1),
            thigh_magnitude.mean(axis=1),
            thigh_magnitude.std(axis=1),
        ]
    )

    blocks = [
        ("mean", mean),
        ("std", std),
        ("min", minimum),
        ("max", maximum),
        ("median", median),
        ("energy", energy),
        ("abs_mean", abs_mean),
        ("iqr", iqr),
        ("skew", skewness),
        ("kurtosis", kurt),
    ]
    feature_parts = [block for _, block in blocks]
    feature_names = [f"{prefix}_{column}" for prefix, block in blocks for column in FEATURE_COLUMNS]

    feature_parts.append(same_axis_corr)
    feature_names.extend(["corr_back_thigh_x", "corr_back_thigh_y", "corr_back_thigh_z"])

    feature_parts.append(magnitude_features)
    feature_names.extend(["back_mag_mean", "back_mag_std", "thigh_mag_mean", "thigh_mag_std"])

    return np.concatenate(feature_parts, axis=1).astype(np.float32), feature_names
