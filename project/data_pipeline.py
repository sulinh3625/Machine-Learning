from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .constants import FEATURE_COLUMNS, LABEL_NAMES


def ensure_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def get_subject_paths(data_dir: Path) -> list[Path]:
    paths = sorted(data_dir.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"Không tìm thấy file CSV trong {data_dir}")
    return paths


def split_subjects(subject_paths: list[Path], seed: int) -> tuple[list[str], list[str], list[str]]:
    subject_names = [path.stem for path in subject_paths]
    rng = random.Random(seed)
    rng.shuffle(subject_names)

    train_count = 16
    val_count = 3
    train_subjects = sorted(subject_names[:train_count])
    val_subjects = sorted(subject_names[train_count : train_count + val_count])
    test_subjects = sorted(subject_names[train_count + val_count :])
    return train_subjects, val_subjects, test_subjects


def majority_vote(label_windows: np.ndarray, class_count: int) -> np.ndarray:
    return np.asarray(
        [np.bincount(window, minlength=class_count).argmax() for window in label_windows],
        dtype=np.int32,
    )


def create_windows(
    features: np.ndarray,
    labels: np.ndarray,
    window_size: int,
    step_size: int,
    class_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(features) < window_size:
        return (
            np.empty((0, window_size, features.shape[1]), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        )

    feature_windows = np.lib.stride_tricks.sliding_window_view(
        features, window_shape=(window_size, features.shape[1])
    )[:, 0]
    feature_windows = feature_windows[::step_size].astype(np.float32, copy=False)

    label_windows = np.lib.stride_tricks.sliding_window_view(labels, window_shape=window_size)[::step_size]
    window_labels = majority_vote(label_windows, class_count)
    return feature_windows, window_labels


def sample_windows(X: np.ndarray, y: np.ndarray, limit: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if len(y) <= limit:
        return X, y
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    target_counts = np.floor(counts / counts.sum() * limit).astype(int)
    target_counts = np.maximum(target_counts, 1)
    target_counts = np.minimum(target_counts, counts)

    while target_counts.sum() > limit:
        candidates = np.where(target_counts > 1)[0]
        if len(candidates) == 0:
            break
        reduce_index = candidates[np.argmax(target_counts[candidates])]
        target_counts[reduce_index] -= 1

    while target_counts.sum() < limit:
        capacity = counts - target_counts
        candidates = np.where(capacity > 0)[0]
        if len(candidates) == 0:
            break
        add_index = candidates[np.argmax(capacity[candidates])]
        target_counts[add_index] += 1

    sampled_indices = []
    for class_value, sample_count in zip(classes, target_counts):
        class_indices = np.flatnonzero(y == class_value)
        sampled_indices.append(rng.choice(class_indices, size=int(sample_count), replace=False))

    indices = np.concatenate(sampled_indices)
    indices.sort()
    return X[indices], y[indices]


def load_dataset(config: ExperimentConfig) -> tuple[dict[str, Any], pd.DataFrame]:
    subject_paths = get_subject_paths(config.data_dir)
    train_subjects, val_subjects, test_subjects = split_subjects(subject_paths, config.seed)
    config.train_subjects = tuple(train_subjects)
    config.val_subjects = tuple(val_subjects)
    config.test_subjects = tuple(test_subjects)

    label_values = set()
    subject_frames: list[tuple[str, pd.DataFrame]] = []
    subject_summary: list[dict[str, Any]] = []

    for path in subject_paths:
        frame = pd.read_csv(
            path,
            usecols=FEATURE_COLUMNS + ["label"],
            dtype={**{column: np.float32 for column in FEATURE_COLUMNS}, "label": np.int32},
        )
        subject_name = path.stem
        label_values.update(frame["label"].unique().tolist())
        subject_frames.append((subject_name, frame))
        subject_summary.append({"subject": subject_name, "rows": int(len(frame))})

    label_order = sorted(label_values)
    label_to_index = {label: index for index, label in enumerate(label_order)}
    class_names = [f"{label}: {LABEL_NAMES.get(label, f'Class {label}')}" for label in label_order]

    split_payload: dict[str, list[np.ndarray]] = {"train": [], "val": [], "test": []}
    label_payload: dict[str, list[np.ndarray]] = {"train": [], "val": [], "test": []}

    for subject_name, frame in subject_frames:
        encoded_labels = frame["label"].map(label_to_index).to_numpy(dtype=np.int32)
        subject_features = frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        windows, window_labels = create_windows(
            subject_features,
            encoded_labels,
            config.window_size,
            config.step_size,
            class_count=len(label_order),
        )
        if subject_name in train_subjects:
            split_name = "train"
        elif subject_name in val_subjects:
            split_name = "val"
        else:
            split_name = "test"
        split_payload[split_name].append(windows)
        label_payload[split_name].append(window_labels)

    dataset = {
        "X_train": np.concatenate(split_payload["train"], axis=0),
        "y_train": np.concatenate(label_payload["train"], axis=0),
        "X_val": np.concatenate(split_payload["val"], axis=0),
        "y_val": np.concatenate(label_payload["val"], axis=0),
        "X_test": np.concatenate(split_payload["test"], axis=0),
        "y_test": np.concatenate(label_payload["test"], axis=0),
        "label_order": label_order,
        "class_names": class_names,
        "train_subjects": train_subjects,
        "val_subjects": val_subjects,
        "test_subjects": test_subjects,
        "split_counts": {
            "train": int(sum(len(item) for item in label_payload["train"])),
            "val": int(sum(len(item) for item in label_payload["val"])),
            "test": int(sum(len(item) for item in label_payload["test"])),
        },
    }

    if config.quick:
        dataset["X_train"], dataset["y_train"] = sample_windows(
            dataset["X_train"], dataset["y_train"], config.quick_train_windows, config.seed
        )
        dataset["X_val"], dataset["y_val"] = sample_windows(
            dataset["X_val"], dataset["y_val"], config.quick_val_windows, config.seed + 1
        )
        dataset["X_test"], dataset["y_test"] = sample_windows(
            dataset["X_test"], dataset["y_test"], config.quick_test_windows, config.seed + 2
        )

    return dataset, pd.DataFrame(subject_summary)


def standardize_windows(
    X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (
        ((X_train - mean) / std).astype(np.float32),
        ((X_val - mean) / std).astype(np.float32),
        ((X_test - mean) / std).astype(np.float32),
    )
