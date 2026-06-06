from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf

from .data_download import download_data_from_url, find_csv_dir, find_data_url


@dataclass
class ExperimentConfig:
    data_dir: Path
    output_dir: Path
    seed: int = 42
    window_size: int = 128
    step_size: int = 64
    quick: bool = False
    quick_train_windows: int = 5000
    quick_val_windows: int = 1500
    quick_test_windows: int = 1500
    cnn_epochs: int = 12
    cnn_lstm_epochs: int = 14
    batch_size: int = 256
    tune_sample_size: int = 25000
    train_subjects: tuple[str, ...] | None = None
    val_subjects: tuple[str, ...] | None = None
    test_subjects: tuple[str, ...] | None = None


def resolve_data_dir(source_root: Path, arg_path: Path | None, data_url: str | None) -> Path:
    target_dir = arg_path if arg_path is not None else source_root / "data"
    csv_dir = find_csv_dir(target_dir)
    if csv_dir is not None:
        return csv_dir

    readme_url = find_data_url(source_root / "README.md")
    resolved_url = data_url or readme_url
    if resolved_url:
        print("Không tìm thấy CSV local. Tự động tải dữ liệu HARTH từ link trong README/--data-url...")
        downloaded_dir = download_data_from_url(resolved_url, target_dir)
        csv_dir = find_csv_dir(downloaded_dir)
        if csv_dir is not None:
            return csv_dir

    checked = str(target_dir)
    raise FileNotFoundError(
        "Không tìm thấy dữ liệu HARTH.\n"
        f"Đã kiểm tra: {checked}\n"
        "Cách sửa nhanh:\n"
        "1) Đặt các file CSV vào Source/Task2/data, hoặc\n"
        "2) Dán link tải vào README theo dòng DATA_URL=<link>, hoặc\n"
        "3) Chạy với --data-url <link> hoặc --data-dir <thư_mục_csv>."
    )


def parse_args() -> ExperimentConfig:
    source_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="HARTH activity recognition project")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=source_root / "output")
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--step-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--cnn-epochs", type=int, default=12)
    parser.add_argument("--cnn-lstm-epochs", type=int, default=14)
    parser.add_argument("--tune-sample-size", type=int, default=25000)
    parser.add_argument("--data-url", type=str, default=None)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    return ExperimentConfig(
        data_dir=resolve_data_dir(source_root, args.data_dir, args.data_url),
        output_dir=args.output_dir,
        seed=args.seed,
        window_size=args.window_size,
        step_size=args.step_size,
        batch_size=args.batch_size,
        cnn_epochs=4 if args.quick else args.cnn_epochs,
        cnn_lstm_epochs=5 if args.quick else args.cnn_lstm_epochs,
        tune_sample_size=2000 if args.quick else args.tune_sample_size,
        quick=args.quick,
    )


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
