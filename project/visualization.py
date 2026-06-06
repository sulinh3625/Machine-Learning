from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline


sns.set_theme(style="whitegrid")


def plot_eda(subject_summary: pd.DataFrame, dataset: dict[str, Any], output_path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.barplot(data=subject_summary, x="subject", y="rows", ax=axes[0], color="#2f7ed8")
    axes[0].set_title("Số dòng dữ liệu theo subject")
    axes[0].set_xlabel("Subject")
    axes[0].set_ylabel("Rows")
    axes[0].tick_params(axis="x", rotation=70)

    counts = pd.Series(dataset["y_train"]).value_counts().sort_index()
    count_frame = pd.DataFrame(
        {"label": [dataset["class_names"][index] for index in counts.index], "count": counts.values}
    )
    sns.barplot(data=count_frame, x="label", y="count", ax=axes[1], color="#55a868")
    axes[1].set_title("Phân bố nhãn trên tập train")
    axes[1].set_xlabel("Label")
    axes[1].set_ylabel("Windows")
    axes[1].tick_params(axis="x", rotation=75)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(estimator: Any, feature_names: list[str], output_path) -> None:
    model = estimator
    if isinstance(estimator, Pipeline):
        model = estimator.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return
    importances = model.feature_importances_
    top_indices = np.argsort(importances)[-20:]
    frame = pd.DataFrame(
        {"feature": np.asarray(feature_names)[top_indices], "importance": importances[top_indices]}
    ).sort_values("importance")
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(frame["feature"], frame["importance"], color="#dd8452")
    ax.set_title("Top 20 Feature Importances")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_result_comparison(results_frame: pd.DataFrame, output_path) -> None:
    ranking = results_frame.sort_values("F1-Score", ascending=False)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=ranking, x="Model", y="F1-Score", ax=ax, palette="viridis")
    ax.set_title("So sánh Macro F1-score giữa các mô hình")
    ax.set_xlabel("Model")
    ax.set_ylabel("Macro F1-score")
    ax.tick_params(axis="x", rotation=65)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(y_true, y_pred, class_names: list[str], output_path, title: str) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(matrix, cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.xticks(rotation=80)
    plt.yticks(rotation=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_learning_curves(artefacts: dict[str, dict[str, Any]], output_path) -> None:
    histories = {name: payload["history"] for name, payload in artefacts.items() if "history" in payload}
    if not histories:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for name, history in histories.items():
        axes[0].plot(history["loss"], label=f"{name} train")
        axes[0].plot(history["val_loss"], linestyle="--", label=f"{name} val")
        axes[1].plot(history["sparse_categorical_accuracy"], label=f"{name} train")
        axes[1].plot(history["val_sparse_categorical_accuracy"], linestyle="--", label=f"{name} val")
    axes[0].set_title("Learning Curves - Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[1].set_title("Learning Curves - Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Sparse Accuracy")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
