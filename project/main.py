from __future__ import annotations

import warnings

import pandas as pd
from sklearn.metrics import classification_report

from .config import parse_args, set_global_seed
from .data_pipeline import ensure_output_dir, load_dataset, sample_windows, standardize_windows
from .deep_pipeline import run_deep_pipeline
from .features import extract_statistical_features
from .ml_pipeline import run_ml_pipeline
from .visualization import (
    plot_confusion,
    plot_eda,
    plot_feature_importance,
    plot_learning_curves,
    plot_result_comparison,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="Precision loss occurred in moment calculation")


def main() -> None:
    config = parse_args()
    ensure_output_dir(config.output_dir)
    set_global_seed(config.seed)

    print("Loading HARTH dataset...")
    dataset, subject_summary = load_dataset(config)
    plot_eda(subject_summary, dataset, config.output_dir / "eda_analysis.png")

    X_train_scaled, X_val_scaled, X_test_scaled = standardize_windows(
        dataset["X_train"],
        dataset["X_val"],
        dataset["X_test"],
    )

    print("Extracting statistical features for ML branch...")
    X_train_features, feature_names = extract_statistical_features(X_train_scaled)
    X_val_features, _ = extract_statistical_features(X_val_scaled)
    X_test_features, _ = extract_statistical_features(X_test_scaled)

    ml_results, ml_artefacts = run_ml_pipeline(
        X_train_features,
        dataset["y_train"],
        X_val_features,
        dataset["y_val"],
        X_test_features,
        dataset["y_test"],
        feature_names,
        config.seed,
        config.tune_sample_size,
        sample_windows,
        config.quick,
    )

    dl_results, dl_artefacts = run_deep_pipeline(
        X_train_scaled,
        dataset["y_train"],
        X_val_scaled,
        dataset["y_val"],
        X_test_scaled,
        dataset["y_test"],
        len(dataset["class_names"]),
        config.cnn_epochs,
        config.cnn_lstm_epochs,
        config.batch_size,
    )
    plot_learning_curves(dl_artefacts, config.output_dir / "deep_learning_curves.png")

    results_frame = pd.DataFrame(ml_results + dl_results)
    results_frame["RunMode"] = "quick" if config.quick else "full"
    results_frame["WindowSize"] = config.window_size
    results_frame["StepSize"] = config.step_size
    results_frame = results_frame.sort_values("F1-Score", ascending=False)
    results_frame.to_csv(config.output_dir / "all_results.csv", index=False)
    plot_result_comparison(results_frame, config.output_dir / "final_comparison.png")

    importance_source = ml_artefacts.get("Tuned XGBoost", ml_artefacts.get("XGBoost"))
    if importance_source is not None:
        plot_feature_importance(
            importance_source["estimator"],
            feature_names,
            config.output_dir / "feature_importance.png",
        )

    all_artefacts = {**ml_artefacts, **dl_artefacts}
    best_model_name = results_frame.iloc[0]["Model"]
    best_pred = all_artefacts[best_model_name]["y_pred"]
    class_report = classification_report(
        dataset["y_test"],
        best_pred,
        labels=list(range(len(dataset["class_names"]))),
        target_names=dataset["class_names"],
        zero_division=0,
        output_dict=True,
    )
    pd.DataFrame(class_report).transpose().to_csv(config.output_dir / "best_classification_report.csv")
    plot_confusion(
        dataset["y_test"],
        best_pred,
        dataset["class_names"],
        config.output_dir / "best_confusion_matrix.png",
        f"Confusion Matrix - {best_model_name}",
    )

    print("\nFinished. Top models:")
    print(results_frame[["Model", "Accuracy", "F1-Score"]].head(5).to_string(index=False))
    print(f"\nSaved outputs to: {config.output_dir}")


if __name__ == "__main__":
    main()
