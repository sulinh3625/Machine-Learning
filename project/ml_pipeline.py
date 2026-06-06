from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from .evaluation import score_estimator


def build_xgboost_classifier(seed: int, quick: bool = False) -> XGBClassifier:
    return XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        learning_rate=0.08,
        max_depth=4 if quick else 6,
        n_estimators=80 if quick else 220,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=2,
        reg_lambda=1.2,
        tree_method="hist",
        random_state=seed,
        n_jobs=1,
        verbosity=0,
    )


def build_ml_models(seed: int, quick: bool = False) -> dict[str, Any]:
    return {
        "Decision Tree (Baseline)": DecisionTreeClassifier(
            max_depth=6,
            min_samples_leaf=16,
            class_weight="balanced",
            random_state=seed,
        ),
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=700, solver="lbfgs", class_weight="balanced")),
            ]
        ),
        "KNN": Pipeline(
            [("scaler", StandardScaler()), ("model", KNeighborsClassifier(n_neighbors=7, weights="distance"))]
        ),
        "Linear SVM": Pipeline(
            [("scaler", StandardScaler()), ("model", LinearSVC(C=1.2, class_weight="balanced", dual="auto"))]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=80 if quick else 260,
            max_depth=12 if quick else None,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=1,
        ),
        "XGBoost": build_xgboost_classifier(seed, quick=quick),
    }


def run_feature_selection(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
) -> dict[str, Any]:
    k_candidates = [18, 24, 36, 48, min(len(feature_names), 60)]
    best_score = -1.0
    best_selector: SelectKBest | None = None

    for k in sorted(set(k for k in k_candidates if k < len(feature_names))):
        selector = SelectKBest(score_func=mutual_info_classif, k=k)
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_val_selected = selector.transform(X_val)
        estimator = Pipeline(
            [("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=700, class_weight="balanced"))]
        )
        estimator.fit(X_train_selected, y_train)
        val_pred = estimator.predict(X_val_selected)
        val_f1 = f1_score(y_val, val_pred, average="macro", zero_division=0)
        if val_f1 > best_score:
            best_score = val_f1
            best_selector = selector

    if best_selector is None:
        raise RuntimeError("Feature selection không chọn được cấu hình hợp lệ.")

    X_trainval_selected = best_selector.transform(np.concatenate([X_train, X_val], axis=0))
    y_trainval = np.concatenate([y_train, y_val], axis=0)
    X_test_selected = best_selector.transform(X_test)
    estimator = Pipeline(
        [("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=700, class_weight="balanced"))]
    )
    estimator.fit(X_trainval_selected, y_trainval)
    metrics, y_pred, y_score = score_estimator(estimator, X_test_selected, y_test)

    return {
        "result": {"Model": "SelectKBest + Logistic Regression", **metrics},
        "artefact": {"estimator": estimator, "selector": best_selector, "y_pred": y_pred, "y_score": y_score},
    }


def run_xgboost_tuning(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    sample_size: int,
    sampler: Any,
    quick: bool,
) -> dict[str, Any]:
    X_tune, y_tune = sampler(X_train, y_train, sample_size, seed + 20)
    search = RandomizedSearchCV(
        estimator=build_xgboost_classifier(seed, quick=quick),
        param_distributions={
            "max_depth": [4, 5, 6, 7],
            "learning_rate": [0.03, 0.05, 0.08, 0.1],
            "n_estimators": [60, 80, 120] if quick else [160, 220, 280],
            "subsample": [0.75, 0.85, 0.95],
            "colsample_bytree": [0.75, 0.85, 0.95],
            "min_child_weight": [1, 2, 4],
        },
        n_iter=3 if quick else 6,
        cv=2 if quick else 3,
        scoring="f1_macro",
        verbose=0,
        random_state=seed,
        n_jobs=1,
    )
    search.fit(X_tune, y_tune)
    best_params = search.best_params_

    tuned_model = build_xgboost_classifier(seed, quick=quick)
    tuned_model.set_params(**best_params)
    X_trainval = np.concatenate([X_train, X_val], axis=0)
    y_trainval = np.concatenate([y_train, y_val], axis=0)
    tuned_model.fit(X_trainval, y_trainval)
    metrics, y_pred, y_score = score_estimator(tuned_model, X_test, y_test)

    return {
        "result": {"Model": "Tuned XGBoost", **metrics},
        "artefact": {
            "estimator": tuned_model,
            "y_pred": y_pred,
            "y_score": y_score,
            "best_params": best_params,
            "best_cv_score": float(search.best_score_),
        },
    }


def run_ml_pipeline(
    X_train_features: np.ndarray,
    y_train: np.ndarray,
    X_val_features: np.ndarray,
    y_val: np.ndarray,
    X_test_features: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    seed: int,
    tune_sample_size: int,
    sampler: Any,
    quick: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    X_trainval_features = np.concatenate([X_train_features, X_val_features], axis=0)
    y_trainval = np.concatenate([y_train, y_val], axis=0)
    results: list[dict[str, Any]] = []
    artefacts: dict[str, dict[str, Any]] = {}

    models = build_ml_models(seed, quick=quick)
    for model_name, estimator in models.items():
        print(f"[ML] Training {model_name}...")
        fitted = clone(estimator)
        fitted.fit(X_trainval_features, y_trainval)
        metrics, y_pred, y_score = score_estimator(fitted, X_test_features, y_test)
        results.append({"Model": model_name, **metrics})
        artefacts[model_name] = {"estimator": fitted, "y_pred": y_pred, "y_score": y_score}

    print("[ML] Running SelectKBest + Logistic Regression...")
    feature_selection_result = run_feature_selection(
        X_train_features,
        y_train,
        X_val_features,
        y_val,
        X_test_features,
        y_test,
        feature_names,
    )
    results.append(feature_selection_result["result"])
    artefacts[feature_selection_result["result"]["Model"]] = feature_selection_result["artefact"]

    print("[ML] Training Voting ensemble...")
    voting_estimators = [
        ("lr", clone(models["Logistic Regression"])),
        ("rf", clone(models["Random Forest"])),
        ("xgb", clone(models["XGBoost"])),
    ]
    voting = VotingClassifier(estimators=voting_estimators, voting="soft", n_jobs=1)
    voting.fit(X_trainval_features, y_trainval)
    metrics, y_pred, y_score = score_estimator(voting, X_test_features, y_test)
    results.append({"Model": "Soft Voting", **metrics})
    artefacts["Soft Voting"] = {"estimator": voting, "y_pred": y_pred, "y_score": y_score}

    print("[ML] Training Stacking ensemble...")
    stacking = StackingClassifier(
        estimators=voting_estimators,
        final_estimator=LogisticRegression(max_iter=500),
        stack_method="predict_proba",
        cv=2 if quick else 3,
        n_jobs=1,
    )
    stacking.fit(X_trainval_features, y_trainval)
    metrics, y_pred, y_score = score_estimator(stacking, X_test_features, y_test)
    results.append({"Model": "Stacking", **metrics})
    artefacts["Stacking"] = {"estimator": stacking, "y_pred": y_pred, "y_score": y_score}

    print("[ML] Running hyperparameter tuning for XGBoost...")
    tuned_result = run_xgboost_tuning(
        X_train_features,
        y_train,
        X_val_features,
        y_val,
        X_test_features,
        y_test,
        seed,
        tune_sample_size,
        sampler,
        quick,
    )
    results.append(tuned_result["result"])
    artefacts[tuned_result["result"]["Model"]] = tuned_result["artefact"]

    return results, artefacts
