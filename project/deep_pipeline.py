from __future__ import annotations

from typing import Any

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from .evaluation import evaluate_predictions


def make_tf_dataset(X: np.ndarray, y: np.ndarray, batch_size: int, training: bool, augment: bool) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    if training:
        dataset = dataset.shuffle(buffer_size=min(len(y), 10000), seed=42, reshuffle_each_iteration=True)
    if training and augment:
        dataset = dataset.map(augment_signal, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def augment_signal(x: tf.Tensor, y: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    noise = tf.random.normal(tf.shape(x), mean=0.0, stddev=0.02)
    scale = tf.random.uniform((1, tf.shape(x)[-1]), minval=0.9, maxval=1.1)
    x = x * scale + noise
    return x, y


def compute_class_weights(y_train: np.ndarray) -> dict[int, float]:
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    return {int(label): float(weight) for label, weight in zip(classes, weights)}


def build_cnn_model(input_shape: tuple[int, int], class_count: int) -> tf.keras.Model:
    regularizer = tf.keras.regularizers.l2(1e-4)
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv1D(64, 5, padding="same", kernel_regularizer=regularizer)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Conv1D(128, 5, padding="same", kernel_regularizer=regularizer)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Conv1D(128, 3, padding="same", kernel_regularizer=regularizer)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=regularizer)(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    outputs = tf.keras.layers.Dense(class_count, activation="softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    return model


def build_cnn_lstm_model(input_shape: tuple[int, int], class_count: int) -> tf.keras.Model:
    regularizer = tf.keras.regularizers.l2(1e-4)
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv1D(64, 5, padding="same", kernel_regularizer=regularizer)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Conv1D(128, 3, padding="same", kernel_regularizer=regularizer)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.LSTM(96, dropout=0.2, recurrent_dropout=0.0)(x)
    x = tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=regularizer)(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    outputs = tf.keras.layers.Dense(class_count, activation="softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=8e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    return model


def train_deep_model(
    model_name: str,
    builder: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_count: int,
    epochs: int,
    batch_size: int,
    augment: bool,
) -> dict[str, Any]:
    print(f"[DL] Training {model_name}...")
    tf.keras.backend.clear_session()

    train_ds = make_tf_dataset(X_train, y_train, batch_size, training=True, augment=augment)
    val_ds = make_tf_dataset(X_val, y_val, batch_size, training=False, augment=False)
    test_ds = make_tf_dataset(X_test, y_test, batch_size, training=False, augment=False)

    model = builder((X_train.shape[1], X_train.shape[2]), class_count)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5),
    ]
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        verbose=0,
        callbacks=callbacks,
        class_weight=compute_class_weights(y_train),
    )
    y_score = model.predict(test_ds, verbose=0)
    y_pred = y_score.argmax(axis=1)
    metrics = evaluate_predictions(y_test, y_pred, y_score)
    return {
        "result": {"Model": model_name, **metrics},
        "artefact": {"model": model, "y_pred": y_pred, "y_score": y_score, "history": history.history},
    }


def run_deep_pipeline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_count: int,
    cnn_epochs: int,
    cnn_lstm_epochs: int,
    batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    results: list[dict[str, Any]] = []
    artefacts: dict[str, dict[str, Any]] = {}

    cnn_result = train_deep_model(
        "1D CNN",
        build_cnn_model,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        class_count,
        cnn_epochs,
        batch_size,
        augment=False,
    )
    results.append(cnn_result["result"])
    artefacts["1D CNN"] = cnn_result["artefact"]

    aug_result = train_deep_model(
        "1D CNN + Augmentation",
        build_cnn_model,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        class_count,
        cnn_epochs,
        batch_size,
        augment=True,
    )
    results.append(aug_result["result"])
    artefacts["1D CNN + Augmentation"] = aug_result["artefact"]

    cnn_lstm_result = train_deep_model(
        "CNN-LSTM",
        build_cnn_lstm_model,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        class_count,
        cnn_lstm_epochs,
        batch_size,
        augment=True,
    )
    results.append(cnn_lstm_result["result"])
    artefacts["CNN-LSTM"] = cnn_lstm_result["artefact"]

    return results, artefacts
