"""DNN model definition for TSR key-frequency classification."""

from __future__ import annotations

from typing import Iterable, Sequence

import tensorflow as tf
from tensorflow.keras import Model, regularizers
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Input


def build_tsr_dnn(
    input_dim: int,
    num_classes: int,
    hidden_units: Sequence[int] = (512, 256, 128, 64),
    dropout: float = 0.35,
    l2_strength: float = 1e-4,
    learning_rate: float = 1e-3,
    optimizer_name: str = "adamw",
) -> Model:
    """Build and compile a dense neural network for tabular TSR features.

    Batch normalization stabilizes the hidden activations after feature scaling.
    Dropout and L2 regularization are both included because TSR key-frequency
    matrices are often high-dimensional relative to the number of labeled
    protein/ligand samples.
    """

    if input_dim <= 0:
        raise ValueError("input_dim must be positive.")
    if num_classes <= 1:
        raise ValueError("num_classes must be greater than one for classification.")

    inputs = Input(shape=(input_dim,), name="tsr_features")
    x = inputs
    for layer_index, units in enumerate(hidden_units, start=1):
        x = Dense(
            units,
            kernel_regularizer=regularizers.l2(l2_strength),
            use_bias=False,
            name=f"dense_{units}",
        )(x)
        x = BatchNormalization(name=f"batch_norm_{layer_index}")(x)
        x = tf.keras.layers.Activation("relu", name=f"relu_{layer_index}")(x)
        x = Dropout(dropout, name=f"dropout_{layer_index}")(x)

    # Named explicitly so evaluate_dnn_tsr.py can extract embeddings later.
    penultimate = Dense(
        64,
        activation="relu",
        kernel_regularizer=regularizers.l2(l2_strength),
        name="penultimate",
    )(x)
    outputs = Dense(num_classes, activation="softmax", name="class_probabilities")(penultimate)

    model = Model(inputs=inputs, outputs=outputs, name="tsr_key_frequency_dnn")
    optimizer = _build_optimizer(optimizer_name, learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _build_optimizer(optimizer_name: str, learning_rate: float):
    """Use AdamW when available, otherwise fall back to Adam."""

    optimizer_name = optimizer_name.lower()
    if optimizer_name == "adamw":
        try:
            return tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=1e-5)
        except AttributeError:
            return tf.keras.optimizers.Adam(learning_rate=learning_rate)
    if optimizer_name == "adam":
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)
    raise ValueError("--optimizer must be adam or adamw")


class MacroF1Callback(tf.keras.callbacks.Callback):
    """Compute validation macro F1 at the end of each epoch.

    Keras does not provide a built-in multiclass macro F1 metric that exactly
    matches sklearn's reporting, so this callback computes it from validation
    predictions and injects it into the epoch logs. It then appears in
    history.csv and can be plotted like a normal training metric.
    """

    def __init__(self, validation_data):
        super().__init__()
        self.validation_data_for_f1 = validation_data

    def on_epoch_end(self, epoch, logs=None):
        from sklearn.metrics import f1_score

        logs = logs or {}
        x_val, y_val = self.validation_data_for_f1
        y_prob = self.model.predict(x_val, verbose=0)
        y_pred = y_prob.argmax(axis=1)
        logs["val_macro_f1"] = f1_score(y_val, y_pred, average="macro", zero_division=0)

