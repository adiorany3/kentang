"""Compatibility loader for the potato leaf disease Keras HDF5 model.

The original `keras_model.h5` is a legacy nested Sequential/Functional model.
On modern Keras it may fail during model deserialization with errors such as:
"Layer ... expects 1 input(s), but it received 2 input tensors".

To avoid that fragile deserialization path, this module rebuilds the known
architecture and loads only the weights from the HDF5 file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import tensorflow as tf

TARGET_SIZE = (224, 224)
NUM_CLASSES = 4
MOBILENET_ALPHA = 0.35


class ModelLoadError(RuntimeError):
    """Raised when the legacy H5 model cannot be rebuilt."""


def _read_dataset(group: h5py.Group, name: str) -> np.ndarray:
    if name not in group:
        available = ", ".join(group.keys())
        raise ModelLoadError(f"Dataset `{name}` tidak ditemukan. Tersedia: {available}")
    return np.asarray(group[name])


def _set_layer_weights_from_group(layer: tf.keras.layers.Layer, group: h5py.Group) -> bool:
    """Set weights for one MobileNetV2 layer from a same-named HDF5 group.

    Returns True when weights are assigned, False for weightless layers.
    """
    if not layer.weights:
        return False

    layer_type = layer.__class__.__name__

    if layer_type == "Conv2D":
        layer.set_weights([_read_dataset(group, "kernel:0")])
        return True

    if layer_type == "DepthwiseConv2D":
        layer.set_weights([_read_dataset(group, "depthwise_kernel:0")])
        return True

    if layer_type == "BatchNormalization":
        layer.set_weights(
            [
                _read_dataset(group, "gamma:0"),
                _read_dataset(group, "beta:0"),
                _read_dataset(group, "moving_mean:0"),
                _read_dataset(group, "moving_variance:0"),
            ]
        )
        return True

    # Fallback for future-proofing. Keras weight objects usually expose names
    # ending in "kernel", "bias", etc.; the legacy file uses ":0" suffixes.
    values = []
    for weight in layer.weights:
        key = weight.name.split("/")[-1]
        legacy_key = key if key.endswith(":0") else f"{key}:0"
        if legacy_key not in group:
            raise ModelLoadError(
                f"Layer `{layer.name}` memiliki weight `{weight.name}`, "
                f"tetapi `{legacy_key}` tidak ada di file H5."
            )
        values.append(np.asarray(group[legacy_key]))

    layer.set_weights(values)
    return True


def _validate_groups(weights_root: h5py.Group, required_groups: Iterable[str]) -> None:
    missing = [name for name in required_groups if name not in weights_root]
    if missing:
        raise ModelLoadError(
            "Struktur H5 tidak sesuai. Group yang hilang: " + ", ".join(missing)
        )


def build_compatible_model(model_path: str | Path) -> tf.keras.Model:
    """Rebuild the model architecture and load legacy H5 weights.

    Architecture inferred from the provided model:
    MobileNetV2 alpha=0.35, include_top=False -> GlobalAveragePooling2D ->
    Dense(100, relu) -> Dense(4, softmax, no bias).
    """
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"File model tidak ditemukan: {model_path}")

    image_input = tf.keras.Input(shape=(224, 224, 3), name="image_input")

    mobilenet = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        alpha=MOBILENET_ALPHA,
        include_top=False,
        weights=None,
    )

    mobilenet.trainable = False

    x = mobilenet(image_input, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(
        name="global_average_pooling2d_GlobalAveragePooling2D1"
    )(x)
    x = tf.keras.layers.Dense(100, activation="relu", name="dense_Dense1")(x)
    output = tf.keras.layers.Dense(
        NUM_CLASSES,
        activation="softmax",
        use_bias=False,
        name="dense_Dense2",
    )(x)

    model = tf.keras.Model(image_input, output, name="kentang_compat_model")

    with h5py.File(model_path, "r") as h5_file:
        if "model_weights" not in h5_file:
            raise ModelLoadError("File H5 tidak memiliki group `model_weights`.")

        weights_root = h5_file["model_weights"]
        _validate_groups(weights_root, ["sequential_1", "sequential_3"])

        feature_weights = weights_root["sequential_1"]
        classifier_weights = weights_root["sequential_3"]

        assigned = 0
        for layer in mobilenet.layers:
            if not layer.weights:
                continue
            if layer.name not in feature_weights:
                raise ModelLoadError(f"Weight untuk layer MobileNet `{layer.name}` tidak ditemukan.")
            if _set_layer_weights_from_group(layer, feature_weights[layer.name]):
                assigned += 1

        dense1_group = classifier_weights["dense_Dense1"]
        dense2_group = classifier_weights["dense_Dense2"]

        model.get_layer("dense_Dense1").set_weights(
            [
                _read_dataset(dense1_group, "kernel:0"),
                _read_dataset(dense1_group, "bias:0"),
            ]
        )
        model.get_layer("dense_Dense2").set_weights(
            [_read_dataset(dense2_group, "kernel:0")]
        )

    # Build a small dummy pass to ensure the graph is callable after weights are set.
    _ = model(np.zeros((1, 224, 224, 3), dtype=np.float32), training=False)
    return model


def preprocess_pil_image(image) -> np.ndarray:
    """Convert a PIL image to the model input tensor."""
    image = image.convert("RGB")
    image = image.resize(TARGET_SIZE)
    image_array = np.asarray(image, dtype=np.float32)
    image_array = (image_array / 127.5) - 1.0
    return np.expand_dims(image_array, axis=0)
