"""Utilities for MNIST inference using the systolic array tiling driver.

Handles weight loading, bias addition, ReLU, and requantization in software.
The matmul is performed via the hardware tiling driver.
"""

import numpy as np
from pathlib import Path

EXPORT_DIR = Path(__file__).parent.parent / "mnist" / "export"


def load_layer(layer_num):
    """Load quantized weights, bias, and scale for a layer.

    Returns:
        weights: int8 numpy array (out_features, in_features)
        bias: float32 numpy array (out_features,)
        w_scale: float32 scalar
    """
    weights = np.load(EXPORT_DIR / f"layer{layer_num}_weights.npy")
    bias = np.load(EXPORT_DIR / f"layer{layer_num}_bias.npy")
    w_scale = np.load(EXPORT_DIR / f"layer{layer_num}_scale.npy")[0]
    return weights, bias, w_scale


def load_input_scale():
    return np.load(EXPORT_DIR / "input_scale.npy")[0]


def load_test_data():
    """Load test images and labels.

    Returns:
        images: int8 array (10, 784)
        labels: int64 array (10,)
    """
    images = np.load(EXPORT_DIR / "test_images.npy")
    labels = np.load(EXPORT_DIR / "test_labels.npy")
    return images, labels


def relu_int32(x):
    """ReLU on float/int array."""
    return np.maximum(x, 0)


def bias_add(acc_float, bias):
    """Add bias to dequantized accumulator values."""
    return acc_float + bias.astype(np.float64)


def requantize_to_int8(x_float):
    """Symmetric requantization of float values to int8.

    Returns:
        x_int8: int8 array
        scale: float32 scalar (for next layer)
    """
    max_abs = np.abs(x_float).max()
    if max_abs == 0:
        return np.zeros_like(x_float, dtype=np.int8), 1.0
    scale = float(max_abs) / 127.0
    x_int8 = np.clip(np.round(x_float / scale), -128, 127).astype(np.int8)
    return x_int8, scale


async def run_inference(driver, image_int8, input_scale):
    """Run full 3-layer MLP inference on a single image.

    Args:
        driver: TilingDriver instance
        image_int8: int8 array of shape (784,)
        input_scale: float, input quantization scale

    Returns:
        logits: float64 array of shape (10,) — raw logits
        predicted_class: int
    """
    x_int8 = image_int8.copy()
    current_scale = input_scale

    for layer_idx in range(1, 4):
        weights, bias_vals, w_scale = load_layer(layer_idx)
        out_features, in_features = weights.shape

        # Matmul via hardware: x (1 x in_features) @ W^T (in_features x out_features)
        # Reshape for tiling driver
        x_2d = x_int8.reshape(1, -1)
        w_t = weights.T  # (in_features, out_features)

        acc_int32 = await driver.matmul(x_2d, w_t)  # (1, out_features)
        acc_int32 = acc_int32.flatten().astype(np.float64)

        # Dequantize and add bias
        acc_float = acc_int32 * current_scale * w_scale
        acc_float = bias_add(acc_float, bias_vals)

        if layer_idx < 3:  # ReLU for layers 1 and 2
            acc_float = relu_int32(acc_float)
            x_int8, current_scale = requantize_to_int8(acc_float)
        else:
            # Last layer: return raw logits
            return acc_float, int(np.argmax(acc_float))
