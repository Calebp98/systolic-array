"""Tests for MNIST quantization pipeline."""

import numpy as np
import pytest
from pathlib import Path

EXPORT_DIR = Path(__file__).parent / "export"


def _check_exported():
    """Skip if export files don't exist (need to run train.py + quantize.py first)."""
    if not (EXPORT_DIR / "layer1_weights.npy").exists():
        pytest.skip("Export files not found. Run: python train.py && python quantize.py")


class TestExportedFiles:
    def test_weight_shapes(self):
        _check_exported()
        w1 = np.load(EXPORT_DIR / "layer1_weights.npy")
        w2 = np.load(EXPORT_DIR / "layer2_weights.npy")
        w3 = np.load(EXPORT_DIR / "layer3_weights.npy")
        assert w1.shape == (128, 784)
        assert w2.shape == (64, 128)
        assert w3.shape == (10, 64)

    def test_weight_dtype(self):
        _check_exported()
        for i in range(1, 4):
            w = np.load(EXPORT_DIR / f"layer{i}_weights.npy")
            assert w.dtype == np.int8

    def test_bias_shapes(self):
        _check_exported()
        b1 = np.load(EXPORT_DIR / "layer1_bias.npy")
        b2 = np.load(EXPORT_DIR / "layer2_bias.npy")
        b3 = np.load(EXPORT_DIR / "layer3_bias.npy")
        assert b1.shape == (128,)
        assert b2.shape == (64,)
        assert b3.shape == (10,)

    def test_scale_files(self):
        _check_exported()
        for i in range(1, 4):
            s = np.load(EXPORT_DIR / f"layer{i}_scale.npy")
            assert s.shape == (1,)
            assert s.dtype == np.float32
            assert s[0] > 0

    def test_input_scale(self):
        _check_exported()
        s = np.load(EXPORT_DIR / "input_scale.npy")
        assert s.shape == (1,)
        assert s[0] > 0

    def test_test_images(self):
        _check_exported()
        imgs = np.load(EXPORT_DIR / "test_images.npy")
        labels = np.load(EXPORT_DIR / "test_labels.npy")
        assert imgs.shape == (10, 784)
        assert imgs.dtype == np.int8
        assert labels.shape == (10,)
        assert set(labels.tolist()) == set(range(10))

    def test_weight_range(self):
        _check_exported()
        for i in range(1, 4):
            w = np.load(EXPORT_DIR / f"layer{i}_weights.npy")
            assert w.min() >= -128
            assert w.max() <= 127


class TestQuantizedAccuracy:
    def test_accuracy_above_threshold(self):
        _check_exported()
        from quantize import verify_quantized_accuracy
        quant_acc, float_acc = verify_quantized_accuracy()
        assert float_acc > 0.95, f"Float model accuracy {float_acc} too low"
        assert quant_acc > 0.93, f"Quantized accuracy {quant_acc} too low (expect >93%)"

    def test_roundtrip_weights(self):
        """Verify we can load and use the exported weights."""
        _check_exported()
        for i in range(1, 4):
            w = np.load(EXPORT_DIR / f"layer{i}_weights.npy")
            s = np.load(EXPORT_DIR / f"layer{i}_scale.npy")
            # Dequantized weights should be reasonable floats
            w_float = w.astype(np.float32) * s[0]
            assert np.isfinite(w_float).all()
            assert np.abs(w_float).max() < 10.0  # weights shouldn't be huge
