"""Cocotb tests for MNIST inference using the systolic array with software
bias/ReLU/requantize (Phase 3: software-only post-processing).
"""

import cocotb
from cocotb.clock import Clock
import numpy as np
from tiling_driver import TilingDriver, reset
from mnist_utils import (
    load_layer, load_input_scale, load_test_data, run_inference
)


async def setup(dut):
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset(dut)
    return TilingDriver(dut)


@cocotb.test()
async def test_weight_shapes(dut):
    """Verify exported weight shapes match expected MLP architecture."""
    _ = await setup(dut)

    w1, b1, s1 = load_layer(1)
    w2, b2, s2 = load_layer(2)
    w3, b3, s3 = load_layer(3)

    assert w1.shape == (128, 784), f"Layer 1 weights: {w1.shape}"
    assert b1.shape == (128,), f"Layer 1 bias: {b1.shape}"
    assert w2.shape == (64, 128), f"Layer 2 weights: {w2.shape}"
    assert b2.shape == (64,), f"Layer 2 bias: {b2.shape}"
    assert w3.shape == (10, 64), f"Layer 3 weights: {w3.shape}"
    assert b3.shape == (10,), f"Layer 3 bias: {b3.shape}"

    assert w1.dtype == np.int8
    assert w2.dtype == np.int8
    assert w3.dtype == np.int8

    input_scale = load_input_scale()
    assert input_scale > 0


@cocotb.test()
async def test_single_image(dut):
    """Run inference on a single image and verify it classifies correctly."""
    driver = await setup(dut)

    images, labels = load_test_data()
    input_scale = load_input_scale()

    # Test first image (digit 0)
    logits, pred = await run_inference(driver, images[0], input_scale)

    assert logits.shape == (10,), f"Logits shape: {logits.shape}"
    assert pred == labels[0], f"Predicted {pred}, expected {labels[0]}"
    cocotb.log.info(f"Image 0 (digit {labels[0]}): predicted {pred}, logits max={logits.max():.2f}")


@cocotb.test()
async def test_all_digits(dut):
    """Run inference on 10 images (one per digit), expect >= 8/10 correct."""
    driver = await setup(dut)

    images, labels = load_test_data()
    input_scale = load_input_scale()

    correct = 0
    for i in range(10):
        logits, pred = await run_inference(driver, images[i], input_scale)
        if pred == labels[i]:
            correct += 1
        cocotb.log.info(f"Digit {labels[i]}: predicted {pred} {'OK' if pred == labels[i] else 'WRONG'}")

    cocotb.log.info(f"Accuracy: {correct}/10")
    assert correct >= 8, f"Only {correct}/10 correct, expected >= 8"


@cocotb.test()
async def test_batch_of_4(dut):
    """Run inference on a batch of 4 images."""
    driver = await setup(dut)

    images, labels = load_test_data()
    input_scale = load_input_scale()

    correct = 0
    for i in range(4):
        _, pred = await run_inference(driver, images[i], input_scale)
        if pred == labels[i]:
            correct += 1

    cocotb.log.info(f"Batch accuracy: {correct}/4")
    assert correct >= 3, f"Only {correct}/4 correct, expected >= 3"
