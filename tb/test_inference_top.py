"""Cocotb tests for the integrated inference pipeline (inference_top).

Tests the systolic_array + bias_add + relu + requantize pipeline,
and runs MNIST end-to-end inference with RTL post-processing.
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, ClockCycles
import numpy as np
from pathlib import Path

from tiling_driver import pack_matrix, tick, TilingDriver, TILE, ACC_WIDTH, DATA_WIDTH

N = TILE


def pack_bias_int32(bias_vec):
    """Pack 4 int32 bias values into an integer."""
    val = 0
    mask = (1 << ACC_WIDTH) - 1
    for j in range(N):
        val |= (int(bias_vec[j]) & mask) << (j * ACC_WIDTH)
    return val


def unpack_int32(signal):
    """Read 4x4 int32 result from an ACC_WIDTH-packed signal."""
    raw = int(signal.value)
    C = np.zeros((N, N), dtype=np.int64)
    mask = (1 << ACC_WIDTH) - 1
    for i in range(N):
        for j in range(N):
            idx = i * N + j
            v = (raw >> (idx * ACC_WIDTH)) & mask
            if v >= (1 << (ACC_WIDTH - 1)):
                v -= 1 << ACC_WIDTH
            C[i, j] = v
    return C


def unpack_result_int8(dut):
    """Read 4x4 int8 result from result_quant."""
    raw = int(dut.result_quant.value)
    C = np.zeros((N, N), dtype=np.int32)
    mask = (1 << DATA_WIDTH) - 1
    for i in range(N):
        for j in range(N):
            idx = i * N + j
            v = (raw >> (idx * DATA_WIDTH)) & mask
            if v >= (1 << (DATA_WIDTH - 1)):
                v -= 1 << DATA_WIDTH
            C[i, j] = v
    return C


async def reset(dut):
    dut.rst_n.value = 0
    dut.start.value = 0
    dut.a_data.value = 0
    dut.b_data.value = 0
    dut.bias_data.value = 0
    dut.enable_bias.value = 0
    dut.enable_relu.value = 0
    dut.enable_requant.value = 0
    dut.shift_amount.value = 0
    await ClockCycles(dut.clk, 3)
    dut.rst_n.value = 1
    await tick(dut)


async def run_pipeline(dut, A, B, bias=None, enable_bias=False,
                       enable_relu=False, enable_requant=False, shift=0):
    """Run one 4x4 tile through the inference pipeline."""
    dut.a_data.value = pack_matrix(A)
    dut.b_data.value = pack_matrix(B)

    if bias is not None:
        dut.bias_data.value = pack_bias_int32(bias)
    else:
        dut.bias_data.value = 0

    dut.enable_bias.value = 1 if enable_bias else 0
    dut.enable_relu.value = 1 if enable_relu else 0
    dut.enable_requant.value = 1 if enable_requant else 0
    dut.shift_amount.value = shift

    dut.start.value = 1
    await tick(dut)
    dut.start.value = 0

    for _ in range(30):
        await tick(dut)
        if int(dut.done.value) == 1:
            break
    else:
        raise RuntimeError("Timed out waiting for pipeline to complete")


async def setup(dut):
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset(dut)


@cocotb.test()
async def test_matmul_only(dut):
    """Pipeline with no post-processing should match raw matmul."""
    await setup(dut)

    A = np.array([[1, 2, 3, 4], [5, 6, 7, 8],
                  [9, 10, 11, 12], [13, 14, 15, 16]], dtype=np.int8)
    B = np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                  [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.int8)

    await run_pipeline(dut, A, B)
    got = unpack_int32(dut.result_acc)
    expected = A.astype(np.int32) @ B.astype(np.int32)
    np.testing.assert_array_equal(got, expected)


@cocotb.test()
async def test_bias_add(dut):
    """Pipeline with bias addition."""
    await setup(dut)

    A = np.eye(4, dtype=np.int8)
    B = np.eye(4, dtype=np.int8)
    bias = np.array([10, 20, 30, 40], dtype=np.int32)

    await run_pipeline(dut, A, B, bias=bias, enable_bias=True)
    got = unpack_int32(dut.result_post)
    # Identity @ Identity = Identity, then + bias per column
    expected = np.eye(4, dtype=np.int64)
    for j in range(4):
        expected[:, j] += bias[j]
    np.testing.assert_array_equal(got, expected)


@cocotb.test()
async def test_bias_relu(dut):
    """Pipeline with bias + ReLU."""
    await setup(dut)

    A = np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                  [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.int8)
    B = np.array([[5, -3, 0, 0], [0, 0, 0, 0],
                  [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.int8)
    bias = np.array([-10, 5, 0, 0], dtype=np.int32)

    await run_pipeline(dut, A, B, bias=bias, enable_bias=True, enable_relu=True)
    got = unpack_int32(dut.result_post)

    # Matmul result
    matmul_result = A.astype(np.int64) @ B.astype(np.int64)
    # Add bias
    for j in range(4):
        matmul_result[:, j] += bias[j]
    # ReLU
    expected = np.maximum(matmul_result, 0)
    np.testing.assert_array_equal(got, expected)


@cocotb.test()
async def test_full_pipeline(dut):
    """Full pipeline: matmul + bias + relu + requantize."""
    await setup(dut)

    A = np.array([[10, 20, 0, 0], [5, -5, 0, 0],
                  [0, 0, 1, 0], [0, 0, 0, -1]], dtype=np.int8)
    B = np.array([[1, 2, 0, 0], [3, 4, 0, 0],
                  [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.int8)
    bias = np.array([0, 0, 0, 0], dtype=np.int32)

    await run_pipeline(dut, A, B, bias=bias, enable_bias=True,
                       enable_relu=True, enable_requant=True, shift=0)

    got_int8 = unpack_result_int8(dut)

    # Software reference
    acc = A.astype(np.int32) @ B.astype(np.int32)
    acc = np.maximum(acc, 0)  # ReLU
    expected = np.clip(acc, -128, 127).astype(np.int32)
    np.testing.assert_array_equal(got_int8, expected)


@cocotb.test()
async def test_requantize_with_shift(dut):
    """Requantize with non-zero shift."""
    await setup(dut)

    # Values that will produce larger accumulators
    A = np.array([[50, 50, 50, 50], [100, 0, 0, 0],
                  [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.int8)
    B = np.array([[10, 0, 0, 0], [10, 0, 0, 0],
                  [10, 0, 0, 0], [10, 0, 0, 0]], dtype=np.int8)

    shift = 4
    await run_pipeline(dut, A, B, enable_requant=True, shift=shift)

    got_int8 = unpack_result_int8(dut)
    acc = A.astype(np.int32) @ B.astype(np.int32)

    # Software reference: arithmetic right shift then clamp
    expected = np.zeros_like(acc)
    for i in range(N):
        for j in range(N):
            shifted = int(acc[i, j]) >> shift
            expected[i, j] = max(-128, min(127, shifted))

    np.testing.assert_array_equal(got_int8, expected)


@cocotb.test()
async def test_mnist_single_tile_vs_software(dut):
    """Compare a single 4x4 tile through the pipeline vs software reference."""
    await setup(dut)

    rng = np.random.RandomState(42)
    A = rng.randint(-10, 10, (4, 4)).astype(np.int8)
    B = rng.randint(-10, 10, (4, 4)).astype(np.int8)
    bias = rng.randint(-50, 50, 4).astype(np.int32)

    await run_pipeline(dut, A, B, bias=bias, enable_bias=True,
                       enable_relu=True, enable_requant=True, shift=0)

    got_int8 = unpack_result_int8(dut)

    # Software reference
    acc = A.astype(np.int32) @ B.astype(np.int32)
    for j in range(4):
        acc[:, j] += bias[j]
    acc = np.maximum(acc, 0)
    expected = np.clip(acc, -128, 127).astype(np.int32)

    np.testing.assert_array_equal(got_int8, expected)


# ---- MNIST end-to-end with RTL post-processing (tiling still in software) ----

class InferenceTopTilingDriver(TilingDriver):
    """Tiling driver that uses inference_top instead of raw systolic_array.

    Overrides multiply_tile to go through the inference pipeline with
    post-processing disabled (raw int32 accumulation for tiled matmul).
    """

    async def multiply_tile(self, A_tile, B_tile):
        """Run one 4x4 tile through inference_top with no post-processing."""
        await run_pipeline(self.dut, A_tile, B_tile)
        return unpack_int32(self.dut.result_acc)


@cocotb.test()
async def test_mnist_inference_rtl(dut):
    """Full MNIST inference using inference_top for matmul, SW post-processing."""
    await setup(dut)
    driver = InferenceTopTilingDriver(dut)

    from mnist_utils import load_test_data, load_input_scale, run_inference
    images, labels = load_test_data()
    input_scale = load_input_scale()

    correct = 0
    for i in range(10):
        _, pred = await run_inference(driver, images[i], input_scale)
        if pred == labels[i]:
            correct += 1
        cocotb.log.info(f"Digit {labels[i]}: predicted {pred} {'OK' if pred == labels[i] else 'WRONG'}")

    cocotb.log.info(f"RTL Pipeline Accuracy: {correct}/10")
    assert correct >= 8, f"Only {correct}/10 correct, expected >= 8"


@cocotb.test()
async def test_back_to_back_pipeline(dut):
    """Test back-to-back pipeline invocations without reset.

    Critical test: ensures the FSM correctly handles consecutive operations
    without needing an explicit reset between them. This is required for
    real inference workloads with tiled matrix operations.
    """
    await setup(dut)

    # First operation
    A1 = np.eye(4, dtype=np.int8)
    B1 = np.ones((4, 4), dtype=np.int8) * 5
    await run_pipeline(dut, A1, B1, enable_requant=True, shift=0)
    got1 = unpack_result_int8(dut)
    expected1 = np.ones((4, 4), dtype=np.int32) * 5
    expected1[np.eye(4, dtype=bool)] = 5  # Identity pattern
    np.testing.assert_array_equal(got1, expected1)

    # Second operation immediately after (no reset)
    # Use a different pattern: diagonal matrix
    A2 = np.array([[2, 0, 0, 0], [0, 3, 0, 0], [0, 0, 4, 0], [0, 0, 0, 5]], dtype=np.int8)
    B2 = np.eye(4, dtype=np.int8)
    await run_pipeline(dut, A2, B2, enable_requant=True, shift=0)
    got2 = unpack_result_int8(dut)
    expected2 = A2.astype(np.int32) @ B2.astype(np.int32)
    np.testing.assert_array_equal(got2, expected2)

    # Third operation with different post-processing
    A3 = np.array([[10, 0, 0, 0], [0, 10, 0, 0], [0, 0, 10, 0], [0, 0, 0, 10]], dtype=np.int8)
    B3 = np.eye(4, dtype=np.int8)
    bias3 = np.array([-5, -10, -15, 5], dtype=np.int32)
    await run_pipeline(dut, A3, B3, bias=bias3, enable_bias=True,
                      enable_relu=True, enable_requant=True, shift=0)
    got3 = unpack_result_int8(dut)

    matmul3 = A3.astype(np.int32) @ B3.astype(np.int32)
    with_bias3 = matmul3.copy()
    for j in range(4):
        with_bias3[:, j] += bias3[j]
    expected3 = np.maximum(with_bias3, 0).astype(np.int32)
    np.testing.assert_array_equal(got3, expected3)


@cocotb.test()
async def test_enable_toggling(dut):
    """Test dynamic enable/disable of post-processing stages.

    Verifies that enable signals can be changed between operations and
    the pipeline correctly applies only the requested transformations.
    """
    await setup(dut)

    # Use a simpler test case: diagonal matrix * identity = diagonal matrix
    A = np.array([[5, 0, 0, 0], [0, -3, 0, 0], [0, 0, 2, 0], [0, 0, 0, -1]], dtype=np.int8)
    B = np.eye(4, dtype=np.int8)

    bias_pos = np.array([5, 5, 5, 5], dtype=np.int32)

    # Run 1: Only bias (with positive bias on diagonal matrix)
    await run_pipeline(dut, A, B, bias=bias_pos, enable_bias=True,
                      enable_relu=False, enable_requant=False)
    got = unpack_int32(dut.result_post)
    matmul_result = A.astype(np.int64) @ B.astype(np.int64)
    expected = matmul_result.copy()
    for j in range(4):
        expected[:, j] += bias_pos[j]
    np.testing.assert_array_equal(got, expected)

    # Run 2: Bias + ReLU
    await run_pipeline(dut, A, B, bias=bias_pos, enable_bias=True,
                      enable_relu=True, enable_requant=False)
    got = unpack_int32(dut.result_post)
    expected_with_bias = matmul_result.copy()
    for j in range(4):
        expected_with_bias[:, j] += bias_pos[j]
    expected = np.maximum(expected_with_bias, 0)
    np.testing.assert_array_equal(got, expected)

    # Run 3: No bias, only ReLU (should zero out the negative values)
    await run_pipeline(dut, A, B, enable_bias=False, enable_relu=True,
                      enable_requant=False)
    got = unpack_int32(dut.result_post)
    expected = np.maximum(matmul_result, 0)
    np.testing.assert_array_equal(got, expected)

    # Run 4: All disabled (raw matmul)
    await run_pipeline(dut, A, B, enable_bias=False, enable_relu=False,
                      enable_requant=False)
    got = unpack_int32(dut.result_acc)
    expected = matmul_result
    np.testing.assert_array_equal(got, expected)


@cocotb.test()
async def test_reset_during_pipeline(dut):
    """Test reset asserted during pipeline execution.

    Tests that reset during the matmul phase properly halts and clears
    the pipeline, allowing clean restart.
    """
    await setup(dut)

    A = np.ones((4, 4), dtype=np.int8) * 10
    B = np.ones((4, 4), dtype=np.int8) * 10

    # Start pipeline
    dut.a_data.value = pack_matrix(A)
    dut.b_data.value = pack_matrix(B)
    dut.enable_bias.value = 0
    dut.enable_relu.value = 0
    dut.enable_requant.value = 0
    dut.shift_amount.value = 0
    dut.start.value = 1
    await tick(dut)
    dut.start.value = 0

    # Wait a few cycles into computation
    for _ in range(5):
        await tick(dut)

    # Assert reset mid-computation
    dut.rst_n.value = 0
    await tick(dut)
    dut.rst_n.value = 1
    await tick(dut)

    # Verify clean state
    assert int(dut.state_out.value) == 0, "Should be in IDLE after reset"
    assert int(dut.done.value) == 0, "done should be 0 after reset"

    # Run a fresh pipeline operation to verify functionality
    A2 = np.eye(4, dtype=np.int8)
    B2 = np.eye(4, dtype=np.int8) * 7
    await run_pipeline(dut, A2, B2)
    got = unpack_int32(dut.result_acc)
    expected = np.eye(4, dtype=np.int64) * 7
    np.testing.assert_array_equal(got, expected)


@cocotb.test()
async def test_requantize_shift_range(dut):
    """Test requantize with various shift amounts in the inference pipeline.

    Previously only tested shift=0 and shift=4 in the pipeline context.
    This test exercises higher shift values.
    """
    await setup(dut)

    # Create inputs that produce large accumulators
    A = np.ones((4, 4), dtype=np.int8) * 100
    B = np.ones((4, 4), dtype=np.int8) * 100

    # Test various shifts
    for shift in [0, 4, 8, 12, 16]:
        await run_pipeline(dut, A, B, enable_requant=True, shift=shift)
        got = unpack_result_int8(dut)

        # Software reference
        acc = A.astype(np.int32) @ B.astype(np.int32)  # Each element = 4 * 100 * 100 = 40000
        shifted = (acc >> shift).astype(np.int32)
        expected = np.clip(shifted, -128, 127).astype(np.int32)

        np.testing.assert_array_equal(got, expected,
            err_msg=f"Mismatch at shift={shift}")


@cocotb.test()
async def test_pipeline_with_overflow_clamping(dut):
    """Test that requantize properly clamps values that overflow int8 range.

    Uses inputs designed to produce accumulators that exceed [-128, 127]
    even after shifting, verifying saturation logic.
    """
    await setup(dut)

    # Maximum values: 127 * 127 * 4 = 64516 per element
    A = np.ones((4, 4), dtype=np.int8) * 127
    B = np.ones((4, 4), dtype=np.int8) * 127

    # Shift by 8: 64516 >> 8 = 252, should clamp to 127
    await run_pipeline(dut, A, B, enable_requant=True, shift=8)
    got = unpack_result_int8(dut)
    expected = np.ones((4, 4), dtype=np.int32) * 127
    np.testing.assert_array_equal(got, expected)

    # Minimum values: (-128) * (-128) * 4 = 65536 (positive overflow)
    A = np.ones((4, 4), dtype=np.int8) * (-128)
    B = np.ones((4, 4), dtype=np.int8) * (-128)

    # Shift by 8: 65536 >> 8 = 256, should clamp to 127
    await run_pipeline(dut, A, B, enable_requant=True, shift=8)
    got = unpack_result_int8(dut)
    expected = np.ones((4, 4), dtype=np.int32) * 127
    np.testing.assert_array_equal(got, expected)

    # Mixed signs: large negative result
    A = np.ones((4, 4), dtype=np.int8) * 127
    B = np.ones((4, 4), dtype=np.int8) * (-128)

    # Shift by 8: -65024 >> 8 = -254, should clamp to -128
    await run_pipeline(dut, A, B, enable_requant=True, shift=8)
    got = unpack_result_int8(dut)
    expected = np.ones((4, 4), dtype=np.int32) * (-128)
    np.testing.assert_array_equal(got, expected)
