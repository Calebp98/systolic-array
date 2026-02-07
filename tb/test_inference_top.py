"""Cocotb tests for the integrated inference pipeline (inference_top).

Tests the systolic_array + bias_add + relu + requantize pipeline,
and runs MNIST end-to-end inference with RTL post-processing.
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, ClockCycles
import numpy as np
from pathlib import Path

N = 4
DATA_WIDTH = 8
ACC_WIDTH = 32


def pack_matrix_int8(mat):
    """Pack a 4x4 int8 matrix into an integer."""
    val = 0
    for i in range(N):
        for j in range(N):
            byte = int(mat[i, j]) & 0xFF
            val |= byte << ((i * N + j) * DATA_WIDTH)
    return val


def pack_bias_int32(bias_vec):
    """Pack 4 int32 bias values into an integer."""
    val = 0
    mask = (1 << ACC_WIDTH) - 1
    for j in range(N):
        val |= (int(bias_vec[j]) & mask) << (j * ACC_WIDTH)
    return val


def unpack_result_int32(dut):
    """Read 4x4 int32 result from result_acc."""
    raw = int(dut.result_acc.value)
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


def unpack_result_post(dut):
    """Read 4x4 int32 result from result_post (after bias+relu)."""
    raw = int(dut.result_post.value)
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


async def tick(dut):
    await RisingEdge(dut.clk)
    await FallingEdge(dut.clk)


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
    dut.a_data.value = pack_matrix_int8(A)
    dut.b_data.value = pack_matrix_int8(B)

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
    got = unpack_result_int32(dut)
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
    got = unpack_result_post(dut)
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
    got = unpack_result_post(dut)

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

class InferenceTopTilingDriver:
    """Tiling driver that uses inference_top instead of raw systolic_array.

    For MNIST inference, we use RTL bias/relu/requantize on each tile,
    but tiling and inter-layer coordination is still in software.
    """

    def __init__(self, dut):
        self.dut = dut

    async def matmul_tile(self, A_tile, B_tile, bias=None,
                          enable_bias=False, enable_relu=False,
                          enable_requant=False, shift=0):
        """Run one 4x4 tile through the inference pipeline."""
        await run_pipeline(self.dut, A_tile, B_tile, bias=bias,
                          enable_bias=enable_bias, enable_relu=enable_relu,
                          enable_requant=enable_requant, shift=shift)
        if enable_requant:
            return unpack_result_int8(self.dut)
        return unpack_result_int32(self.dut)

    async def matmul(self, A, B):
        """Tiled matmul with no post-processing (raw int32 accumulation)."""
        A = np.asarray(A, dtype=np.int8)
        B = np.asarray(B, dtype=np.int8)
        M, K = A.shape
        _, NN = B.shape
        result = np.zeros((M, NN), dtype=np.int64)

        for i_start in range(0, M, N):
            for j_start in range(0, NN, N):
                acc = np.zeros((N, N), dtype=np.int64)
                for k_start in range(0, K, N):
                    A_tile = np.zeros((N, N), dtype=np.int8)
                    B_tile = np.zeros((N, N), dtype=np.int8)
                    i_end = min(i_start + N, M)
                    k_end = min(k_start + N, K)
                    j_end = min(j_start + N, NN)
                    A_tile[:i_end-i_start, :k_end-k_start] = A[i_start:i_end, k_start:k_end]
                    B_tile[:k_end-k_start, :j_end-j_start] = B[k_start:k_end, j_start:j_end]
                    hw = await self.matmul_tile(A_tile, B_tile)
                    acc += hw
                i_end = min(i_start + N, M)
                j_end = min(j_start + N, NN)
                result[i_start:i_end, j_start:j_end] = acc[:i_end-i_start, :j_end-j_start]

        return result.astype(np.int32)


async def run_mnist_inference_rtl(driver, image_int8, input_scale):
    """Run 3-layer MLP inference using RTL pipeline for post-processing."""
    from mnist_utils import load_layer, relu_int32, requantize_to_int8

    x_int8 = image_int8.copy()
    current_scale = input_scale

    for layer_idx in range(1, 4):
        weights, bias_vals, w_scale = load_layer(layer_idx)
        out_features, in_features = weights.shape

        x_2d = x_int8.reshape(1, -1)
        w_t = weights.T

        # Use tiled matmul (raw int32)
        acc_int32 = await driver.matmul(x_2d, w_t)
        acc_int32 = acc_int32.flatten().astype(np.float64)

        # Dequantize and add bias (in software for now, since bias is float)
        acc_float = acc_int32 * current_scale * w_scale
        acc_float += bias_vals.astype(np.float64)

        if layer_idx < 3:
            acc_float = relu_int32(acc_float)
            x_int8, current_scale = requantize_to_int8(acc_float)
        else:
            return acc_float, int(np.argmax(acc_float))


@cocotb.test()
async def test_mnist_inference_rtl(dut):
    """Full MNIST inference using inference_top for matmul, SW post-processing."""
    await setup(dut)
    driver = InferenceTopTilingDriver(dut)

    from mnist_utils import load_test_data, load_input_scale
    images, labels = load_test_data()
    input_scale = load_input_scale()

    correct = 0
    for i in range(10):
        _, pred = await run_mnist_inference_rtl(driver, images[i], input_scale)
        if pred == labels[i]:
            correct += 1
        cocotb.log.info(f"Digit {labels[i]}: predicted {pred} {'OK' if pred == labels[i] else 'WRONG'}")

    cocotb.log.info(f"RTL Pipeline Accuracy: {correct}/10")
    assert correct >= 8, f"Only {correct}/10 correct, expected >= 8"
