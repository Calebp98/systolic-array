"""Cocotb tests for the tiling driver — arbitrary-sized matrix multiplication
on the 4x4 systolic array hardware.
"""

import cocotb
from cocotb.clock import Clock
import numpy as np
from tiling_driver import TilingDriver, reset


def ref_matmul(A, B):
    """Numpy reference matmul with int32 accumulation."""
    return (A.astype(np.int32) @ B.astype(np.int32))


async def setup(dut):
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset(dut)
    return TilingDriver(dut)


@cocotb.test()
async def test_4x4_no_tiling(dut):
    """4x4 multiply — single tile, no tiling needed."""
    driver = await setup(dut)
    A = np.array([[1, 2, 3, 4], [5, 6, 7, 8],
                  [9, 10, 11, 12], [13, 14, 15, 16]], dtype=np.int8)
    B = np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                  [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.int8)
    got = await driver.matmul(A, B)
    expected = ref_matmul(A, B)
    np.testing.assert_array_equal(got, expected)


@cocotb.test()
async def test_8x8_multi_tile(dut):
    """8x8 multiply — requires 2x2x2 = 8 hardware matmuls."""
    driver = await setup(dut)
    rng = np.random.RandomState(42)
    A = rng.randint(-10, 10, (8, 8)).astype(np.int8)
    B = rng.randint(-10, 10, (8, 8)).astype(np.int8)
    got = await driver.matmul(A, B)
    expected = ref_matmul(A, B)
    np.testing.assert_array_equal(got, expected)


@cocotb.test()
async def test_5x5_non_aligned(dut):
    """5x5 multiply — non-aligned, requires padding."""
    driver = await setup(dut)
    rng = np.random.RandomState(123)
    A = rng.randint(-5, 5, (5, 5)).astype(np.int8)
    B = rng.randint(-5, 5, (5, 5)).astype(np.int8)
    got = await driver.matmul(A, B)
    expected = ref_matmul(A, B)
    np.testing.assert_array_equal(got, expected)


@cocotb.test()
async def test_non_square(dut):
    """8x4 @ 4x8 — non-square matrices."""
    driver = await setup(dut)
    rng = np.random.RandomState(7)
    A = rng.randint(-10, 10, (8, 4)).astype(np.int8)
    B = rng.randint(-10, 10, (4, 8)).astype(np.int8)
    got = await driver.matmul(A, B)
    expected = ref_matmul(A, B)
    np.testing.assert_array_equal(got, expected)


@cocotb.test()
async def test_16x16_many_tiles(dut):
    """16x16 multiply — many tiles."""
    driver = await setup(dut)
    rng = np.random.RandomState(99)
    A = rng.randint(-5, 5, (16, 16)).astype(np.int8)
    B = rng.randint(-5, 5, (16, 16)).astype(np.int8)
    got = await driver.matmul(A, B)
    expected = ref_matmul(A, B)
    np.testing.assert_array_equal(got, expected)


@cocotb.test()
async def test_signed_values(dut):
    """Signed values at extremes."""
    driver = await setup(dut)
    A = np.array([[127, -128, 1, -1], [0, 50, -50, 100]], dtype=np.int8)
    B = np.array([[1, 0], [-1, 0], [2, -2], [3, 3]], dtype=np.int8)
    got = await driver.matmul(A, B)
    expected = ref_matmul(A, B)
    np.testing.assert_array_equal(got, expected)


@cocotb.test()
async def test_wide_inner_dim(dut):
    """1x16 @ 16x1 — long inner dimension, single output element."""
    driver = await setup(dut)
    A = np.ones((1, 16), dtype=np.int8)
    B = np.arange(1, 17, dtype=np.int8).reshape(16, 1)
    got = await driver.matmul(A, B)
    expected = ref_matmul(A, B)
    np.testing.assert_array_equal(got, expected)
    assert got[0, 0] == 136  # sum(1..16) = 136
