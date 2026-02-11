"""Tiling driver for arbitrary-sized matrix multiplication on a 4x4 systolic array.

Splits (M x K) @ (K x N) into 4x4 tiles, accumulates partial products via
hardware matmuls, and assembles the full result.
"""

import numpy as np
from cocotb.triggers import RisingEdge, FallingEdge, ClockCycles

TILE = 4
DATA_WIDTH = 8
ACC_WIDTH = 32


def pack_matrix(mat):
    """Pack a 4x4 matrix of int8 values into an integer for the DUT port."""
    val = 0
    for i in range(TILE):
        for j in range(TILE):
            byte = int(mat[i, j]) & 0xFF
            val |= byte << ((i * TILE + j) * DATA_WIDTH)
    return val


def unpack_result(dut):
    """Read the 4x4 result matrix from the DUT as signed 32-bit integers."""
    raw = int(dut.result.value)
    C = np.zeros((TILE, TILE), dtype=np.int64)
    mask32 = (1 << ACC_WIDTH) - 1
    for i in range(TILE):
        for j in range(TILE):
            idx = i * TILE + j
            v = (raw >> (idx * ACC_WIDTH)) & mask32
            if v >= (1 << (ACC_WIDTH - 1)):
                v -= 1 << ACC_WIDTH
            C[i, j] = v
    return C


async def tick(dut):
    """Advance one clock cycle and settle."""
    await RisingEdge(dut.clk)
    await FallingEdge(dut.clk)


async def reset(dut):
    """Reset the systolic array."""
    dut.rst_n.value = 0
    dut.start.value = 0
    dut.a_data.value = 0
    dut.b_data.value = 0
    await ClockCycles(dut.clk, 3)
    dut.rst_n.value = 1
    await tick(dut)


async def run_multiply_4x4(dut, A_tile, B_tile):
    """Run a single 4x4 hardware multiply. Returns int32 result as numpy array."""
    dut.a_data.value = pack_matrix(A_tile)
    dut.b_data.value = pack_matrix(B_tile)

    dut.start.value = 1
    await tick(dut)
    dut.start.value = 0

    for _ in range(30):
        await tick(dut)
        if int(dut.state_out.value) == 0:
            break
    else:
        raise RuntimeError("Timed out waiting for multiply to complete")

    return unpack_result(dut)


class TilingDriver:
    """Drives arbitrary-sized matrix multiplications on a 4x4 systolic array."""

    def __init__(self, dut):
        self.dut = dut

    async def multiply_tile(self, A_tile, B_tile):
        """Run a single 4x4 hardware multiply. Subclasses can override."""
        return await run_multiply_4x4(self.dut, A_tile, B_tile)

    async def matmul(self, A, B):
        """Multiply A (M x K) by B (K x N) using tiled 4x4 hardware matmuls.

        Args:
            A: numpy array of int8, shape (M, K)
            B: numpy array of int8, shape (K, N)

        Returns:
            numpy array of int32, shape (M, N)
        """
        A = np.asarray(A, dtype=np.int8)
        B = np.asarray(B, dtype=np.int8)
        M, K = A.shape
        K2, N = B.shape
        assert K == K2, f"Inner dimensions must match: {K} != {K2}"

        result = np.zeros((M, N), dtype=np.int64)

        # Tile over output rows and columns
        for i_start in range(0, M, TILE):
            for j_start in range(0, N, TILE):
                acc = np.zeros((TILE, TILE), dtype=np.int64)

                # Accumulate over K tiles
                for k_start in range(0, K, TILE):
                    # Extract tiles with zero-padding for edge cases
                    A_tile = np.zeros((TILE, TILE), dtype=np.int8)
                    B_tile = np.zeros((TILE, TILE), dtype=np.int8)

                    i_end = min(i_start + TILE, M)
                    k_end = min(k_start + TILE, K)
                    j_end = min(j_start + TILE, N)

                    A_tile[:i_end - i_start, :k_end - k_start] = \
                        A[i_start:i_end, k_start:k_end]
                    B_tile[:k_end - k_start, :j_end - j_start] = \
                        B[k_start:k_end, j_start:j_end]

                    hw_result = await self.multiply_tile(A_tile, B_tile)
                    acc += hw_result

                # Write accumulated result back (only valid portion)
                i_end = min(i_start + TILE, M)
                j_end = min(j_start + TILE, N)
                result[i_start:i_end, j_start:j_end] = \
                    acc[:i_end - i_start, :j_end - j_start]

        return result.astype(np.int32)
