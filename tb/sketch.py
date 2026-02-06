"""Scratch pad for one-off systolic array experiments.

Run with:  make sketch
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, ClockCycles

N = 4
DATA_WIDTH = 8
ACC_WIDTH = 32


def matmul(A, B):
    C = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            for k in range(N):
                C[i][j] += A[i][k] * B[k][j]
    return C


def pack_matrix(mat):
    val = 0
    for i in range(N):
        for j in range(N):
            val |= (mat[i][j] & 0xFF) << ((i * N + j) * DATA_WIDTH)
    return val


def unpack_result(dut):
    raw = int(dut.result.value)
    C = [[0] * N for _ in range(N)]
    mask = (1 << ACC_WIDTH) - 1
    for i in range(N):
        for j in range(N):
            v = (raw >> ((i * N + j) * ACC_WIDTH)) & mask
            if v >= (1 << (ACC_WIDTH - 1)):
                v -= 1 << ACC_WIDTH
            C[i][j] = v
    return C


def print_matrix(label, M):
    print(f"  {label}:")
    for row in M:
        print("    [" + " ".join(f"{v:6d}" for v in row) + " ]")


async def tick(dut):
    await RisingEdge(dut.clk)
    await FallingEdge(dut.clk)


async def reset(dut):
    dut.rst_n.value = 0
    dut.start.value = 0
    dut.a_data.value = 0
    dut.b_data.value = 0
    await ClockCycles(dut.clk, 3)
    dut.rst_n.value = 1
    await tick(dut)


async def run_multiply(dut, A, B):
    dut.a_data.value = pack_matrix(A)
    dut.b_data.value = pack_matrix(B)
    dut.start.value = 1
    await tick(dut)
    dut.start.value = 0
    for _ in range(30):
        await tick(dut)
        if int(dut.state_out.value) == 0:
            break


# ── Put your experiments below ──────────────────────────────────────


@cocotb.test()
async def sketch(dut):
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset(dut)

    # Define whatever matrices you want to try
    A = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
    ]
    B = [
        [1, 2, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 5],
        [3, 0, 0, 1],
    ]

    await run_multiply(dut, A, B)

    hw = unpack_result(dut)
    sw = matmul(A, B)

    print()
    print_matrix("A", A)
    print_matrix("B", B)
    print_matrix("Hardware C", hw)
    print_matrix("Expected C", sw)

    match = hw == sw
    print(f"\n  Match: {match}")
    assert match, "Hardware result != software reference"
