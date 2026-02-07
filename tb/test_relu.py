"""Cocotb tests for the combinational ReLU module."""

import cocotb
from cocotb.triggers import Timer

DATA_WIDTH = 32
N_ELEM = 4


def pack_values(vals):
    """Pack N_ELEM int32 values into a single integer."""
    packed = 0
    mask = (1 << DATA_WIDTH) - 1
    for i, v in enumerate(vals):
        packed |= (v & mask) << (i * DATA_WIDTH)
    return packed


def unpack_values(raw, n=N_ELEM, width=DATA_WIDTH):
    """Unpack N_ELEM signed values from a single integer."""
    mask = (1 << width) - 1
    vals = []
    for i in range(n):
        v = (raw >> (i * width)) & mask
        if v >= (1 << (width - 1)):
            v -= 1 << width
        vals.append(v)
    return vals


async def check_relu(dut, inputs, expected):
    dut.data_in.value = pack_values(inputs)
    await Timer(1, units="ns")
    got = unpack_values(int(dut.data_out.value))
    for i in range(N_ELEM):
        assert got[i] == expected[i], f"ReLU[{i}]: got {got[i]}, expected {expected[i]}"


@cocotb.test()
async def test_all_positive(dut):
    await check_relu(dut, [1, 2, 3, 4], [1, 2, 3, 4])


@cocotb.test()
async def test_all_negative(dut):
    await check_relu(dut, [-1, -2, -100, -128], [0, 0, 0, 0])


@cocotb.test()
async def test_mixed(dut):
    await check_relu(dut, [10, -5, 0, -1000], [10, 0, 0, 0])


@cocotb.test()
async def test_zero(dut):
    await check_relu(dut, [0, 0, 0, 0], [0, 0, 0, 0])


@cocotb.test()
async def test_edge_values(dut):
    int32_max = 2**31 - 1
    int32_min = -(2**31)
    await check_relu(dut, [int32_max, int32_min, 1, -1], [int32_max, 0, 1, 0])
