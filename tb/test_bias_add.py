"""Cocotb tests for the combinational bias_add module."""

import cocotb
from cocotb.triggers import Timer

DATA_WIDTH = 32
N_ELEM = 4


def pack_values(vals):
    packed = 0
    mask = (1 << DATA_WIDTH) - 1
    for i, v in enumerate(vals):
        packed |= (v & mask) << (i * DATA_WIDTH)
    return packed


def unpack_values(raw, n=N_ELEM, width=DATA_WIDTH):
    mask = (1 << width) - 1
    vals = []
    for i in range(n):
        v = (raw >> (i * width)) & mask
        if v >= (1 << (width - 1)):
            v -= 1 << width
        vals.append(v)
    return vals


async def check_bias_add(dut, data, bias, expected):
    dut.data_in.value = pack_values(data)
    dut.bias.value = pack_values(bias)
    await Timer(1, units="ns")
    got = unpack_values(int(dut.data_out.value))
    for i in range(N_ELEM):
        assert got[i] == expected[i], f"bias_add[{i}]: got {got[i]}, expected {expected[i]}"


@cocotb.test()
async def test_positive_bias(dut):
    await check_bias_add(dut, [10, 20, 30, 40], [1, 2, 3, 4], [11, 22, 33, 44])


@cocotb.test()
async def test_negative_bias(dut):
    await check_bias_add(dut, [10, 20, 30, 40], [-1, -2, -3, -4], [9, 18, 27, 36])


@cocotb.test()
async def test_zero_bias(dut):
    await check_bias_add(dut, [100, -200, 0, 42], [0, 0, 0, 0], [100, -200, 0, 42])


@cocotb.test()
async def test_negative_data(dut):
    await check_bias_add(dut, [-10, -20, -30, -40], [10, 20, 30, 40], [0, 0, 0, 0])


@cocotb.test()
async def test_large_values(dut):
    await check_bias_add(dut, [1000000, -1000000, 0, 500000],
                         [500000, 500000, -500000, -500000],
                         [1500000, -500000, -500000, 0])
