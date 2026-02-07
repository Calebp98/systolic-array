"""Cocotb tests for the requantize module (int32 >> shift → clamped int8)."""

import cocotb
from cocotb.triggers import Timer

IN_WIDTH = 32
OUT_WIDTH = 8
N_ELEM = 4


def pack_int32s(vals):
    packed = 0
    mask = (1 << IN_WIDTH) - 1
    for i, v in enumerate(vals):
        packed |= (v & mask) << (i * IN_WIDTH)
    return packed


def unpack_int8s(raw, n=N_ELEM):
    mask = (1 << OUT_WIDTH) - 1
    vals = []
    for i in range(n):
        v = (raw >> (i * OUT_WIDTH)) & mask
        if v >= (1 << (OUT_WIDTH - 1)):
            v -= 1 << OUT_WIDTH
        vals.append(v)
    return vals


def sw_requantize(val, shift):
    """Software reference: arithmetic right shift then clamp to [-128, 127]."""
    shifted = val >> shift  # Python >> is arithmetic for negative ints
    return max(-128, min(127, shifted))


async def check_requantize(dut, inputs, shift, expected):
    dut.data_in.value = pack_int32s(inputs)
    dut.shift_amount.value = shift
    await Timer(1, units="ns")
    got = unpack_int8s(int(dut.data_out.value))
    for i in range(N_ELEM):
        assert got[i] == expected[i], \
            f"requantize[{i}]: input={inputs[i]}, shift={shift}, got {got[i]}, expected {expected[i]}"


@cocotb.test()
async def test_shift_8(dut):
    inputs = [256, 512, -256, 0]
    shift = 8
    expected = [sw_requantize(v, shift) for v in inputs]
    await check_requantize(dut, inputs, shift, expected)


@cocotb.test()
async def test_shift_0(dut):
    inputs = [50, -50, 127, -128]
    expected = [50, -50, 127, -128]
    await check_requantize(dut, inputs, 0, expected)


@cocotb.test()
async def test_clamp_positive(dut):
    inputs = [50000, 200, 128, 1000]
    shift = 0
    expected = [127, 127, 127, 127]
    await check_requantize(dut, inputs, shift, expected)


@cocotb.test()
async def test_clamp_negative(dut):
    inputs = [-50000, -200, -129, -1000]
    shift = 0
    expected = [-128, -128, -128, -128]
    await check_requantize(dut, inputs, shift, expected)


@cocotb.test()
async def test_various_shifts(dut):
    for shift in [1, 4, 8, 12, 16]:
        inputs = [1024, -1024, 32768, -32768]
        expected = [sw_requantize(v, shift) for v in inputs]
        await check_requantize(dut, inputs, shift, expected)


@cocotb.test()
async def test_matches_software(dut):
    """Compare against software reference for a range of values."""
    import random
    rng = random.Random(42)
    for shift in [0, 4, 8, 12]:
        inputs = [rng.randint(-100000, 100000) for _ in range(N_ELEM)]
        expected = [sw_requantize(v, shift) for v in inputs]
        await check_requantize(dut, inputs, shift, expected)
